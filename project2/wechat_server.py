from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agent_graph import run_graph_agent
from agent_workflow import run_agent
from tool_call_logger import append_agent_run


DEFAULT_TOKEN = "project2-agent-token"
MAX_WECHAT_TEXT_BYTES = 1900
RUNNERS = {
    "workflow": run_agent,
    "graph": run_graph_agent,
}


def check_signature(token: str, signature: str, timestamp: str, nonce: str) -> bool:
    """Validate WeChat server signature."""
    if not all([token, signature, timestamp, nonce]):
        return False
    raw = "".join(sorted([token, timestamp, nonce]))
    expected = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return hmac.compare_digest(expected, signature)


def _cdata(value: Any) -> str:
    text = "" if value is None else str(value)
    return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _truncate_utf8(text: str, max_bytes: int = MAX_WECHAT_TEXT_BYTES) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    suffix = "\n\n回复较长，已自动截断；请联系人工客服继续确认。"
    budget = max_bytes - len(suffix.encode("utf-8"))
    if budget <= 0:
        return suffix
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def parse_wechat_xml(body: bytes) -> dict[str, str]:
    root = ET.fromstring(body)
    return {child.tag: child.text or "" for child in root}


def build_text_reply(to_user: str, from_user: str, content: str) -> bytes:
    reply = _truncate_utf8(content)
    xml = (
        "<xml>"
        f"<ToUserName>{_cdata(to_user)}</ToUserName>"
        f"<FromUserName>{_cdata(from_user)}</FromUserName>"
        f"<CreateTime>{int(time.time())}</CreateTime>"
        f"<MsgType>{_cdata('text')}</MsgType>"
        f"<Content>{_cdata(reply)}</Content>"
        "</xml>"
    )
    return xml.encode("utf-8")


def build_agent_reply(question: str, mode: str) -> tuple[str, dict[str, Any]]:
    runner = RUNNERS.get(mode, run_graph_agent)
    result = runner(question)
    append_agent_run(
        result=result,
        execution_mode=f"wechat_{mode}",
        tool_arguments=result.get("tool_arguments", {}),
    )
    return result["customer_reply"], result


class WeChatAgentHandler(BaseHTTPRequestHandler):
    server_version = "Project2WeChatAgent/0.1"

    @property
    def token(self) -> str:
        return getattr(self.server, "wechat_token", DEFAULT_TOKEN)

    @property
    def agent_mode(self) -> str:
        return getattr(self.server, "agent_mode", "graph")

    def _send_text(self, status_code: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_xml(self, status_code: int, body: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/xml; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> dict[str, str]:
        parsed = urllib.parse.urlparse(self.path)
        values = urllib.parse.parse_qs(parsed.query)
        return {key: value[0] for key, value in values.items() if value}

    def _signature_ok(self, query: dict[str, str]) -> bool:
        return check_signature(
            token=self.token,
            signature=query.get("signature", ""),
            timestamp=query.get("timestamp", ""),
            nonce=query.get("nonce", ""),
        )

    def do_GET(self) -> None:
        query = self._query()
        if self._signature_ok(query):
            self._send_text(200, query.get("echostr", ""))
            return
        self._send_text(403, "invalid signature")

    def do_POST(self) -> None:
        query = self._query()
        if not self._signature_ok(query):
            self._send_text(403, "invalid signature")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        try:
            message = parse_wechat_xml(body)
            msg_type = message.get("MsgType", "")
            from_user = message.get("FromUserName", "")
            to_user = message.get("ToUserName", "")

            if msg_type != "text":
                reply = "目前微信助手第一版先支持文本咨询。请直接发送配件型号、配件名称、数量、城市或订单号。"
            else:
                question = message.get("Content", "").strip()
                if not question:
                    reply = "请发送需要咨询的挖机配件问题，例如：小松PC200原厂液压泵要1件，有没有现货？"
                else:
                    reply, _ = build_agent_reply(question, self.agent_mode)

            self._send_xml(200, build_text_reply(from_user, to_user, reply))
        except Exception as exc:  # noqa: BLE001 - webhook must not leak stack traces to users.
            fallback = "抱歉，微信助手暂时没有处理成功，请转人工客服确认。"
            try:
                message = parse_wechat_xml(body)
                self._send_xml(
                    200,
                    build_text_reply(
                        to_user=message.get("FromUserName", ""),
                        from_user=message.get("ToUserName", ""),
                        content=fallback,
                    ),
                )
            except Exception:  # noqa: BLE001
                self._send_text(500, fallback)

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format, *args)


def run_server(host: str, port: int, token: str, mode: str, quiet: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), WeChatAgentHandler)
    server.wechat_token = token
    server.agent_mode = mode
    server.quiet = quiet
    print(f"WeChat Agent webhook listening on http://{host}:{port}")
    print(f"Token: {token}")
    print(f"Agent mode: {mode}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local WeChat webhook adapter for project2 Agent.")
    parser.add_argument("--host", default=os.getenv("WECHAT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WECHAT_PORT", "8510")))
    parser.add_argument("--token", default=os.getenv("WECHAT_TOKEN", DEFAULT_TOKEN))
    parser.add_argument("--mode", choices=sorted(RUNNERS), default=os.getenv("WECHAT_AGENT_MODE", "graph"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        token=args.token,
        mode=args.mode,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
