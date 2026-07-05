from __future__ import annotations

import argparse
import hashlib
import time
import urllib.parse
import urllib.request


def make_signature(token: str, timestamp: str, nonce: str) -> str:
    raw = "".join(sorted([token, timestamp, nonce]))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def make_url(base_url: str, token: str, timestamp: str, nonce: str, echostr: str | None = None) -> str:
    query = {
        "signature": make_signature(token, timestamp, nonce),
        "timestamp": timestamp,
        "nonce": nonce,
    }
    if echostr is not None:
        query["echostr"] = echostr
    return f"{base_url}?{urllib.parse.urlencode(query)}"


def build_text_message(content: str) -> bytes:
    xml = f"""
<xml>
  <ToUserName><![CDATA[agent_server]]></ToUserName>
  <FromUserName><![CDATA[test_user]]></FromUserName>
  <CreateTime>{int(time.time())}</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[{content}]]></Content>
  <MsgId>1000000000000001</MsgId>
</xml>
""".strip()
    return xml.encode("utf-8")


def request_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read().decode("utf-8")


def post_xml(url: str, body: bytes) -> str:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/xml; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate WeChat GET verification and text POST.")
    parser.add_argument("--url", default="http://127.0.0.1:8510")
    parser.add_argument("--token", default="project2-agent-token")
    parser.add_argument(
        "--question",
        default="小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？",
    )
    args = parser.parse_args()

    timestamp = str(int(time.time()))
    nonce = "codex-local-test"
    verify_url = make_url(args.url, args.token, timestamp, nonce, echostr="wechat-ok")
    post_url = make_url(args.url, args.token, timestamp, nonce)

    print("GET verification response:")
    print(request_text(verify_url))
    print()
    print("POST text response:")
    print(post_xml(post_url, build_text_message(args.question)))


if __name__ == "__main__":
    main()
