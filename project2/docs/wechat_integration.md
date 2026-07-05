# 微信聊天助手接入说明

## 当前版本做了什么

当前已经新增一个微信 webhook 适配层：

```text
project2/wechat_server.py
```

它负责：

- 处理微信服务器 GET 验签。
- 处理微信 POST 文本消息 XML。
- 把用户文本交给项目二 Agent。
- 把 Agent 回复包装成微信被动回复文本 XML。
- 把本轮运行写入 `logs/agent_runs.csv`，方便后续回放。

第一版只支持文本消息。图片、语音、菜单事件、客服消息接口可以后续再做。

## 本地启动

```powershell
cd "D:\new things\项目1\day1\project2"
& "..\.venv\Scripts\python.exe" wechat_server.py --port 8510 --token project2-agent-token --mode graph
```

参数说明：

- `--port 8510`：本地 webhook 端口。
- `--token project2-agent-token`：微信后台服务器配置里的 Token，本地测试可先用这个。
- `--mode graph`：使用 LangGraph 版本 Agent。也可以用 `--mode workflow`。

## 本地模拟微信请求

另开一个终端运行：

```powershell
cd "D:\new things\项目1\day1\project2"
& "..\.venv\Scripts\python.exe" tests\simulate_wechat.py --url http://127.0.0.1:8510 --token project2-agent-token
```

预期结果：

- GET verification response 返回 `wechat-ok`。
- POST text response 返回一段 `<xml>...</xml>`，里面包含 Agent 的客服回复。

也可以传自定义问题：

```powershell
& "..\.venv\Scripts\python.exe" tests\simulate_wechat.py --question "订单号 A20260616001，买错了能不能退货？"
```

## 真实微信公众号接入需要什么

微信后台通常需要填写：

```text
URL:   https://你的公网域名或隧道地址
Token: project2-agent-token
```

注意：

- 微信服务器不能访问你的 `127.0.0.1`。
- 真正接入时必须有公网 HTTPS 地址。
- 临时测试可以用 Cloudflare Tunnel 或 ngrok 把本地 `8510` 暴露出去。
- 正式展示建议部署到云服务器、Render、Railway 等能提供公网 HTTP 服务的平台。

更完整的真实接入 checklist 见：

```text
docs/wechat_real_connection_checklist.md
```

当前这台电脑暂未检测到 `cloudflared` 或 `ngrok`。如果要立刻连真实微信，需要先安装其中一个隧道工具，或把 `wechat_server.py` 部署到公网服务器。

## 当前不建议做个人微信自动化

个人微信自动化通常依赖非官方协议或桌面端模拟，容易受版本、风控和登录状态影响，也不适合写进简历项目。

更推荐的路线：

1. 先做微信公众号或企业微信 webhook。
2. 后面需要客服能力时，再接官方客服消息接口或企业微信应用消息。
3. 保持 Agent 后端独立，让微信、网页、未来 API 都只是不同入口。

## 面试表达

可以这样讲：

> 我把微信助手设计成一个渠道适配层，而不是把业务逻辑写死在微信代码里。微信层只负责验签、解析 XML、回复 XML；真正的意图识别、工具调用和运行日志仍然复用项目二 Agent。这样后续如果换成网页、企业微信或 API，也不用重写核心业务流程。
