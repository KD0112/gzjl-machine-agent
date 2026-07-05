# 微信真实接入 Checklist

## 结论：现在不一定需要企业微信

当前项目最适合先接“微信公众号测试号”或“微信公众号开发者配置”，不需要先建立企业微信。

三种路线对比：

| 路线 | 是否推荐现在做 | 适合场景 | 说明 |
| --- | --- | --- | --- |
| 微信公众号测试号 | 推荐 | 本地/面试/演示验证 | 最快验证 webhook，适合当前项目 |
| 正式微信公众号 | 后续推荐 | 给外部客户使用 | 需要公众号主体和服务器配置 |
| 企业微信 | 暂时不必 | 公司内部员工、客户群、企业应用 | 更偏企业内部系统，回调和权限更复杂 |

如果你只是想证明“微信聊天助手可以接入 Agent”，先用微信公众号测试号最合适。

## 本地已经完成的部分

已经有：

- `wechat_server.py`：本地微信 webhook 服务。
- `tests/simulate_wechat.py`：模拟微信服务器请求。
- `start_wechat_server.ps1`：启动本地 webhook。
- `start_wechat_tunnel.ps1`：启动本地公网隧道。

本地 webhook 地址：

```text
http://127.0.0.1:8510
```

但这个地址只能本机访问，不能直接填到微信后台。

## 真实连接微信需要什么

你需要准备：

1. 一个微信公众号测试号或正式公众号。
2. 一个公网 HTTPS URL，指向本地 `8510` 或云端服务。
3. 一个 Token，例如：

```text
project2-agent-token
```

微信后台填写：

```text
URL:   https://你的公网地址
Token: project2-agent-token
```

EncodingAESKey 第一版可以先用明文模式或平台默认配置，后续如果启用安全模式，需要再做消息加解密。

## 步骤 1：启动本地微信服务

```powershell
cd "D:\new things\项目1\day1\project2"
.\start_wechat_server.ps1 -Port 8510 -Token project2-agent-token -Mode graph
```

如果 PowerShell 提示禁止运行脚本，先临时执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 步骤 2：启动公网隧道

如果已安装 Cloudflare Tunnel 或 ngrok：

```powershell
cd "D:\new things\项目1\day1\project2"
.\start_wechat_tunnel.ps1 -Port 8510
```

脚本会优先使用 `cloudflared`，找不到再尝试 `ngrok`。

启动后会看到一个公网 HTTPS 地址，例如：

```text
https://xxxx.trycloudflare.com
```

或：

```text
https://xxxx.ngrok-free.app
```

把这个 HTTPS 地址填到微信后台的 URL。

## 步骤 3：微信后台配置

在公众号测试号或公众号后台的服务器配置里填写：

```text
URL:   https://xxxx.trycloudflare.com
Token: project2-agent-token
```

提交时，微信会发 GET 请求验证签名。当前 `wechat_server.py` 已经支持这个验签流程。

## 步骤 4：发送微信消息验收

配置成功后，在测试号二维码里扫码关注，然后发送：

```text
小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？
```

预期回复包含：

- 贵阳仓库存
- 参考报价区间
- 贵阳物流时效
- 人工确认提示

再发送：

```text
订单号 A20260616001，买错了能不能退货？
```

预期回复包含：

- 售后工单号
- 问题类型：退货
- 补充照片或视频
- 人工客服核对提示

## 本地模拟命令

即使没有真实微信公众号，也可以先模拟：

```powershell
cd "D:\new things\项目1\day1\project2"
& "..\.venv\Scripts\python.exe" tests\simulate_wechat.py --url http://127.0.0.1:8510 --token project2-agent-token
```

## 后续如果选择企业微信

企业微信不是不能做，但它通常涉及：

- 企业微信应用
- 回调 URL
- Token
- EncodingAESKey
- 消息加解密
- 企业内部成员或客户联系权限

当前 `wechat_server.py` 做的是公众号风格的明文 XML webhook。若要正式接企业微信，需要新增企业微信消息加解密适配层，不建议作为当前第一步。
