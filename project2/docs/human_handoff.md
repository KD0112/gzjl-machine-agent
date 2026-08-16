# 人工客服接管设计与验收

## 审批与接管的区别

工具审批表示 Agent 已经知道要调用哪个工具，只等待人授权。

人工接管表示 Agent 无法可靠完成问题，需要真人客服直接处理并回复客户。

## 服务单状态

```text
queued    待领取
claimed   处理中
resolved  已完成
```

每个服务单记录 `handoff_id`、`thread_id`、原因、优先级、渠道、客户标识、上下文、负责人和人工回复。

## 上下文包

- 客户原始问题。
- 意图、槽位、置信度和解析来源。
- 已调用、跳过和未接入工具。
- 工具参数、结果和错误。
- 审批记录。
- 执行轨迹。
- Agent 建议回复。
- 渠道和客户标识。

人工客服不需要让客户重复描述已有信息。

## Outbox

网页渠道的人工回复直接显示在调试台。

微信等异步渠道的人工回复写入 `outbox_messages`：

```text
pending -> delivered
```

当前项目只实现可靠入队和幂等去重。真实发送仍需要微信公众号或企业微信的客服消息 API、access token、错误重试和回执处理，未配置凭证时不会标记为 delivered。

## 网页验收

1. 打开 Agent 调试台，选择 LangGraph。
2. 打开人工客服接管。
3. 输入“我要找人工客服确认PC200液压泵”。
4. 状态应为 `waiting_human`，页面显示 `handoff_id`。
5. 左侧切换“人工客服工作台”。
6. 在待领取列表选择服务单，查看转接原因和 Agent 上下文。
7. 输入客服名称并领取。
8. 输入真实回复，点击“回复客户并完成”。
9. 服务单变为 `resolved`，图状态变为 `completed`。
10. 执行轨迹包含 `evaluate_handoff`、`create_handoff`、`human_response` 和 `build_response`。

## 自动验收

```powershell
& "..\.venv\Scripts\python.exe" -m unittest tests.test_handoff_runtime -v
& "..\.venv\Scripts\python.exe" tests\evaluate_handoffs.py
```

预期：

```text
Ran 6 tests
OK

Total: 9
Passed: 9
Pass rate: 100.0%
```
