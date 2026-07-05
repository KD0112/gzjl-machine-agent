# 项目二演示路径：多工具销售 Agent

## 演示目标

用 3-5 分钟讲清楚项目二不是普通聊天机器人，而是一个可以解析意图、校验参数、调用确定性工具、记录执行轨迹和运行日志的多工具 Agent。

## 启动调试台

```powershell
cd "D:\new things\项目1\day1\project2"
& "..\.venv\Scripts\streamlit.exe" run app.py --server.port 8503
```

打开：

```text
http://127.0.0.1:8503
```

## 演示 1：混合意图工具调用

问题：

```text
小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？
```

看点：

- `intents` 同时识别出 `inventory`、`quote`、`logistics`。
- `slots` 抽取出品牌、型号、配件、品质档位、数量、城市。
- `called_tools` 依次包含 `inventory_tool`、`quote_tool`、`logistics_tool`。
- `执行轨迹` 显示 `parse -> inventory_tool -> quote_tool -> logistics_tool -> build_response`。
- 客户回复中只展示库存、参考报价和物流时效，不暴露内部 JSON 字段。

## 演示 2：信息不足追问

问题：

```text
小松 PC200 的液压泵有没有现货？多少钱？发到贵阳要多久？
```

看点：

- 缺少 `quality_level` 和 `quantity`。
- 状态变为 `need_more_info`。
- 系统不会贸然调用库存或报价工具，而是追问原厂/副厂/经济型和数量。
- `执行轨迹` 显示 `parse -> guard_missing_fields -> build_response`，证明缺信息时工具被拦截。

## 演示 3：售后高风险动作只生成草稿

问题：

```text
订单号 A20260616001，买错了能不能退货？
```

看点：

- 识别 `after_sales` 意图并调用 `ticket_tool`。
- 生成售后工单草稿，提示补充照片或视频。
- 不直接承诺退货、退款或换货结论，必须人工确认。

## 调试台展示顺序

1. 先看“客户侧回复”，说明客户看到的是自然语言结果。
2. 再看“执行轨迹”，说明 Agent 每一步为什么继续、拦截或调用工具。
3. 再看“解析结果”，说明 Agent 如何理解意图和槽位。
4. 再看“工具参数”，说明工具调用前参数已结构化并由 Pydantic schema 校验。
5. 再看“工具结果”，说明库存、报价、物流来自确定性工具，不是模型编造。
6. 最后看“历史记录”，说明每一轮问题都能回放，用于 badcase 分析。

## 命令行评测

```powershell
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --mode workflow
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --mode graph
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --cases tests\agent_observability_cases.jsonl --mode workflow
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --cases tests\agent_observability_cases.jsonl --mode graph
```

当前目标结果：

```text
Total: 30
Passed: 30
Pass rate: 100.0%

Total: 5
Passed: 5
Pass rate: 100.0%
```

## 面试表达

项目一解决的是“基于企业知识库回答问题”，项目二解决的是“按业务流程调用工具办事”。我没有让模型直接编库存、价格或售后结论，而是把库存查询、报价草稿、物流估算和售后工单拆成确定性工具，并在 workflow 层做参数校验、缺失字段追问、工具调度、执行轨迹和日志记录。LangGraph 版本把 State、Node、Edge 显式化，方便后续扩展多轮状态管理和失败分支。
