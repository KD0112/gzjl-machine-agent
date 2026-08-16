# 项目二演示路径：多工具销售 Agent

## 演示目标

用 10 分钟讲清楚项目二不是普通聊天机器人，而是一个能组合 LangChain、LangGraph、多模态证据、RAG、确定性工具、人工审批和人工客服接管的可恢复 Agent。

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

先切换到“LangGraph”，打开人工审批、人工客服接管和 RAG，解析模式选择“混合解析”。

问题：

```text
小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？
```

看点：

- `intents` 同时识别出 `inventory`、`quote`、`logistics`。
- `slots` 抽取出品牌、型号、配件、品质档位、数量、城市。
- 库存查询先自动执行，到了 `quote_tool` 后状态变为 `waiting_approval`。
- 页面显示报价参数、`thread_id` 和批准/修改/拒绝选项。
- 选择“批准”后，图从 SQLite checkpoint 恢复，继续执行报价和物流。
- `called_tools` 最终依次包含 `inventory_tool`、`quote_tool`、`logistics_tool`。
- `执行轨迹` 显示 `parse -> inventory_tool -> approval -> quote_tool -> logistics_tool -> build_response`。
- 客户回复中只展示库存、参考报价和物流时效，不暴露内部 JSON 字段。

## 演示 2：图片证据与客户确认

保持 LangGraph 模式，点击“新建会话”，上传：

```text
tests/fixtures/multimodal/synthetic_part_label.png
```

问题：

```text
请识别图片上的品牌、配件名称和零件号，先不要查询价格。
```

看点：

- 图片先通过 JPG/PNG/WebP 格式、大小、解码、像素和元数据清理。
- `inspect_image` 输出 `ImageInspectionResult`，页面进入 `waiting_image_confirmation`。
- 候选字段应包含 `KOMATSU`、`液压泵` 和 `708-2L-00300`。
- 点击“确认并继续”后从同一 checkpoint 恢复，状态变为 `completed`。
- 本轮只是图片字段提取，不调用价格工具、不转人工，也不承诺适配。
- “图片证据”页展示 Provider、模型、置信度、确认记录和已合并槽位，但不展示 API Key 或图片二进制。
- 再说明模糊图会拒绝猜零件号并要求重拍或转人工。

## 演示 3：上下文与多轮记忆

先关闭人工审批和人工接管，点击“新建会话”，依次输入：

```text
小松PC200原厂液压泵要1件，有没有现货？
```

```text
这个多少钱？
```

看点：

- 两轮 `thread_id` 相同，`turn_count` 从 1 变为 2。
- 第二轮从 checkpoint 继承机型、配件、品质和数量，只调用 `quote_tool`。
- `slot_sources` 显示字段来自 `conversation`，不是模型猜测。
- “上下文 / 记忆”展示近期消息、摘要、Token 预算和长期客户事实。
- 新建会话后，同一客户可读取白名单长期记忆；更换客户 ID 后不能读取。

## 演示 4：信息不足追问

问题：

```text
小松 PC200 的液压泵有没有现货？多少钱？发到贵阳要多久？
```

看点：

- 缺少 `quality_level` 和 `quantity`。
- 状态变为 `need_more_info`。
- 系统不会贸然调用库存或报价工具，而是追问原厂/副厂/经济型和数量。
- `执行轨迹` 显示 `parse -> guard_missing_fields -> build_response`，证明缺信息时工具被拦截。

## 演示 5：售后高风险动作只生成草稿

问题：

```text
订单号 A20260616001，买错了能不能退货？
```

看点：

- 识别 `after_sales` 意图，但在调用 `ticket_tool` 前暂停。
- 可以先选择“拒绝”，证明 `called_tools` 中不会出现 `ticket_tool`；再新开一轮选择“批准”。
- 生成售后工单草稿，提示补充照片或视频。
- 批准工具后图进入第二个暂停点 `waiting_human`，证明工具审批与客服接管不是一回事。
- 不直接承诺退货、退款或换货结论，必须由人工客服回复。

## 演示 6：LangChain + 项目一 RAG

问题：

```text
PC200液压泵能不能适配，应该核对什么信息？
```

看点：

- `parse_source` 显示规则或 LangChain 混合解析来源。
- `knowledge_tool` 复用项目一 Chroma、中文 Embedding、Top-K 和距离阈值。
- 工具结果显示来源、排名、距离和证据片段。
- 适配属于高风险问题，即使检索有结果，也会进入人工接管。

## 演示 7：人工客服工作台

1. 左侧切换“人工客服工作台”。
2. 查看服务单优先级、转接原因、工具结果和 Agent 建议回复。
3. 输入客服名称并领取。
4. 输入真实回复后点击“回复客户并完成”。
5. 服务单变为 `resolved`，原 LangGraph 线程变为 `completed`。

重点说明：Checkpoint 保存图状态；人工服务单 SQLite 保存业务队列；微信回复 outbox 保存待发送消息，三者职责不同。

## 演示 8：Checkpoint 和恢复

在“Checkpoint / 审批”标签查看：

- `StateSnapshot` 中保存的 `current_tool`、`tool_queue` 和审批请求。
- `next` 显示图下一步等待执行的节点。
- checkpoint 历史记录每一次 State 更新。
- 工具执行后生成的 `idempotency_key`。

命令行也可以先启动并退出进程：

```powershell
& "..\.venv\Scripts\python.exe" agent_graph.py "PC200原厂液压泵要1件，价格多少？" --thread-id demo-quote-001 --manual-approval
```

再用另一个进程恢复：

```powershell
& "..\.venv\Scripts\python.exe" agent_graph.py --thread-id demo-quote-001 --resume approve --comment "参数已核对"
```

## 调试台展示顺序

1. 先看“客户侧回复”，说明客户看到的是自然语言结果。
2. 再看“执行轨迹”，说明 Agent 每一步为什么继续、拦截或调用工具。
3. 再看“解析结果”，说明 Agent 如何理解意图和槽位。
4. 再看“工具参数”，说明工具调用前参数已结构化并由 Pydantic schema 校验。
5. 再看“工具结果”，说明库存、报价、物流来自确定性工具，不是模型编造。
6. 查看“图片证据”，说明视觉结果是受控候选证据并经过客户确认。
7. 查看“Checkpoint / 审批”，说明图能够暂停、持久化和恢复。
8. 切到人工客服工作台，展示队列、领取、上下文和人工回复。
9. 最后看“历史记录”，说明每一轮问题都能回放，用于 badcase 分析。

## 命令行评测

```powershell
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --mode workflow
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --mode graph
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --cases tests\agent_observability_cases.jsonl --mode workflow
& "..\.venv\Scripts\python.exe" tests\evaluate_agent.py --cases tests\agent_observability_cases.jsonl --mode graph
& "..\.venv\Scripts\python.exe" -m unittest tests.test_langgraph_runtime -v
& "..\.venv\Scripts\python.exe" -m unittest tests.test_context_memory -v
& "..\.venv\Scripts\python.exe" -m unittest tests.test_handoff_runtime -v
& "..\.venv\Scripts\python.exe" -m unittest tests.test_langchain_integration -v
& "..\.venv\Scripts\python.exe" -m unittest tests.test_multimodal_runtime -v
& "..\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
& "..\.venv\Scripts\python.exe" tests\evaluate_handoffs.py
& "..\.venv\Scripts\python.exe" tests\evaluate_multimodal.py
```

当前目标结果：

```text
Total: 30
Passed: 30
Pass rate: 100.0%

Total: 5
Passed: 5
Pass rate: 100.0%

Ran 6 tests
OK

Ran 7 tests
OK

Ran 6 tests
OK

Ran 7 tests
OK

Total: 9
Passed: 9
Pass rate: 100.0%

Ran 45 tests
OK

Multimodal: 4/4 synthetic API smoke cases
```

## 面试表达

项目二使用 LangGraph 负责显式状态、动态工具队列、checkpoint 和两类人工中断；上下文管理器按优先级和 Token 预算选择信息，checkpoint 保存线程内短期记忆，独立 SQLite 仓库保存受治理的跨线程客户事实；LangChain 在节点内负责模型接口、结构化语义解析和 StructuredTool；项目一 RAG 作为 knowledge_tool 复用。
