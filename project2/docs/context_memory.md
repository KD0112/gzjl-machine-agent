# 上下文管理与多轮记忆

## 三类数据的边界

| 数据 | 存储位置 | 作用范围 | 是否直接进入上下文 |
| --- | --- | --- | --- |
| 短期会话状态 | `logs/langgraph_checkpoints.sqlite3` | 同一 `thread_id` | 经过预算和压缩后进入 |
| 长期客户事实 | `logs/agent_memory.sqlite3` | 同一 `customer_id` 的不同线程 | 只加载有效白名单事实 |
| 执行与业务日志 | CSV、服务单数据库、checkpoint 历史 | 调试、评测、审计 | 不整段进入模型上下文 |

RAG 是企业知识，不属于客户记忆。工具结果也不是记忆，只在当前任务需要时进入受控上下文。

## 上下文优先级

1. 安全规则。
2. 当前客户问题。
3. 已确认槽位和有效客户记忆。
4. RAG 证据。
5. 当前工具结果。
6. 近期消息。
7. 较早对话摘要。

当前问题与历史值冲突时，当前问题优先，同时在
`parse_result.context_conflicts` 保留冲突记录。RAG、工具结果和历史消息全部标记为
不可信数据；疑似 Prompt Injection 会写入
`context_snapshot.injection_signals`。

## 自动验收

在 `project2` 目录执行：

```powershell
& "..\.venv\Scripts\python.exe" -m unittest tests.test_context_memory -v
```

预期结果：

```text
Ran 7 tests
OK
```

一次运行全部运行时和集成测试：

```powershell
& "..\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

预期结果：

```text
Ran 66 tests
OK
```

## 网页验收：同一线程多轮对话

1. 启动调试台，选择 `LangGraph`。
2. 关闭“人工审批演示”和“人工客服接管”，客户 ID 使用
   `demo-customer-001`。
3. 点击“新建会话”。
4. 第一轮输入：`小松PC200原厂液压泵要1件，有没有现货？`
5. 第二轮输入：`这个多少钱？`
6. 两轮页面显示的 `thread_id` 应相同。
7. 第二轮 `turn_count` 应为 `2`，`called_tools` 应只有 `quote_tool`。
8. 第二轮 `parse_result.slots` 应包含 `PC200`、`液压泵`、`原厂` 和数量 `1`。
9. 第二轮 `slot_sources` 中这些字段应显示 `conversation`。
10. “上下文 / 记忆”页签应显示四条近期消息：客户、Agent、客户、Agent。

这证明第二轮不是重新猜测，而是从同一个 SQLite checkpoint 恢复会话状态。

## 网页验收：跨线程长期记忆

1. 保持客户 ID 为 `demo-customer-001`，点击“新建会话”。
2. 输入：`PC200液压泵要1件，多少钱？`
3. 如果前一会话已经确认过“原厂”，本轮应补齐
   `quality_level=原厂`，其 `slot_sources` 应为 `long_term_memory`。
4. 将客户 ID 改为 `demo-customer-002`，系统会自动新建线程。
5. 再输入相同问题，系统应追问品质档位，不能读取客户 001 的记忆。

这证明长期记忆按 `customer_id` 隔离，而不是所有客户共享。

## 网页验收：上下文限制

连续发送五轮以上问题后，查看“上下文 / 记忆”：

- `messages` 最多保留配置的近期消息数，默认 8 条。
- 被移出的较早消息进入 `conversation_summary`。
- `context_snapshot.estimated_tokens` 不得超过 `max_tokens`。
- `context_dropped_messages` 应记录累计压缩数量。
- `execution_trace` 中应包含 `build_context`。

默认预算可在 `.env` 中调整：

```text
AGENT_CONTEXT_MAX_TOKENS=1400
AGENT_CONTEXT_RECENT_MESSAGES=8
AGENT_CONTEXT_SUMMARY_CHARS=1000
```

## 默认到底能记多少轮

- 默认保留最近 8 条 message；正常一轮是一条客户消息加一条 Agent 消息，因此约等于最近 4 轮完整问答原文。
- 单条消息进入上下文前最多保留 600 字符。
- 更早消息不会全部消失，而是滚动压缩到最多 1000 字符的 `conversation_summary`，但摘要是有损的。
- `turn_count` 没有固定硬上限；模型每轮真正看到的内容仍受约 1400 token 总预算限制。
- 长期客户记忆每轮最多加载 8 条有效白名单事实，不等于加载 8 轮聊天。

所以面试时应表述为：“会话可以继续增长，默认保留最近约 4 轮原文，更早内容进入滚动摘要；关键品牌、机型和偏好另存为受治理的结构化记忆。”不能说成“系统最多只能聊 4 轮”，也不能声称旧对话永久逐字进入模型。

## checkpoint 查询与多会话边界

已知 `thread_id` 时，系统可以读取最新 State 和 checkpoint 历史，也能在进程重启或人工中断后继续执行。当前 Streamlit 的 checkpoint 页展示当前线程最近 20 个快照；这个 20 是 UI 查询限制，不是数据库保存上限。

当前仍不支持在 checkpoint 中全文搜索所有客户问题，但已经有独立会话目录。点击“新建会话”会创建新的 UUID；页面可以按 `customer_id` 列出、打开、重命名、归档和恢复旧线程。打开后使用 `load_graph_thread()` 从最新 checkpoint 恢复 State，并继续使用原 `thread_id`。

会话目录和 checkpoint 分库，且两层都会校验客户归属。完整实现和网页验收见 `conversation_sessions.md`；技术选型见 `web_session_architecture.md`。

当前 checkpoint 没有 TTL 和自动清理。单实例本地演示可以继续使用 SQLite；生产部署需要配置持久数据库、客户隔离、保留周期、归档和删除策略。

## 长期记忆治理

系统只自动保存以下已确认字段：

- `brand`
- `machine_model`
- `quality_level`
- `city`

普通聊天、RAG 内容、工具输出、订单号和联系方式不会自动写入长期记忆。每条记忆包含
`customer_id`、`fact_type`、`fact_value`、`source`、`confidence`、
`created_at`、`updated_at`、`expires_at`、`status` 和 `revision`。

`MemoryRepository` 提供：

- `correct_fact()`：纠正事实并增加版本。
- `delete_fact()`：软删除。
- `expire_due()`：按保留期限过期。
- `list_events()`：查看创建、刷新、纠错、删除和过期审计事件。

手机号、身份证、银行卡、密码、Token 和 API Key 模式会被策略拒绝。
