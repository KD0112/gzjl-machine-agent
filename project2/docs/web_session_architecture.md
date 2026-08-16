# 网页、会话与可观测性架构说明

更新时间：2026-07-28

## 结论先行

当前阶段继续保留 Streamlit。它已经承担客户对话、图片上传、内部 JSON、执行轨迹、checkpoint 和人工客服工作台，最适合作品集演示和本地调试。

FastAPI 和 LangSmith 不应被当成 Streamlit 的同类替代品：

| 组件 | 解决的问题 | 当前建议 |
| --- | --- | --- |
| Streamlit | 演示 UI、调试台、人工工作台 | 保留为当前主网页 |
| FastAPI | 稳定 HTTP API、鉴权、多客户端接入、前后端解耦 | 等需要真实外部接入时增加，不必现在重写页面 |
| LangSmith | Trace、线程观察、线上/离线评测 | 可选接入，不能替代网页或 API；先做好脱敏和采样 |

推荐的长期结构是：

```text
Streamlit 演示/运营台
          |
       FastAPI
          |
AgentService -> LangGraph -> 工具/RAG/模型/数据库
          |
  本地日志 + 可选 LangSmith
```

当前不需要为追求技术名词而把已经可运行的 Streamlit 全部改写。只有当微信、独立前端、移动端或第三方系统需要共同调用 Agent 时，FastAPI 才成为高优先级。

## 当前 checkpoint 能力

项目使用 `SqliteSaver`，数据库是 `logs/langgraph_checkpoints.sqlite3`：

- 每个 `thread_id` 是一条独立会话。
- 已知 `thread_id` 时，`get_graph_state()` 可以读取最新 State。
- 已知 `thread_id` 时，`get_graph_history()` 可以查看历史 checkpoint 快照。
- 同一个 `thread_id` 可以在进程重启后继续，也能从人工审批或人工客服中断处恢复。
- 同一 `thread_id` 不允许切换 `customer_id`，防止客户上下文串线。

当前已经新增产品层会话目录，但仍不是“对话全文搜索系统”：

- 不能在 checkpoint 中按客户问题全文搜索所有历史对话。
- 可以按 `customer_id` 列出最近会话并直接切换、重命名、归档和恢复归档。
- 网页的“历史记录”是最近 50 条运行日志，不等于可恢复的会话目录。
- 网页的 checkpoint 页默认展示当前线程最近 20 个快照，不等于只保存 20 个快照。
- 当前没有 checkpoint TTL 或自动清理策略；生产环境还需要保留周期、归档和删除机制。

本地 SQLite 能支持作品集和单实例演示。Streamlit Community Cloud 当前没有配置外部持久数据库，因此不应把容器内 SQLite 当成跨重建、重新部署均有保证的生产存储。正式部署应把 checkpoint 和会话目录迁移到 Postgres 等持久数据库。

## 最多能记忆多少轮

必须区分“近期原文”“较早摘要”“checkpoint 历史”和“长期客户事实”：

| 内容 | 当前默认值 | 实际含义 |
| --- | --- | --- |
| 近期消息 | 8 条 message | 正常一轮包含一条客户消息和一条 Agent 消息，约为最近 4 轮完整问答 |
| 单条消息 | 600 字符 | 超出后进入上下文前会截断 |
| 会话摘要 | 1000 字符 | 较早消息滚动压缩为有损摘要 |
| 模型上下文预算 | 约 1400 tokens | 超预算时按优先级丢弃低优先级区段 |
| 长期记忆 | 最多加载 8 条有效事实 | 仅品牌、机型、品质和城市等白名单客户事实 |
| 会话轮数 | 没有固定硬上限 | `turn_count` 可继续增长，但旧对话不会永久以逐字原文进入模型 |

因此准确说法不是“最多只能记 4 轮”，而是“默认保留最近约 4 轮原文，更早内容进入 1000 字符滚动摘要，并在 1400 token 预算内按优先级装配”。如果面试官问长期对话，应该主动说明摘要是有损的，关键业务事实还需要结构化槽位或受治理的长期记忆保存。

对应环境变量：

```text
AGENT_CONTEXT_MAX_TOKENS=1400
AGENT_CONTEXT_RECENT_MESSAGES=8
AGENT_CONTEXT_MESSAGE_CHARS=600
AGENT_CONTEXT_SUMMARY_CHARS=1000
AGENT_CONTEXT_MEMORY_ITEMS=8
```

不建议只把这些数字调大。更合理的做法是根据长对话评测调整，并监控指代正确率、事实冲突率、上下文 Token 和延迟。

## 消息是否经过格式处理

已经处理，不是把 Streamlit 原始文本和全部历史直接拼给模型：

1. `make_message()` 将角色归一化为 `user`、`assistant` 或 `human_agent`，清理空白，并加入 `turn_index`、`request_id` 和时间。
2. `compact_messages()` 限制单条长度和近期消息数量，把较早消息滚动压缩进 `conversation_summary`。
3. `build_context_snapshot()` 按“安全规则、当前问题、已确认客户信息、RAG、工具结果、近期消息、摘要”装配上下文。
4. RAG、工具结果和历史内容都标记为不可信数据，不能覆盖系统规则；疑似 Prompt Injection 会留下信号。
5. LangChain 语义解析使用 `ChatPromptTemplate` 的 system/human 消息和 Pydantic structured output；视觉链路使用标准 `HumanMessage` 传文字与图片。

当前 State 中的对话消息是项目自定义字典，不是全量使用 LangChain `BaseMessage`。对当前显式 LangGraph workflow 是合适的，但如果后续增加通用 tool calling、sub-agent 或 LangSmith Messages 视图，建议补：

- `ConversationMessage` Pydantic schema。
- `to_langchain_message()` / `from_langchain_message()` 转换器。
- 对 `HumanMessage`、`AIMessage`、`ToolMessage` 和 `tool_call_id` 的显式映射。
- 继续让 RAG 证据、工具结果和客户消息分区存储，不能为了格式统一而把所有内容混成聊天记录。

## 多会话现状

第一阶段已经完成：

- 点击“新建会话”会生成新的 UUID，并立即写入会话目录。
- `conversation_repository.py` 使用独立 SQLite 保存稳定会话索引：

```text
thread_id
customer_id
title
status
channel
created_at
updated_at
last_message_preview
archived_at
```

Streamlit 侧栏已经支持：

- 新建会话。
- 最近会话列表。
- 按 `customer_id` 隔离后的切换。
- 重命名和归档。
- 选择旧线程后通过公开服务函数加载最新 State。

`load_graph_thread()` 和会话仓库都校验 `customer_id`。已有 6 条专项测试覆盖双线程隔离、跨进程恢复、跨客户拒绝、重命名和归档。首次升级还会从运行日志提取旧 `thread_id`，再通过公开 checkpoint API 回填目录，不读取 LangGraph 内部表。

完整验收见 `conversation_sessions.md`。

## 后续阶段

### 需要外部客户端时再加 FastAPI

建议最小接口：

```text
POST /v1/chat
GET  /v1/threads
GET  /v1/threads/{thread_id}
POST /v1/threads/{thread_id}/resume
POST /v1/threads/{thread_id}/image-confirm
GET  /health
```

届时把 Agent 调用抽到 `AgentService`，Streamlit 和 FastAPI 复用同一服务层，并增加鉴权、限流、幂等键、超时和结构化错误码。微信本地 `ThreadingHTTPServer` 也可以再迁移成 FastAPI router。

### 可选 LangSmith

项目已有本地执行轨迹、工具日志、模型日志和离线评测，所以 LangSmith 是增强项，不是当前阻塞项。接入时应：

- 用环境变量开关，不影响离线运行。
- 将 `thread_id`、`customer_id` 的脱敏标识和运行版本写入 metadata。
- 对客户文本、电话、订单、图片和工具输出做脱敏或禁止上传。
- 先对测试环境或采样流量开启，再决定是否长期保留。

## 当前优先级

1. 完成 40 张公开迁移图片的双人盲审与第三人裁决，不能把预跑成功率当准确率。
2. 加入经授权脱敏的真实挖机图片并分层评测，修正 badcase。
3. 增加受控只读网页搜索。
4. 出现真实外部接入需求时增加 FastAPI。
5. 需要统一 Trace 和线上评测时可选接入 LangSmith。

## 官方资料

- Streamlit Session State：<https://docs.streamlit.io/develop/concepts/architecture/session-state>
- Streamlit Community Cloud：<https://docs.streamlit.io/deploy/streamlit-community-cloud>
- FastAPI Features：<https://fastapi.tiangolo.com/features/>
- LangSmith Observability：<https://docs.langchain.com/langsmith/observability-concepts>
- LangSmith Evaluation：<https://docs.langchain.com/langsmith/evaluation>
