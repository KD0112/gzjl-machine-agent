# LangChain 与 LangGraph 组合说明

## 为什么不是二选一

LangGraph 解决长流程编排，LangChain 提供模型、Prompt、structured output、工具 schema 和 Retriever 抽象。项目二采用“LangGraph 主控、LangChain 节点内使用”的组合。

## 当前实现

`langchain_adapter.py` 提供：

```text
rules   只使用规则，适合确定性回归
hybrid  高置信度规则直出，低置信度调用 LangChain
llm     强制调用 LangChain，失败仍回退规则
```

模型输出必须符合 `AgentParsePlan`：

```text
intents
slots
confidence
reason
```

未明确提供的槽位必须为 null。结果还会经过业务必填字段检查，不会因为模型输出 JSON 就直接执行工具。

## 为什么不直接使用 create_agent

当前图已经显式实现：

- 工具风险审批。
- 动态工具队列。
- SQLite checkpoint。
- 节点重试和错误处理。
- 幂等复用。
- 人工客服接管。

直接替换成通用 Agent 循环会隐藏这些业务边界。当前做法既展示 LangChain 组件能力，又保留 LangGraph 的可解释流程。

## StructuredTool

`langchain_tools.py` 将五个确定性函数包装为标准工具：

```text
inventory_tool
quote_tool
logistics_tool
ticket_tool
knowledge_tool
```

所有参数 schema 继续复用 `schemas.py`，不存在两套参数定义。

`get_langchain_tool_map()` 是生产工具注册表。`tool_dispatcher.py` 不再维护第二套 handler 表，而是在 Pydantic 校验、审批、重试和幂等策略通过后调用：

```text
StructuredTool.invoke(validated_arguments)
```

因此 LangChain 负责工具协议和调用封装，LangGraph 继续负责“何时允许执行”。

## 项目一 RAG 的 LangChain pipeline

项目一通过 `knowledge_tool` 复用，当前组件链路是：

```text
Path / pypdf loader
  -> LangChain Document
  -> RecursiveCharacterTextSplitter
  -> HuggingFaceEmbeddings
  -> Chroma VectorStore
  -> ScoredVectorStoreRetriever
  -> ChatPromptTemplate
  -> ChatOpenAI
  -> StrOutputParser
```

Retriever 会保留 `retrieval_rank`、`retrieval_distance` 和 `retrieval_provider`，所以组件化后仍能支持原有距离阈值、来源展示和 Top-K 评测。

Embedding、VectorStore 和 Retriever 由 `rag_components.py` 统一创建。更换模型或数据库时，业务层 API `answer_with_metadata()` 和项目二 `query_knowledge()` 不需要变化；但索引指纹变化后必须重新建库，并重新评测距离阈值和 Top-K。

## 稳定性策略

- 业务回归固定使用 rules。
- 调试台默认 hybrid。
- 模型没有 Key、超时或结构化输出校验失败时回退规则。
- `parse_source` 和 `confidence` 写入 State、轨迹和运行日志。
- 模型只做语义计划，不直接写库存、价格、工单或外部消息。

## 消息格式与上下文转换

项目并不是把 Streamlit 原始聊天记录直接交给模型。`context_manager.py` 先完成：

1. 将角色归一化为 `user`、`assistant` 或 `human_agent`，补充轮次、请求 ID 和时间。
2. 截断超长单条消息，保留默认 8 条近期消息，把较早消息滚动压缩到会话摘要。
3. 按安全规则、当前问题、已确认客户信息、RAG、工具结果、近期消息和摘要的优先级装配受控上下文。
4. 将 RAG、工具结果和历史内容标记为不可信数据，并记录 Prompt Injection 信号。

文本解析链再通过 `ChatPromptTemplate` 转换为 system/human 消息，并用 `with_structured_output(AgentParsePlan)` 约束结果。视觉链使用标准 `HumanMessage` 传入文本和图片。因此“给模型的对话”已经做过格式化、裁剪、分区和结构化校验。

当前 State 仍保存项目自定义 message 字典，而不是统一保存 LangChain `BaseMessage`。这对显式业务 workflow 足够清晰；如果以后增加通用 tool calling、sub-agent 或 LangSmith Messages 视图，再补 `ConversationMessage` Pydantic schema，以及 `HumanMessage`、`AIMessage`、`ToolMessage` 和 `tool_call_id` 的双向转换。不要把 RAG 和工具结果为了格式统一而混入普通聊天历史。
