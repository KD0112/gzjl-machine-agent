# 挖机配件多工具销售 Agent 流程图

更新时间：2026-06-17

## 1. 当前业务闭环

```mermaid
flowchart TD
    A["客户问题<br/>例如：小松PC200原厂液压泵要1件，有没有现货，多少钱，发到贵阳要多久？"]
    B["agent_parser<br/>意图识别 + 槽位抽取"]
    C{"缺少关键信息？"}
    D["response_builder<br/>生成追问话术"]
    E["agent_workflow / agent_graph<br/>选择并调度工具"]
    F["inventory_tool<br/>查询库存"]
    G["quote_tool<br/>生成参考报价"]
    H["logistics_tool<br/>估算物流时效和运费"]
    T["ticket_tool<br/>生成售后工单草稿"]
    I["response_builder<br/>生成真人客服口吻回复"]
    X["execution_trace<br/>记录解析、拦截、工具调用和回复生成"]
    J["Streamlit 调试台<br/>展示客户回复 + 执行轨迹 + 内部 JSON"]

    A --> B
    B --> C
    C -- "是" --> D --> J
    C -- "否" --> E
    E --> F
    E --> G
    E --> H
    E --> T
    F --> I
    G --> I
    H --> I
    T --> I
    B --> X
    D --> X
    E --> X
    F --> X
    G --> X
    H --> X
    T --> X
    I --> X
    I --> J
    X --> J
```

## 2. LangGraph 版本节点图

```mermaid
flowchart LR
    START(["START"])
    P["parse_node<br/>解析客户问题"]
    R{"route_after_parse<br/>是否缺信息？"}
    INV["call_inventory_node<br/>库存工具节点"]
    Q["call_quote_node<br/>报价工具节点"]
    L["call_logistics_node<br/>物流工具节点"]
    T["call_ticket_node<br/>售后工单节点"]
    U["mark_unsupported_tools_node<br/>标记未接入工具"]
    RESP["build_response_node<br/>生成客户回复"]
    END(["END"])

    START --> P
    P --> R
    R -- "缺信息" --> RESP
    R -- "信息完整" --> INV --> Q --> L --> T --> U --> RESP
    RESP --> END
```

## 3. State / Node / Edge 对应关系

```text
State：保存当前任务状态
- question：原始客户问题
- parse_result：意图、槽位、缺失字段
- tool_results：库存、报价、物流工具返回值
- called_tools：本轮实际调用的工具
- unsupported_tools：识别到了但暂未接入的工具
- customer_reply：最终给客户看的回复
- execution_trace：内部执行轨迹，用于解释每一步决策和工具调用

Node：每个处理步骤
- parse_node：调用 agent_parser.parse_customer_question()
- call_inventory_node：调用 inventory_tool
- call_quote_node：调用 quote_tool
- call_logistics_node：调用 logistics_tool
- call_ticket_node：调用 ticket_tool
- build_response_node：调用 response_builder.build_customer_reply()

Edge：节点之间的流转规则
- START -> parse_node
- 如果缺字段：parse_node -> build_response_node
- 如果信息完整：parse_node -> 工具节点 -> build_response_node
- build_response_node -> END
```

## 4. 面试表达

项目二第一版先用手写 workflow 跑通业务闭环，确认库存、报价、物流、售后工单工具和客服话术都稳定。

随后我把同一套逻辑迁移到 LangGraph：把任务状态抽象成 State，把每个步骤抽象成 Node，把“信息不足先追问、信息完整再调用工具”的判断抽象成 Edge。迁移后仍然复用原来的 30 条测试集，workflow 模式和 LangGraph 模式都能达到 30/30，通过率 100%。

这说明 LangGraph 不是为了套框架，而是在业务闭环已经稳定后，用图结构增强流程可维护性、可解释性和后续扩展能力。

新增 `execution_trace` 后，手写 workflow 和 LangGraph 都会输出同样结构的执行轨迹。面试时可以直接展示 `parse -> call_tool -> build_response` 或 `parse -> guard_missing_fields -> build_response`，说明系统为什么调用工具，为什么不调用工具，以及客户侧回复如何生成。
