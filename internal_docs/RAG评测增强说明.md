# RAG 评测增强说明

项目一现在有两组本地评测：

```powershell
python evaluate_cases.py --k 3
```

默认 30 条客服回归测试，主要检查分类、主动追问、工单处理人和检索关键词。

```powershell
python evaluate_cases.py --cases tests/rag_observability_cases.jsonl --k 5
```

RAG 可解释性测试，覆盖内部调试台的 5 条演示路径和 9 个知识库模块。它会额外检查：

- Top-K 是否命中指定知识库文件。
- metadata 的 `category` 是否符合预期。
- metadata 的 `risk_level` 是否符合预期。
- 缺失字段是否包含关键追问信息。
- 工单草稿是否包含指定业务关键词。

非默认测试不会覆盖 `reports/evaluation_summary.md`，会生成独立文件：

```text
reports/evaluation_summary_rag_observability_cases.md
```

## 面试讲法

我没有只看大模型回答好不好，而是把 RAG 链路拆成可测试指标：意图分类、检索命中文档、metadata 风险等级、主动追问、工单处理人和工单字段。这样每次扩充知识库或调整分类规则后，都可以用脚本回归，避免系统表面能答、内部链路却失控。
