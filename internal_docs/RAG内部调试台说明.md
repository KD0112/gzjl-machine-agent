---
title: RAG内部调试台说明
company: 贵州劲龙机械
category: 调试说明
version: V1.0
updated_date: 2026-07-05
risk_level: 低
---

# RAG 内部调试台说明

内部调试台用于面试展示和本地排查，不面向客户展示。客户侧页面继续使用 `app.py`，内部调试页使用 `rag_debug_app.py`。

## 启动命令

```powershell
cd "D:\new things\项目1\day1"
.\.venv\Scripts\streamlit.exe run rag_debug_app.py --server.port 8504 --server.fileWatcherType none
```

打开：

```text
http://127.0.0.1:8504
```

## 推荐演示问题

```text
电器件通电了还能换货吗？
液压泵异响怎么排查？
没有零件号能不能配液压泵？
你们能不能给三档方案？
PC200液压泵有没有现货？
```

## 调试台展示重点

- 问题分类：显示问题类型、类型代码、是否需要主动追问。
- Top-K 检索概览：显示命中文档、标题、类别、风险等级和内容预览。
- 检索片段详情：展开后可以查看每个 chunk 的 metadata 和正文。
- 工单草稿：展示建议处理人、缺失信息、下一步动作和风险提示。
- 回答 / Prompt：默认不调用大模型，可以勾选后生成 DeepSeek 回答；同时展示构造后的 prompt。

## 面试表达

客户侧页面隐藏 Top-K、引用来源和工单 JSON，是为了保证真实咨询体验；内部调试台保留这些信息，是为了让系统可解释、可排查、可评估。这样可以说明项目不是只做一个聊天界面，而是有清晰的 RAG 检索链路和业务流程追踪。

更完整的逐题讲解稿见：

```text
internal_docs/RAG调试台演示脚本.md
```
