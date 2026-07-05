# 企业智能客服 Agent + 知识库 RAG 系统

这是一个面向工程机械配件销售场景的 AI Agent / RAG 项目。

## 封版状态

本项目已完成作品集 MVP 闭环，当前建议作为“企业客服 RAG Agent 项目”写入简历，不再继续反复美化前端。下一阶段建议进入“多工具销售 Agent”项目，补充库存查询、报价计算、物流估算、工单写入和工具调用日志。

关键信息：

- 线上网站：`https://gzjl-machine-agent-7353.streamlit.app`
- GitHub 仓库：`https://github.com/KD0112/gzjl-machine-agent`
- 最新评测：30 条测试用例，分类、主动追问、工单处理人、检索关键词命中均为 30/30
- 封版总结：`D:\new things\项目1\md\项目一_封版总结.md`
- 简历初稿：`D:\new things\项目1\md\Agent实习简历初稿.md`

当前链路：

```text
企业文档 -> 文档切分 -> embedding -> Chroma 向量库
用户问题 -> 问题分类 -> Top-K 检索 -> DeepSeek 回答
        -> 主动追问 -> 工单草稿 -> 引用来源展示
```

## 已实现功能

- DeepSeek API 调用，API key 从 `.env` 读取，避免泄露。
- 读取 `docs/` 里的企业知识库文档并建立 Chroma 向量库。
- 知识库已从单一主文档扩展为多份业务文档，覆盖产品目录、型号适配、报价档位、售后政策、物流签收、故障诊断 FAQ、客户常见问题和转人工规则。
- Markdown 文档支持在文件头部写入 `title`、`category`、`risk_level` 等 metadata，建库时会写入向量库，方便后续做分类过滤和内部调试。
- 使用 `BAAI/bge-small-zh-v1.5` 做中文 embedding。
- 根据用户问题检索 Top-K 知识片段。
- 调用 DeepSeek 生成基于知识库的客服回答。
- 展示引用来源，降低模型编造风险。
- 规则型问题分类：配件询价、配件匹配、故障诊断、售后质保、物流交付、库存采购、综合咨询。
- 信息不足时主动追问。
- 生成结构化工单草稿：客户诉求、问题类型、处理人、优先级、设备/机型、涉及配件、缺失信息、下一步动作、风险提示、证据来源。
- 30 条客服测试集与批量评估脚本。

## 1. 填写 API Key

打开 `.env`，填写：

```env
DEEPSEEK_API_KEY=你的key
```

`.env` 已经写入 `.gitignore`，不要上传到 GitHub。

## 2. 创建环境并安装依赖

```powershell
cd "D:\new things\项目1\day1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果 PowerShell 禁止运行 `.ps1`：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3. 测试 DeepSeek

```powershell
python test_deepseek.py
```

能正常返回内容，说明 API key、模型名和网络可用。

## 4. 建立向量库

```powershell
python build_index.py
```

这一步会读取 `docs/` 里的文档，解析 Markdown 头部 metadata，切成 chunk，用 embedding 模型转成向量，然后写入 `chroma_db/`。脚本会先清理旧的本地向量库，避免重复写入旧 chunk。

当前知识库结构：

```text
docs/
  贵州劲龙机械.md
  01_产品目录与配件分类.md
  02_型号适配与件号确认规则.md
  03_报价与品质档位说明.md
  04_售后质保与退换货政策.md
  05_物流发货与签收规则.md
  06_故障诊断FAQ.md
  07_客户常见问题FAQ.md
  08_人工客服转接规则.md
```

## 5. 命令行问答

```powershell
python rag_chat.py
```

核心逻辑在 `rag_chat.py`：

- `classify_question()`：识别业务问题类型。
- `retrieve()`：向量库 Top-K 检索。
- `build_prompt()`：根据问题类型构造回答模板。
- `generate_ticket_draft()`：生成结构化工单草稿。
- `answer_with_metadata()`：返回回答、检索来源、分类结果和工单草稿。

## 6. 网页展示

```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.port 8502
```

打开：

```text
http://localhost:8502
```

当前网页是客户侧聊天界面，会展示：

- 连续聊天消息
- 面向客户的自然语言回答
- 联系电话 `18750528881`

技术细节仍然在后端保留，但默认不展示给客户：

- Top-K 固定为 3，不在页面展示。
- 问题分类、检索来源、工单草稿仍由后端生成，用于内部流程和面试讲解。
- 客户页面不会展示引用来源和工单 JSON，避免干扰真实咨询体验。

## 6.5 内部 RAG 调试台

客户侧页面默认隐藏 Top-K、引用来源和工单草稿。面试展示或本地排查时，可以启动内部调试台：

```powershell
.\.venv\Scripts\streamlit.exe run rag_debug_app.py --server.port 8504 --server.fileWatcherType none
```

打开：

```text
http://127.0.0.1:8504
```

内部调试台会展示：

- 问题分类和主动追问判断
- Top-K 检索片段
- 每个片段来源文档和 metadata：`title`、`category`、`risk_level`
- 工单草稿
- 构造后的 prompt
- 可选 DeepSeek 最终回答

详细说明见 `internal_docs/RAG内部调试台说明.md`，逐题演示话术见 `internal_docs/RAG调试台演示脚本.md`。

## 7. 批量测试

默认测试不会调用 DeepSeek，不会消耗大模型 token：

```powershell
python evaluate_cases.py --k 3
```

RAG 可解释性测试会额外检查命中文档、metadata、风险等级、缺失字段和工单关键词：

```powershell
python evaluate_cases.py --cases tests/rag_observability_cases.jsonl --k 5
```

输出文件：

- `reports/evaluation_summary.md`
- `reports/evaluation_summary_rag_observability_cases.md`
- `reports/evaluation_时间戳.csv`

如果要测试最终回答质量，可以调用大模型：

```powershell
python evaluate_cases.py --k 3 --with-llm
```

注意：`--with-llm` 会对 30 条测试问题逐条调用 DeepSeek。

## 当前测试结果

最近一次默认评估：

- 测试条数：30
- 问题分类准确率：30/30
- 主动追问判断准确率：30/30
- 工单处理人准确率：30/30
- 检索关键词命中率：30/30

RAG 可解释性测试会覆盖 9 条重点链路，用来证明检索来源、metadata 和工单草稿可被自动检查。说明见 `internal_docs/RAG评测增强说明.md`。

最近一次 RAG 可解释性评估：

- 测试条数：9
- 检索来源命中率：9/9
- metadata category 命中率：8/8
- metadata risk_level 命中率：8/8
- 缺失字段关键词命中率：7/7
- 工单关键词命中率：9/9

这个结果只代表当前测试集上的规则层验证，不代表所有真实客户问题都不会出错。真实上线仍然需要持续收集 badcase。

## 8. Streamlit Cloud 部署

当前项目已经按 Streamlit Cloud 部署做了准备：

- `.env` 不会上传 GitHub，DeepSeek API Key 需要配置在 Streamlit Cloud Secrets。
- `chroma_db/` 不会上传，云端首次启动时会根据 `docs/` 自动建立向量库。
- 客户页面默认隐藏 Top-K、引用来源和工单草稿，只展示连续对话体验。
- 部署细节见 `DEPLOY_STREAMLIT.md`。

当前线上地址：

```text
https://gzjl-machine-agent-7353.streamlit.app
```
