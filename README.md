# 企业智能客服 Agent + 知识库 RAG 系统

这是一个面向工程机械配件销售场景的 AI Agent / RAG 项目。

## 当前状态

项目一 RAG 已完成三个阶段：文档上传与版本管理、持久化多会话与结构化引用、真实流式输出与受限语义缓存。客户页当前以“底盘件与销轴紧固件知识 V1.0”为主要展示主题，原有综合配件知识仍作为补充检索资料。

关键信息：

- 线上网站：`https://gzjl-machine-agent-7353.streamlit.app`
- GitHub 仓库：`https://github.com/KD0112/gzjl-machine-agent`
- 最新评测：30 条测试用例，分类、主动追问、工单处理人、检索关键词命中均为 30/30
- 封版总结：`D:\new things\项目1\md\项目一_封版总结.md`
- 简历初稿：`D:\new things\项目1\md\Agent实习简历初稿.md`

当前链路：

```text
企业文档/网页上传 -> 解析、去重、版本管理 -> 文档切分 -> Chroma 知识库
用户问题 -> 会话历史改写 -> 分类 -> Top-K 检索 -> 缓存安全判断
        -> DeepSeek 真实流式回答 -> 结构化引用 -> SQLite 持久化
```

## 已实现功能

- DeepSeek API 调用，API key 从 `.env` 读取，避免泄露。
- 读取 `docs/` 里的企业知识库文档并建立 Chroma 向量库。
- 知识库已从单一主文档扩展为多份业务文档，覆盖产品目录、型号适配、报价档位、售后政策、物流签收、故障诊断 FAQ、客户常见问题和转人工规则。
- Markdown 文档支持在文件头部写入 `title`、`category`、`risk_level` 等 metadata，建库时会写入向量库，方便后续做分类过滤和内部调试。
- 使用 `BAAI/bge-small-zh-v1.5` 做中文 embedding。
- 使用共享 `rag_components.py` 工厂创建 Embedding、VectorStore 和 Retriever，避免建库与查询各维护一套配置。
- 使用显式 `ScoredVectorStoreRetriever` 返回 `Document`，并把 rank、distance 和 provider 写入 metadata。
- 根据用户问题检索 Top-K 知识片段。
- 检索会记录 Chroma 距离分数、同义词扩展、Top-K rank 和低置信状态，便于内部解释和评测。
- 支持业务同义词扩展，例如 `主泵/大泵/泵总成 -> 液压泵`、`喷油嘴/油嘴 -> 喷油器`、`CAT/卡特彼勒 -> 卡特`。
- 低置信或无检索结果时，会返回“暂未查询到明确记录”的兜底话术，并建议补充信息或转人工。
- 使用 `ChatPromptTemplate + ChatOpenAI + StrOutputParser` 组成 LCEL chain，调用 DeepSeek 生成基于知识库的客服回答。
- 展示引用来源，降低模型编造风险。
- 支持 PDF、DOCX、XLSX、TXT、MD 和常见图片 OCR 上传，记录解析、版本和索引状态。
- 使用 SQLite 持久化用户、多个会话、消息、业务状态和结构化引用。
- 短追问会结合受限最近历史改写为完整检索问题，不同用户和会话严格隔离。
- 使用 ChatModel 原生 `stream()` 实现真实流式回答，完成后才落库并展示引用。
- 使用独立 Chroma collection 保存受限语义缓存；价格、库存、物流、售后、个人信息和历史依赖问题不会进入缓存。
- 知识库 fingerprint、Prompt 版本、模型名称、TTL 或引用文档状态变化时，旧缓存自动失效。
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

默认 chunk 参数是 `chunk_size=500`、`chunk_overlap=80`。也可以显式指定：

```powershell
python build_index.py --chunk-size 500 --chunk-overlap 80
```

小规模知识库可以全量重建；如果只是新增或更新少量文档，也可以尝试增量更新：

```powershell
python build_index.py --incremental
```

索引完成后会生成 `chroma_db/index_manifest.json`，记录 embedding provider、模型、向量维度、normalize、VectorStore、距离度量、chunk 参数、chunk 数量和稳定 chunk ids。增量更新前会比较配置指纹；配置变化或旧 manifest 缺少指纹时会自动安全全量重建，避免新旧向量混用。

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
- `retrieve_with_metadata()`：返回检索片段、距离分数、同义词扩展和低置信状态。
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
- 回答完成后的“参考依据”
- 多会话切换与历史恢复
- 联系电话 `18750528881`

技术细节仍然在后端保留，但默认不展示给客户：

- Top-K 固定为 3，不在页面展示。
- 问题分类、检索来源、工单草稿仍由后端生成，用于内部流程和面试讲解。
- 客户页面只展示文档名和页码、Sheet 或章节，不展示向量距离、Chunk ID 和内部工单 JSON。

## 6.5 内部 RAG 调试台

客户侧页面默认隐藏 Top-K、向量距离、Chunk ID 和工单草稿。面试展示或本地排查时，可以启动内部调试台：

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
- 检索状态、距离分数和同义词扩展
- 每个片段来源文档和 metadata：`title`、`category`、`risk_level`
- 工单草稿
- 构造后的 prompt
- 可选 DeepSeek 最终回答

详细说明见 `internal_docs/RAG内部调试台说明.md`，逐题演示话术见 `internal_docs/RAG调试台演示脚本.md`。

## 7. 批量测试

先验证 LangChain RAG 组件、Retriever metadata、索引指纹和 LCEL chain：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

默认测试不会调用 DeepSeek，不会消耗大模型 token：

```powershell
python evaluate_cases.py --k 3
```

RAG 可解释性测试会额外检查命中文档、metadata、风险等级、缺失字段和工单关键词：

```powershell
python evaluate_cases.py --cases tests/rag_observability_cases.jsonl --k 5
```

如果要对比不同 Top-K，可以运行：

```powershell
python evaluate_cases.py --cases tests/rag_observability_cases.jsonl --k-list 1,3,5,8
```

输出文件：

- `reports/evaluation_summary.md`
- `reports/evaluation_summary_rag_observability_cases.md`
- `reports/evaluation_时间戳.csv`
- `reports/topk_comparison_时间戳.md`

如果要测试最终回答质量，可以调用大模型：

```powershell
python evaluate_cases.py --k 3 --with-llm
```

注意：`--with-llm` 会对 30 条测试问题逐条调用 DeepSeek。

## 当前测试结果

最近一次项目一自动化测试：

- 单元与集成测试：54/54
- 覆盖文档解析、OCR、去重、版本、增量索引、会话隔离、历史改写、结构化引用、真实流式接口和语义缓存
- 自动化测试使用 Fake Model 和临时向量库，不调用真实外部模型 API

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

RAG 生产级问题、补充改动和自验收标准见 `internal_docs/RAG生产级改进与验收标准.md`。

## 8. Streamlit Cloud 部署

当前项目已经按 Streamlit Cloud 部署做了准备：

- `.env` 不会上传 GitHub，DeepSeek API Key 需要配置在 Streamlit Cloud Secrets。
- `chroma_db/` 不会上传，云端首次启动时会根据 `docs/` 自动建立向量库。
- 客户页面默认隐藏 Top-K、内部距离和工单草稿，只展示连续对话、客户可读引用和缓存命中标记。
- 部署细节见 `DEPLOY_STREAMLIT.md`。

当前线上地址：

```text
https://gzjl-machine-agent-7353.streamlit.app
```
