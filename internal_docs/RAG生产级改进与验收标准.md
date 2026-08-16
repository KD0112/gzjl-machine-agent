# RAG 生产级改进与验收标准

这份文档用于面试讲解和自测验收。当前项目仍然是作品集级 MVP，但已经把 RAG 从“能回答”推进到“可解释、可评测、可拒答、可演进”。

## 2026-07-27 LangChain 组件化升级

本轮把原先分散在 `build_index.py` 和 `rag_chat.py` 的组件创建逻辑集中到 `rag_components.py`：

- `create_embeddings()`：统一 Embedding provider、模型、device 和 normalize。
- `create_vector_store()`：统一 VectorStore provider、collection 和持久化目录。
- `ScoredVectorStoreRetriever`：成为在线查询的显式 Retriever，并把 rank、distance、provider 写入 `Document.metadata`。
- `create_retriever()`：统一 Top-K Retriever 创建入口。
- `build_index_fingerprint()`：记录影响向量兼容性的配置。

项目一生成链已改为：

```text
ScoredVectorStoreRetriever
  -> Document context formatter
  -> ChatPromptTemplate
  -> ChatOpenAI（DeepSeek OpenAI-compatible API）
  -> StrOutputParser
```

项目二的五个 `StructuredTool` 现在不只是 schema 展示：`tool_dispatcher.execute_tool_with_args()` 会在审批、重试和幂等策略通过后调用标准 `StructuredTool.invoke()`。

索引 manifest 新增：

```text
embedding_provider / embedding_model / embedding_dimension
embedding_device / embedding_normalize
vector_db_provider / vector_db_collection / distance_metric
chunk_size / chunk_overlap
```

运行 `python build_index.py --incremental` 时，如果指纹变化或旧 manifest 缺少指纹，系统会自动执行安全全量重建，避免维度冲突或新旧 embedding 混用。

由于 `langchain-community` 已停止维护，本项目不再依赖其中的简单 Loader：TXT/Markdown 使用 `Path` 读取，PDF 使用 `pypdf`，统一转换为 LangChain `Document` 后再进入 splitter。Chroma、HuggingFace、text splitter 和 ChatModel 继续使用各自的独立 LangChain 集成包。

## 阅读前先统一两个概念

1. 本项目的 `chunk_size` 当前按字符长度计算，不应直接说成 token 数。

`RecursiveCharacterTextSplitter` 默认使用 Python 的 `len()` 计算文本长度，所以当前 `chunk_size=500` 更准确的说法是“每个 chunk 最多约 500 个字符”。只有显式接入 tokenizer 作为 `length_function` 后，才能说是 500 token。

2. `similarity score` 和 Chroma 返回的 `distance` 方向相反。

- 通用语境中的相似度分数通常越大越相似。
- 本项目 `similarity_search_with_score()` 返回的是 `retrieval_distance`，通常越小越相似。
- 当前判断条件是：最佳距离小于或等于 `RAG_MAX_DISTANCE` 才允许进入回答链路。
- 面试时必须先说明使用的是“相似度”还是“距离”，不能只说“score 高于阈值”。

## 本次已补充的 5 个改动

1. 检索分数与低置信兜底

- `rag_chat.retrieve_with_metadata()` 改为使用 `similarity_search_with_score()`。
- 每个 chunk metadata 会追加 `retrieval_rank`、`retrieval_distance`、`retrieval_query`、`retrieval_aliases`。
- 默认 `RAG_MAX_DISTANCE=1.0`，距离越小越相似；超过阈值会标记为 `low_confidence`。
- `answer_with_metadata()` 在 `no_docs` 或 `low_confidence` 时不调用大模型直接编答案，而是返回“暂未查询到明确记录 + 补充信息/转人工”的兜底话术。

2. 同义词扩展

- 新增 `QUERY_ALIASES`，覆盖：
  - `主泵`、`大泵`、`泵总成` -> `液压泵`
  - `喷油嘴`、`油嘴` -> `喷油器`
  - `CAT`、`卡特彼勒` -> `卡特`
- 检索前会把同义词补充进 query，内部调试台和评测报告会展示扩展结果。

3. Top-K 评测

- `evaluate_cases.py` 新增 `--k-list`，可以一次对比多个 K。
- 报告会展示分类、追问、处理人、检索命中、来源命中、metadata 命中、低置信数量和距离统计。
- 面试回答不再说“我感觉 K=3 合适”，而是说“客户侧默认 K=3，内部评测用 K=5，并用 k-list 做对比选择”。

4. Chunk 参数与索引可演进

- `settings.py` 新增：
  - `RAG_CHUNK_SIZE`
  - `RAG_CHUNK_OVERLAP`
  - `RAG_MAX_DISTANCE`
- `build_index.py` 支持：
  - `--chunk-size`
  - `--chunk-overlap`
  - `--incremental`
- 每个 chunk 会写入稳定 `chunk_id`、`chunk_index`、`chunk_size`、`chunk_overlap`。
- `chroma_db/index_manifest.json` 会记录文档数、chunk 数、embedding 模型和 chunk ids，方便后续增量更新和审计。

5. 内部调试台可解释性

- `rag_debug_app.py` 会展示：
  - 检索状态 `ok/no_docs/low_confidence`
  - 最佳候选距离
  - Top-K 距离分数
  - 同义词扩展
  - chunk id / chunk 参数
- 面试演示时可以打开“完整 JSON”，证明系统不是黑盒回答。

## 面试高频问题回答

### chunk size / overlap 怎么选？为什么？

当前默认是 `chunk_size=500`、`chunk_overlap=80`。本项目按字符长度切分。原因是客服知识库多为短规则、FAQ、报价边界和售后条款，约 500 个字符通常能保留一个较完整的业务规则，80 个字符的 overlap 能减少切分边界导致的上下文丢失。这个值是当前工程基线，不应表述成已经证明全局最优；后续要用 `--chunk-size`、`--chunk-overlap` 和同一套评测集做对比实验。

### 为什么用 bge-small-zh-v1.5？

它对中文语义检索友好，体积小，CPU 也能跑，适合作品集和 Streamlit 演示。限制是效果和吞吐不如更大的 embedding 模型，领域同义词、型号、零件号仍然需要业务词表和测试集补强。

### Top-K 为什么是 3 或 5？怎么验证最优？

客户侧默认 K=3，减少噪声和 prompt 冗余；内部调试和评测可用 K=5，便于观察更多候选证据。验证方式是跑：

```powershell
python evaluate_cases.py --cases tests/rag_observability_cases.jsonl --k-list 1,3,5,8
```

看来源命中率、metadata 命中率、低置信数量、失败数和距离统计，而不是只看最终回答顺不顺。

### 如果检索不到正确文档怎么办？

系统现在会标记 `no_docs` 或 `low_confidence`，低置信时不让大模型继续编确定性答案，而是返回“暂未查询到明确记录”，并要求客户补充型号、配件、照片、订单等信息，必要时转人工。

### 如何降低幻觉？答案是否必须引用来源？

客户侧不一定展示引用来源，否则会影响真实客服体验；内部调试台和评测必须保留来源、metadata 和距离分数。降低幻觉的关键是：证据不足时拒绝确定回答、prompt 明确禁止编造、内部保留可审计证据、对售后/适配/库存等高风险场景转人工。

### 怎么评估 RAG 质量？只看测试通过率够吗？

不够。当前评测至少拆成：检索关键词命中、来源命中、metadata category/risk 命中、分类准确、追问准确、工单字段命中、低置信数量。后续还可以补 LLM judge 或人工标注，评估回答忠实度、完整性、可执行性和拒答质量。

### 新知识加入后，如何增量更新向量库？

当前小规模知识库仍可全量重建。代码已补 `--incremental` 和 `index_manifest.json`：新增或修改文档后可以根据稳定 chunk id 添加新 chunk、删除旧 chunk。生产环境还要补文档版本、审批流、回滚和线上索引切换。

### 如何处理“液压泵 / 主泵 / 泵总成”等同义词？

现在用 `QUERY_ALIASES` 做 query expansion，并把扩展结果写入 metadata。这样客户说“主泵”，检索时会补充“液压泵”，提升命中文档概率。更大规模时可以把同义词表放进独立配置或业务词库。

### 客户问不在知识库里的问题，系统怎么拒答或转人工？

低置信时系统会输出固定兜底话术，不给确定结论。比如政治、手机、优惠券这类越界问题，应该提示暂未查询到明确记录或不属于当前挖机配件咨询范围，并建议转人工或补充业务信息。

### Chroma 适合生产吗？数据量变大怎么办？

Chroma 适合本地原型、小规模知识库和作品集演示。生产中如果数据量变大、并发变高、需要权限隔离和稳定运维，可以迁移到 Chroma Server/Cloud、Qdrant、Milvus、Elasticsearch/OpenSearch 或云厂商向量检索服务。

## RAG 核心参数专项复习

### 1. Chunk 是什么？

离线建库链路是：

```text
业务文档
  -> 文本切分为 chunks
  -> 每个 chunk 生成 embedding
  -> embedding 和 metadata 写入 Chroma
```

在线问答链路是：

```text
用户问题
  -> query embedding
  -> 检索相关 chunks
  -> 距离阈值判断
  -> 注入 Prompt
  -> DeepSeek 回答或低置信兜底
```

Chunk 是向量检索的最小知识单元。它既要足够完整，能保留一个业务事实或操作步骤，又要足够聚焦，避免一个向量混入多个无关主题。

### 2. Chunk 太大或太小分别有什么问题？

Chunk 太大：

- 一个 chunk 可能同时包含液压泵、发动机、电器和售后规则，语义不集中。
- Top-K 注入的总字符数和 token 数增加，延迟、费用和上下文噪声随之增加。
- 即使命中正确文档，模型也要从更长内容中寻找局部答案。

Chunk 太小：

- 原因、现象和处理步骤可能被切到不同 chunk。
- 用户只召回“液压泵压力不足”，却没有召回相邻的“检查吸油管和油液”。
- chunk 数量增加，向量库体积和检索候选数量也会增加。

Overlap 是相邻 chunk 之间重复保留的内容，用来降低句子或业务规则在切分边界处断裂的风险。Overlap 太小会丢上下文，太大会制造重复向量、增加存储和检索冗余。

### 3. Chunk 参数应该怎样做实验？

固定知识库、embedding 模型、测试集和 Top-K，只改变 chunk 参数，例如：

| 实验组 | chunk size | overlap | 当前状态 |
|---|---:|---:|---|
| A | 200 | 40 | 待实测 |
| B | 500 | 80 | 当前基线 |
| C | 800 | 120 | 待实测 |
| D | 1000 | 200 | 待实测 |

每组重新建库后至少比较：

- 来源命中率或 Hit@K。
- 真正标注多篇相关文档后的 Recall@K。
- MRR，观察正确来源是否排在更前面。
- Top-K 上下文总长度和重复率。
- 最终回答的忠实度、完整性和延迟。

不能把示例实验表中的百分比当作本项目实测结果。当前项目只确定了 `500/80` 是可用基线，chunk 参数矩阵仍属于后续实验。

### 4. Top-K 是参数，Recall@K 是指标

Top-K 表示检索器返回多少个候选 chunk。例如 K=5，就是把排名最靠前的 5 个 chunk 交给后续流程。

K 太小可能漏掉原因、处理方法或政策限制；K 太大可能引入无关片段，增加 prompt 成本并污染回答。因此 K 不是越大越好，应在召回、精度、成本和延迟之间平衡。

本项目 2026-07-26 的 9 条 RAG 可解释性用例结果：

| K | 检索关键词 | 来源命中 | 失败数 | 结论 |
|---:|---:|---:|---:|---|
| 1 | 9/9 | 6/9 | 4 | 候选太少 |
| 3 | 9/9 | 9/9 | 1 | 来源已找全，但一条工单关键词不完整 |
| 5 | 9/9 | 9/9 | 0 | 当前评测集表现最好 |
| 8 | 9/9 | 9/9 | 0 | 相比 K=5 没有新增收益 |

因此当前选择是：

- 客户侧默认 K=3，优先控制噪声、延迟和上下文长度。
- 内部调试与面试演示使用 K=5，保证可解释字段和工单信息更完整。
- 暂不选择 K=8，因为当前指标没有提升。

### 5. 当前项目测到的是 Hit@K，还是严格 Recall@K？

两者要区分：

- Hit@K：Top-K 中只要出现至少一个预期来源，就记为命中。
- Recall@K：Top-K 找回的相关文档数 / 该问题全部相关文档数。

当前 `expected_source_keywords` 多数只标注一个预期来源，所以报告里的“来源命中率”更接近 query-level Hit@K，而不是严格意义上的多相关文档 Recall@K。

例如一个问题有 5 篇人工标注的相关文档，Top-5 找回其中 3 篇：

```text
Recall@5 = 3 / 5 = 60%
```

如果只判断 Top-5 是否至少包含 1 篇正确文档，那么这是 Hit@5。面试时可以说“项目当前已经实现 Hit@K/来源命中评测，后续补充多相关文档标注后可以计算严格 Recall@K”。

### 6. MRR 和 Precision@K 分别看什么？

- `MRR` 关注第一篇正确文档排在第几位。正确文档排名为 1、2、3 时，倒数排名分别是 1、1/2、1/3，再对所有问题取平均。
- `Precision@K` 关注 Top-K 中有多少比例是真正相关内容。
- `Recall@K` 关注所有相关资料中有多少被召回。

故障排查可能更看重 Recall，因为答案需要多个原因和步骤；售后政策、价格规则和适配结论更看重 Precision 和高排名，因为错引资料的业务风险更高。

### 7. Distance 和 threshold 怎么理解？

Top-K 永远会尽力返回最接近的候选，但“最接近”不等于“真的相关”。当知识库只有挖机配件资料时，用户问飞机发动机，系统仍然可能返回某个机械维修 chunk。

本项目读取 Chroma distance：

```text
distance 越小 -> 通常越相似
distance <= RAG_MAX_DISTANCE -> 允许使用证据
distance > RAG_MAX_DISTANCE -> low_confidence
```

阈值太严格会产生假阴性：知识库明明有答案，却被系统拒答。阈值太宽会产生假阳性：知识库没有答案，系统却把无关片段交给 LLM，增加幻觉风险。

当前默认 `RAG_MAX_DISTANCE=1.0`。已有 30 条业务测试的最佳距离范围约为 `0.4299-0.9554`，说明已知正样本没有被误拦截，但这不能证明阈值已经最优。还必须加入知识库外负样本，统计正负样本距离分布，再选择阈值。

### 8. RAG 质量为什么不能只看“测试通过率”？

完整评估至少分两层。

检索层：

- Hit@K / Recall@K：正确资料有没有被召回。
- Precision@K：召回内容里有多少是真相关。
- MRR：正确资料排得是否足够靠前。
- 低置信误拒率与误放率。
- 来源、metadata、同义词和距离是否可追踪。

生成层：

- Faithfulness：回答是否都能被检索证据支持。
- Answer relevance：是否真正回答了客户问题。
- Completeness：原因、步骤、限制和追问是否完整。
- Citation correctness：内部引用是否对应真实来源。
- Refusal quality：资料不足时是否正确拒答、补问或转人工。
- 业务安全：是否编造价格、库存、适配、退款或维修结论。

当前 `evaluate_cases.py` 默认不调用 DeepSeek，所以 `Failed cases: 0` 只证明规则层和检索层通过，不等于最终自然语言回答完全正确。

### 9. 当前项目还需要补哪些 RAG 实验？

| 项目 | 当前状态 | 后续补充 |
|---|---|---|
| 30 条业务回归 | 已通过，30/30 | 持续加入真实 badcase |
| 9 条可解释性评测 | K=5 已通过，9/9 | 扩大到更多边界问题 |
| Top-K 对比 | 已完成 1/3/5/8 | 增加延迟、上下文长度和生成质量 |
| 同义词扩展 | 已实现 | 增加“主泵/大泵/泵总成”等自动化用例 |
| 低置信兜底 | 已实现 | 增加知识库外负样本和阈值扫描 |
| 增量索引 | 已实现 | 测试新增、修改、删除和回滚 |
| Chunk 参数矩阵 | 尚未完成 | 比较 200/40、500/80、800/120 等组合 |
| 最终回答评估 | 基础能力已有 | 增加人工标注或 LLM judge |

### 10. 面试官问“你的 RAG 如何优化”，推荐回答

> 我把优化分成数据、检索、生成和评估四层。数据层根据 FAQ、售后规则和维修资料的结构选择 chunk size 与 overlap，并计划通过固定测试集做参数矩阵实验。检索层使用中文 embedding、Top-K、业务同义词扩展和 Chroma distance 阈值；当前对 K=1、3、5、8 做过对比，K=5 在 9 条可解释性用例上零失败，客户侧仍保留 K=3 来控制噪声。生成层在低置信时不调用模型编造答案，并对价格、库存、售后和适配结论设置业务边界。评估层把检索和生成拆开，当前已经有来源命中、metadata、工单字段和距离报告，后续继续补严格 Recall@K、MRR、faithfulness、answer relevance 和拒答质量。

## 自己验收标准

在项目根目录运行：

```powershell
cd "D:\new things\项目1\day1"
```

1. 语法检查通过

```powershell
python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['settings.py','build_index.py','rag_chat.py','rag_debug_app.py','evaluate_cases.py']]; print('syntax ok')"
```

2. 重建索引成功，并生成 manifest

```powershell
python build_index.py --chunk-size 500 --chunk-overlap 80
```

验收点：

- 控制台出现 `Loaded documents`、`Written chunks`、`Manifest`。
- 文件 `chroma_db/index_manifest.json` 存在。
- manifest 里能看到 `embedding_model`、`chunk_size`、`chunk_overlap`、`chunk_ids`。

3. 增量索引命令可运行

```powershell
python build_index.py --incremental
```

验收点：

- 控制台出现 `Total chunks`、`Added chunks`、`Deleted chunks`。
- 没有报错。

4. 默认 30 条回归通过

```powershell
python evaluate_cases.py --k 3
```

验收点：

- `Failed cases: 0`
- `reports/evaluation_summary.md` 中分类、追问、处理人、检索关键词仍为 30/30。
- `低置信/无结果条数` 应为 0/30。

5. RAG 可解释性评测通过

```powershell
python evaluate_cases.py --cases tests/rag_observability_cases.jsonl --k 5
```

验收点：

- `Failed cases: 0`
- 来源命中率 9/9。
- metadata category/risk_level、缺失字段、工单关键词保持通过。

6. Top-K 对比报告生成

```powershell
python evaluate_cases.py --cases tests/rag_observability_cases.jsonl --k-list 1,3,5,8
```

验收点：

- `reports/topk_comparison_rag_observability_cases_时间戳.md` 生成。
- 表格中能看到 K=1、3、5、8 的对比结果。

7. 低置信兜底可触发

```powershell
python -c "from rag_chat import answer_with_metadata; r=answer_with_metadata('美国总统是谁？'); print(r['answer_source'], r['retrieval']['status'], r['answer'])"
```

验收点：

- 输出 `fallback low_confidence`。
- 回答包含“暂未查询到明确记录”。
- 不会编造与挖机配件无关的答案。

8. 内部调试台可解释字段可见

```powershell
streamlit run rag_debug_app.py --server.port 8504 --server.fileWatcherType none
```

验收点：

- 页面顶部能看到“检索状态”。
- Top-K 表格中能看到 `distance` 和 `aliases`。
- “完整 JSON”里能看到 `retrieval`、`retrieval_distance`、`retrieval_query`。

## 怎么确认刚刚的修改都已经改好？

最直接看三类证据：

1. 代码证据：`git diff` 能看到 `settings.py`、`build_index.py`、`rag_chat.py`、`rag_debug_app.py`、`evaluate_cases.py` 和本文件的改动。
2. 运行证据：上面的重建索引、默认评测、RAG 评测、Top-K 对比都能跑完。
3. 页面证据：内部调试台能展示检索状态、距离、同义词、完整 JSON；越界问题能触发 fallback。
