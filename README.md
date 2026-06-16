# 企业智能客服 Agent + 知识库 RAG 系统

这是一个面向工程机械配件销售场景的 AI Agent / RAG 项目。

当前链路：

```text
企业文档 -> 文档切分 -> embedding -> Chroma 向量库
用户问题 -> 问题分类 -> Top-K 检索 -> DeepSeek 回答
        -> 主动追问 -> 工单草稿 -> 引用来源展示
```

## 已实现功能

- DeepSeek API 调用，API key 从 `.env` 读取，避免泄露。
- 读取 `docs/` 里的企业知识库文档并建立 Chroma 向量库。
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

这一步会读取 `docs/` 里的文档，切成 chunk，用 embedding 模型转成向量，然后写入 `chroma_db/`。

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

## 7. 批量测试

默认测试不会调用 DeepSeek，不会消耗大模型 token：

```powershell
python evaluate_cases.py --k 3
```

输出文件：

- `reports/evaluation_summary.md`
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

这个结果只代表当前测试集上的规则层验证，不代表所有真实客户问题都不会出错。真实上线仍然需要持续收集 badcase。

## 8. Streamlit Cloud 部署

当前项目已经按 Streamlit Cloud 部署做了准备：

- `.env` 不会上传 GitHub，DeepSeek API Key 需要配置在 Streamlit Cloud Secrets。
- `chroma_db/` 不会上传，云端首次启动时会根据 `docs/` 自动建立向量库。
- 客户页面默认隐藏 Top-K、引用来源和工单草稿，只展示连续对话体验。
- 部署细节见 `DEPLOY_STREAMLIT.md`。

推荐线上地址：

```text
https://gzjl-machine-agent.streamlit.app
```
