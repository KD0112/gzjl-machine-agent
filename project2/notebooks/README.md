# 项目二断点学习与验收 Notebook

这组 Notebook 把挖机配件多工具客服 Agent 拆成 16 个可独立运行的章节。每章都包含：

- 这一部分解决什么业务问题；
- 对应的真实代码入口；
- 可以单独执行的最小测试；
- `[PASS]` / `[FAIL]` 验收断言；
- 什么结果才算成功；
- 面试官可能追问的实现细节、项目化参考答案和代码落点；
- 当前边界与下一步生产化方向。

Notebook 默认不调用付费或不稳定的线上模型。需要真实模型的单元会明确标为可选，并由 `RUN_LIVE_MODEL_TESTS` 控制。

## 第一次使用

在 PowerShell 中进入项目：

```powershell
cd "D:\new things\项目1\day1"
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& ".\.venv\Scripts\Activate.ps1"
python -m pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name project1-agent --display-name "Python (.venv 项目1 Agent)"
```

然后用 VS Code 打开：

```text
D:\new things\项目1\day1\project2\notebooks
```

打开 `00_总目录与项目架构.ipynb`，右上角内核选择：

```text
Python (.venv 项目1 Agent)
```

建议先点击“重启内核”，再从上到下逐格运行。学习时不要一开始使用“全部运行”，每运行一格就读本格上方的说明和下方的输出。

## 推荐顺序

| 阶段 | Notebook | 你要掌握的内容 |
| --- | --- | --- |
| 0 | `00_总目录与项目架构.ipynb` | 项目目标、目录、两条主链路和总架构 |
| 1 | `01_环境安全与一键体检.ipynb` | 环境、密钥、67条测试和两套30条回归 |
| 2 | `02_RAG知识库构建与检索.ipynb` | loader、splitter、embedding、Chroma、Retriever |
| 3 | `03_RAG评测与生产化.ipynb` | Chunk、Top-K、引用、拒答、增量更新和生产迁移 |
| 4 | `04_LangChain组件与消息格式.ipynb` | Document、Retriever、StructuredTool、Prompt、消息转换 |
| 5 | `05_多工具调用与Pydantic.ipynb` | 意图、槽位、工具参数、Pydantic与确定性工具 |
| 6 | `06_LangGraph状态路由与Checkpoint.ipynb` | State、node、conditional edge、暂停、恢复 |
| 7 | `07_上下文工程与分层记忆.ipynb` | Token预算、摘要、短期/长期记忆、注入防护 |
| 8 | `08_多会话目录与旧会话恢复.ipynb` | 新建、列表、重命名、归档、恢复旧会话 |
| 9 | `09_人工审批与客服接管.ipynb` | 工具审批和人工客服两类 interrupt |
| 10 | `10_可观测日志Harness与ModelRouter.ipynb` | 模型路由、预算、重试、脱敏和日志 |
| 11 | `11_多模态图片识别.ipynb` | 图片校验、视觉Schema、拒识、确认、人工接管 |
| 12 | `12_双人标注Gold与图片评测.ipynb` | 双盲标注、冲突裁决、字段准确率和拒识率 |
| 13 | `13_Streamlit部署与架构选型.ipynb` | Streamlit、FastAPI和LangSmith各自职责 |
| 14 | `14_端到端演示与面试题库.ipynb` | 完整演示路径、90秒项目介绍和追问题库 |
| 15 | `15_Skills_WebSearch_MultiAgent后续路线.ipynb` | Skills、网页搜索、Multi-Agent与生产路线 |

## 如何判断一格成功

1. 单元左侧出现执行序号，例如 `[3]`，而不是一直显示 `[*]`。
2. 输出包含 `[PASS]`，且没有红色 Traceback。
3. 表格中的实际值符合本格上方写出的成功标准。
4. 测试单元最后出现 `OK`，数量与文字说明一致。
5. 涉及真实模型的可选单元被跳过不等于失败；默认离线验收不依赖外网。

出现 `[FAIL]` 时，先看该行的 `actual` 和 `expected`，再向上看本格引用的源码路径。不要跳到下一格，否则后面的失败可能只是前面状态没有建立。

## 一键复验

重新生成全部 Notebook：

```powershell
cd "D:\new things\项目1\day1\project2"
& "..\.venv\Scripts\python.exe" "notebooks\build_notebooks.py"
```

自动执行全部 Notebook 并保存输出：

```powershell
& "..\.venv\Scripts\python.exe" "notebooks\execute_notebooks.py" --timeout 300
```

只执行一本：

```powershell
& "..\.venv\Scripts\python.exe" "notebooks\execute_notebooks.py" --pattern "06_*.ipynb" --timeout 300
```

完整执行后的总结果在 [ACCEPTANCE.md](ACCEPTANCE.md)。机器可读 JSON 和详细失败堆栈保存在 `project2/reports/`。

## 复习方法

第一遍只做 00、01、02、05、06、14，先建立“RAG提供证据，LangGraph控制流程，工具执行业务”的主线。

第二遍补 03、04、07、08、09、10，重点回答为什么这样设计、失败时怎么办、如何恢复和评测。

第三遍完成 11、12、13、15，理解多模态数据闭环、部署边界和后续路线。面试前只需要重新运行 01 和 14，并挑一个 RAG、一个 checkpoint、一个人工接管案例现场演示。

每本最后的“参考答案”不是要求逐字背诵。建议先只看问题口头回答，再展开答案核对是否包含：业务风险、设计选择、代码位置、测试证据、当前边界和下一步。
