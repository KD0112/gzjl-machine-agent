# 多模态图片证据 MVP

更新时间：2026-07-27

## 当前结论

项目二已经完成第一版多模态闭环，不再只是规划：

```text
Streamlit 上传
  -> 文件校验与清洗
  -> 独立证据仓库
  -> LangGraph inspect_image
  -> 智谱视觉模型
  -> ImageInspectionResult
  -> 质量/置信度门控
  -> 客户确认 interrupt
  -> 同一 checkpoint 恢复
  -> 合并已确认槽位或转人工
```

当前文本模型仍为 DeepSeek；视觉模型通过 `ModelRouter` 独立使用智谱
`glm-4.1v-thinking-flash`。业务代码不依赖具体 Provider，后续可用同一评测集对比其他视觉模型。

## 已实现文件

- `schemas.py`：`ImageInspectionResult`、`ImageConfirmation` 和 `part_number` 槽位。
- `image_evidence.py`：图片校验、重编码、质量信号、客户隔离、过期和删除。
- `vision_service.py`：视觉提示词、JSON 解析、Pydantic 校验和安全门控。
- `agent_graph.py`：`inspect_image`、`confirm_image`、checkpoint 恢复和图片专用终止路径。
- `app.py`：最多 3 张图片上传、候选字段确认/编辑/拒绝/转人工、图片证据调试页。
- `handoff_policy.py`：视觉失败、低质量、客户拒绝和主动转人工原因。
- `tool_call_logger.py`：只记录证据 ID、结构化结果和路由，不记录图片二进制。
- `tests/test_multimodal_runtime.py`：10 条离线运行时测试。
- `tests/evaluate_multimodal.py`：真实视觉 API 评测脚本。
- `tests/multimodal_cases.jsonl`：首批 4 条合成图片用例。

## 结构化结果

`ImageInspectionResult` 包含：

```text
image_type
extracted_text
brand
machine_model
part_name_candidate
part_number
visible_damage
observed_features
image_quality
confidence
warnings
required_followups
safe_for_auto_merge
```

模型只能描述可见证据，不得根据外观直接给出最终适配、内部故障、质保责任、价格或库存结论。未知字段必须为 `null`，零件号看不清时不得补全。

## 文件安全与数据治理

当前支持 JPG、PNG 和 WebP，并同时检查：

- 文件扩展名、声明 MIME 和实际解码格式一致。
- 默认不超过 8 MB、2000 万像素，最小边不低于 96 像素。
- 拒绝损坏文件、动画图片、格式伪装和异常尺寸。
- EXIF 方向纠正后重新编码，移除原始元数据。
- 使用亮度、对比度和清晰度信号辅助低质量拒识。

原图不进入 LangGraph State、checkpoint、长期记忆或 JSONL 模型日志。图片保存在
`logs/image_evidence/`，SQLite 只保存元数据；读取时必须同时匹配 `evidence_id` 和
`customer_id`，默认保留 24 小时并支持过期清理和主动删除。

## LangGraph 状态与路由

新增 State 字段包括：

```text
attachments
vision_results
vision_status
vision_error
vision_model_runtime
image_confirmation_request
image_confirmation_decisions
confirmed_visual_slots
```

关键规则：

1. 有图片时先进入 `inspect_image`，无图片仍走原文本路径。
2. 高质量候选进入 `confirm_image` interrupt，客户确认后用 `Command(resume=...)` 恢复。
3. 图片只填补文本没有明确提供的槽位；文本和图片冲突时保留当前文本并记录冲突。
4. 客户可确认、编辑、拒绝或转人工，未经确认的候选不进入业务槽位。
5. 模糊图、无关图、低置信度或 Provider 失败不猜测，要求重拍或创建人工服务单。
6. 仅要求图片字段提取时使用 `image_inspection` 内部意图，确认后直接回复字段；询问适配、故障、价格或库存时才继续工具和人工策略。

## 当前评测

2026-07-27 自动验收结果：

- 多模态离线运行时测试：10/10。
- 全部运行时与集成测试：66/66。
- workflow 业务回归：30/30。
- LangGraph 业务回归：30/30。
- 人工接管策略：9/9。
- 首批真实 API 合成图片冒烟评测：4/4。
- 40 张开放许可真实机械候选预跑：40/40，API/解析错误 0，
  P50 9.83 秒、P95 17.53 秒。

4 张合成图片覆盖清晰设备铭牌、液压泵零件标签、裂纹/漏油可见证据和模糊反光铭牌。40 张开放许可真实候选覆盖旧牌、序列牌、液压泵、腐蚀、泄漏和困难负例，已经证明连续批处理、结构化响应重试和断点复用可用；候选字段尚未完成双人金标，因此仍不能宣称真实挖机配件识别准确率。

报告位置：

```text
reports/multimodal_evaluation_summary.md
reports/multimodal_evaluation_summary_multimodal_real_candidates.md
reports/multimodal_evaluation_*.csv
reports/multimodal_evaluation_*.jsonl
```

## 自动验收

```powershell
cd "D:\new things\项目1\day1\project2"
& "..\.venv\Scripts\python.exe" -m unittest tests.test_multimodal_runtime -v
& "..\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
& "..\.venv\Scripts\python.exe" tests\evaluate_multimodal.py
```

最后一条会调用真实视觉 API，需要本地 `.env` 已配置有效 `ZHIPU_API_KEY`，并会消耗免费额度或产生平台用量。

## 网页验收

1. 打开 `http://127.0.0.1:8503`，选择 `LangGraph`。
2. 上传 `tests/fixtures/multimodal/synthetic_part_label.png`。
3. 输入“请识别图片上的品牌、配件名称和零件号，先不要查询价格。”
4. 点击“运行 Agent”，确认页面进入“待图片确认”。
5. 核对候选字段后点击“确认并继续”。
6. 预期状态为“完成”，回复包含 `KOMATSU`、`液压泵` 和 `708-2L-00300`，且不调用价格工具、不转人工。
7. 打开“图片证据”，核对 Provider、模型、置信度、客户确认和已合并字段。
8. 再上传模糊样例，确认系统不输出零件号并要求重拍或转人工。

## 下一步

1. 完成 40 张公开迁移候选的双人标注与裁决，生成正式字段报告。
2. 再补 20-30 张经授权脱敏的真实挖机客户图片，与公开迁移集分层报告。
3. 针对序列号/型号误映射、反光和损坏误判修正提示词与质量门控。
4. 为订单截图补敏感信息检测/遮盖，再决定是否允许进入售后证据流。
5. 图片评测稳定后，再实现带官方域名白名单和引用的只读 `web_search_tool`。

## 面试表达

> 我没有用多模态模型直接判断适配或自动报价，而是把图片当成不可信业务证据。上传文件先经过格式、大小、解码和元数据清理，再由独立视觉路由输出 Pydantic 结构；LangGraph 在关键字段合并前暂停，等待客户确认。原图不进入 checkpoint，低质量和低置信度结果拒绝猜测并可转人工。我还收集了 40 张开放许可真实机械图片，补了来源、隐私、双人金标门禁、字段混淆矩阵和断点续跑。当前 40/40 只是候选预跑，正式准确率必须等双人金标完成后再报告。
