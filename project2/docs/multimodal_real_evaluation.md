# 真实图片候选集与字段级评测

## 当前完成度

截至 2026-07-27，仓库已经具备一套可复现的真实图片评测流程：

- 40 张开放许可真实图片候选，包含 29 张铭牌/旧牌、4 张液压泵实物和
  7 张腐蚀、泄漏、密封损坏或困难负例。
- 下载后统一限制到 1600px、重编码并移除 EXIF；记录来源页、作者、许可、
  原图地址、尺寸、文件哈希和采集时间。
- 生成候选联系表、许可归属表和双人标注 CSV。
- 提供双人盲审分包、独立结果合并、冲突字段识别和第三人裁决工作表。
- `gold` 构建器会在模型调用前检查许可、隐私、双人标注、裁决人、字段可读性
  和拒识标签。
- 评测器输出字段 TP/FP/FN/TN、accuracy、precision、recall、F1、拒识混淆矩阵、
  幻觉零件号、场景分层和 P50/P95 延迟。
- 40 张候选在线预跑 40/40，API/解析错误 0，P50 9.83 秒、P95 17.53 秒。
- 首轮批处理发现非 JSON 和 `JSONDecodeError` 漂移后，解析/Pydantic 校验已移入
  Harness；断点验证成功复用 39 张，只重跑 1 张。

这些公开图片是机械视觉迁移集，不是挖机客户业务集。它们适合验证 OCR、旧牌、
反光、低对比、损坏证据和拒识流程，但不能用于宣称真实挖机配件生产准确率。

## 文件

```text
tests/collect_public_multimodal.py
tests/multimodal_real_candidates.jsonl
tests/multimodal_real_attribution.csv
tests/multimodal_real_annotation_template.csv
tests/prepare_multimodal_double_review.py
tests/build_multimodal_gold.py
tests/evaluate_multimodal.py
reports/multimodal_real_candidates_contact_sheet.jpg
reports/multimodal_real_review_workbook.csv
reports/multimodal_reviewer_a.csv
reports/multimodal_reviewer_b.csv
reports/multimodal_adjudication_workbook.csv
```

图片二进制位于本地忽略目录：

```text
tests/fixtures/multimodal_real/sanitized/
```

仓库提交来源和哈希，不提交这批图片。换机器后重新运行采集脚本即可恢复。

## 数据门禁

候选样本默认是 `evaluation_status=candidate`，字段可读性为 `unreviewed`。
评测器可以用 `--include-candidates` 做模型预跑，但字段全部记为 `Skipped`，
这个结果不是准确率。

正式样本必须同时满足：

1. `license_manual_verified=true`，人工打开来源页核对许可。
2. `privacy_approved=true`，确认无客户姓名、电话、地址、人脸、车牌或定位信息。
3. Reviewer A 和 Reviewer B 使用两份不含模型预测的独立文件，不能互看答案。
4. 合并脚本保留两份原始判断，并自动输出 `conflict_fields`。
5. 有第三位裁决人处理差异，填写最终字段和 `adjudication_reason`。
6. 每个字段明确标成 `readable`、`unreadable` 或 `not_present`。
7. `readable` 字段必须给出金标；不可读字段不得靠来源标题猜值。
8. `should_reject` 必须明确为 true 或 false。

公开图片已经移除 EXIF，联系表中没有保留清晰人物图；这不替代最终逐张隐私复核。

## 标注口径

CSV 中多个可接受值使用 `|` 分隔，例如：

```text
KOMATSU|小松
hydraulic pump|主泵|液压泵
```

- `brand` 和 `part_name_candidate` 支持同义词包含匹配。
- `machine_model` 和 `part_number` 使用规范化后的严格匹配。
- 错误但非空的字段同时计一次 FP 和一次 FN。
- 在 `unreadable` 或 `not_present` 时输出零件号，计作幻觉 FP。
- `predicted_reject` 由 `safe_for_auto_merge=false` 表示。
- 未复核字段跳过，不会被当作正确。

## 40 张图片的双人标注与裁决

在 `project2` 目录运行：

```powershell
& "..\.venv\Scripts\python.exe" tests\collect_public_multimodal.py --target 40
```

先打开联系表确认 40 张图片均可访问：

```text
reports/multimodal_real_candidates_contact_sheet.jpg
```

### 第一步：确定真实人员

需要三个人：

- Reviewer A：独立标注。
- Reviewer B：独立标注，不能看 A 的结果。
- Adjudicator：第三人裁决分歧。

三个人必须填写真实且不同的名字或固定工号。模型不能充当其中任何一位。

### 第二步：生成两份盲审文件

把命令中的名字改为真实审核员：

```powershell
& "..\.venv\Scripts\python.exe" tests\prepare_multimodal_double_review.py packets `
  --reviewer-a "审核员A姓名" `
  --reviewer-b "审核员B姓名"
```

生成：

```text
reports/multimodal_reviewer_a.csv
reports/multimodal_reviewer_b.csv
```

这两份文件没有 `model_*` 列，避免模型答案锚定人工判断。分别交给 A、B，不要让两人共享文件。

### 第三步：A、B 独立填写

每个人逐张查看本地图片和 `landing_url`，填写：

- `license_manual_verified`：只有打开来源页核对许可后才能填 `true`。
- `privacy_approved`：无姓名、电话、地址、人脸、车牌、定位等隐私才填 `true`。
- `expected_image_types`：从 `nameplate|part_label|part|damage|document|irrelevant|unknown` 选择，可多选。
- 四组 `*_visibility`：只能是 `readable`、`unreadable`、`not_present`。
- `expected_*_any`：只有对应字段为 `readable` 才填写；多个可接受值用 `|` 分隔。
- `expected_damage_keywords`：只写图片肉眼可见的漏油、裂纹、磨损、缺件、变形等，不推断故障原因。
- `should_reject`：图片质量 `poor/unusable`、类型为 `document/irrelevant/unknown`，或低置信且没有任何可用证据时填 `true`；清晰损坏证据图不能因为没有零件号就自动拒识。
- `reviewer_notes`：记录反光、遮挡、模糊和判断依据。

禁止从文件名、来源标题或模型预跑结果猜图片中看不见的字段。

### 第四步：合并两份独立结果

两人都完成 40 行后运行：

```powershell
& "..\.venv\Scripts\python.exe" tests\prepare_multimodal_double_review.py merge
```

生成：

```text
reports/multimodal_adjudication_workbook.csv
```

- 双方一致且完整的字段会自动写入最终列。
- 不一致的最终字段保持空白，并列入 `conflict_fields`。
- 任一审核员漏填的必填项列入 `incomplete_fields`。
- `reviewer_a_*` 和 `reviewer_b_*` 永久保留原始判断，不能覆盖。

### 第五步：第三人裁决

Adjudicator 打开裁决表：

1. 先筛选 `incomplete_fields`，退回对应审核员补齐。
2. 再筛选 `conflict_fields`，重新查看图片和来源页。
3. 在标准最终列填写裁决值，不能修改 `reviewer_a_*`、`reviewer_b_*`。
4. 每行填写真实 `adjudicator`。
5. 有冲突的行必须填写 `adjudication_reason`。
6. 可在 `annotation_notes` 记录统一口径。

### 第六步：构建正式 gold

未填完时命令必须失败并列出缺失门禁：

```powershell
& "..\.venv\Scripts\python.exe" tests\build_multimodal_gold.py `
  --annotations reports\multimodal_adjudication_workbook.csv
```

成功时应显示：

```text
Built 40 gold cases.
```

### 第七步：运行正式评测

```powershell
& "..\.venv\Scripts\python.exe" tests\evaluate_multimodal.py `
  --cases tests\multimodal_real_gold.jsonl `
  --require-gold
```

结果位于：

```text
reports/multimodal_evaluation_summary_multimodal_real_gold.md
reports/multimodal_evaluation_*.csv
```

这时才能报告字段 precision/recall/F1 和拒识率。40/40 API 成功只代表接口稳定，不代表识别准确。

### 模型预标注的正确用途

`prepare_multimodal_review.py` 仍可把模型输出合成审核工作表，但应在人工 gold 冻结后用于 badcase 分析，不能在 Reviewer A/B 盲审前展示：

```powershell
& "..\.venv\Scripts\python.exe" tests\prepare_multimodal_review.py `
  --predictions reports\multimodal_evaluation_YYYYMMDD_HHMMSS.jsonl
```

只做候选预跑时使用：

```powershell
& "..\.venv\Scripts\python.exe" tests\evaluate_multimodal.py `
  --cases tests\multimodal_real_candidates.jsonl `
  --include-candidates
```

报告会明确写“不是准确率”。

批次中断或只有少量坏例时，可复用上一份成功结果：

```powershell
& "..\.venv\Scripts\python.exe" tests\evaluate_multimodal.py `
  --cases tests\multimodal_real_candidates.jsonl `
  --include-candidates `
  --resume-jsonl reports\multimodal_evaluation_YYYYMMDD_HHMMSS.jsonl
```

只有图片 SHA-256 和 Prompt SHA-256 都相同时才会复用。

## 建议验收门槛

以下是下一轮基线目标，不是当前已取得结果：

- 可读零件号 precision 不低于 95%，严格匹配 recall 不低于 85%。
- 不存在或不可读的零件号幻觉率为 0。
- 应拒识样本 recall 不低于 95%。
- 清晰可用样本的错误拒识率不高于 15%。
- API/schema 成功率不低于 98%。
- 演示环境 P95 延迟先以 20 秒为观察线，再根据真实网络和配额调整。

每个数字都必须同时报告分母、业务/迁移数据占比和场景分布。

## 下一步

工程代码已经接近第一版封版，剩余关键工作是证据质量：

1. 完成 40 张公开迁移集的双人标注和裁决，生成第一份正式字段报告。
2. 再补 20-30 张经授权脱敏的真实挖机客户图片，单独标成
   `domain_match=excavator`，不要与公开迁移集混报。
3. 对零件号幻觉、反光、远距离小字和损坏误判做 badcase 修正。
4. 冻结 `dataset_version`、prompt 版本、模型版本和验收阈值，作为简历演示基线。
5. 完成这一步后，再做只读官方网页搜索；完整多 Agent 仍可后置。
