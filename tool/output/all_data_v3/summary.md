# all_data_v3 训练/评测分布

| 指标 | 训练 | 评测 | 全量 |
|------|------|------|------|
| case 数 | 6251 | 695 | 6946 |
| jsonl 行 | 18143 | 2017 | 20160 |
| 占比 | 90.0% | 10.0% | 100% |

## 实体存在性 (case 级)

| 类别 | 训练 | 评测 | 全量 |
|------|------|------|------|
| yes | 5946 (95.1%) | 661 (95.1%) | 6607 (95.1%) |
| no | 305 (4.9%) | 34 (4.9%) | 339 (4.9%) |

## 几何关系 (case 级)

| 类别 | 训练 | 评测 | 全量 |
|------|------|------|------|
| aligned | 4743 (75.9%) | 526 (75.7%) | 5269 (75.9%) |
| misaligned | 1203 (19.2%) | 135 (19.4%) | 1338 (19.3%) |
| (no_entity) | 305 (4.9%) | 34 (4.9%) | 339 (4.9%) |

## 障碍类型 (case 级)

| 类别 | 训练 | 评测 | 全量 |
|------|------|------|------|
| other_obstacle | 2757 (44.1%) | 306 (44.0%) | 3063 (44.1%) |
| wheel_stop | 1030 (16.5%) | 114 (16.4%) | 1144 (16.5%) |
| parking_curb | 0 (0%) | 0 (0%) | 0 (0%) |
| hard_curb | 890 (14.2%) | 99 (14.2%) | 989 (14.2%) |
| speed_bump | 714 (11.4%) | 80 (11.5%) | 794 (11.4%) |
| ground_irregularity | 555 (8.9%) | 62 (8.9%) | 617 (8.9%) |
| (no_entity) | 305 (4.9%) | 34 (4.9%) | 339 (4.9%) |

## jsonl 样本结构

每个 case 在 `dataset.jsonl` 中展开为 **3 条** 训练样本（不同 user 任务）：

| 任务 | 训练 jsonl | 评测 jsonl |
|------|-----------|-----------|
| 实体存在性 | 6251 | 695 |
| 几何关系 | 5946 | 661 |
| 障碍物类型 | 5946 | 661 |
| **合计** | **18143** | **2017** |

划分方式：`split_and_check.py` 按 `label.csv` 三元组分层抽样约 **10%** case 进 `val_dataset.jsonl`（`SEED=42`）。训练/评测在各标签维度比例几乎一致（分层有效）。

## 饼图输出

| 文件 | 说明 |
|------|------|
| `split_cases.png` | case 数：训练 6251 vs 评测 695 |
| `train_entity_existence.png` / `val_entity_existence.png` | 实体存在性 |
| `train_geometry_relation.png` / `val_geometry_relation.png` | 几何关系（仅 entity=yes） |
| `train_object_type_yes.png` / `val_object_type_yes.png` | 障碍类型（仅 entity=yes） |

数据源：`label.csv`（case 级）；对应 `name,count` 文本在同目录 `*.txt`。

重新出图示例：

```bash
cd /home/jiangzirou/avp_promptkit
python3 tool/plot_pie.py -i tool/output/all_data_v3/train_object_type_yes.txt \
  -o tool/output/all_data_v3/train_object_type_yes.png \
  --title "train 障碍类型·有实体 (n=5946)"
```

