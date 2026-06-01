#!/usr/bin/env python3
"""将 label.csv 中废弃的 curb_like 迁移为 parking_curb / hard_curb，并可选重生成 JSONL。

默认：所有 curb_like → hard_curb（占位，需人工将车位路沿改回 parking_curb）。

用法::

  # 占位迁移 + 导出待复核列表
  python tool/migrate_curb_like_labels.py \\
    --label-csv /mnt/public-data/user/ziroujiang/all_data_v3/label.csv

  # 按映射表精确迁移（case_id,new_type 每行）
  python tool/migrate_curb_like_labels.py \\
    --label-csv .../label.csv \\
    --mapping-csv tool/output/curb_like_mapping.csv

  # 迁移后重生成 dataset / train / val（需同目录有 images/crop/yuyan）
  python tool/migrate_curb_like_labels.py \\
    --label-csv .../all_data_v3/label.csv \\
    --regenerate-jsonl \\
    --resplit-val
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_tool_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tool_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from prompts_engine.context.object_type_catalog import (  # noqa: E402
    OBJECT_TYPE_ORDER,
    coerce_legacy_object_type,
    normalize_object_type_label,
)
from tool.export_feishu_labels_csv import (  # noqa: E402
    generate_training_jsonl,
    read_label_csv,
)

DEPRECATED = {"curb_like", "路沿/台阶", "路沿台阶", "路沿"}


def _load_mapping(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "case_id" not in (reader.fieldnames or []):
            raise ValueError("mapping CSV 需含列 case_id, object_type")
        ocol = "object_type" if "object_type" in reader.fieldnames else "new_object_type"
        for row in reader:
            cid = str(row.get("case_id", "")).strip()
            ot = str(row.get(ocol, "")).strip()
            if cid and ot in OBJECT_TYPE_ORDER:
                out[cid] = ot
    return out


def migrate_label_csv(
    label_path: str,
    *,
    default: str,
    mapping: dict[str, str] | None,
    review_path: str | None,
    dry_run: bool,
) -> dict[str, int]:
    rows = read_label_csv(label_path)
    stats = {"curb_like": 0, "mapped": 0, "unchanged": 0}
    review_rows: list[dict[str, str]] = []

    for row in rows:
        ot = str(row.get("object_type", "")).strip()
        n = normalize_object_type_label(ot)
        if n not in DEPRECATED and ot != "curb_like":
            stats["unchanged"] += 1
            continue
        stats["curb_like"] += 1
        cid = str(row.get("case_id", "")).strip()
        new_ot = mapping.get(cid) if mapping else None
        if not new_ot:
            new_ot = coerce_legacy_object_type(ot, default=default)
        row["object_type"] = new_ot
        stats["mapped"] += 1
        review_rows.append(
            {
                "case_id": cid,
                "old_object_type": ot or "curb_like",
                "new_object_type": new_ot,
                "note": "请核对：路边车位开口→parking_curb，否则→hard_curb",
            }
        )

    if dry_run:
        print(f"[dry-run] 将迁移 {stats['curb_like']} 条 curb_like → 见映射/default={default}")
        return stats

    if stats["curb_like"]:
        with open(label_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id",
                    "entity_existence",
                    "geometry_relation",
                    "object_type",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"已写回 {label_path}，迁移 {stats['mapped']} 条")

    if review_path and review_rows:
        os.makedirs(os.path.dirname(os.path.abspath(review_path)) or ".", exist_ok=True)
        with open(review_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
            w.writeheader()
            w.writerows(review_rows)
        print(f"待复核列表: {review_path} ({len(review_rows)} 条)")

    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="迁移 curb_like → parking_curb / hard_curb")
    p.add_argument("--label-csv", required=True)
    p.add_argument(
        "--default",
        default="hard_curb",
        choices=["parking_curb", "hard_curb"],
        help="无 mapping 时 curb_like 默认目标（建议先 hard_curb 再人工改车位路沿）",
    )
    p.add_argument("--mapping-csv", help="case_id,object_type 精确映射")
    p.add_argument(
        "--review-csv",
        default=os.path.join(_tool_dir, "output", "curb_like_migration_review.csv"),
    )
    p.add_argument("--regenerate-jsonl", action="store_true")
    p.add_argument(
        "--resplit-val",
        action="store_true",
        help="调用 split_and_check 重划分 train/val（需 label.csv 在同目录）",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    mapping = _load_mapping(args.mapping_csv) if args.mapping_csv else None
    stats = migrate_label_csv(
        args.label_csv,
        default=args.default,
        mapping=mapping,
        review_path=args.review_csv,
        dry_run=args.dry_run,
    )
    print(stats)

    if args.dry_run:
        return 0

    base = os.path.dirname(os.path.abspath(args.label_csv))
    if args.regenerate_jsonl:
        rows = read_label_csv(args.label_csv)
        counts = generate_training_jsonl(rows, base)
        print("已生成 JSONL:", counts)

    if args.resplit_val:
        split_py = os.path.join(base, "split_and_check.py")
        if os.path.isfile(split_py):
            import subprocess

            r = subprocess.run([sys.executable, split_py], cwd=base)
            if r.returncode != 0:
                print(
                    f"[WARN] split_and_check 退出码 {r.returncode}（若有孤儿文件可忽略），"
                    "请确认 train/val jsonl 已更新",
                )
            else:
                print("已重划分 train_dataset.jsonl / val_dataset.jsonl")
        else:
            print(f"[WARN] 未找到 {split_py}，跳过重划分")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
