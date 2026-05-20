#!/usr/bin/env python3
"""
混合多个训练数据目录到统一输出目录。

每个源目录需为扁平结构（与 raw_data_01-04 一致）::
  images/{case_id}.jpg
  crop/{case_id}.jpg
  yuyan/{case_id}.jpg
  label.csv
  dataset.jsonl

输出目录::
  <dst>/images|crop|yuyan/   ← 从各源复制图片（跳过 .json）
  <dst>/label.csv            ← 合并各源 label.csv
  <dst>/dataset.jsonl        ← 由合并后的 label 重新生成（图像路径相对 dst 根目录）

用法::

    python tool/mix_train_data.py

    python tool/mix_train_data.py \
      --source /mnt/public-data/user/ziroujiang/raw_data_01-04 \
      --source /mnt/public-data/user/ziroujiang/generate_ground_irregularity \
      --source /mnt/public-data/user/ziroujiang/trigger60000 \
      --dst /mnt/public-data/user/ziroujiang/train_data_v2

    # 仅合并标签与 JSONL，不复制图片
    python tool/mix_train_data.py --no-copy

    # 合并已有 dataset.jsonl（不重新生成）
    python tool/mix_train_data.py --merge-jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Sequence, Set, Tuple

_tool_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tool_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tool.export_feishu_labels_csv import (  # noqa: E402
    generate_training_jsonl,
    read_label_csv,
)

IMAGE_SUBDIRS = ("images", "crop", "yuyan")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_FIELDS = (
    "case_id",
    "entity_existence",
    "geometry_relation",
    "object_type",
)

DEFAULT_SOURCES = [
    "/mnt/public-data/user/ziroujiang/generate_ground_irregularity",
    "/mnt/public-data/user/ziroujiang/raw_data_01-04",
    "/mnt/public-data/user/ziroujiang/trigger60000",
]
DEFAULT_DST = "/mnt/public-data/user/ziroujiang/train_data_v2"


def _is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def _copy_image_subdirs(
    src_root: str,
    dst_root: str,
    *,
    dry_run: bool,
    on_conflict: str,
) -> Dict[str, int]:
    """从 src 复制 images/crop/yuyan 下的图片到 dst。返回各子目录复制数量。"""
    stats = {sub: 0 for sub in IMAGE_SUBDIRS}
    skipped_conflict = 0
    skipped_non_image = 0

    for sub in IMAGE_SUBDIRS:
        src_dir = os.path.join(src_root, sub)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(dst_root, sub)
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)

        for name in sorted(os.listdir(src_dir)):
            src_path = os.path.join(src_dir, name)
            if not os.path.isfile(src_path):
                continue
            if not _is_image_file(name):
                skipped_non_image += 1
                continue

            dst_path = os.path.join(dst_dir, name)
            if os.path.exists(dst_path):
                if on_conflict == "skip":
                    skipped_conflict += 1
                    continue
                if on_conflict == "error":
                    raise FileExistsError(
                        f"冲突: {dst_path} 已存在（来自更早的源目录）"
                    )
                # on_conflict == "overwrite": 继续覆盖

            if not dry_run:
                shutil.copy2(src_path, dst_path)
            stats[sub] += 1

    stats["_skipped_conflict"] = skipped_conflict
    stats["_skipped_non_image"] = skipped_non_image
    return stats


def _merge_label_csvs(
    sources: Sequence[str],
    *,
    on_conflict: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """按 sources 顺序合并 label.csv。返回 (rows, warnings)。"""
    merged: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    warnings: List[str] = []

    for src_root in sources:
        label_path = os.path.join(src_root, "label.csv")
        if not os.path.isfile(label_path):
            warnings.append(f"[WARN] 无 label.csv: {label_path}")
            continue
        rows = read_label_csv(label_path)
        for row in rows:
            cid = str(row.get("case_id", "")).strip()
            if not cid:
                continue
            if cid in merged:
                prev_src = merged[cid].get("_source", "?")
                msg = (
                    f"[WARN] case_id 重复: {cid} "
                    f"({prev_src} → {src_root})"
                )
                if on_conflict == "error":
                    raise ValueError(msg)
                warnings.append(msg)
                if on_conflict == "skip":
                    continue
            row = {k: str(row.get(k, "")).strip() for k in LABEL_FIELDS}
            row["_source"] = src_root
            if cid not in merged:
                order.append(cid)
            merged[cid] = row

    out_rows: List[Dict[str, str]] = []
    for cid in order:
        r = {k: merged[cid][k] for k in LABEL_FIELDS}
        out_rows.append(r)
    return out_rows, warnings


def _write_label_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(LABEL_FIELDS))
        w.writeheader()
        w.writerows(rows)


def _filter_rows_with_images(
    rows: List[Dict[str, str]], dst_root: str
) -> Tuple[List[Dict[str, str]], List[str]]:
    """仅保留三张图均存在的样本。"""
    kept: List[Dict[str, str]] = []
    missing: List[str] = []
    for row in rows:
        cid = row["case_id"]
        paths = [
            os.path.join(dst_root, "images", f"{cid}.jpg"),
            os.path.join(dst_root, "crop", f"{cid}.jpg"),
            os.path.join(dst_root, "yuyan", f"{cid}.jpg"),
        ]
        if all(os.path.isfile(p) for p in paths):
            kept.append(row)
        else:
            missing.append(cid)
    return kept, missing


def _merge_dataset_jsonl(
    sources: Sequence[str],
    dst_path: str,
    *,
    valid_case_ids: Set[str],
    dry_run: bool,
) -> int:
    """拼接各源 dataset.jsonl，仅保留 valid_case_ids 中的样本。"""
    lines: List[str] = []
    seen: Set[str] = set()

    for src_root in sources:
        jsonl_path = os.path.join(src_root, "dataset.jsonl")
        if not os.path.isfile(jsonl_path):
            continue
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                images = obj.get("images") or []
                if not images:
                    continue
                # case_id 从第一张图文件名推断
                base = os.path.basename(str(images[0]))
                cid = os.path.splitext(base)[0]
                if cid not in valid_case_ids or cid in seen:
                    continue
                # 规范化路径为相对 dst 根目录
                obj["images"] = [
                    f"images/{cid}.jpg",
                    f"crop/{cid}.jpg",
                    f"yuyan/{cid}.jpg",
                ]
                lines.append(json.dumps(obj, ensure_ascii=False))
                seen.add(cid)

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    return len(lines)


def _remove_extra_jsonl(dst_root: str, keep: str = "dataset.jsonl") -> None:
    for name in (
        "entity_existence.jsonl",
        "geometry_relation.jsonl",
        "object_type.jsonl",
    ):
        p = os.path.join(dst_root, name)
        if os.path.isfile(p) and name != keep:
            os.remove(p)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="混合多个训练数据目录到统一输出目录"
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="源数据目录，可多次指定；按顺序合并",
    )
    parser.add_argument("--dst", default=DEFAULT_DST, help="输出目录")
    parser.add_argument(
        "--on-conflict",
        choices=("overwrite", "skip", "error"),
        default="overwrite",
        help="case_id 或图片文件名冲突时的策略（默认后者覆盖前者）",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="不复制图片，仅合并 label.csv / dataset.jsonl",
    )
    parser.add_argument(
        "--merge-jsonl",
        action="store_true",
        help="拼接各源 dataset.jsonl；默认根据合并后的 label.csv 重新生成",
    )
    parser.add_argument(
        "--all-jsonl",
        action="store_true",
        help="同时生成 entity_existence / geometry_relation / object_type JSONL",
    )
    parser.add_argument(
        "--no-filter-missing-images",
        action="store_true",
        help="不过滤缺少三张图的样本",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写入")
    args = parser.parse_args()

    sources = args.source or list(DEFAULT_SOURCES)
    dst_root = os.path.abspath(args.dst)

    for src in sources:
        if not os.path.isdir(src):
            print(f"[ERROR] 源目录不存在: {src}", file=sys.stderr)
            return 1

    if not args.dry_run and not args.no_copy:
        os.makedirs(dst_root, exist_ok=True)

    # 1) 复制图片
    total_copy = {sub: 0 for sub in IMAGE_SUBDIRS}
    if not args.no_copy:
        for src in sources:
            print(f"[copy] {src} → {dst_root}")
            stats = _copy_image_subdirs(
                src,
                dst_root,
                dry_run=args.dry_run,
                on_conflict=args.on_conflict,
            )
            for sub in IMAGE_SUBDIRS:
                total_copy[sub] += stats[sub]
                if stats[sub]:
                    print(f"  {sub}/: +{stats[sub]}")
            if stats["_skipped_conflict"]:
                print(f"  跳过冲突: {stats['_skipped_conflict']}")
    else:
        print("[copy] 已跳过 (--no-copy)")

    # 2) 合并 label.csv
    rows, warnings = _merge_label_csvs(sources, on_conflict=args.on_conflict)
    for w in warnings:
        print(w)
    print(f"[label] 合并 {len(rows)} 条")

    if not args.no_filter_missing_images and not args.dry_run:
        rows, missing = _filter_rows_with_images(rows, dst_root)
        if missing:
            print(f"[label] 缺少图片被过滤: {len(missing)} 条")
        print(f"[label] 保留 {len(rows)} 条（三张图齐全）")
    elif args.dry_run:
        print(f"[label] dry-run 跳过缺图过滤（实际运行后会按三张图齐全过滤）")

    label_path = os.path.join(dst_root, "label.csv")
    if not args.dry_run:
        _write_label_csv(label_path, rows)
        print(f"[label] 已写入: {label_path}")

    valid_ids = {r["case_id"] for r in rows}

    # 3) dataset.jsonl
    dataset_path = os.path.join(dst_root, "dataset.jsonl")
    if args.merge_jsonl:
        n = _merge_dataset_jsonl(
            sources,
            dataset_path,
            valid_case_ids=valid_ids,
            dry_run=args.dry_run,
        )
        print(f"[dataset] 合并写入 {n} 条 → {dataset_path}")
    else:
        if args.dry_run:
            print(f"[dataset] 将按 {len(rows)} 条 label 重新生成")
        else:
            counts = generate_training_jsonl(rows, dst_root)
            if not args.all_jsonl:
                _remove_extra_jsonl(dst_root)
            print(
                f"[dataset] 已生成 dataset.jsonl: {counts.get('dataset', 0)} 条"
            )
            if args.all_jsonl:
                for k in ("entity_existence", "geometry_relation", "object_type"):
                    print(f"  {k}.jsonl: {counts.get(k, 0)} 条")

    # 汇总
    print("\n======== 完成 ========")
    print(f"输出目录: {dst_root}")
    if not args.no_copy:
        for sub in IMAGE_SUBDIRS:
            print(f"  {sub}/: {total_copy[sub]} 张（本次复制）")
    print(f"  label.csv: {len(rows)} 条")
    if args.dry_run:
        print("  (dry-run，未实际写入)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
