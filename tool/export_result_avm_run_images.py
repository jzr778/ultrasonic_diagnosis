#!/usr/bin/env python3
"""
从 result_avm 的误检/正常结果中收集 jpg，合并 draw_image 下的 yuyan_draw.jpg，
复制到 {AVP_BASE}/{RUN_NAME}/误检|正常/，文件名带 tag 与时间戳。

用法:
  python tool/export_result_avm_run_images.py
  python tool/export_result_avm_run_images.py --run-name run_0401_yuyan

修改脚本顶部 RUN_NAME，或使用 --run-name 覆盖。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# ========= 手动配置（与命令行 --run-name 二选一，CLI 优先）=========
RUN_NAME = "run_0416_v5"

AVP_BASE = "/mnt/public-data/user/ziroujiang/avp"
RESULT_SRC = os.path.join(AVP_BASE, "result_avm")
DRAW_SRC = os.path.join(AVP_BASE, "draw_image")

def dst_name_for(tag: str, ts: str, basename: str) -> str:
    name_part = os.path.splitext(basename)[0]
    if name_part == "avm":
        return f"{tag}_{ts}.jpg"
    return f"{tag}_{ts}_{name_part}.jpg"


def copy_one(
    category: str,
    out_dir: str,
    tag: str,
    ts: str,
    src_file: str,
    basename: str,
) -> None:
    dn = dst_name_for(tag, ts, basename)
    dst_file = os.path.join(out_dir, dn)
    shutil.copy2(src_file, dst_file)
    sub = os.path.basename(out_dir)
    print(f"  {category} -> {sub}/{dn}")


def run(run_name: str) -> None:
    dst_root = os.path.join(AVP_BASE, run_name)
    mapping = {
        "misdetected": os.path.join(dst_root, "误检"),
        "normal": os.path.join(dst_root, "正常"),
    }

    for out_dir in mapping.values():
        os.makedirs(out_dir, exist_ok=True)

    for category, out_dir in mapping.items():
        cat_dir = os.path.join(RESULT_SRC, category)
        if not os.path.isdir(cat_dir):
            continue
        for tag in sorted(os.listdir(cat_dir)):
            tag_dir = os.path.join(cat_dir, tag)
            if not os.path.isdir(tag_dir):
                continue
            for ts in sorted(os.listdir(tag_dir)):
                ts_dir = os.path.join(tag_dir, ts)
                if not os.path.isdir(ts_dir):
                    continue
                for f in os.listdir(ts_dir):
                    if not f.endswith(".jpg"):
                        continue
                    if f.startswith("panoramic_"):
                        continue
                    copy_one(
                        category,
                        out_dir,
                        tag,
                        ts,
                        os.path.join(ts_dir, f),
                        f,
                    )
                draw_ts = os.path.join(DRAW_SRC, tag, ts)
                yuyan = os.path.join(draw_ts, "yuyan_draw.jpg")
                if os.path.isfile(yuyan):
                    copy_one(category, out_dir, tag, ts, yuyan, "yuyan_draw.jpg")

    print("\n--- 统计 ---")
    for label, out_dir in mapping.items():
        if not os.path.isdir(out_dir):
            print(f"{os.path.basename(out_dir)}: 0 张图片（目录不存在）")
            continue
        cnt = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
        print(f"{os.path.basename(out_dir)}: {cnt} 张图片")
    print(f"\n输出根目录: {dst_root}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 result_avm 收集 jpg 与 yuyan_draw 到 {AVP_BASE}/{RUN_NAME}/"
    )
    parser.add_argument(
        "--run-name",
        default=RUN_NAME,
        help=f"导出目录名（默认: 脚本内 RUN_NAME={RUN_NAME!r}）",
    )
    args = parser.parse_args()
    if not args.run_name.strip():
        print("错误: --run-name 不能为空", file=sys.stderr)
        return 1
    run(args.run_name.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
