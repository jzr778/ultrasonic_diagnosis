#!/usr/bin/env python3
"""
跑 pipeline.py 的 Step1–5（默认跳过 Step6 VLM 诊断）。仅把 ``id_mapping.json`` 与 pipeline
``logs/`` 写入 ``tool/output/``；其余中间产物（``samples`` / ``generate`` / ``read_data`` /
``draw_image``）仍落在 ``config.DATA_BASE``（默认 ``/mnt/public-data/user/ziroujiang/avp``）。
最后把 ``draw_image`` 下已绘制的 AVM 与单路鱼眼图按 ``{tag_id}_{timestamp_us}.jpg`` 命名复制到
目标根目录（默认 ``/mnt/public-data/user/ziroujiang/raw_data``）。同时对每张 AVM 按超声质心裁
剪到同级 ``crop/`` 子目录（复用 ``tool/crop_read_data_chaosheng.py``）：

  <dst>/images/{tag}_{ts}.jpg   ← draw_image/<tag>/<ts>/avm.jpg
  <dst>/yuyan/{tag}_{ts}.jpg    ← draw_image/<tag>/<ts>/yuyan_draw.jpg（缺失则跳过）
  <dst>/crop/{tag}_{ts}.jpg      ← 在 AVM 上按超声质心裁剪（默认 150×150，与 images/yuyan 同名）

默认**过滤 FS_CAR**（给 pipeline 传入 ``--ignore-fs-types FS_CAR``），可通过 ``--no-filter-fs-car`` 关闭；
也可用 ``--extra-ignore-fs-types`` 追加其他 freespaceType。

用法::

    # 标准：id_mapping.json 与 logs 写到 tool/output，Step1–5，复制到 raw_data
    # 1-3月数据
    python tool/collect_raw_data.py -p iffcom -v A9umFAhvg 

    # 只重绘 Step5 并复制（需 tool/output/id_mapping.json 已存在，或显式 --id-mapping）
    python tool/collect_raw_data.py --skip-steps 1 2 3 4

    # 另存最终图片到其它目录；不过滤 FS_CAR
    python tool/collect_raw_data.py --dst-root /path/to/out --no-filter-fs-car

    # 仅跑 pipeline 不复制
    python tool/collect_raw_data.py --no-copy
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Sequence

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

DEFAULT_DST_ROOT = "/mnt/public-data/user/ziroujiang/raw_data"
DEFAULT_WORK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _build_ignore_fs_types(
    filter_fs_car: bool, extra: Sequence[str]
) -> List[str]:
    ignore: List[str] = []
    if filter_fs_car:
        ignore.append("FS_CAR")
    for item in extra:
        item = (item or "").strip()
        if item and item not in ignore:
            ignore.append(item)
    return ignore


def _run_pipeline(
    project_key: str,
    view_id: str,
    skip_steps: Sequence[int],
    ignore_fs_types: Sequence[str],
    id_mapping: str,
    log_dir: str,
    extra_args: Sequence[str],
) -> None:
    pipeline_py = os.path.join(_project_root, "pipeline.py")
    if not os.path.isfile(pipeline_py):
        raise FileNotFoundError(f"找不到 {pipeline_py}")
    merged_skip = sorted({int(s) for s in skip_steps} | {6})
    if 1 in merged_skip and not os.path.isfile(id_mapping):
        raise SystemExit(
            f"错误: 已跳过 Step1，但 --id-mapping 指定的文件不存在: {id_mapping}\n"
            "请显式传 --id-mapping 或先跑一次 Step1 生成。"
        )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(id_mapping)) or ".", exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        pipeline_py,
        "-p",
        project_key,
        "-v",
        view_id,
        "--id-mapping",
        id_mapping,
        "--log-dir",
        log_dir,
    ]
    if merged_skip:
        cmd += ["--skip-steps", *[str(s) for s in merged_skip]]
    if ignore_fs_types:
        cmd += ["--ignore-fs-types", *ignore_fs_types]
    cmd += list(extra_args or [])

    print(f"[collect_raw_data] id_mapping = {id_mapping}", flush=True)
    print(f"[collect_raw_data] log_dir    = {log_dir}", flush=True)
    print(f"[collect_raw_data] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _copy_drawn_images(
    draw_image_dir: str,
    dst_root: str,
    copy_avm: bool = True,
    copy_yuyan: bool = True,
    only_tag_ids: Sequence[str] | None = None,
) -> None:
    images_dir = os.path.join(dst_root, "images")
    yuyan_dir = os.path.join(dst_root, "yuyan")
    if copy_avm:
        os.makedirs(images_dir, exist_ok=True)
    if copy_yuyan:
        os.makedirs(yuyan_dir, exist_ok=True)

    if not os.path.isdir(draw_image_dir):
        print(f"[collect_raw_data] draw_image 目录不存在: {draw_image_dir}", file=sys.stderr)
        return

    tag_filter = set(str(t) for t in (only_tag_ids or []) if str(t).isdigit())

    n_avm = n_yuyan = 0
    for tag in sorted(os.listdir(draw_image_dir)):
        tag_dir = os.path.join(draw_image_dir, tag)
        if not os.path.isdir(tag_dir) or not tag.isdigit():
            continue
        if tag_filter and tag not in tag_filter:
            continue
        for ts in sorted(os.listdir(tag_dir)):
            ts_dir = os.path.join(tag_dir, ts)
            if not os.path.isdir(ts_dir) or not ts.isdigit():
                continue
            stem = f"{tag}_{ts}"
            if copy_avm:
                src_avm = os.path.join(ts_dir, "avm.jpg")
                if os.path.isfile(src_avm):
                    dst_avm = os.path.join(images_dir, f"{stem}.jpg")
                    shutil.copy2(src_avm, dst_avm)
                    n_avm += 1
            if copy_yuyan:
                src_yu = os.path.join(ts_dir, "yuyan_draw.jpg")
                if os.path.isfile(src_yu):
                    dst_yu = os.path.join(yuyan_dir, f"{stem}.jpg")
                    shutil.copy2(src_yu, dst_yu)
                    n_yuyan += 1
    print(
        f"[collect_raw_data] 复制完成: images/={n_avm}, yuyan/={n_yuyan}  →  {dst_root}",
        flush=True,
    )


def _crop_drawn_images(
    draw_image_dir: str,
    read_data_dir: str,
    dst_root: str,
    size: int,
    ignore_fs_types: Sequence[str],
    only_tag_ids: Sequence[str] | None = None,
) -> None:
    """对 draw_image 下已绘制的 avm.jpg 按超声质心中心裁剪，输出到 <dst_root>/crop/。

    复用 tool/crop_read_data_chaosheng.py 的完整裁剪逻辑（子进程调用）。
    """
    crop_dir = os.path.join(dst_root, "crop")
    os.makedirs(crop_dir, exist_ok=True)

    if not os.path.isdir(draw_image_dir):
        print(
            f"[collect_raw_data] draw_image 目录不存在，跳过裁剪: {draw_image_dir}",
            file=sys.stderr,
        )
        return

    tag_filter = set(str(t) for t in (only_tag_ids or []) if str(t).isdigit())
    cases: List[str] = []
    for tag in sorted(os.listdir(draw_image_dir)):
        tag_dir = os.path.join(draw_image_dir, tag)
        if not os.path.isdir(tag_dir) or not tag.isdigit():
            continue
        if tag_filter and tag not in tag_filter:
            continue
        for ts in sorted(os.listdir(tag_dir)):
            ts_dir = os.path.join(tag_dir, ts)
            if not os.path.isdir(ts_dir) or not ts.isdigit():
                continue
            if not os.path.isfile(os.path.join(ts_dir, "avm.jpg")):
                continue
            cases.append(f"{tag}_{ts}")

    if not cases:
        print("[collect_raw_data] 无可裁剪 case（draw_image 下没有 avm.jpg）", flush=True)
        return

    crop_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "crop_read_data_chaosheng.py"
    )
    if not os.path.isfile(crop_script):
        print(f"[collect_raw_data] 未找到 {crop_script}，跳过裁剪", file=sys.stderr)
        return

    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", prefix="collect_crop_cases_", delete=False, encoding="utf-8"
    )
    try:
        for c in cases:
            tmp.write(c + "\n")
        tmp.flush()
        tmp.close()

        cmd = [
            sys.executable,
            crop_script,
            "--tags-file",
            tmp.name,
            "--read-data",
            read_data_dir,
            "--draw-image-dir",
            draw_image_dir,
            "--crop-dir",
            crop_dir,
            "--size",
            str(size),
        ]
        if ignore_fs_types:
            cmd += ["--ignore-fs-types", *ignore_fs_types]
        print(
            f"[collect_raw_data] crop: {len(cases)} cases → {crop_dir}  "
            f"(size={size}, ignore={list(ignore_fs_types) or '-'})",
            flush=True,
        )
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def main() -> int:
    p = argparse.ArgumentParser(
        description="跑 pipeline Step1–5 并收集已绘制 AVM / 单路鱼眼图到 raw_data 目录"
    )
    p.add_argument(
        "-p",
        "--project-key",
        default=config.FEISHU_PROJECT_KEY,
        help="飞书项目 Key，透传给 pipeline.py",
    )
    p.add_argument(
        "-v",
        "--view-id",
        default="U9zPLpFvR",
        help="飞书视图 ID（默认: U9zPLpFvR）",
    )
    p.add_argument(
        "--skip-steps",
        nargs="*",
        type=int,
        default=[],
        help="额外跳过的 pipeline 步骤（本工具总是跳过 Step6）",
    )
    p.add_argument(
        "--work-root",
        default=DEFAULT_WORK_ROOT,
        help=(
            "id_mapping.json 与 pipeline logs 的保存根目录（默认: tool/output）。"
            "仅影响这两项；samples/generate/read_data/draw_image 不受影响"
        ),
    )
    p.add_argument(
        "--id-mapping",
        default=None,
        help="tag_id ↔ feishu_id 映射 JSON；默认 <work-root>/id_mapping.json",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="pipeline 日志目录（默认: <work-root>/logs）",
    )
    p.add_argument(
        "--no-filter-fs-car",
        dest="filter_fs_car",
        action="store_false",
        help="关闭默认的 FS_CAR 过滤（默认开启：绘图时忽略 FS_CAR）",
    )
    p.set_defaults(filter_fs_car=True)
    p.add_argument(
        "--extra-ignore-fs-types",
        nargs="*",
        default=[],
        help="在 FS_CAR 之外追加的 freespaceType 忽略项",
    )
    p.add_argument(
        "--draw-image-dir",
        default=config.DRAW_IMAGE_DIR,
        help=f"绘图输出根目录（默认: {config.DRAW_IMAGE_DIR}）",
    )
    p.add_argument(
        "--dst-root",
        default=DEFAULT_DST_ROOT,
        help=f"目标根目录；AVM 放 images/ 子目录，鱼眼放 yuyan/（默认: {DEFAULT_DST_ROOT}）",
    )
    p.add_argument(
        "--no-pipeline",
        dest="run_pipeline",
        action="store_false",
        help="不执行 pipeline，仅复制现有 draw_image 结果",
    )
    p.set_defaults(run_pipeline=True)
    p.add_argument(
        "--no-copy",
        dest="do_copy",
        action="store_false",
        help="执行 pipeline 但不复制（只想生成 draw_image）",
    )
    p.set_defaults(do_copy=True)
    p.add_argument(
        "--no-avm",
        dest="copy_avm",
        action="store_false",
        help="不复制 AVM 到 images/",
    )
    p.set_defaults(copy_avm=True)
    p.add_argument(
        "--no-yuyan",
        dest="copy_yuyan",
        action="store_false",
        help="不复制 yuyan_draw 到 yuyan/",
    )
    p.set_defaults(copy_yuyan=True)
    p.add_argument(
        "--no-crop",
        dest="do_crop",
        action="store_false",
        help="不生成裁剪图（默认开启：按超声质心裁剪到 <dst-root>/crop/）",
    )
    p.set_defaults(do_crop=True)
    p.add_argument(
        "--crop-size",
        type=int,
        default=150,
        help="裁剪正方形边长（像素，默认 150）",
    )
    p.add_argument(
        "--read-data-dir",
        default=config.READ_DATA_DIR,
        help=f"read_data 根目录，用于算质心（默认: {config.READ_DATA_DIR}）",
    )
    p.add_argument(
        "--only-tags",
        nargs="*",
        default=[],
        help="仅复制这些 tag_id 的已绘制结果（拷贝阶段过滤，不影响 pipeline）",
    )
    p.add_argument(
        "--pipeline-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="其余参数原样透传给 pipeline.py（放在本命令末尾）",
    )

    args = p.parse_args()

    work_root = os.path.abspath(args.work_root)
    id_mapping = args.id_mapping or os.path.join(work_root, "id_mapping.json")
    log_dir = args.log_dir or os.path.join(work_root, "logs")
    draw_image_dir = args.draw_image_dir

    ignore_fs_types = _build_ignore_fs_types(args.filter_fs_car, args.extra_ignore_fs_types)

    if args.run_pipeline:
        _run_pipeline(
            project_key=args.project_key,
            view_id=args.view_id,
            skip_steps=args.skip_steps,
            ignore_fs_types=ignore_fs_types,
            id_mapping=id_mapping,
            log_dir=log_dir,
            extra_args=args.pipeline_extra,
        )
    else:
        print("[collect_raw_data] --no-pipeline：跳过 pipeline，只做复制", flush=True)

    if args.do_copy:
        _copy_drawn_images(
            draw_image_dir=draw_image_dir,
            dst_root=args.dst_root,
            copy_avm=args.copy_avm,
            copy_yuyan=args.copy_yuyan,
            only_tag_ids=args.only_tags,
        )
    else:
        print("[collect_raw_data] --no-copy：跳过复制", flush=True)

    if args.do_crop:
        _crop_drawn_images(
            draw_image_dir=draw_image_dir,
            read_data_dir=args.read_data_dir,
            dst_root=args.dst_root,
            size=args.crop_size,
            ignore_fs_types=ignore_fs_types,
            only_tag_ids=args.only_tags,
        )
    else:
        print("[collect_raw_data] --no-crop：跳过裁剪", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
