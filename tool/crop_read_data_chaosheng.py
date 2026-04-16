#!/usr/bin/env python3
"""
从 read_data 读取给定 tag 列表下的各 case（时间戳目录），计算超声障碍在 AVM 上的质心像素坐标，
将「crop_{tag}_{时间戳}: 坐标」写入清单文件（默认 ``<crop-dir>/chaosheng_centroids.txt``，与裁剪图同目录），并按质心裁剪固定尺寸（默认 150×150）保存到 ``crop-dir``。
**优先**使用已有 ``draw_image/<tag>/<ts>/avm.jpg``（Step5 结果）；**若无**，则从 ``generate`` 取原始 AVM，按与 Step5 相同的 ``draw_obstacles_on_bev`` 现场叠绘后再裁剪，保证输出为「已绘制」的局部图。

输出图片文件名：``crop_{tag_id}_{timestamp_us}.jpg``（同一帧多个超声质心时仅按**第一个**质心裁剪并写该名，其余在 stderr 提示）。

用法示例::

  # tag 列表 JSON：对象形如 ``{"129678972": 0, ...}`` 时取键为 tag_id；或 JSON 数组 ``[129678972, ...]``
  python tool/crop_read_data_chaosheng.py --tag-root /home/jiangzirou/avp_promptkit/get_data/test.json

  # tag 列表目录：其下每个子目录名为 tag_id（纯数字）
  python tool/crop_read_data_chaosheng.py --tag-root /path/to/folder_with_tag_subdirs

  # 或直接指定多个 tag
  python tool/crop_read_data_chaosheng.py --tags 130072435 130072425

  # 指定清单文件名与裁剪输出目录
  python tool/crop_read_data_chaosheng.py --tags 130072435 --manifest my_centroids.txt --crop-dir /tmp/crops

依赖：需能访问 DR 元数据接口以解析 Heavy bag 与 AVM 路径（与 vlm 绘图一致）；失败时可加 ``--scan-generate-dir`` 在 generate 下全局扫描时间戳匹配。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from get_data.get_meta_data import get_meta_data
from vlm.panoramic_projector import PanoramicProjector

DEFAULT_CROP_ROOT = "/mnt/public-data/user/ziroujiang/datasets/crop"
DEFAULT_MANIFEST_BASENAME = "chaosheng_centroids.txt"

AVM_MATCH_TOLERANCE_US = 50_000
FOCAL_LENGTH = 162.6
CAMERA_HEIGHT = 3.44


def _discover_tags_from_dir(tag_root: str) -> List[int]:
    tags: List[int] = []
    if not os.path.isdir(tag_root):
        return tags
    for name in sorted(os.listdir(tag_root)):
        path = os.path.join(tag_root, name)
        if os.path.isdir(path) and name.isdigit():
            tags.append(int(name))
    return tags


def _load_tags_from_json(path: str) -> List[int]:
    """从 JSON 读取 tag_id 列表。

    支持：
    - ``{"129678972": 0, ...}``：取所有键（须可解析为整数）；
    - ``[129678972, "130072435", ...]``：数组元素为整数或可转整数的字符串。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[int] = []
    if isinstance(data, dict):
        for k in data.keys():
            out.append(int(str(k).strip()))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                tid = item.get("tag_id")
                if tid is not None:
                    out.append(int(tid))
                continue
            out.append(int(str(item).strip()))
    else:
        raise ValueError(f"不支持的 JSON 顶层类型: {type(data).__name__}")
    return out


def _load_tags_file(path: str) -> List[int]:
    out: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line.split()[0]))
    return out


def _build_avm_index_from_meta(tag_id: int, generate_dir: str) -> List[Tuple[int, str]]:
    """与 avp_vlm_pipeline_avm.draw_single_tag 一致：仅 Heavy bag 目录下的 AVM 图。"""
    meta = get_meta_data(tag_id=tag_id)
    if not meta or not meta.get("body"):
        return []
    bag_list = meta["body"][0].get("bagsName") or []
    bag_list = sorted(b for b in bag_list if "Heavy" in b)
    bag_list = [item.split(".")[0] for item in bag_list]
    all_avm_files: List[Tuple[int, str]] = []
    for bag in bag_list:
        bag_path = os.path.join(generate_dir, bag)
        if not os.path.isdir(bag_path):
            continue
        for fname in os.listdir(bag_path):
            base, ext = os.path.splitext(fname)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                avm_ts = int(base)
            except ValueError:
                continue
            all_avm_files.append((avm_ts, os.path.join(bag_path, fname)))
    return all_avm_files


def _build_avm_index_scan_generate(generate_dir: str) -> List[Tuple[int, str]]:
    """不依赖元数据：扫描 generate 下所有子目录中的时间戳命名图像。"""
    all_avm_files: List[Tuple[int, str]] = []
    if not os.path.isdir(generate_dir):
        return all_avm_files
    for bag in os.listdir(generate_dir):
        bag_path = os.path.join(generate_dir, bag)
        if not os.path.isdir(bag_path):
            continue
        for fname in os.listdir(bag_path):
            base, ext = os.path.splitext(fname)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                avm_ts = int(base)
            except ValueError:
                continue
            all_avm_files.append((avm_ts, os.path.join(bag_path, fname)))
    return all_avm_files


def _match_avm_path(
    ts_us: int,
    all_avm_files: Sequence[Tuple[int, str]],
    tolerance: int = AVM_MATCH_TOLERANCE_US,
) -> Optional[str]:
    best: Optional[str] = None
    best_diff = tolerance + 1
    for avm_ts, fpath in all_avm_files:
        diff = abs(avm_ts - ts_us)
        if diff < best_diff:
            best_diff = diff
            best = fpath
    return best if best_diff <= tolerance else None


def _centroids_for_chaosheng(
    projector: PanoramicProjector,
    chaosheng: List[Dict[str, Any]],
    ground: Dict[str, Any],
    image_height: int,
    image_width: int,
    skip_fs_car: bool,
) -> List[Tuple[int, int]]:
    """与 PanoramicProjector.draw_obstacles_on_bev 超声段质心一致。"""
    pos: List[Tuple[int, int]] = []
    for obstacle in chaosheng:
        if skip_fs_car and obstacle.get("freespaceType", "") == "FS_CAR":
            continue
        polygon_area = obstacle.get("polygonArea", {}).get("point", [])
        if not polygon_area:
            continue
        points_3d = []
        for point in polygon_area:
            points_3d.append(
                [float(point.get("x", 0)), float(point.get("y", 0)), float(point.get("z", 0))]
            )
        pts = np.array(points_3d, dtype=np.float32)
        if len(pts) == 0:
            continue
        try:
            points_2d = projector.transform_sensor_to_avm_image(
                pts,
                ground,
                virtual_camera_focal_length=FOCAL_LENGTH,
                virtual_camera_height=CAMERA_HEIGHT,
                image_height=image_height,
                image_width=image_width,
            )
        except Exception:
            continue
        points_2d_int = points_2d.astype(np.int32)
        valid_points = (
            points_2d_int[:-1]
            if len(points_2d_int) > 1 and np.array_equal(points_2d_int[0], points_2d_int[-1])
            else points_2d_int
        )
        if len(valid_points) < 2:
            continue
        center = np.mean(valid_points, axis=0)
        pos.append((int(round(center[0])), int(round(center[1]))))
    return pos


def render_bev_from_raw_avm(
    projector: PanoramicProjector,
    avm_bgr: np.ndarray,
    obstacle: List[Dict[str, Any]],
    chaosheng: List[Dict[str, Any]],
    ground: Dict[str, Any],
    planning_point: List,
    pose: Dict[str, Any],
    vehicle2sensing: Dict[str, Any],
    car_config: Dict[str, Any],
    chaosheng_pixel_radius: Optional[int],
    ignore_fs_types: Optional[List[str]],
) -> np.ndarray:
    """与 avp_vlm_pipeline_avm Step5 中 ``draw_obstacles_on_bev`` 分支一致（仅 BEV 叠绘，不含鱼眼）。"""
    planning_point = projector.world2vehicle2sensing_planning(
        planning_point, pose, vehicle2sensing
    )
    to_tail = car_config["back_edge_to_center"]
    for point in planning_point:
        point[0] -= to_tail
    if len(planning_point) > 0:
        planning_point_df = pd.DataFrame(planning_point, columns=["x", "y", "z"])
        planning_point_df = planning_point_df.drop_duplicates()
        planning_point = planning_point_df.values.tolist()
    ignore_fs = set(ignore_fs_types or [])
    has_non_fs_car = any(o.get("freespaceType", "") != "FS_CAR" for o in chaosheng)
    if not has_non_fs_car:
        return avm_bgr.copy()
    bev_img, _pos, _yellow = projector.draw_obstacles_on_bev(
        avm_bgr,
        obstacle,
        chaosheng,
        ground,
        FOCAL_LENGTH,
        CAMERA_HEIGHT,
        planning_point,
        chaosheng_pixel_radius=chaosheng_pixel_radius,
        ignore_camera_freespace_types=ignore_fs if ignore_fs else None,
    )
    return bev_img


def _crop_centered(
    image: np.ndarray,
    cx: int,
    cy: int,
    size: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    if x2 - x1 < size:
        x1 = max(0, x2 - size)
    if y2 - y1 < size:
        y1 = max(0, y2 - size)
    x1 = max(0, min(x1, w - size))
    y1 = max(0, min(y1, h - size))
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    return image[y1:y2, x1:x2].copy()


def run(args: argparse.Namespace) -> int:
    tags: List[int] = []
    if args.tag_root:
        tr = os.path.abspath(os.path.expanduser(args.tag_root))
        if os.path.isfile(tr) and tr.lower().endswith(".json"):
            try:
                tags.extend(_load_tags_from_json(tr))
            except (ValueError, OSError, json.JSONDecodeError) as e:
                print(f"错误: 读取 tag JSON 失败 {tr}: {e}", file=sys.stderr)
                return 1
        elif os.path.isdir(tr):
            tags.extend(_discover_tags_from_dir(tr))
        else:
            print(
                f"错误: --tag-root 既不是 .json 文件也不是目录: {args.tag_root}",
                file=sys.stderr,
            )
            return 1
    if args.tags_file:
        tags.extend(_load_tags_file(args.tags_file))
    if args.tags:
        tags.extend(args.tags)
    tags = sorted(set(tags))
    if not tags:
        print(
            "错误: 未指定任何 tag（使用 --tag-root 目录或 .json / --tags-file / --tags）",
            file=sys.stderr,
        )
        return 1

    read_data = os.path.abspath(args.read_data)
    generate_dir = os.path.abspath(args.generate_dir)
    draw_image_dir = os.path.abspath(args.draw_image_dir)
    crop_root = os.path.abspath(args.crop_dir)
    os.makedirs(crop_root, exist_ok=True)

    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
    else:
        manifest_path = os.path.join(crop_root, DEFAULT_MANIFEST_BASENAME)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    projector = PanoramicProjector()

    lines_out: List[str] = []
    n_crop = 0

    for tag_id in tags:
        data_path = os.path.join(read_data, str(tag_id))
        required = [
            "vehicle2sensing.json",
            "ground.json",
            "cameras_parameters.json",
            "car_config.json",
        ]
        missing = [f for f in required if not os.path.isfile(os.path.join(data_path, f))]
        if missing:
            print(f"[WARN] tag={tag_id} 缺少 {missing}，跳过", file=sys.stderr)
            continue

        with open(os.path.join(data_path, "vehicle2sensing.json"), "r", encoding="utf-8") as f:
            vehicle2sensing = json.load(f)
        with open(os.path.join(data_path, "ground.json"), "r", encoding="utf-8") as f:
            ground = json.load(f)
        with open(os.path.join(data_path, "car_config.json"), "r", encoding="utf-8") as f:
            car_config = json.load(f)

        if args.scan_generate_dir:
            all_avm_files = _build_avm_index_scan_generate(generate_dir)
        else:
            all_avm_files = _build_avm_index_from_meta(tag_id, generate_dir)
            if not all_avm_files:
                all_avm_files = _build_avm_index_scan_generate(generate_dir)
                if all_avm_files:
                    print(
                        f"[WARN] tag={tag_id} 元数据无 Heavy AVM，已回退为扫描 generate 目录",
                        file=sys.stderr,
                    )

        subdirs = [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()
        ]
        for ts in sorted(subdirs, key=int):
            item_path = os.path.join(data_path, ts)
            cs_path = os.path.join(item_path, "chaosheng.json")
            if not os.path.isfile(cs_path):
                continue
            with open(cs_path, "r", encoding="utf-8") as f:
                chaosheng = json.load(f)
            ignore_fs = set(args.ignore_fs_types or [])
            if ignore_fs:
                chaosheng = [
                    o for o in chaosheng if o.get("freespaceType", "") not in ignore_fs
                ]
            with open(os.path.join(item_path, "obstacle.json"), "r", encoding="utf-8") as f:
                obstacle = json.load(f)
            with open(os.path.join(item_path, "pose.json"), "r", encoding="utf-8") as f:
                pose = json.load(f)
            plan_path = os.path.join(item_path, "plan.json")
            if os.path.isfile(plan_path):
                with open(plan_path, "r", encoding="utf-8") as f:
                    planning_point = json.load(f)
            else:
                planning_point = []

            projector.apply_chaosheng_z_from_camera_ground_plane(chaosheng, obstacle)
            obstacle = projector.world2vehicle2sensing(obstacle, pose, vehicle2sensing)
            chaosheng = projector.world2vehicle2sensing_chaosheng(
                chaosheng, pose, vehicle2sensing
            )

            drawn_avm = os.path.join(draw_image_dir, str(tag_id), ts, "avm.jpg")
            use_cached_draw = (not args.from_generate_only) and os.path.isfile(drawn_avm)
            if use_cached_draw:
                img = cv2.imread(drawn_avm)
                if img is None:
                    print(
                        f"[WARN] tag={tag_id} ts={ts} 读已绘制图失败: {drawn_avm}，尝试现场绘制",
                        file=sys.stderr,
                    )
                    use_cached_draw = False

            if not use_cached_draw:
                raw_path = _match_avm_path(int(ts), all_avm_files)
                if not raw_path or not os.path.isfile(raw_path):
                    print(
                        f"[WARN] tag={tag_id} ts={ts} generate 无匹配 AVM（容差 {AVM_MATCH_TOLERANCE_US}μs），跳过",
                        file=sys.stderr,
                    )
                    continue
                raw = cv2.imread(raw_path)
                if raw is None:
                    print(
                        f"[WARN] tag={tag_id} ts={ts} 读原始 AVM 失败: {raw_path}",
                        file=sys.stderr,
                    )
                    continue
                img = render_bev_from_raw_avm(
                    projector,
                    raw,
                    obstacle,
                    chaosheng,
                    ground,
                    planning_point,
                    pose,
                    vehicle2sensing,
                    car_config,
                    args.chaosheng_pixel_radius,
                    args.ignore_fs_types,
                )

            if img is None or img.size == 0:
                print(f"[WARN] tag={tag_id} ts={ts} 无有效图像，跳过", file=sys.stderr)
                continue
            ih, iw = img.shape[:2]

            centroids = _centroids_for_chaosheng(
                projector,
                chaosheng,
                ground,
                ih,
                iw,
                skip_fs_car=not args.include_fs_car,
            )
            if not centroids:
                continue
            if len(centroids) > 1:
                print(
                    f"[WARN] tag={tag_id} ts={ts} 共 {len(centroids)} 个超声质心，"
                    f"仅按第一个裁剪 → crop_{tag_id}_{ts}.jpg",
                    file=sys.stderr,
                )
            cx, cy = centroids[0]
            stem = f"crop_{tag_id}_{ts}"
            lines_out.append(f"{stem}: {cx},{cy}")
            patch = _crop_centered(img, cx, cy, args.size)
            if patch.size == 0:
                continue
            out_path = os.path.join(crop_root, f"{stem}.jpg")
            cv2.imwrite(out_path, patch)
            n_crop += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")

    print(f"清单已写入: {manifest_path}（共 {len(lines_out)} 行）")
    print(f"裁剪图已写入: {crop_root}（共 {n_crop} 张，{args.size}×{args.size}）")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="从 read_data 计算超声质心并裁剪 AVM 区域"
    )
    p.add_argument(
        "--tag-root",
        type=str,
        default="",
        help="tag 来源：若为目录则其下纯数字子目录名为 tag；若为 .json 文件则解析键或数组为 tag 列表",
    )
    p.add_argument(
        "--tags-file",
        type=str,
        default="",
        help="每行一个 tag_id 的文本文件",
    )
    p.add_argument(
        "--tags",
        type=int,
        nargs="*",
        default=[],
        help="命令行直接列出 tag_id",
    )
    p.add_argument(
        "--read-data",
        type=str,
        default=config.READ_DATA_DIR,
        help=f"read_data 根目录（默认: {config.READ_DATA_DIR}）",
    )
    p.add_argument(
        "--generate-dir",
        type=str,
        default=config.GENERATE_DIR,
        help=f"原始 AVM 目录；无 draw_image/avm.jpg 时按时间戳匹配（默认: {config.GENERATE_DIR}）",
    )
    p.add_argument(
        "--draw-image-dir",
        type=str,
        default=config.DRAW_IMAGE_DIR,
        help=f"已绘制 AVM 根目录，优先使用 <tag>/<ts>/avm.jpg（默认: {config.DRAW_IMAGE_DIR}）",
    )
    p.add_argument(
        "--from-generate-only",
        action="store_true",
        help="不读 draw_image；始终用 generate 原始图并按 Step5 叠绘后再裁剪",
    )
    p.add_argument(
        "--chaosheng-pixel-radius",
        type=int,
        default=30,
        help="与 Step5 一致：BEV 超声-相机关联半径（像素），传给 draw_obstacles_on_bev",
    )
    p.add_argument(
        "--ignore-fs-types",
        type=str,
        nargs="*",
        default=[],
        help="与 Step5 一致：忽略的 freespaceType（超声过滤 + 相机黄线过滤）",
    )
    p.add_argument(
        "--crop-dir",
        type=str,
        default=DEFAULT_CROP_ROOT,
        help=f"裁剪输出根目录（默认: {DEFAULT_CROP_ROOT}）",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=f"case:坐标 清单文件路径（默认: <crop-dir>/{DEFAULT_MANIFEST_BASENAME}）",
    )
    p.add_argument(
        "--size",
        type=int,
        default=150,
        help="裁剪正方形边长（像素）",
    )
    p.add_argument(
        "--include-fs-car",
        action="store_true",
        help="包含 FS_CAR 超声段（默认与绘图一致：跳过 FS_CAR）",
    )
    p.add_argument(
        "--scan-generate-dir",
        action="store_true",
        help="不调用元数据，仅在 generate 下全局扫描时间戳图像做匹配",
    )
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
