#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""遍历 read_data 下某 tag 的每帧 JSON，打印超声与相机障碍的 freespaceType、类型及顶点个数（不打印顶点坐标）。

- **超声**：``chaosheng.json``（与绘图一致时可带 ``--ignore-fs-types`` 先剔除部分类型）。
- **相机**：``obstacle.json`` 中 ``sensorType == "CAMERA"`` 的条目。

默认启用 BEV 像素筛选 **R=30**（与 pipeline ``chaosheng_pixel_radius`` 一致），**仅用于相机**：
只列出「任一顶点到**全体**超声多边形顶点 BEV 点云」最小像素距 ≤ R 的 CAMERA 条目（与
``draw_obstacles_on_bev(..., chaosheng_pixel_radius=R)`` 一致）。**超声始终全量列出**（仍可先
``--ignore-fs-types`` 剔除）。传 ``--near-camera-pixels 0`` 则相机也不筛选、全量打印。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, List, Set

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vlm.panoramic_projector import PanoramicProjector


def polygon_vertex_count(ob: dict[str, Any]) -> int:
    """polygonArea.point 顶点个数。"""
    pts = (ob.get("polygonArea") or {}).get("point")
    return len(pts) if pts else 0


def _project_ultrasonic_polygon_pixels(
    projector: PanoramicProjector,
    item: dict[str, Any],
    ground: dict[str, Any],
    focal_length: float,
    camera_height: float,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    poly = (item.get("polygonArea") or {}).get("point") or []
    if not poly:
        return np.empty((0, 2), dtype=np.float32)
    rows = []
    for p in poly:
        rows.append(
            [
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("z", 0.0)),
            ]
        )
    pts = np.array(rows, dtype=np.float32)
    try:
        return projector.transform_sensor_to_avm_image(
            pts,
            ground,
            focal_length,
            camera_height,
            image_height,
            image_width,
        )
    except Exception:
        return np.empty((0, 2), dtype=np.float32)


def bev_camera_indices_near_ultrasonic_pixels(
    chaosheng: List[dict[str, Any]],
    obstacles: List[dict[str, Any]],
    pose: dict[str, Any],
    vehicle2sensing: dict[str, Any],
    ground: dict[str, Any],
    pixel_radius: float,
    focal_length: float = 162.6,
    camera_height: float = 3.44,
    image_height: int = 800,
    image_width: int = 640,
) -> Set[int]:
    """与 ``draw_obstacles_on_bev`` 的 chaosheng_pixel_radius 一致：返回 obstacle 列表中下标 j，
    其 CAMERA 障碍在 BEV 上至少有一顶点到全体超声多边形顶点像素距 ≤ pixel_radius。"""
    projector = PanoramicProjector()
    ch = copy.deepcopy(chaosheng)
    obs = copy.deepcopy(obstacles)
    projector.apply_chaosheng_z_from_camera_ground_plane(ch, obs)
    obs = projector.world2vehicle2sensing(obs, pose, vehicle2sensing)
    ch = projector.world2vehicle2sensing_chaosheng(ch, pose, vehicle2sensing)

    all_uv_ultra: List[np.ndarray] = []
    for u in ch:
        if u.get("sensorType") != "ULTRASONIC":
            continue
        uv = _project_ultrasonic_polygon_pixels(
            projector,
            u,
            ground,
            focal_length,
            camera_height,
            image_height,
            image_width,
        )
        if uv.size:
            all_uv_ultra.append(uv)
    chaosheng_img_pts = (
        np.vstack(all_uv_ultra)
        if all_uv_ultra
        else np.empty((0, 2), dtype=np.float32)
    )

    cam_related: Set[int] = set()
    if len(chaosheng_img_pts) > 0:
        for j, c in enumerate(obs):
            if c.get("sensorType") != "CAMERA":
                continue
            if not (c.get("polygonArea") or {}).get("point"):
                continue
            if projector._obstacle_near_chaosheng_pixels(
                c,
                chaosheng_img_pts,
                pixel_radius,
                ground,
                focal_length,
                camera_height,
                image_height,
                image_width,
            ):
                cam_related.add(j)

    return cam_related


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--read-dir",
        default="/mnt/public-data/user/ziroujiang/avp/read_data",
        help="read_data 根目录（含 tag 子目录及 vehicle2sensing.json / ground.json）",
    )
    parser.add_argument(
        "--tags",
        type=int,
        nargs="+",
        default=[129346680],
        help="tag id 列表",
    )
    parser.add_argument(
        "--near-camera-pixels",
        type=float,
        default=30.0,
        metavar="R",
        help="仅筛选相机：BEV 上与全体超声顶点距≤R 的 CAMERA（同 chaosheng_pixel_radius，默认 30）；"
        "超声始终全量。设为 0 或负数则相机也不筛选",
    )
    parser.add_argument(
        "--ignore-fs-types",
        nargs="*",
        default=[],
        help="与绘图一致：这些 freespaceType 的超声条目在筛选前剔除（可选）",
    )
    args = parser.parse_args()
    read_dir = args.read_dir
    ignore_fs = set(args.ignore_fs_types or [])
    pixel_radius_arg = (
        None if args.near_camera_pixels <= 0 else args.near_camera_pixels
    )

    for tag in args.tags:
        tag_dir = os.path.join(read_dir, str(tag))
        if not os.path.isdir(tag_dir):
            print("tag=%d: 目录不存在\n" % tag)
            continue
        v2s_path = os.path.join(tag_dir, "vehicle2sensing.json")
        ground_path = os.path.join(tag_dir, "ground.json")
        effective_radius = pixel_radius_arg
        vehicle2sensing = None
        ground = None
        if effective_radius is not None:
            if not os.path.isfile(v2s_path) or not os.path.isfile(ground_path):
                print(
                    "tag=%d: 缺少标定 %s / %s，本 tag 无法按 R 筛相机，相机打印全量（超声仍全量）\n"
                    % (tag, v2s_path, ground_path)
                )
                effective_radius = None
            else:
                with open(v2s_path, encoding="utf-8") as f:
                    vehicle2sensing = json.load(f)
                with open(ground_path, encoding="utf-8") as f:
                    ground = json.load(f)

        timestamps = sorted(
            [d for d in os.listdir(tag_dir) if os.path.isdir(os.path.join(tag_dir, d))],
            key=int,
        )
        print("=== tag=%d (%d 个时间戳) ===" % (tag, len(timestamps)))
        if effective_radius is not None:
            print(
                "  筛选: 仅相机，BEV R=%.1f（与绘图 chaosheng_pixel_radius：相机顶点距全体超声顶点）"
                % effective_radius
            )
        for ts in timestamps:
            cs_path = os.path.join(tag_dir, ts, "chaosheng.json")
            if not os.path.isfile(cs_path):
                print("  ts=%s: 无 chaosheng.json" % ts)
                continue
            with open(cs_path, encoding="utf-8") as f:
                data = json.load(f)
            if ignore_fs:
                data = [o for o in data if o.get("freespaceType", "") not in ignore_fs]

            ob_path = os.path.join(tag_dir, ts, "obstacle.json")
            obstacle_missing = not os.path.isfile(ob_path)
            obstacles: List[dict[str, Any]] = []
            if not obstacle_missing:
                with open(ob_path, encoding="utf-8") as f:
                    obstacles = json.load(f)

            cam_related: Set[int] | None = None
            if effective_radius is not None:
                pose_path = os.path.join(tag_dir, ts, "pose.json")
                if not os.path.isfile(ob_path) or not os.path.isfile(pose_path):
                    print(
                        "  ts=%s: 缺少 obstacle.json 或 pose.json，无法按 R 筛相机，相机全量"
                        % ts
                    )
                else:
                    with open(pose_path, encoding="utf-8") as f:
                        pose = json.load(f)
                    cam_related = bev_camera_indices_near_ultrasonic_pixels(
                        data,
                        obstacles,
                        pose,
                        vehicle2sensing,
                        ground,
                        effective_radius,
                    )

            # ---------- 超声 (chaosheng.json)，不做 R 筛选 ----------
            total_u = len(data)
            print("  ts=%s — 超声障碍 %d 条（全量）" % (ts, total_u))

            for i, o in enumerate(data):
                ft = o.get("freespaceType", "(空)")
                n = polygon_vertex_count(o)
                print(
                    "    [超声 %d] freespaceType=%s, 顶点数=%d"
                    % (i, ft if ft else "(空)", n)
                )

            # ---------- 相机 (obstacle.json, CAMERA) ----------
            cam_rows = [
                (j, o)
                for j, o in enumerate(obstacles)
                if o.get("sensorType") == "CAMERA"
            ]
            if obstacle_missing:
                print("  ts=%s — 相机障碍: 无 obstacle.json" % ts)
            elif not cam_rows:
                print(
                    "  ts=%s — 相机障碍: obstacle.json 中无 sensorType=CAMERA 条目"
                    % ts
                )
            else:
                total_c = len(cam_rows)
                if cam_related is not None:
                    n_rel = len([j for j, _ in cam_rows if j in cam_related])
                    print(
                        "  ts=%s — 相机障碍 CAMERA %d 条，R=%.0fpx 内与超声点云相关 %d 条"
                        % (ts, total_c, effective_radius, n_rel)
                    )
                else:
                    print("  ts=%s — 相机障碍 CAMERA %d 条" % (ts, total_c))

                for j, o in cam_rows:
                    if cam_related is not None and j not in cam_related:
                        continue
                    ft = o.get("freespaceType", "(空)")
                    ot = o.get("type", "(空)")
                    mt = o.get("modelType", "(空)")
                    oid = o.get("id", "(空)")
                    n = polygon_vertex_count(o)
                    print(
                        "    [相机 %d] freespaceType=%s, type=%s, modelType=%s, id=%s, 顶点数=%d"
                        % (j, ft if ft else "(空)", ot, mt, oid, n)
                    )
        print()


if __name__ == "__main__":
    main()
