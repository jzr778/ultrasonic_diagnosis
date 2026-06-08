#!/usr/bin/env python3
"""
选「主相机帧离 target_ts 最近」的 Heavy bag 
→ 在主相机时间线上以 target 为中心、每隔 5 帧采 1 帧、共取 10 帧参考时间戳 
→ 对 4 路鱼眼各做最近邻时间匹配 
→ 从 bag 中解码对应帧图像 
→ 4 路拼 AVM + 主方位单路落盘


给定 case_id={tag_id}_{timestamp_us}：
1) 在该 tag 对应 Heavy bag 中定位目标时间戳附近 N 帧（默认 10 帧）；
2) 抽取每帧 4 路鱼眼并按 offline_avm_generate 输入结构落盘；
3) 调用 offline_avm_generate 生成 AVM；
4) 默认对生成的 AVM 做与 **pipeline Step5**（``avp_vlm_pipeline_avm.draw_single_tag`` → ``render_bev_from_raw_avm`` → ``draw_obstacles_on_bev``）**相同的坐标变换与绘制**：规划白线、绿色邻车、**黄色** PARK_FREESPACE；**唯一刻意省略的是超声障碍的红色线段/点**（仍用超声多边形参与 ``chaosheng_pixel_radius`` 邻近筛选，与 Step5 一致）。若需含超声红，与 Step5 完全一致，请加 ``--pipeline-draw-ultrasonic-red``。
   **每帧叠绘所用 obstacle/pose/plan/chaosheng 默认从所选 Heavy bag 同名 Light bag 现场抽取最近邻**，让每一帧都有自己的坐标，不会沿用 case_id 时刻的快照（可加 ``--no-per-frame-payload`` 退回 ``read_data/<tag>/<ts>/`` 目录方案）。**即使本帧超声 ``chaosheng`` 为空或无可画 freespace，仍会叠绘规划线/相机障碍等**（与 Step5 默认 gate 不同）。默认用 **case_id 时间戳** 下 ``read_data/<tag>/<ts>/chaosheng.json`` 的原始超声经本帧 pose 投影为锚，**仅绘制距该投影 ≤50px（``--case-chaosheng-anchor-radius``）的相机障碍**；加 ``--no-case-chaosheng-anchor-filter`` 可关闭。静态资源 ``vehicle2sensing/ground/car_config`` 仍从 ``read_data/<tag>/`` 复用。
5) 可选 ``--mark-avm``：**所有** case 的 AVM 均生成并落盘结束后，再统一打开 OpenCV 逐张标注（多 case 不会中途打断批量生成）。

默认输出目录:
  /mnt/public-data/user/ziroujiang/generate_raw_data

批量：使用 ``--case-list /path/to/case_id.txt``（每行一个 ``{tag}_{timestamp_us}``，支持 ``#`` 注释）。
**任一 bag 读失败**（不存在、超时、冷存储、网络错误等）**立即放弃该 case**，不尝试其它 Heavy bag、不回退 read_data；单条失败会写入
``<output-root>/.generate_avm_failures.jsonl``，**再次跑同一列表时自动跳过已记录 case**（``--retry-failed`` 可强制重试）。

远端读 bag 已优化：按文件名预选 1–3 个 Heavy 候选 + ``read_messages(start_time/end_time)`` 只扫 target 附近时间窗（默认 ±45s），四路时间戳一次扫完，不再对 9 个 Heavy 整包遍历。

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python tool/generate_avm_from_case.py --case-list /mnt/public-data/user/ziroujiang/generate_raw_data/case_id.txt --mark-avm 

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python tool/generate_avm_from_case.py --mark-avm

生成 + 标注 + 扁平导出（与 ``generate_ground_irregularity`` 同结构）::

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python tool/generate_avm_from_case.py \
      --case-list /mnt/public-data/user/ziroujiang/generate_raw_data/case_id.txt \
      --mark-avm \
      --export-flat /mnt/public-data/user/ziroujiang/generate_parking_curb

成功生成后会写入 ``<case_out>/.generate_avm_manifest.json``（记录 ``ref_ts``、``frames``、``frame_step``、``main_camera`` 等）。下次若 manifest 与当前 CLI 一致且对应 **每帧** ``avm/*.jpg`` 与 ``yuyan/*.jpg`` 均已存在且非空，则 **整 case 跳过**：**不打开 Heavy bag、不调 get_meta_data**（加 ``--force-regenerate`` 强制重做）。尚无 manifest 的旧目录仍会走 bag 流程。

``--export-flat`` 在全部 case 的 avm/crop/yuyan 就绪后，将 ``<output_root>/<case>/avm|crop|yuyan`` 汇总为 ``<export-flat>/images|crop|yuyan``（仅导出三者齐全的帧）。
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import os
import re
from datetime import datetime
import shutil
import subprocess
import sys
import tempfile
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from get_data.dr_client import DRFILE_AVAILABLE, download_trip_config
from get_data.get_meta_data import get_meta_data

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)
from avm_marker import interactive_red_mark_session

try:
    from crop_read_data_chaosheng import (
        CAMERA_HEIGHT,
        FOCAL_LENGTH,
        render_bev_from_raw_avm,
    )
except ImportError:
    render_bev_from_raw_avm = None  # type: ignore[misc,assignment]
    FOCAL_LENGTH = 162.6  # type: ignore[assignment]
    CAMERA_HEIGHT = 3.44  # type: ignore[assignment]

try:
    from vlm.panoramic_projector import PanoramicProjector
except ImportError:
    PanoramicProjector = None  # type: ignore[misc,assignment]

try:
    from vlm.avp_vlm_pipeline_avm import avm_positions_to_yuyan_camera
except ImportError:
    avm_positions_to_yuyan_camera = None  # type: ignore[misc,assignment]

try:
    from vlm.avp_vlm_pipeline_avm import (
        get_direction_from_position,
        _AVM_DIRECTION_TO_YUYAN_CAM,
    )
except ImportError:
    get_direction_from_position = None  # type: ignore[misc,assignment]
    _AVM_DIRECTION_TO_YUYAN_CAM = {}  # type: ignore[misc]

try:
    from vlm.avp_vlm_pipeline_avm import draw_single_tag as _draw_single_tag
except ImportError:
    _draw_single_tag = None  # type: ignore[misc,assignment]

try:
    from dpbag import strip_header
    from dpbag.bag.bag import DpBag
except ImportError:
    strip_header = None  # type: ignore[assignment]
    DpBag = None  # type: ignore[assignment]

try:
    # 依赖 get_data/proto 的 PYTHONPATH 注入（config.PROTO_LOCAL_DIR）
    sys.path.insert(0, config.PROTO_LOCAL_DIR)
    from drivers.sensor_image_pb2 import CompressedImage
except Exception:
    CompressedImage = None  # type: ignore[assignment]

try:
    import get_data.perception_obstacles_compat  # noqa: F401  # 注册截断的 PerceptionObstacles 类
    from perception.deeproute_perception_obstacle_pb2 import PerceptionObstacles
    from drivers.gnss.ins_pb2 import Ins
    from planning.planning_pb2 import ADCTrajectory
    from google.protobuf.json_format import MessageToDict
except Exception:
    PerceptionObstacles = None  # type: ignore[assignment]
    Ins = None  # type: ignore[assignment]
    ADCTrajectory = None  # type: ignore[assignment]
    MessageToDict = None  # type: ignore[assignment]


DEFAULT_OUTPUT_ROOT = "/mnt/public-data/user/ziroujiang/generate_raw_data"
DEFAULT_FRAMES = 10
DEFAULT_FRAME_STEP = 5
# 与 crop_read_data_chaosheng / pipeline Step5 AVM 匹配 read_data 时间戳目录一致
READ_DATA_TS_TOLERANCE_US = 50_000
# 落盘于 ``<case_out>/``，用于下次在无 manifest 校验下也能跳过打开 bag（须与 CLI 参数一致）
GENERATE_AVM_MANIFEST_BASENAME = ".generate_avm_manifest.json"
GENERATE_AVM_MANIFEST_VERSION = 1
_FISHEYE_ALL_DIR = ".fisheye_all"
_FLAT_EXPORT_SUBDIRS = ("images", "crop", "yuyan")
_FLAT_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# 以 case_id 对应 read_data 子目录中的原始 chaosheng 为锚，只绘制其投影周边（像素）内的相机障碍
DEFAULT_CASE_CHAOSHENG_ANCHOR_RADIUS_PX = 50
# 从 bag 文件名推断分片时长（无法从相邻 bag 推断时的回退值，默认 5 分钟）
DEFAULT_BAG_SEGMENT_SPAN_US = 300_000_000
# 远端扫 bag 时在 target_ts 两侧各扩展的时间窗（秒），避免整包 read_messages
DEFAULT_BAG_SCAN_MARGIN_SEC = 45
# 按文件名预选后最多尝试打开的 Heavy bag 个数（读失败仍立即放弃该 case）
DEFAULT_BAG_CANDIDATE_LIMIT = 3

_COLD_STORAGE_MARKERS = ("prod-cold", "STORAGE_ACCESS_ERROR")


class ColdStorageError(RuntimeError):
    """Bag 数据在冷存储中，无法直接读取。"""
    pass


class BagNotFoundError(RuntimeError):
    """远端 bag 不存在或无法访问（DATA_NOT_FOUND 等）。"""
    pass


_BAG_NOT_FOUND_MARKERS = (
    "DATA_NOT_FOUND",
    "Can not found bag",
    "Can not find bag",
    "NOT_FOUND",
)


def _is_bag_not_found(exc: BaseException) -> bool:
    msg = str(exc)
    return any(m in msg for m in _BAG_NOT_FOUND_MARKERS)


def _check_cold_storage(exc: Exception) -> None:
    """检测异常是否由冷存储引起，是则抛出 ColdStorageError。"""
    msg = str(exc)
    for marker in _COLD_STORAGE_MARKERS:
        if marker in msg:
            raise ColdStorageError(
                f"数据在冷存储中: {msg[:200]}"
            ) from exc


def _fail_bag(bag_name: str, exc: Exception) -> None:
    """任意 bag 访问失败：立即中止当前 case（不尝试其它 bag）。"""
    _check_cold_storage(exc)
    if _is_bag_not_found(exc):
        raise BagNotFoundError(f"bag 失败: {bag_name}, err={exc}") from exc
    raise RuntimeError(f"bag 失败: {bag_name}, err={exc}") from exc


@dataclass
class PlannedFrame:
    ref_ts: int
    per_cam_ts: Dict[str, int]


def parse_case_id(case_id: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d+)_(\d+)", case_id.strip())
    if not m:
        raise ValueError("case_id 需形如 {tag_id}_{timestamp_us}")
    return int(m.group(1)), int(m.group(2))


def normalize_case_id_line(raw: str) -> str:
    """单行解析为 case_id：``#`` 后为注释丢弃；逗号分隔时取第一列（兼容 CSV / 行尾逗号）。"""
    s = raw.strip()
    if not s:
        return ""
    if s.startswith("#"):
        return ""
    s = s.split("#", 1)[0].strip()
    if not s:
        return ""
    # "tag_ts," 或 "tag_ts, col2..." → 只要第一列
    return s.split(",", maxsplit=1)[0].strip()


def _append_failure_log(log_path: str, record: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_failure_case_ids(log_path: str) -> Set[str]:
    """读取失败日志中已记录的 case_id（去重）。"""
    ids: Set[str] = set()
    if not os.path.isfile(log_path):
        return ids
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = str(rec.get("case_id") or "").strip()
            if cid:
                ids.add(cid)
    return ids


def _print_batch_summary(
    *,
    total: int,
    success_cases: List[str],
    failed_records: List[Dict[str, str]],
    cold_storage_cases: List[str],
    skipped_prior_failures: List[str],
    failures_log: str,
) -> None:
    print(f"\n{'=' * 70}")
    print(
        f"[汇总] case 共 {total} 个：成功 {len(success_cases)}，"
        f"失败 {len(failed_records)}，冷存储跳过 {len(cold_storage_cases)}，"
        f"历史失败跳过 {len(skipped_prior_failures)}"
    )
    if skipped_prior_failures:
        print(
            f"  历史失败跳过示例: {skipped_prior_failures[0]}"
            + (
                f" … 等 {len(skipped_prior_failures)} 个"
                if len(skipped_prior_failures) > 1
                else ""
            )
        )
        print(f"  记录文件: {failures_log}")
    if success_cases:
        print(f"  成功示例: {success_cases[0]}" + (f" … 等 {len(success_cases)} 个" if len(success_cases) > 1 else ""))
    if failed_records:
        print("  失败列表:")
        for rec in failed_records[:20]:
            print(f"    {rec['case_id']}: [{rec.get('kind', 'error')}] {rec.get('error', '')[:120]}")
        if len(failed_records) > 20:
            print(f"    … 另有 {len(failed_records) - 20} 条，见 {failures_log}")
    if failures_log and (failed_records or cold_storage_cases):
        print(f"  失败明细已追加: {failures_log}")
    print(f"{'=' * 70}")


def load_case_ids_from_file(path: str) -> List[str]:
    """每行一个 case_id；忽略空行与 ``#`` 起始行；支持行尾逗号或 CSV 首列。"""
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"case 列表文件不存在: {abs_path}")
    out: List[str] = []
    with open(abs_path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("#"):
                continue
            s = normalize_case_id_line(raw)
            if not s:
                continue
            try:
                parse_case_id(s)
            except ValueError as e:
                raise ValueError(f"{abs_path}:{lineno}: 非法 case_id {s!r}: {e}") from None
            out.append(s)
    return out


def ensure_runtime_deps() -> None:
    if DpBag is None or strip_header is None:
        raise RuntimeError(
            "缺少 dpbag 依赖，请在可用环境中运行（例如 avp conda 环境）"
        )
    if CompressedImage is None:
        raise RuntimeError(
            "缺少 CompressedImage proto，请确认 get_data/proto 已生成并可导入"
        )


def warn_if_mark_avm_no_gui() -> None:
    """Linux 无 DISPLAY/WAYLAND 时 OpenCV 窗口通常不会出现。"""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    print(
        "[mark-avm] 警告：当前 Linux 未设置 DISPLAY / WAYLAND_DISPLAY，"
        "OpenCV 绘图窗口大概率无法显示。请在本机图形桌面终端运行，或使用 ``ssh -Y`` / ``ssh -X`` 转发图形。\n"
        "补充：若命令行未加 ``--mark-avm``，脚本默认不会弹出标注窗口。",
        file=sys.stderr,
    )


def ts_us_to_filename(ts_us: int) -> str:
    sec = ts_us // 1_000_000
    usec = ts_us % 1_000_000
    return f"{sec}_{usec:06d}.jpg"


def extract_bag_prefix(bag_name: str) -> str:
    return os.path.basename(bag_name).split(".")[0]


def extract_yyyymm_from_bag_prefix(bag_prefix: str) -> str:
    m = re.search(r"_(\d{8})_", bag_prefix)
    if not m:
        raise ValueError(f"无法从 bag 名解析 YYYYMM: {bag_prefix}")
    return m.group(1)[:6]


def parse_bag_name_start_us(bag_name: str) -> Optional[int]:
    """从 bag 文件名解析分片起始墙钟时间（μs），如 ``..._20260314_124415.Heavy...``。"""
    m = re.search(r"_(\d{8})_(\d{6})\.", os.path.basename(bag_name))
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return int(dt.timestamp() * 1_000_000)
    except ValueError:
        return None


def _bag_segment_end_us(
    heavy_bags: Sequence[str], idx: int, *, default_span_us: int = DEFAULT_BAG_SEGMENT_SPAN_US
) -> int:
    start = parse_bag_name_start_us(heavy_bags[idx])
    if start is None:
        return 0
    if idx + 1 < len(heavy_bags):
        nxt = parse_bag_name_start_us(heavy_bags[idx + 1])
        if nxt is not None and nxt > start:
            return nxt - 1
    return start + int(default_span_us)


def rank_heavy_bags_by_filename(
    heavy_bags: Sequence[str], target_ts: int
) -> List[str]:
    """按文件名分片与 target_ts 的接近程度排序 Heavy bag（不访问远端）。"""
    scored: List[Tuple[int, str]] = []
    for i, bag in enumerate(heavy_bags):
        start = parse_bag_name_start_us(bag)
        if start is None:
            scored.append((10**30, bag))
            continue
        end = _bag_segment_end_us(heavy_bags, i)
        if start <= target_ts <= end:
            mid = (start + end) // 2
            scored.append((abs(target_ts - mid), bag))
        else:
            scored.append((min(abs(target_ts - start), abs(target_ts - end)), bag))
    scored.sort(key=lambda x: x[0])
    return [b for _, b in scored]


def scan_time_window_us(
    target_ts: int,
    frames: int,
    frame_step: int,
    *,
    margin_sec: float = DEFAULT_BAG_SCAN_MARGIN_SEC,
) -> Tuple[int, int]:
    """为 read_messages 计算 start_time/end_time（μs）。"""
    step_span = int(max(1, frames - 1) * max(1, frame_step) * 200_000)
    margin = int(max(5.0, margin_sec) * 1_000_000)
    half = step_span + margin
    return int(target_ts) - half, int(target_ts) + half


def narrow_time_window_us(
    ts_list: Sequence[int], *, pad_us: int = 5_000_000
) -> Tuple[int, int]:
    if not ts_list:
        return 0, 2**63 - 1
    lo, hi = min(ts_list), max(ts_list)
    return lo - int(pad_us), hi + int(pad_us)


def parse_filename_ts_us(name: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.fullmatch(r"(\d+)_(\d{6})", stem)
    if not m:
        return None
    return int(m.group(1)) * 1_000_000 + int(m.group(2))


def _read_yuyan_camera_from_index(tag_id: int, ts_us: int) -> Optional[str]:
    """从 ``draw_image/<tag>/<ts>/index_avm.json`` 读取 ``yuyan_camera``。"""
    base = os.path.join(config.DRAW_IMAGE_DIR, str(tag_id), str(ts_us))
    for fname in ("index_avm.json", "index.json"):
        idx_path = os.path.join(base, fname)
        if not os.path.isfile(idx_path):
            continue
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cam = payload.get("yuyan_camera")
            if cam in config.CAMERA_NAMES:
                return str(cam)
        except Exception:
            continue
    return None


def _run_step5_for_tag(tag_id: int) -> None:
    """对单个 tag_id 执行 Step 5（draw_single_tag），生成 index_avm.json。"""
    if _draw_single_tag is None:
        raise RuntimeError(
            "无法导入 vlm.avp_vlm_pipeline_avm.draw_single_tag，"
            "请确认 vlm 模块可用"
        )
    draw_args = argparse.Namespace(
        data_path=config.READ_DATA_DIR,
        chaosheng_pixel_radius=30,
        yuyan=True,
        ignore_fs_types=[],
    )
    print(f"  [INFO] tag_id={tag_id}: index_avm.json 不存在，运行 Step 5 生成 ...",
          flush=True)
    result = _draw_single_tag(tag_id, draw_args)
    if "draw_success" not in result:
        detail = result
        raise RuntimeError(
            f"Step 5 draw_single_tag 未成功: tag_id={tag_id}, result={detail}"
        )
    print(f"  [INFO] tag_id={tag_id}: Step 5 完成", flush=True)


def infer_main_camera_from_pipeline(tag_id: int, ts_us: int) -> Optional[str]:
    """读取 ``index_avm.json`` 的 ``yuyan_camera``；若文件不存在则先跑 Step 5 生成。"""
    cam = _read_yuyan_camera_from_index(tag_id, ts_us)
    if cam is not None:
        return cam
    _run_step5_for_tag(tag_id)
    cam = _read_yuyan_camera_from_index(tag_id, ts_us)
    if cam is not None:
        return cam
    raise RuntimeError(
        f"Step 5 已执行但仍未生成 yuyan_camera: "
        f"tag_id={tag_id}, ts={ts_us}"
    )


def _collect_avm_paths_from_existing_dirs(
    output_root: str, skip_if_crop_exists: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """扫描 output_root 下已有 case 子目录，收集 AVM 图片路径。

    返回 (avm_paths_to_mark, skipped_case_ids, selected_case_ids)
    """
    avm_paths: List[str] = []
    skipped: List[str] = []
    selected: List[str] = []
    if not os.path.isdir(output_root):
        return avm_paths, skipped, selected
    for name in sorted(os.listdir(output_root)):
        case_dir = os.path.join(output_root, name)
        if not os.path.isdir(case_dir):
            continue
        avm_dir = os.path.join(case_dir, "avm")
        if not os.path.isdir(avm_dir):
            continue
        if skip_if_crop_exists and os.path.isdir(os.path.join(case_dir, "crop")):
            skipped.append(name)
            continue
        imgs = sorted(
            f for f in os.listdir(avm_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not imgs:
            continue
        selected.append(name)
        for img_name in imgs:
            avm_paths.append(os.path.join(avm_dir, img_name))
    return avm_paths, skipped, selected


def _do_crop_from_avm_paths(
    avm_paths: List[str], crop_id_json: str, crop_size: int = 150,
) -> int:
    """对 AVM 路径列表做中心裁剪，返回裁剪数量。"""
    crop_id_map = _load_crop_id_json(crop_id_json)
    if not crop_id_map:
        print("[crop] crop_id.json 为空或不存在，跳过裁剪")
        return 0
    crop_half = crop_size // 2
    total = 0
    case_dirs_seen: set = set()
    for avm_path in avm_paths:
        stem = os.path.splitext(os.path.basename(avm_path))[0]
        center_str = crop_id_map.get(stem)
        if not center_str:
            continue
        try:
            coords = json.loads(center_str)
            cx, cy = int(coords[0]), int(coords[1])
        except (json.JSONDecodeError, IndexError, ValueError, TypeError):
            print(f"[crop] WARN: {stem} 坐标解析失败: {center_str!r}", file=sys.stderr)
            continue
        avm_img = cv2.imread(avm_path)
        if avm_img is None:
            continue
        h, w = avm_img.shape[:2]
        x1 = max(cx - crop_half, 0)
        y1 = max(cy - crop_half, 0)
        x2 = min(cx + crop_half, w)
        y2 = min(cy + crop_half, h)
        if x2 <= x1 or y2 <= y1:
            print(f"[crop] WARN: {stem} 裁剪区域为空 cx={cx} cy={cy}", file=sys.stderr)
            continue
        case_dir = os.path.dirname(os.path.dirname(avm_path))
        crop_out_dir = os.path.join(case_dir, "crop")
        if case_dir not in case_dirs_seen:
            os.makedirs(crop_out_dir, exist_ok=True)
            case_dirs_seen.add(case_dir)
        crop_img = avm_img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(crop_out_dir, f"{stem}.jpg"), crop_img)
        total += 1
    return total


def _reassign_yuyan_from_crop_id(
    output_root: str, crop_id_json: str,
) -> int:
    """根据 crop_id.json 中每张 AVM 的 OpenCV 标记坐标，逐帧选择对应的鱼眼相机并汇入 ``yuyan/``。

    从 ``_FISHEYE_ALL_DIR/panoramic_X/`` 取图写入 ``yuyan/``，完成后删除临时目录。
    返回成功复制的文件数。
    """
    if get_direction_from_position is None or not _AVM_DIRECTION_TO_YUYAN_CAM:
        print("[yuyan-reassign] WARN: 方位推断模块不可用，跳过", file=sys.stderr)
        return 0

    crop_id_map = _load_crop_id_json(crop_id_json)
    if not crop_id_map:
        print("[yuyan-reassign] crop_id.json 为空或不存在，跳过")
        return 0

    total = 0
    case_dirs = sorted(
        d for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
        and os.path.isdir(os.path.join(output_root, d, "avm"))
    )

    for case_name in case_dirs:
        case_dir = os.path.join(output_root, case_name)
        fisheye_dir = os.path.join(case_dir, _FISHEYE_ALL_DIR)
        if not os.path.isdir(fisheye_dir):
            continue
        avm_dir = os.path.join(case_dir, "avm")
        yuyan_dir = os.path.join(case_dir, "yuyan")
        avm_files = [
            f for f in os.listdir(avm_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not avm_files:
            continue

        frame_cameras: List[str] = []
        assignments: List[Tuple[str, str]] = []  # (stem, cam)

        for fname in sorted(avm_files):
            stem = os.path.splitext(fname)[0]
            center_str = crop_id_map.get(stem)
            if not center_str:
                continue
            try:
                coords = json.loads(center_str)
                cx, cy = int(coords[0]), int(coords[1])
            except (json.JSONDecodeError, IndexError, ValueError, TypeError):
                print(f"[yuyan-reassign] WARN: {stem} 坐标解析失败: {center_str!r}", file=sys.stderr)
                continue
            direction = get_direction_from_position(cx, cy)
            cam = _AVM_DIRECTION_TO_YUYAN_CAM.get(direction, "panoramic_1")
            frame_cameras.append(cam)
            assignments.append((stem, cam))

        if not assignments:
            continue

        os.makedirs(yuyan_dir, exist_ok=True)
        copied = 0
        for stem, cam in assignments:
            src = os.path.join(fisheye_dir, cam, f"{stem}.jpg")
            if not os.path.isfile(src):
                print(
                    f"[yuyan-reassign] WARN: {case_name}/{_FISHEYE_ALL_DIR}/{cam}/{stem}.jpg 不存在，跳过",
                    file=sys.stderr,
                )
                continue
            dst = os.path.join(yuyan_dir, f"{stem}.jpg")
            shutil.copy2(src, dst)
            copied += 1

        shutil.rmtree(fisheye_dir, ignore_errors=True)

        cam_summary = {}
        for cam in frame_cameras:
            cam_summary[cam] = cam_summary.get(cam, 0) + 1
        cam_str = ", ".join(f"{c}×{n}" for c, n in sorted(cam_summary.items()))
        print(f"[yuyan-reassign] {case_name}: {copied} 张 → yuyan/ ({cam_str})")
        total += copied

    return total


def _is_nonempty_image(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _iter_case_dirs_with_avm(output_root: str) -> List[str]:
    """返回含 ``avm/`` 的 case 子目录绝对路径（跳过隐藏目录）。"""
    if not os.path.isdir(output_root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(output_root)):
        if name.startswith("."):
            continue
        case_dir = os.path.join(output_root, name)
        if os.path.isdir(case_dir) and os.path.isdir(os.path.join(case_dir, "avm")):
            out.append(case_dir)
    return out


def export_output_root_to_flat(
    output_root: str,
    export_root: str,
    *,
    on_conflict: str = "overwrite",
    dry_run: bool = False,
) -> Dict[str, int]:
    """将 ``output_root`` 下各 case 的 avm/crop/yuyan 整理为扁平训练目录。

    目标布局与 ``generate_ground_irregularity`` 一致::
      <export_root>/images/{tag}_{ref_ts}.jpg  ← 来自各 case 的 avm/
      <export_root>/crop/...
      <export_root>/yuyan/...

    仅导出 avm、crop、yuyan 三者均存在且非空的帧；缺一则计入 incomplete 并跳过。
    """
    stats: Dict[str, int] = {sub: 0 for sub in _FLAT_EXPORT_SUBDIRS}
    stats["incomplete"] = 0
    stats["conflict_skip"] = 0
    stats["cases_scanned"] = 0

    if not dry_run:
        for sub in _FLAT_EXPORT_SUBDIRS:
            os.makedirs(os.path.join(export_root, sub), exist_ok=True)

    for case_dir in _iter_case_dirs_with_avm(output_root):
        stats["cases_scanned"] += 1
        avm_dir = os.path.join(case_dir, "avm")
        for fname in sorted(os.listdir(avm_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _FLAT_IMAGE_EXTS:
                continue
            stem = os.path.splitext(fname)[0]
            src_avm = os.path.join(avm_dir, fname)
            src_crop = os.path.join(case_dir, "crop", f"{stem}.jpg")
            src_yuyan = os.path.join(case_dir, "yuyan", f"{stem}.jpg")
            if not (
                _is_nonempty_image(src_avm)
                and _is_nonempty_image(src_crop)
                and _is_nonempty_image(src_yuyan)
            ):
                stats["incomplete"] += 1
                print(
                    f"[export-flat] 不完整跳过: {stem} "
                    f"(avm={_is_nonempty_image(src_avm)}, "
                    f"crop={_is_nonempty_image(src_crop)}, "
                    f"yuyan={_is_nonempty_image(src_yuyan)})",
                    file=sys.stderr,
                )
                continue

            dst_avm = os.path.join(export_root, "images", fname)
            dst_crop = os.path.join(export_root, "crop", f"{stem}.jpg")
            dst_yuyan = os.path.join(export_root, "yuyan", f"{stem}.jpg")
            dst_triple = (dst_avm, dst_crop, dst_yuyan)
            if on_conflict == "skip" and any(os.path.exists(p) for p in dst_triple):
                stats["conflict_skip"] += 1
                continue
            if on_conflict == "error":
                for p in dst_triple:
                    if os.path.exists(p):
                        raise FileExistsError(f"冲突: {p} 已存在")
            if not dry_run:
                shutil.copy2(src_avm, dst_avm)
                shutil.copy2(src_crop, dst_crop)
                shutil.copy2(src_yuyan, dst_yuyan)
            stats["images"] += 1
            stats["crop"] += 1
            stats["yuyan"] += 1

    print(
        f"[export-flat] 扫描 case: {stats['cases_scanned']}，"
        f"导出 images={stats['images']} crop={stats['crop']} yuyan={stats['yuyan']}，"
        f"不完整跳过={stats['incomplete']}，冲突跳过={stats['conflict_skip']}"
    )
    print(f"[export-flat] 目标目录: {export_root}")
    return stats


def _mark_existing_and_crop(args: argparse.Namespace) -> int:
    """--mark-existing 模式：扫描已有子目录，跳过有 crop/ 的，标注+裁剪。"""
    avm_paths, skipped, selected = _collect_avm_paths_from_existing_dirs(
        args.output_root, skip_if_crop_exists=True,
    )
    print(f"[mark-existing] 扫描 {args.output_root}")
    print(f"  已有 crop/ 跳过: {len(skipped)} 个 case")
    print(f"  待标注: {len(selected)} 个 case, {len(avm_paths)} 张 AVM")
    if skipped:
        for s in skipped:
            print(f"    [跳过] {s}")

    if not avm_paths:
        print("[mark-existing] 无需标注的 AVM")
        if getattr(args, "export_flat", None):
            export_output_root_to_flat(
                args.output_root,
                args.export_flat,
                on_conflict=getattr(args, "export_flat_conflict", "overwrite"),
            )
        return 0

    # OpenCV 逐张标注
    crop_id_path = args.crop_id_json
    print(
        "\n[mark-existing] 左键拖动红色 | s 保存覆盖并记录质心 | z/r 撤销/清空 | "
        "n 下一张（自动记录质心） | q/ESC 结束"
    )
    for i, p in enumerate(avm_paths, start=1):
        img = cv2.imread(p)
        if img is None or img.size == 0:
            print(f"[mark-existing] 跳过（无法读取）: {p}", file=sys.stderr)
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        title = f"mark AVM [{i}/{len(avm_paths)}]: {p}"
        r = interactive_red_mark_session(
            img,
            save_path=p,
            window_title=title,
            thickness=args.mark_thickness,
            allow_next=True,
            case_id=stem,
            crop_id_path=crop_id_path,
        )
        if r == "quit":
            break
    cv2.destroyAllWindows()

    # crop 后处理
    total_cropped = _do_crop_from_avm_paths(avm_paths, args.crop_id_json)
    print(f"\n[crop] 150×150 中心裁剪完成: {total_cropped} 张")

    total_yuyan = _reassign_yuyan_from_crop_id(args.output_root, args.crop_id_json)
    if total_yuyan:
        print(f"[yuyan-reassign] 逐帧鱼眼重分配完成: {total_yuyan} 张 → yuyan/")

    if getattr(args, "export_flat", None):
        export_output_root_to_flat(
            args.output_root,
            args.export_flat,
            on_conflict=getattr(args, "export_flat_conflict", "overwrite"),
        )
    return 0


def _load_crop_id_json(path: str) -> Dict[str, str]:
    """加载 crop_id.json，格式 {case_id: "[x,y]"}。"""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[crop] WARN: 加载 {path} 失败: {e}", file=sys.stderr)
        return {}


_READ_DATA_REQUIRED_JSON = ("chaosheng.json", "obstacle.json", "pose.json")


def _read_data_ts_dir_complete(ts_dir: str) -> bool:
    if not os.path.isdir(ts_dir):
        return False
    return all(
        os.path.isfile(os.path.join(ts_dir, n)) and os.path.getsize(os.path.join(ts_dir, n)) > 0
        for n in _READ_DATA_REQUIRED_JSON
    )


def list_read_data_snapshot_timestamps_sorted(tag_read_root: str) -> List[int]:
    """列出 ``read_data/<tag>/`` 下障碍感知快照齐全的数字子目录时间戳，升序。"""
    if not os.path.isdir(tag_read_root):
        return []
    out: List[int] = []
    for name in os.listdir(tag_read_root):
        if not name.isdigit():
            continue
        ts_dir = os.path.join(tag_read_root, name)
        if _read_data_ts_dir_complete(ts_dir):
            out.append(int(name))
    out.sort()
    return out


def pick_read_data_snapshot_for_frame(
    tag_read_root: str,
    frame_ref_ts_us: int,
    *,
    snapshot_ts_sorted: Sequence[int],
    max_skew_us: int,
    require_exact_dir_name: bool,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """为本帧 ``frame_ref_ts_us`` 选取用于叠绘的 read_data 子目录名（字符串形式的 ts）。

    - **优先** ``read_data/.../<frame_ref_ts_us>/`` 目录齐全则用之（与该帧对齐）。
    - 否则在 ``snapshot_ts_sorted`` 中二分找最近快照；若 ``|Δ| > max_skew_us`` 则不可用。
    - ``require_exact_dir_name=True`` 时仅允许精确目录，不做最近邻。

    返回 ``(子目录名, |Δts|μs, 说明)``；失败时第一域为 None。
    """
    exact_dir = os.path.join(tag_read_root, str(frame_ref_ts_us))
    if _read_data_ts_dir_complete(exact_dir):
        return str(frame_ref_ts_us), 0, None

    if require_exact_dir_name:
        return (
            None,
            None,
            f"无 read_data/.../{frame_ref_ts_us}/ 或未齐全（已启用 --pipeline-overlay-read-data-exact）",
        )

    if not snapshot_ts_sorted:
        return None, None, "read_data 下无齐全快照目录"

    snaps = list(snapshot_ts_sorted)
    idx = bisect.bisect_left(snaps, frame_ref_ts_us)
    candidates: List[int] = []
    if idx < len(snaps):
        candidates.append(snaps[idx])
    if idx > 0:
        candidates.append(snaps[idx - 1])
    snapshot_ts = min(candidates, key=lambda s: abs(s - frame_ref_ts_us))
    diff_us = abs(snapshot_ts - frame_ref_ts_us)
    if diff_us > max_skew_us:
        return (
            None,
            diff_us,
            f"最近快照 ts={snapshot_ts} 与 ref_ts={frame_ref_ts_us} 相差 {diff_us}μs "
            f"> max_skew={max_skew_us}μs，跳过叠绘以免错位",
        )
    hint = None
    if diff_us > 0:
        hint = (
            f"本帧 ref_ts={frame_ref_ts_us} 使用最近感知快照目录 ts={snapshot_ts}（|Δ|={diff_us}μs）"
        )
    return str(snapshot_ts), diff_us, hint


def derive_light_bag_from_heavy(heavy_bag: str) -> str:
    """``Heavy → Light``（与 ``BagReader.event_heavy_bags`` 反向规则一致）。

    例如 ``YR-C01-35_20260120_062850.Heavy_Topic_Group.bag`` →
    ``YR-C01-35_20260120_062850.Light_Topic_Group.bag``。
    """
    return heavy_bag.replace("Heavy_Topic_Group", "Light_Topic_Group")


def _ensure_light_bag_proto_runtime() -> Optional[str]:
    """若 Light bag 抽取所需 proto 不可用，返回错误说明；可用时返回 None。"""
    if PerceptionObstacles is None or Ins is None or ADCTrajectory is None or MessageToDict is None:
        return (
            "Light bag 抽取所需 proto/工具不可用：PerceptionObstacles / Ins / "
            "ADCTrajectory / google.protobuf.json_format.MessageToDict 任一为 None"
        )
    return None


def extract_per_frame_payloads_from_light_bag(
    light_bag: str,
    target_ts_list: Sequence[int],
    *,
    max_skew_us: int = READ_DATA_TS_TOLERANCE_US,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, int]]]:
    """对每个目标 ``ref_ts``，扫描 Light bag 找最近邻 obstacle / pose / plan / chaosheng。

    输出与 ``save_bag_data.save_data`` 落盘的 JSON 字段同结构（即 ``read_data/<tag>/<ts>/*.json``
    的内存版），可直接喂给 ``PanoramicProjector`` / ``render_bev_from_raw_avm``。

    四路 topic 均在 ``max_skew_us`` 内对本帧 ``ref_ts`` 独立最近邻；``chaosheng`` 可为空列表。

    Returns:
      ``(per_frame_payload, per_frame_diffs)``

      * per_frame_payload[ref_ts] = {'obstacle': [...], 'pose': {...}, 'plan': [...], 'chaosheng': [...]}
        缺字段时不会出现该键；超过 ``max_skew_us`` 的最近邻视作"未匹配到"，对应字段不写入。

      * per_frame_diffs[ref_ts] = {'obstacle': μs, 'pose': μs, 'plan': μs, 'chaosheng': μs}
        仅记录命中（≤ max_skew_us）的字段。
    """
    if DpBag is None or strip_header is None:
        raise RuntimeError("dpbag 不可用，无法读取 Light bag")
    err = _ensure_light_bag_proto_runtime()
    if err:
        raise RuntimeError(err)

    targets = sorted({int(t) for t in target_ts_list})
    payload: Dict[int, Dict[str, Any]] = {t: {} for t in targets}
    diffs: Dict[int, Dict[str, int]] = {t: {} for t in targets}
    best_diff: Dict[Tuple[int, str], int] = {}

    def _update(field: str, ts_us: int, value_factory):
        for evt_t in targets:
            d = abs(ts_us - evt_t)
            if d > max_skew_us:
                continue
            key = (evt_t, field)
            prev = best_diff.get(key)
            if prev is None or d < prev:
                payload[evt_t][field] = value_factory()
                diffs[evt_t][field] = d
                best_diff[key] = d

    topics = [
        config.OBSTACLE_TOPIC,
        config.POSE_TOPIC,
        config.PLANNING_TOPIC,
        config.CHAOSHENG_TOPIC,
    ]
    lo, hi = narrow_time_window_us(
        targets, pad_us=max(int(max_skew_us) * 4, 5_000_000)
    )
    print(
        f"[lightbag-scan] {os.path.basename(light_bag)} window_us=[{lo},{hi}] "
        f"frames={len(targets)}",
        flush=True,
    )
    try:
        with DpBag(bag=light_bag) as bag:
            for topic, msg, _ in bag.read_messages(
                topics=topics,
                dpbag_name=light_bag,
                force_get_data_by_raw=True,
                start_time=lo,
                end_time=hi,
            ):
                raw = strip_header(msg.data)

                if topic == config.OBSTACLE_TOPIC:
                    obj = PerceptionObstacles()
                    obj.ParseFromString(raw)
                    ts = int(obj.time_measurement)
                    _update(
                        "obstacle",
                        ts,
                        lambda obj=obj: [
                            MessageToDict(item) if hasattr(item, "DESCRIPTOR") else item
                            for item in obj.perception_obstacle
                        ],
                    )
                    continue

                if topic == config.POSE_TOPIC:
                    obj = Ins()
                    obj.ParseFromString(raw)
                    ts = int(obj.measurement_time)
                    _update(
                        "pose",
                        ts,
                        lambda obj=obj: {
                            "position": [obj.position.x, obj.position.y, obj.position.z],
                            "euler_angles": [
                                obj.euler_angles.x,
                                obj.euler_angles.y,
                                obj.euler_angles.z,
                            ],
                        },
                    )
                    continue

                if topic == config.PLANNING_TOPIC:
                    obj = ADCTrajectory()
                    obj.ParseFromString(raw)
                    ts = int(obj.header.timestamp_sec * 1e6)
                    _update(
                        "plan",
                        ts,
                        lambda obj=obj: [
                            {
                                "relative_time": pt.relative_time,
                                "x": pt.path_point.x,
                                "y": pt.path_point.y,
                            }
                            for pt in obj.trajectory_point
                        ],
                    )
                    continue

                if topic == config.CHAOSHENG_TOPIC:
                    obj = PerceptionObstacles()
                    obj.ParseFromString(raw)
                    ts = int(obj.time_measurement)
                    items = []
                    for it in obj.perception_obstacle:
                        d = MessageToDict(it) if hasattr(it, "DESCRIPTOR") else it
                        if (
                            d.get("modelType") == "MODEL_PARKING"
                            and d.get("type") == "PLANNING_STOP_OBSTACLE"
                            and d.get("sensorType") == "ULTRASONIC"
                        ):
                            items.append(d)
                    _update("chaosheng", ts, lambda items=items: items)
                    continue
    except Exception as e:
        _fail_bag(light_bag, e)

    return payload, diffs


def _load_static_payload_from_read_data(
    read_data_root: str, tag_id: int
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """加载 ``read_data/<tag>/`` 下与时间戳无关的静态资源。"""
    tag_root = os.path.join(read_data_root, str(tag_id))
    required = ("vehicle2sensing.json", "ground.json", "car_config.json")
    for fn in required:
        if not os.path.isfile(os.path.join(tag_root, fn)):
            return None, f"read_data/{tag_id} 缺少 {fn}"
    try:
        with open(os.path.join(tag_root, "vehicle2sensing.json"), "r", encoding="utf-8") as f:
            vehicle2sensing = json.load(f)
        with open(os.path.join(tag_root, "ground.json"), "r", encoding="utf-8") as f:
            ground = json.load(f)
        with open(os.path.join(tag_root, "car_config.json"), "r", encoding="utf-8") as f:
            car_config = json.load(f)
    except OSError as e:
        return None, f"读静态 JSON 失败: {e}"
    return {
        "vehicle2sensing": vehicle2sensing,
        "ground": ground,
        "car_config": car_config,
    }, None


def _load_frame_payload_from_disk(
    tag_root: str,
    ts_name: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """从 ``read_data/<tag>/<ts>/`` 子目录加载本帧的 obstacle / pose / chaosheng / plan。"""
    item_path = os.path.join(tag_root, ts_name)
    try:
        with open(os.path.join(item_path, "chaosheng.json"), "r", encoding="utf-8") as f:
            chaosheng = json.load(f)
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
    except OSError as e:
        return None, f"读 JSON 失败: {e}"
    return {
        "obstacle": obstacle,
        "pose": pose,
        "plan": planning_point,
        "chaosheng": chaosheng,
    }, None


def _load_case_chaosheng_anchor_pair(
    read_data_root: str, tag_id: int, case_target_ts: int
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """读取 ``read_data/<tag>/<case_id 时间戳>/`` 下原始 ``chaosheng.json``（及同目录 ``obstacle.json`` 供 z 拟合）。"""
    sub = os.path.join(read_data_root, str(tag_id), str(int(case_target_ts)))
    cp = os.path.join(sub, "chaosheng.json")
    op = os.path.join(sub, "obstacle.json")
    if not os.path.isfile(cp):
        return None, None
    try:
        with open(cp, "r", encoding="utf-8") as f:
            ch = json.load(f)
        ob: List[Any] = []
        if os.path.isfile(op):
            with open(op, "r", encoding="utf-8") as f:
                ob = json.load(f)
        if not isinstance(ch, list):
            return None, None
        if not isinstance(ob, list):
            ob = []
        return ch, ob
    except (OSError, json.JSONDecodeError):
        return None, None


def _case_anchor_chaosheng_in_current_pose(
    projector: Any,
    chaosheng_raw: List[Any],
    obstacle_raw: List[Any],
    pose_cur: Dict[str, Any],
    vehicle2sensing: Dict[str, Any],
) -> List[Any]:
    """将 case 目录中的 chaosheng（世界系）用本帧 ``pose_cur`` 变到当前 sensing，与当帧 obstacle 对齐同一车身。"""
    ch = copy.deepcopy(chaosheng_raw)
    obs = copy.deepcopy(obstacle_raw)
    projector.apply_chaosheng_z_from_camera_ground_plane(ch, obs)
    return projector.world2vehicle2sensing_chaosheng(ch, pose_cur, vehicle2sensing)


def overlay_avm_like_pipeline_step5(
    projector: Any,
    avm_bgr: np.ndarray,
    tag_id: int,
    ref_ts_us: int,
    *,
    read_data_root: str,
    chaosheng_pixel_radius: Optional[int],
    ignore_fs_types: Optional[List[str]],
    draw_ultrasonic_red: bool = False,
    snapshot_ts_sorted: Sequence[int] = (),
    max_skew_us: int = READ_DATA_TS_TOLERANCE_US,
    require_exact_dir_name: bool = False,
    frame_payload: Optional[Dict[str, Any]] = None,
    static_payload: Optional[Dict[str, Any]] = None,
    frame_source_label: Optional[str] = None,
    case_anchor_chaosheng_raw: Optional[List[Any]] = None,
    case_anchor_obstacle_raw: Optional[List[Any]] = None,
    case_anchor_pixel_radius: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[str], Optional[str], Optional[int], List]:
    """与 pipeline Step5 同源：按**本帧** ``ref_ts_us`` 取障碍/位姿/规划/超声后再叠绘。

    数据来源优先级（每一帧独立，不会复用 case_id 时刻的坐标）：

    1. ``frame_payload``：现场从 Light bag 抽取的本帧最近邻负载（推荐）；
    2. 否则：``read_data/<tag>/<ts>/`` 子目录（按 ``pick_read_data_snapshot_for_frame``
       规则选 ts，受 ``max_skew_us`` / ``require_exact_dir_name`` 控制）。

    静态资源（``vehicle2sensing`` / ``ground`` / ``car_config``）：
    优先 ``static_payload``，否则从 ``read_data/<tag>/`` 加载。

    若 ``case_anchor_pixel_radius`` 非 None 且提供了 ``case_anchor_chaosheng_raw``：
    用 **case_id 对应目录** 中的原始 chaosheng（经本帧 pose 变换到 sensing）的投影顶点为锚，
    仅绘制与其像素距离 ≤ 该半径的相机障碍；规划线仍全量绘制；超声红仍用本帧 ``chaosheng``。

    Returns:
        ``(bgr, 警告文本, 数据来源标签, |Δts|μs, chaosheng_positions)``；
        未叠绘（数据缺失/超限）时第 3、4 位可能为 None，第 5 位为空列表。
    """
    if render_bev_from_raw_avm is None or PanoramicProjector is None:
        return avm_bgr.copy(), "缺少 crop_read_data_chaosheng 或 PanoramicProjector", None, None, []

    if static_payload is None:
        static_payload, err = _load_static_payload_from_read_data(read_data_root, tag_id)
        if err or static_payload is None:
            return avm_bgr.copy(), err or "缺少静态资源", None, None, []
    vehicle2sensing = static_payload["vehicle2sensing"]
    ground = static_payload["ground"]
    car_config = static_payload["car_config"]

    pick_hint: Optional[str] = None
    diff_us: Optional[int] = None
    src_label: Optional[str] = frame_source_label

    if frame_payload is None:
        tag_root = os.path.join(read_data_root, str(tag_id))
        if not os.path.isdir(tag_root):
            return avm_bgr.copy(), f"read_data/{tag_id} 不存在", None, None, []
        ts_name, diff_us, pick_hint = pick_read_data_snapshot_for_frame(
            tag_root,
            ref_ts_us,
            snapshot_ts_sorted=snapshot_ts_sorted,
            max_skew_us=max_skew_us,
            require_exact_dir_name=require_exact_dir_name,
        )
        if not ts_name:
            return avm_bgr.copy(), pick_hint or "无法选取 read_data 快照目录", None, diff_us, []
        loaded, err = _load_frame_payload_from_disk(tag_root, ts_name)
        if err or loaded is None:
            return avm_bgr.copy(), err or "本帧负载缺失", ts_name, diff_us, []
        frame_payload = loaded
        if src_label is None:
            src_label = f"disk:{ts_name}"
    else:
        if src_label is None:
            src_label = "lightbag-perframe"

    obstacle = frame_payload.get("obstacle") or []
    pose = frame_payload.get("pose") or {}
    chaosheng = frame_payload.get("chaosheng") or []
    planning_point = frame_payload.get("plan") or []

    if not pose:
        return (
            avm_bgr.copy(),
            f"本帧 ref_ts={ref_ts_us} 缺 pose，跳过叠绘以免错位",
            src_label,
            diff_us,
            [],
        )

    ignore_fs = set(ignore_fs_types or [])
    if ignore_fs:
        chaosheng = [
            o for o in chaosheng if o.get("freespaceType", "") not in ignore_fs
        ]

    anchor_for_pixels: Optional[List[Any]] = None
    radius_eff: Optional[int] = chaosheng_pixel_radius
    if case_anchor_pixel_radius is not None and case_anchor_chaosheng_raw is not None:
        anchor_for_pixels = _case_anchor_chaosheng_in_current_pose(
            projector,
            case_anchor_chaosheng_raw,
            case_anchor_obstacle_raw or [],
            pose,
            vehicle2sensing,
        )
        radius_eff = int(case_anchor_pixel_radius)
        h0, w0 = avm_bgr.shape[:2]
        probe = projector._precompute_chaosheng_img_points(
            anchor_for_pixels,
            ground,
            FOCAL_LENGTH,
            CAMERA_HEIGHT,
            h0,
            w0,
        )
        if probe.size == 0:
            print(
                f"[case-anchor] ref_ts={ref_ts_us}: case chaosheng 投影无顶点，本帧退化为不按锚筛选",
                file=sys.stderr,
            )
            anchor_for_pixels = None
            radius_eff = chaosheng_pixel_radius
            if radius_eff is not None and not chaosheng:
                radius_eff = None
    else:
        # chaosheng 为空时不能开 pixel_radius 邻近筛选，否则 draw_obstacles_on_bev 会跳过全部相机障碍
        if radius_eff is not None and not chaosheng:
            radius_eff = None

    projector.apply_chaosheng_z_from_camera_ground_plane(chaosheng, obstacle)
    obstacle = projector.world2vehicle2sensing(obstacle, pose, vehicle2sensing)
    chaosheng = projector.world2vehicle2sensing_chaosheng(
        chaosheng, pose, vehicle2sensing
    )

    try:
        painted, bev_pos, _yellow = render_bev_from_raw_avm(
            projector,
            avm_bgr,
            obstacle,
            chaosheng,
            ground,
            planning_point,
            pose,
            vehicle2sensing,
            car_config,
            radius_eff,
            list(ignore_fs) if ignore_fs else None,
            draw_ultrasonic_red=draw_ultrasonic_red,
            skip_chaosheng_draw_gate=True,
            chaosheng_pixel_anchor_chaosheng=anchor_for_pixels,
        )
    except Exception as e:
        return avm_bgr.copy(), f"render_bev_from_raw_avm 失败: {e}", src_label, diff_us, []

    return painted, pick_hint, src_label, diff_us, bev_pos


def pick_nearby_window(
    sorted_ts: Sequence[int], target_ts: int, n: int, step: int
) -> Tuple[List[int], int]:
    if not sorted_ts:
        return [], 1
    n = max(1, min(n, len(sorted_ts)))
    step = max(1, int(step))
    # 若 requested step 导致无法凑够 n 帧，则自动回退到可行步长
    max_step = max(1, (len(sorted_ts) - 1) // max(1, n - 1))
    eff_step = min(step, max_step)

    idx = bisect.bisect_left(sorted_ts, target_ts)
    if idx == len(sorted_ts):
        idx = len(sorted_ts) - 1
    elif idx > 0:
        if abs(sorted_ts[idx] - target_ts) >= abs(sorted_ts[idx - 1] - target_ts):
            idx -= 1

    max_start = len(sorted_ts) - 1 - eff_step * (n - 1)
    start = idx - eff_step * (n // 2)
    start = max(0, min(start, max_start))
    picked = [sorted_ts[start + i * eff_step] for i in range(n)]
    return picked, eff_step


def nearest_ts(sorted_ts: Sequence[int], target_ts: int) -> int:
    idx = bisect.bisect_left(sorted_ts, target_ts)
    if idx == 0:
        return int(sorted_ts[0])
    if idx >= len(sorted_ts):
        return int(sorted_ts[-1])
    a = sorted_ts[idx - 1]
    b = sorted_ts[idx]
    return int(a if abs(a - target_ts) <= abs(b - target_ts) else b)


def scan_camera_timestamps(
    heavy_bag: str,
    camera_topics: Dict[str, str],
    only_cam: Optional[str] = None,
    *,
    start_us: Optional[int] = None,
    end_us: Optional[int] = None,
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {c: [] for c in camera_topics.values()}
    read_kw: Dict[str, Any] = {
        "topics": list(camera_topics.keys()),
        "dpbag_name": heavy_bag,
        "force_get_data_by_raw": True,
    }
    if start_us is not None:
        read_kw["start_time"] = int(start_us)
    if end_us is not None:
        read_kw["end_time"] = int(end_us)
    win = ""
    if start_us is not None and end_us is not None:
        win = f" window_us=[{start_us},{end_us}]"
    print(
        f"[bag-scan] {os.path.basename(heavy_bag)} cam="
        f"{only_cam or 'all'}{win}",
        flush=True,
    )
    try:
        with DpBag(bag=heavy_bag) as bag:
            for topic, msg, _ in bag.read_messages(**read_kw):
                cam = camera_topics.get(topic)
                if cam is None:
                    continue
                if only_cam and cam != only_cam:
                    continue
                obj = CompressedImage()
                obj.ParseFromString(strip_header(msg.data))
                ts_us = int(obj.header.timestamp_sec * 1e6)
                out[cam].append(ts_us)
    except Exception as e:
        _fail_bag(heavy_bag, e)

    for cam in out:
        out[cam].sort()
    return out


def choose_best_heavy_bag(
    heavy_bags: Sequence[str],
    target_ts: int,
    main_cam: str,
    camera_topics: Dict[str, str],
    *,
    frames: int = DEFAULT_FRAMES,
    frame_step: int = DEFAULT_FRAME_STEP,
    bag_candidate_limit: int = DEFAULT_BAG_CANDIDATE_LIMIT,
    scan_margin_sec: float = DEFAULT_BAG_SCAN_MARGIN_SEC,
) -> Tuple[str, List[int], Dict[str, List[int]]]:
    """预选 Heavy bag + 时间窗内一次扫齐四路时间戳（避免整包遍历）。

    - 先按文件名分片排序，最多尝试 ``bag_candidate_limit`` 个候选（默认 3）。
    - 任一对 bag 的读取若抛错，立即向上抛出，不扫剩余 Heavy。
    - 返回 ``(best_heavy, main_cam_ts_list, all_cam_ts_map)``，后续不再重复扫 bag。
    """
    ranked = rank_heavy_bags_by_filename(heavy_bags, target_ts)
    win_start, win_end = scan_time_window_us(
        target_ts, frames, frame_step, margin_sec=scan_margin_sec
    )
    limit = max(1, int(bag_candidate_limit or 1))
    tried: List[str] = []
    for bag_name in ranked[:limit]:
        tried.append(bag_name)
        ts_map = scan_camera_timestamps(
            bag_name,
            camera_topics,
            only_cam=None,
            start_us=win_start,
            end_us=win_end,
        )
        main_ts = ts_map.get(main_cam) or []
        if not main_ts:
            print(
                f"[bag-scan] {os.path.basename(bag_name)} 时间窗内无 {main_cam} 帧，"
                f"尝试下一候选",
                file=sys.stderr,
            )
            continue
        near = nearest_ts(main_ts, target_ts)
        print(
            f"[bag-scan] 选用 {os.path.basename(bag_name)}："
            f"{main_cam} 窗内 {len(main_ts)} 帧，最近 target Δ={abs(near - target_ts)}μs",
            flush=True,
        )
        return bag_name, main_ts, ts_map
    raise RuntimeError(
        f"候选 Heavy bag 均无 {main_cam} 帧（已试 {len(tried)}/{len(heavy_bags)}："
        f"{', '.join(os.path.basename(b) for b in tried)}）"
    )


def plan_frames(
    selected_ref_ts: Sequence[int],
    cam_ts_map: Dict[str, List[int]],
) -> List[PlannedFrame]:
    planned: List[PlannedFrame] = []
    for ref_ts in selected_ref_ts:
        per_cam: Dict[str, int] = {}
        ok = True
        for cam in config.CAMERA_NAMES:
            cts = cam_ts_map.get(cam) or []
            if not cts:
                ok = False
                break
            per_cam[cam] = nearest_ts(cts, ref_ts)
        if ok:
            planned.append(PlannedFrame(ref_ts=int(ref_ts), per_cam_ts=per_cam))
    return planned


def extract_selected_images(
    heavy_bag: str,
    camera_topics: Dict[str, str],
    need_ts_by_cam: Dict[str, set[int]],
    *,
    start_us: Optional[int] = None,
    end_us: Optional[int] = None,
) -> Dict[str, Dict[int, object]]:
    out: Dict[str, Dict[int, object]] = {c: {} for c in config.CAMERA_NAMES}
    read_kw: Dict[str, Any] = {
        "topics": list(camera_topics.keys()),
        "dpbag_name": heavy_bag,
        "force_get_data_by_raw": True,
    }
    if start_us is not None:
        read_kw["start_time"] = int(start_us)
    if end_us is not None:
        read_kw["end_time"] = int(end_us)
    need_total = sum(len(v) for v in need_ts_by_cam.values())
    print(
        f"[bag-decode] {os.path.basename(heavy_bag)} 待解码 {need_total} 帧",
        flush=True,
    )
    try:
        with DpBag(bag=heavy_bag) as bag:
            for topic, msg, _ in bag.read_messages(**read_kw):
                cam = camera_topics.get(topic)
                if cam is None:
                    continue
                obj = CompressedImage()
                obj.ParseFromString(strip_header(msg.data))
                ts_us = int(obj.header.timestamp_sec * 1e6)
                if ts_us not in need_ts_by_cam.get(cam, set()):
                    continue
                if ts_us in out[cam]:
                    continue
                img = cv2.imdecode(np.frombuffer(obj.data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    out[cam][ts_us] = img
    except Exception as e:
        _fail_bag(heavy_bag, e)
    return out


def write_samples_layout(
    samples_root: str,
    trip_id: int,
    heavy_bag: str,
    planned: Sequence[PlannedFrame],
    images_by_cam_ts: Dict[str, Dict[int, object]],
) -> Tuple[str, str, Dict[int, str]]:
    bag_prefix = extract_bag_prefix(heavy_bag)
    yyyymm = extract_yyyymm_from_bag_prefix(bag_prefix)
    cfg_dir = os.path.join(samples_root, "config", yyyymm, bag_prefix)
    os.makedirs(cfg_dir, exist_ok=True)

    if DRFILE_AVAILABLE:
        for cfg_name in ("cameras.cfg", "ground.cfg"):
            cfg_path = os.path.join(cfg_dir, cfg_name)
            if not os.path.isfile(cfg_path):
                cfg_bytes = download_trip_config(trip_id, cfg_name)
                with open(cfg_path, "wb") as f:
                    f.write(cfg_bytes)
    else:
        raise RuntimeError("drfile 不可用，无法下载 cameras.cfg / ground.cfg")

    # 保存 4 路鱼眼图片
    ts_to_row_name: Dict[int, str] = {}
    for pf in planned:
        ts_to_row_name[pf.ref_ts] = ts_us_to_filename(pf.ref_ts)
        for cam in config.CAMERA_NAMES:
            ts_cam = pf.per_cam_ts[cam]
            img = images_by_cam_ts.get(cam, {}).get(ts_cam)
            if img is None:
                raise RuntimeError(f"缺少图像: cam={cam}, ts={ts_cam}")
            cam_dir = os.path.join(samples_root, cam, yyyymm, bag_prefix)
            os.makedirs(cam_dir, exist_ok=True)
            fn = ts_us_to_filename(ts_cam)
            cv2.imwrite(os.path.join(cam_dir, fn), img)

    # 写 data_index.csv（每一行对应一个 ref_ts）
    csv_path = os.path.join(cfg_dir, "data_index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["TIMESTAMP"] + config.CAMERA_NAMES + ["Data_dir"])
        for pf in planned:
            row = [str(pf.ref_ts)]
            for cam in config.CAMERA_NAMES:
                row.append(ts_us_to_filename(pf.per_cam_ts[cam]))
            row.append(bag_prefix)
            writer.writerow(row)
    return bag_prefix, yyyymm, ts_to_row_name


def run_offline_avm(samples_root: str, bag_name_for_avm: str, out_root: str) -> None:
    run_sh = os.path.join(PROJECT_ROOT, "offline_avm_generate_release", "run_standalone.sh")
    if not os.path.isfile(run_sh):
        raise FileNotFoundError(f"未找到 AVM 生成脚本: {run_sh}")
    # 关键：传 config 目录对应的 bag 名（本工程 samples 使用 bag_prefix），并强制 interval=1 保留每帧
    cmd = [run_sh, "-i", samples_root, "-b", bag_name_for_avm, "-o", out_root, "--interval", "1"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"offline_avm_generate 失败 (code={p.returncode})\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def find_generated_avm_images(generate_root: str, bag_prefix: str) -> List[str]:
    candidates = [
        os.path.join(generate_root, bag_prefix),
        os.path.join(generate_root, f"{bag_prefix}.Heavy_Topic_Group.bag"),
    ]
    imgs: List[str] = []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isfile(p) and n.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(p)
    return imgs


def map_generated_avm_to_ref_ts(
    generated_imgs: Sequence[str], planned: Sequence[PlannedFrame]
) -> Dict[int, str]:
    """尽量按时间戳精确对齐 AVM 输出；若文件名无时间戳则回退按排序顺序对齐。"""
    ref_ts_set = {pf.ref_ts for pf in planned}
    by_ts: Dict[int, str] = {}
    unknown: List[str] = []
    for p in generated_imgs:
        ts = parse_filename_ts_us(p)
        if ts is None:
            unknown.append(p)
            continue
        # 允许与 ref_ts 有轻微偏差，映射到最近的 ref_ts
        nearest = min(ref_ts_set, key=lambda x: abs(x - ts))
        if abs(nearest - ts) <= 200_000:  # 200ms 容差
            by_ts[nearest] = p
        else:
            unknown.append(p)

    if unknown:
        unknown_sorted = sorted(unknown)
        missing = [pf.ref_ts for pf in planned if pf.ref_ts not in by_ts]
        for i, ts in enumerate(missing):
            if i >= len(unknown_sorted):
                break
            by_ts[ts] = unknown_sorted[i]
    return by_ts


def _manifest_path(case_out: str) -> str:
    return os.path.join(case_out, GENERATE_AVM_MANIFEST_BASENAME)


def _output_files_complete_for_refs(
    case_out: str, tag_id: int, ref_ts_list: Sequence[int]
) -> bool:
    """``case_out/avm`` 与 ``case_out/yuyan`` 下各 ``ref_ts`` 对应 jpg 均存在且非空。"""
    avm_dir = os.path.join(case_out, "avm")
    yuyan_dir = os.path.join(case_out, "yuyan")
    for rts in ref_ts_list:
        avm_p = os.path.join(avm_dir, f"{tag_id}_{int(rts)}.jpg")
        if not (os.path.isfile(avm_p) and os.path.getsize(avm_p) > 0):
            return False
        y_p = os.path.join(yuyan_dir, f"{tag_id}_{int(rts)}.jpg")
        if not (os.path.isfile(y_p) and os.path.getsize(y_p) > 0):
            return False
    return True


def _try_skip_without_opening_bag(
    case_id: str,
    args: argparse.Namespace,
    tag_id: int,
    target_ts: int,
    main_camera: str,
) -> Optional[List[str]]:
    """若 manifest 与磁盘输出齐全且与当前 CLI 一致，则返回 AVM 路径列表且不访问 bag / 元数据。"""
    if args.force_regenerate:
        return None
    case_out = os.path.join(args.output_root, f"{tag_id}_{target_ts}")
    mp = _manifest_path(case_out)
    if not os.path.isfile(mp):
        return None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if int(m.get("schema_version", 0)) != GENERATE_AVM_MANIFEST_VERSION:
        return None
    if int(m.get("tag_id", -1)) != tag_id or int(m.get("target_ts", -1)) != target_ts:
        return None
    if int(m.get("frames_requested", -1)) != int(args.frames):
        return None
    if int(m.get("frame_step_requested", -1)) != int(args.frame_step):
        return None
    if str(m.get("main_camera", "")) != str(main_camera):
        return None
    raw_refs = m.get("ref_ts")
    if not isinstance(raw_refs, list) or not raw_refs:
        return None
    ref_ts_list = [int(x) for x in raw_refs]
    if not _output_files_complete_for_refs(case_out, tag_id, ref_ts_list):
        return None
    avm_out_dir = os.path.join(case_out, "avm")
    print(
        f"[skip-existing] case_id={case_id}：manifest 与输出齐全且参数一致，"
        "**不打开 Heavy bag**、不拉元数据",
        flush=True,
    )
    return [os.path.join(avm_out_dir, f"{tag_id}_{rts}.jpg") for rts in ref_ts_list]


def _write_generate_avm_manifest(
    case_out: str,
    *,
    tag_id: int,
    target_ts: int,
    frames_requested: int,
    frame_step_requested: int,
    frame_step_effective: int,
    main_camera: str,
    ref_ts_list: Sequence[int],
) -> None:
    os.makedirs(case_out, exist_ok=True)
    payload = {
        "schema_version": GENERATE_AVM_MANIFEST_VERSION,
        "tag_id": int(tag_id),
        "target_ts": int(target_ts),
        "frames_requested": int(frames_requested),
        "frame_step_requested": int(frame_step_requested),
        "frame_step_effective": int(frame_step_effective),
        "main_camera": str(main_camera),
        "ref_ts": [int(x) for x in ref_ts_list],
    }
    with open(_manifest_path(case_out), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_case_generation(case_id: str, args: argparse.Namespace) -> List[str]:
    tag_id, target_ts = parse_case_id(case_id)
    if args.main_camera == "auto":
        main_camera = infer_main_camera_from_pipeline(tag_id, target_ts)
    else:
        main_camera = args.main_camera

    skip_paths = _try_skip_without_opening_bag(
        case_id, args, tag_id, target_ts, main_camera
    )
    if skip_paths is not None:
        return skip_paths

    meta = get_meta_data(tag_id=tag_id)
    if not meta or not meta.get("body"):
        raise RuntimeError(f"获取 tag 元数据失败: tag_id={tag_id}")
    trip_id = int(meta["body"][0]["tripId"])
    bags = meta["body"][0].get("bagsName", [])
    heavy_bags = sorted(b for b in bags if "Heavy" in b)
    if not heavy_bags:
        raise RuntimeError(f"tag_id={tag_id} 没有 Heavy bag")

    topic_to_cam = dict(zip(config.CAMERA_TOPICS, config.CAMERA_NAMES))
    best_heavy, main_ts, cam_ts_map = choose_best_heavy_bag(
        heavy_bags,
        target_ts,
        main_camera,
        topic_to_cam,
        frames=args.frames,
        frame_step=args.frame_step,
        bag_candidate_limit=args.bag_candidate_limit,
        scan_margin_sec=args.bag_scan_margin_sec,
    )
    selected_ref, eff_step = pick_nearby_window(
        main_ts, target_ts, args.frames, args.frame_step
    )
    if not selected_ref:
        raise RuntimeError("未找到可用参考帧")

    planned = plan_frames(selected_ref, cam_ts_map)
    if not planned:
        raise RuntimeError("四路最近邻规划失败，无可用帧")

    need_ts_by_cam: Dict[str, set[int]] = {c: set() for c in config.CAMERA_NAMES}
    for pf in planned:
        for cam, ts in pf.per_cam_ts.items():
            need_ts_by_cam[cam].add(ts)

    decode_lo, decode_hi = narrow_time_window_us(selected_ref, pad_us=3_000_000)
    images_by_cam_ts = extract_selected_images(
        best_heavy,
        topic_to_cam,
        need_ts_by_cam,
        start_us=decode_lo,
        end_us=decode_hi,
    )

    tmp_root = tempfile.mkdtemp(prefix=f"case_avm_{tag_id}_", dir=args.output_root)
    samples_root = os.path.join(tmp_root, "samples")
    avm_tmp_out = os.path.join(tmp_root, "avm_out")

    written_avm_paths: List[str] = []
    try:
        bag_prefix, yyyymm, _ = write_samples_layout(
            samples_root, trip_id, best_heavy, planned, images_by_cam_ts
        )
        run_offline_avm(samples_root, bag_prefix, avm_tmp_out)
        avm_imgs = find_generated_avm_images(avm_tmp_out, bag_prefix)

        case_out = os.path.join(args.output_root, f"{tag_id}_{target_ts}")
        avm_out_dir = os.path.join(case_out, "avm")
        os.makedirs(avm_out_dir, exist_ok=True)

        pipeline_projector: Optional[Any] = None
        chaosheng_radius_eff: Optional[int] = None
        if not args.skip_pipeline_overlay:
            rpix = int(args.chaosheng_pixel_radius)
            chaosheng_radius_eff = None if rpix <= 0 else rpix
            if PanoramicProjector is not None and render_bev_from_raw_avm is not None:
                pipeline_projector = PanoramicProjector()
            elif PanoramicProjector is None or render_bev_from_raw_avm is None:
                print(
                    "[pipeline-overlay] WARN: PanoramicProjector / render_bev_from_raw_avm 不可用，跳过叠绘",
                    file=sys.stderr,
                )

        snapshot_ts_sorted: List[int] = []
        static_payload: Optional[Dict[str, Any]] = None
        per_frame_payloads: Dict[int, Dict[str, Any]] = {}
        per_frame_diffs: Dict[int, Dict[str, int]] = {}
        per_frame_extract_used = False
        if (
            not args.skip_pipeline_overlay
            and pipeline_projector is not None
            and render_bev_from_raw_avm is not None
        ):
            static_payload, static_err = _load_static_payload_from_read_data(
                args.read_data_root, tag_id
            )
            if static_err or static_payload is None:
                print(
                    f"[pipeline-overlay] WARN: 静态资源加载失败({static_err})，全帧回退到目录式叠绘",
                    file=sys.stderr,
                )

            if not args.no_per_frame_payload:
                proto_err = _ensure_light_bag_proto_runtime()
                if proto_err:
                    raise RuntimeError(f"Light bag proto 不可用: {proto_err}")
                light_bag = derive_light_bag_from_heavy(best_heavy)
                print(f"[per-frame] 现场扫描 Light bag: {light_bag}")
                try:
                    per_frame_payloads, per_frame_diffs = (
                        extract_per_frame_payloads_from_light_bag(
                            light_bag,
                            [pf.ref_ts for pf in planned],
                            max_skew_us=args.pipeline_overlay_max_skew_us,
                        )
                    )
                except Exception as e:
                    _fail_bag(light_bag, e)
                per_frame_extract_used = True
                n_with_pose = sum(
                    1 for v in per_frame_payloads.values() if v.get("pose")
                )
                n_nonempty_ch = sum(
                    1 for v in per_frame_payloads.values() if v.get("chaosheng")
                )
                print(
                    f"[per-frame] 抽取完成：pose {n_with_pose}/{len(planned)}，"
                    f"非空 chaosheng {n_nonempty_ch}/{len(planned)}（空列表仍会叠绘相机/规划）"
                )

            if not per_frame_extract_used:
                snapshot_ts_sorted = list_read_data_snapshot_timestamps_sorted(
                    os.path.join(args.read_data_root, str(tag_id))
                )

        snapshot_ts_for_fallback: List[int] = []
        if (
            per_frame_extract_used
            and not args.no_per_frame_payload_fallback
            and pipeline_projector is not None
        ):
            snapshot_ts_for_fallback = list_read_data_snapshot_timestamps_sorted(
                os.path.join(args.read_data_root, str(tag_id))
            )

        case_anchor_ch_raw: Optional[List[Any]] = None
        case_anchor_obs_raw: Optional[List[Any]] = None
        case_anchor_px: Optional[int] = None
        if (
            not args.skip_pipeline_overlay
            and not args.no_case_chaosheng_anchor_filter
            and pipeline_projector is not None
        ):
            case_anchor_ch_raw, case_anchor_obs_raw = _load_case_chaosheng_anchor_pair(
                args.read_data_root, tag_id, target_ts
            )
            if case_anchor_ch_raw is not None:
                case_anchor_px = int(args.case_chaosheng_anchor_radius)
                print(
                    f"[case-anchor] 使用 read_data/{tag_id}/{target_ts}/chaosheng.json 为像素锚，"
                    f"仅绘制距其投影 ≤{case_anchor_px}px 的相机障碍（规划线仍全量）"
                )
            else:
                print(
                    f"[case-anchor] WARN: 未找到 read_data/{tag_id}/{target_ts}/chaosheng.json，"
                    "未启用锚点筛选",
                    file=sys.stderr,
                )

        # AVM 输出：优先按文件名时间戳对齐，否则回退按排序顺序。
        # 注意这里使用的是“本次四路鱼眼拼接生成”的结果，不复用历史 AVM。
        avm_map = map_generated_avm_to_ref_ts(avm_imgs, planned)
        stitched_avm = 0
        overlay_painted = 0
        overlay_skipped = 0
        all_bev_chaosheng_pos: List = []
        per_frame_pick_log: List[
            Tuple[int, Optional[str], Optional[int], Optional[Dict[str, int]], Optional[str]]
        ] = []
        for pf in planned:
            ref_ts = pf.ref_ts
            src = avm_map.get(ref_ts)
            if not src:
                continue
            out_name = f"{tag_id}_{ref_ts}.jpg"
            dst = os.path.join(avm_out_dir, out_name)
            avm_img = cv2.imread(src)
            if avm_img is None or avm_img.size == 0:
                continue
            out_img = avm_img
            if (
                not args.skip_pipeline_overlay
                and pipeline_projector is not None
                and render_bev_from_raw_avm is not None
            ):
                fp = per_frame_payloads.get(ref_ts) if per_frame_extract_used else None
                fd = per_frame_diffs.get(ref_ts) if per_frame_extract_used else None
                if per_frame_extract_used:
                    if fp and fp.get("pose"):
                        src_lbl = "lightbag-perframe"
                    else:
                        fp = None
                        src_lbl = None
                else:
                    src_lbl = None

                out_img, ov_warn, picked_src, picked_diff, bev_pos = overlay_avm_like_pipeline_step5(
                    pipeline_projector,
                    avm_img,
                    tag_id,
                    ref_ts,
                    read_data_root=args.read_data_root,
                    chaosheng_pixel_radius=chaosheng_radius_eff,
                    ignore_fs_types=list(args.ignore_fs_types or []),
                    draw_ultrasonic_red=args.pipeline_draw_ultrasonic_red,
                    snapshot_ts_sorted=snapshot_ts_sorted,
                    max_skew_us=args.pipeline_overlay_max_skew_us,
                    require_exact_dir_name=args.pipeline_overlay_read_data_exact,
                    frame_payload=fp,
                    static_payload=static_payload,
                    frame_source_label=src_lbl,
                    case_anchor_chaosheng_raw=case_anchor_ch_raw,
                    case_anchor_obstacle_raw=case_anchor_obs_raw,
                    case_anchor_pixel_radius=case_anchor_px,
                )
                all_bev_chaosheng_pos.extend(bev_pos)

                if (
                    picked_src is None
                    and per_frame_extract_used
                    and not args.no_per_frame_payload_fallback
                    and snapshot_ts_for_fallback
                ):
                    out_img2, ov_warn2, picked_src2, picked_diff2, bev_pos2 = overlay_avm_like_pipeline_step5(
                        pipeline_projector,
                        avm_img,
                        tag_id,
                        ref_ts,
                        read_data_root=args.read_data_root,
                        chaosheng_pixel_radius=chaosheng_radius_eff,
                        ignore_fs_types=list(args.ignore_fs_types or []),
                        draw_ultrasonic_red=args.pipeline_draw_ultrasonic_red,
                        snapshot_ts_sorted=snapshot_ts_for_fallback,
                        max_skew_us=args.pipeline_overlay_max_skew_us,
                        require_exact_dir_name=args.pipeline_overlay_read_data_exact,
                        frame_payload=None,
                        static_payload=static_payload,
                        case_anchor_chaosheng_raw=case_anchor_ch_raw,
                        case_anchor_obstacle_raw=case_anchor_obs_raw,
                        case_anchor_pixel_radius=case_anchor_px,
                    )
                    if picked_src2 is not None:
                        out_img = out_img2
                        ov_warn = ov_warn2
                        picked_src = picked_src2
                        picked_diff = picked_diff2
                        all_bev_chaosheng_pos.extend(bev_pos2)

                per_frame_pick_log.append((ref_ts, picked_src, picked_diff, fd, ov_warn))
                if picked_src is not None:
                    overlay_painted += 1
                else:
                    overlay_skipped += 1
                if ov_warn:
                    print(
                        f"[pipeline-overlay] ref_ts={ref_ts}: {ov_warn}",
                        file=sys.stderr,
                    )
            cv2.imwrite(dst, out_img)
            stitched_avm += 1
            written_avm_paths.append(dst)

        if not main_camera and all_bev_chaosheng_pos and avm_positions_to_yuyan_camera is not None:
            inferred_cam, inferred_dir = avm_positions_to_yuyan_camera(all_bev_chaosheng_pos)
            if inferred_cam:
                main_camera = inferred_cam
                print(
                    f"[yuyan] 从叠绘超声质心推断主方位相机: {main_camera}（方位: {inferred_dir}）"
                )

        fisheye_tmp_dir = os.path.join(case_out, _FISHEYE_ALL_DIR)
        for cam_name in config.CAMERA_NAMES:
            cam_out_dir = os.path.join(fisheye_tmp_dir, cam_name)
            os.makedirs(cam_out_dir, exist_ok=True)
            for pf in planned:
                cam_ts = pf.per_cam_ts.get(cam_name)
                if cam_ts is not None and cam_ts in images_by_cam_ts.get(cam_name, {}):
                    img = images_by_cam_ts[cam_name][cam_ts]
                    out_name = f"{tag_id}_{pf.ref_ts}.jpg"
                    cv2.imwrite(os.path.join(cam_out_dir, out_name), img)

        print("=" * 70)
        print(f"case_id: {case_id}")
        print(f"选中 Heavy bag: {best_heavy}")
        print(f"主方位相机(超声): {main_camera or '（未推断到）'}（mark-avm 后按 crop_id 逐帧重选）")
        if eff_step != args.frame_step:
            print(
                f"抽帧步长: 请求 {args.frame_step}，可用帧不足已自动回退为 {eff_step}"
            )
        else:
            print(f"抽帧步长: {eff_step}")
        print(f"计划帧数: {len(planned)}")
        print(f"AVM 拼接落盘: {stitched_avm} 张")
        print(f"四路鱼眼暂存: {fisheye_tmp_dir}（mark-avm 后自动选入 yuyan/ 并清理）")
        if not args.skip_pipeline_overlay and pipeline_projector is not None:
            mode_label = "Light bag 现场抽取" if per_frame_extract_used else "read_data 目录最近邻"
            print(
                f"AVM 叠绘({mode_label}): 成功 {overlay_painted} / 跳过 {overlay_skipped}"
            )
            if per_frame_pick_log:
                print("  逐帧使用的本帧负载来源:")
                for rts, src, diff, field_diffs, _warn in per_frame_pick_log:
                    if src is None:
                        tag_info = "（未叠绘）"
                    elif src == "lightbag-perframe":
                        if field_diffs:
                            details = ", ".join(
                                f"{k}|Δ|={v}μs" for k, v in sorted(field_diffs.items())
                            )
                            tag_info = f"（Light bag 最近邻：{details}）"
                        else:
                            tag_info = "（Light bag 最近邻）"
                    else:
                        if diff == 0:
                            tag_info = "（read_data 精确目录）"
                        else:
                            tag_info = f"（read_data 最近邻 |Δ|={diff}μs）"
                    print(f"    ref_ts={rts} → src={src or 'NONE'} {tag_info}")
        print(f"输出目录: {case_out}")
        if stitched_avm != len(planned):
            print(
                f"[WARN] AVM 数量({stitched_avm}) 与计划帧数({len(planned)})不一致，"
                "已按时间戳/排序结果尽量对齐。"
            )
        print("=" * 70)
        if written_avm_paths and stitched_avm == len(planned):
            _write_generate_avm_manifest(
                case_out,
                tag_id=tag_id,
                target_ts=target_ts,
                frames_requested=args.frames,
                frame_step_requested=args.frame_step,
                frame_step_effective=eff_step,
                main_camera=main_camera or "",
                ref_ts_list=[pf.ref_ts for pf in planned],
            )
        return written_avm_paths
    finally:
        if args.keep_temp:
            print(f"[KEEP TEMP] {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="给定 {tag}_{timestamp_us}，抽取附近 10 帧四路鱼眼并生成 AVM + 主方位单路鱼眼图"
    )
    parser.add_argument(
        "case_id",
        nargs="?",
        default=None,
        help="单个 case_id（形如 {tag_id}_{timestamp_us}）；与 --case-list 二选一",
    )
    parser.add_argument(
        "--case-list",
        metavar="FILE",
        default=None,
        help=(
            "从文本文件批量读取 case_id，每行一条；空行与 # 注释忽略。"
            "示例: /mnt/public-data/user/ziroujiang/generate_raw_data/case_id.txt"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"输出根目录（默认 {DEFAULT_OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"目标时间戳附近帧数（默认 {DEFAULT_FRAMES}）",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help=f"抽帧步长（按主相机帧索引步长，默认 {DEFAULT_FRAME_STEP}；越大相邻帧差异越明显）",
    )
    parser.add_argument(
        "--main-camera",
        default="auto",
        choices=["auto"] + list(config.CAMERA_NAMES),
        help=(
            "与 AVM 主方位一致的单路鱼眼相机；默认 auto：读 "
            "draw_image/<tag>/<ts>/index_avm.json 中 yuyan_camera（与大模型诊断一致），"
            "推断不到则跳过单路鱼眼输出"
        ),
    )
    parser.add_argument(
        "--pipeline-draw-ultrasonic-red",
        action="store_true",
        help=(
            "叠绘时在鸟瞰图上绘制超声障碍的红色标记（与 pipeline Step5 完全一致）。"
            "默认关闭：其余绘制逻辑与 Step5 相同，仅省略超声红"
        ),
    )
    parser.add_argument(
        "--read-data-root",
        default=config.READ_DATA_DIR,
        help=f"read_data 根目录（叠加相机黄/绿线与规划线；超声红线见 --pipeline-draw-ultrasonic-red），默认 {config.READ_DATA_DIR}",
    )
    parser.add_argument(
        "--skip-pipeline-overlay",
        action="store_true",
        help=(
            "关闭整张 pipeline 式叠绘（黄/绿/白线及可选超声红）。默认开启叠绘，逻辑与 Step5 一致，仅默认不画超声红"
        ),
    )
    parser.add_argument(
        "--pipeline-overlay-max-skew-us",
        type=int,
        default=READ_DATA_TS_TOLERANCE_US,
        metavar="US",
        help=(
            "每帧 ref_ts 与所选 read_data 感知快照目录时间戳的最大允许偏差（微秒）；"
            f"超过则跳过叠绘以免错位（默认 {READ_DATA_TS_TOLERANCE_US}，即 50ms）"
        ),
    )
    parser.add_argument(
        "--pipeline-overlay-read-data-exact",
        action="store_true",
        help=(
            "仅当存在齐全的 read_data/<tag>/<本帧 ref_ts>/ 时才叠绘，不做最近邻快照匹配"
            "（仅在禁用现场抽取后才会真正生效）"
        ),
    )
    parser.add_argument(
        "--no-per-frame-payload",
        action="store_true",
        help=(
            "禁用从 Light bag 现场抽取每帧 obstacle/pose/plan/chaosheng（默认开启）；"
            "禁用后回退到 read_data/<tag>/<ts>/ 子目录最近邻方案，多帧可能共享同一快照"
        ),
    )
    parser.add_argument(
        "--no-per-frame-payload-fallback",
        action="store_true",
        help=(
            "现场抽取未匹配到某帧负载时，默认会回退到 read_data 目录最近邻；启用本开关则该帧直接不叠绘"
        ),
    )
    parser.add_argument(
        "--chaosheng-pixel-radius",
        type=int,
        default=30,
        help=(
            "未启用 case 锚点筛选时：与 pipeline Step5 一致，用本帧 chaosheng 做相机障碍像素邻近筛选；≤0 关闭。"
            "启用 ``--case-chaosheng-anchor-radius``（默认）时，本参数不参与相机障碍筛选"
        ),
    )
    parser.add_argument(
        "--case-chaosheng-anchor-radius",
        type=int,
        default=DEFAULT_CASE_CHAOSHENG_ANCHOR_RADIUS_PX,
        metavar="PX",
        help=(
            "以 case_id 时间戳目录 ``read_data/<tag>/<ts>/chaosheng.json`` 原始超声为锚，"
            "仅绘制其投影顶点该像素半径内的相机障碍（规划线仍全量）；默认 50"
        ),
    )
    parser.add_argument(
        "--no-case-chaosheng-anchor-filter",
        action="store_true",
        help="关闭上述锚点筛选，改按本帧 chaosheng 与 --chaosheng-pixel-radius 逻辑（与 Step5 邻近筛选一致）",
    )
    parser.add_argument(
        "--ignore-fs-types",
        nargs="*",
        default=[],
        help="绘图时忽略的超声 freespaceType（与 pipeline 一致），例如 FS_CURB FS_CHOCK",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留中间 samples / avm 临时目录（默认自动删除）",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help=(
            "忽略已有输出与 .generate_avm_manifest.json，强制重新拉元数据、打开 bag、抽帧、拼接与叠绘"
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "仍处理 .generate_avm_failures.jsonl 中已有记录的 case（默认会跳过这些 case）"
        ),
    )
    parser.add_argument(
        "--bag-scan-margin-sec",
        type=float,
        default=DEFAULT_BAG_SCAN_MARGIN_SEC,
        help=(
            "远端扫 Heavy/Light bag 时在 target_ts 两侧扩展的秒数（默认 45），"
            "避免整包 read_messages"
        ),
    )
    parser.add_argument(
        "--bag-candidate-limit",
        type=int,
        default=DEFAULT_BAG_CANDIDATE_LIMIT,
        help="按文件名预选后最多尝试打开的 Heavy bag 数（默认 3；读失败仍立即放弃该 case）",
    )
    parser.add_argument(
        "--mark-avm",
        action="store_true",
        help=(
            "全部 case 的 AVM 生成并写盘结束后，再按顺序弹出 OpenCV 窗口逐张标注（须显式加本开关；默认不弹窗）。"
            "需在图形桌面或有 DISPLAY/WAYLAND 的环境"
        ),
    )
    parser.add_argument(
        "--mark-thickness",
        type=float,
        default=2,
        help="线宽像素",
    )
    parser.add_argument(
        "--crop-id-json",
        default="/mnt/public-data/user/ziroujiang/generate_raw_data/crop_id.json",
        help=(
            "标注时记录笔划质心坐标的 JSON 路径（格式 {case_id: \"[x,y]\"}）；"
            "默认 /mnt/public-data/user/ziroujiang/generate_raw_data/crop_id.json"
        ),
    )
    parser.add_argument(
        "--export-flat",
        metavar="DIR",
        default="",
        help=(
            "将全部 case 的 avm/crop/yuyan 整理为扁平目录 DIR（images/crop/yuyan），"
            "布局与 generate_ground_irregularity 一致；在生成与 mark-avm 结束后执行"
        ),
    )
    parser.add_argument(
        "--export-flat-only",
        action="store_true",
        help="仅从 --output-root 扫描已有 case 并执行 --export-flat，不生成、不标注",
    )
    parser.add_argument(
        "--export-flat-conflict",
        choices=("skip", "overwrite", "error"),
        default="overwrite",
        help="扁平导出时目标文件已存在的处理方式（默认 overwrite）",
    )
    args = parser.parse_args()

    if args.export_flat_only:
        if not args.export_flat:
            parser.error("--export-flat-only 须同时指定 --export-flat DIR")
        export_output_root_to_flat(
            args.output_root,
            args.export_flat,
            on_conflict=args.export_flat_conflict,
        )
        return 0

    if args.mark_avm:
        warn_if_mark_avm_no_gui()

    case_ids: List[str] = []
    if args.case_list:
        case_ids = load_case_ids_from_file(args.case_list)
    elif args.case_id:
        case_ids = [args.case_id]
    elif not args.mark_avm and not args.export_flat:
        parser.error("请提供单个 case_id、--case-list FILE、--mark-avm 或 --export-flat")

    os.makedirs(args.output_root, exist_ok=True)

    failures_log = os.path.join(
        args.output_root, ".generate_avm_failures.jsonl"
    )
    prior_failure_ids: Set[str] = set()
    if case_ids and not args.retry_failed:
        prior_failure_ids = _load_failure_case_ids(failures_log)
        if prior_failure_ids:
            print(
                f"[INFO] 已加载 {len(prior_failure_ids)} 个历史失败 case，"
                f"将跳过（见 {failures_log}；加 --retry-failed 可重试）",
                flush=True,
            )

    failed = 0
    success_cases: List[str] = []
    failed_records: List[Dict[str, str]] = []
    cold_storage_cases: List[str] = []
    skipped_prior_failures: List[str] = []
    if case_ids:
        ensure_runtime_deps()
        for idx, cid in enumerate(case_ids, start=1):
            if len(case_ids) > 1:
                print(f"\n{'=' * 70}\n[{idx}/{len(case_ids)}] case_id={cid}\n{'=' * 70}")
            if cid in prior_failure_ids:
                print(
                    f"[SKIP-FAIL] {cid}: 已在失败记录中，跳过（{failures_log}）",
                    flush=True,
                )
                skipped_prior_failures.append(cid)
                continue
            try:
                run_case_generation(cid, args)
                success_cases.append(cid)
                print(f"[OK] {cid}", flush=True)
            except ColdStorageError as e:
                print(f"[COLD] {cid}: 数据在冷存储中，跳过")
                cold_storage_cases.append(cid)
                failed += 1
                rec = {
                    "case_id": cid,
                    "kind": "cold_storage",
                    "error": str(e),
                }
                failed_records.append(rec)
                _append_failure_log(failures_log, rec)
            except BagNotFoundError as e:
                print(f"[FAIL] {cid}: bag 不可用 — {e}", file=sys.stderr)
                failed += 1
                rec = {"case_id": cid, "kind": "bag_not_found", "error": str(e)}
                failed_records.append(rec)
                _append_failure_log(failures_log, rec)
            except Exception as e:
                print(f"[FAIL] {cid}: {e}", file=sys.stderr)
                failed += 1
                rec = {"case_id": cid, "kind": "error", "error": str(e)}
                failed_records.append(rec)
                _append_failure_log(failures_log, rec)

        _print_batch_summary(
            total=len(case_ids),
            success_cases=success_cases,
            failed_records=failed_records,
            cold_storage_cases=cold_storage_cases,
            skipped_prior_failures=skipped_prior_failures,
            failures_log=failures_log,
        )

    if args.mark_avm:
        avm_paths, skipped_cases, selected_cases = _collect_avm_paths_from_existing_dirs(
            args.output_root, skip_if_crop_exists=True,
        )
        if skipped_cases:
            print(f"\n[mark-avm] 已有 crop/ 跳过: {len(skipped_cases)} 个 case")
        if avm_paths:
            print(
                f"[mark-avm] 待标注: {len(selected_cases)} 个 case，"
                f"共 {len(avm_paths)} 张 AVM，开始统一逐张标注 …"
            )
            print(
                "[mark-avm] 左键拖动红色 | s 保存覆盖并记录质心 | z/r 撤销/清空 | "
                "n 下一张（自动记录质心） | q/ESC 结束"
            )
            crop_id_path = args.crop_id_json
            for i, p in enumerate(avm_paths, start=1):
                again = cv2.imread(p)
                if again is None or again.size == 0:
                    print(f"[mark-avm] 跳过（无法读取）: {p}", file=sys.stderr)
                    continue
                stem = os.path.splitext(os.path.basename(p))[0]
                title = f"mark AVM [{i}/{len(avm_paths)}]: {p}"
                r = interactive_red_mark_session(
                    again,
                    save_path=p,
                    window_title=title,
                    thickness=args.mark_thickness,
                    allow_next=True,
                    case_id=stem,
                    crop_id_path=crop_id_path,
                )
                if r == "quit":
                    break
            cv2.destroyAllWindows()

            total_cropped = _do_crop_from_avm_paths(avm_paths, args.crop_id_json)
            if total_cropped:
                print(f"\n[crop] 150×150 中心裁剪完成: {total_cropped} 张")

            total_yuyan = _reassign_yuyan_from_crop_id(args.output_root, args.crop_id_json)
            if total_yuyan:
                print(f"[yuyan-reassign] 逐帧鱼眼重分配完成: {total_yuyan} 张 → yuyan/")
        else:
            if skipped_cases:
                print(
                    f"[mark-avm] {len(skipped_cases)} 个 case 已有 crop/，无新 AVM 待标注"
                )
            else:
                print(
                    "[mark-avm] 未发现待标注 AVM（请确认生成是否成功，或检查 --output-root）"
                )
            total_yuyan = _reassign_yuyan_from_crop_id(args.output_root, args.crop_id_json)
            if total_yuyan:
                print(f"[yuyan-reassign] 逐帧鱼眼重分配完成: {total_yuyan} 张 → yuyan/")

    if args.export_flat:
        export_output_root_to_flat(
            args.output_root,
            args.export_flat,
            on_conflict=args.export_flat_conflict,
        )

    # 仅当「有 case 列表且全部失败」时返回非 0；部分成功仍返回 0，便于后续 mark/export 继续
    if case_ids and failed == len(case_ids):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

