#!/usr/bin/env python3
"""
对 val_dataset.jsonl 中的 case 运行 pipeline 同款超声误检诊断（chaosheng_wujian_avm + VLM_API）。

与 tool/eas_eval.py --eval 不同：不做实体/几何/类型三分类，而是判断红色超声是否误检。

输入（默认不依赖 read_data）：
  1. draw_image/<tag>/<ts>/avm.jpg + index_avm.json（若存在）
  2. all_data_v3/images|yuyan|crop/{tag}_{ts}.jpg；
     超声质心：crop/chaosheng_centroids.txt，若无该行则在 AVM 上对 crop 做模板匹配

可选 ``--fallback-read-data``：无 crop 质心时回退 read_data+generate 推算 index。

不发飞书评论；结果写入 jsonl 与 misdetected|normal 目录。

用法::

  conda activate avp
  cd /home/jiangzirou/avp_promptkit

  python tool/diagnose_val_dataset.py --limit 5

  python tool/diagnose_val_dataset.py \
    --val-jsonl /mnt/public-data/user/ziroujiang/all_data_v3/val_dataset_v5.jsonl \
    --data-root /mnt/public-data/user/ziroujiang/all_data_v3 \
    -o /mnt/public-data/user/ziroujiang/val/val_misdetect_diagnose/api_predictions_v5.jsonl \
    --resume --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

_tool_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tool_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from vlm.VLM_API import analyze_scenario_from_images
from vlm.avp_vlm_pipeline_avm import (
    YUYAN_CAMERA_LABEL_ZH,
    avm_positions_to_yuyan_camera,
)
from vlm.point2box_mindistance_avm import (
    calculate_segment_center,
    is_segment_misdetected,
)
from prompts_engine.prompt_gen import prompt_gen

logger = logging.getLogger(__name__)

DEFAULT_VAL = "/mnt/public-data/user/ziroujiang/all_data_v3/val_dataset.jsonl"
DEFAULT_DATA_ROOT = "/mnt/public-data/user/ziroujiang/all_data_v3"
DEFAULT_OUT = os.path.join(_project_root, "tool", "output", "val_misdetect_diagnose")
DEFAULT_CENTROIDS_MANIFEST = "crop/chaosheng_centroids.txt"
CROP_MATCH_MIN_SCORE = 0.5


def _parse_case_stem(stem: str) -> Optional[Tuple[int, str]]:
    stem = stem.strip()
    if "_" not in stem:
        return None
    left, _, right = stem.partition("_")
    if left.isdigit() and right.isdigit():
        return int(left), right
    return None


def load_unique_cases_from_val_jsonl(path: str) -> List[Tuple[int, str, str]]:
    """从 val jsonl 的 images[0] 解析唯一 (tag_id, ts_us, case_id)。"""
    seen: set = set()
    out: List[Tuple[int, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("jsonl 第 %s 行解析失败: %s", lineno, e)
                continue
            images = obj.get("images") or []
            if not images:
                continue
            stem = os.path.splitext(os.path.basename(str(images[0])))[0]
            if stem in seen:
                continue
            parsed = _parse_case_stem(stem)
            if not parsed:
                logger.warning("第 %s 行无法解析 case: %s", lineno, stem)
                continue
            tag_id, ts = parsed
            seen.add(stem)
            out.append((tag_id, ts, stem))
    return out


def _load_done_case_ids(path: Path) -> set:
    done: set = set()
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("case_id")
            if cid and not row.get("skip_reason") and not row.get("api_error"):
                done.add(str(cid))
    return done


def load_chaosheng_centroids_manifest(path: str) -> Dict[str, List[int]]:
    """解析 ``{case_id}: cx,cy`` 清单（与 crop_read_data_chaosheng 输出一致）。"""
    out: Dict[str, List[int]] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            stem, _, rest = line.partition(":")
            stem = stem.strip()
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) < 2:
                continue
            try:
                out[stem] = [int(parts[0]), int(parts[1])]
            except ValueError:
                continue
    return out


def _centroid_from_crop_template(
    avm_bgr: Any,
    crop_bgr: Any,
    *,
    min_score: float = CROP_MATCH_MIN_SCORE,
) -> Optional[List[int]]:
    """在 AVM 全图上匹配 crop  patch，返回质心像素 [u, v]。"""
    if avm_bgr is None or crop_bgr is None:
        return None
    ah, aw = avm_bgr.shape[:2]
    ch, cw = crop_bgr.shape[:2]
    if ch > ah or cw > aw or ch < 8 or cw < 8:
        return None
    res = cv2.matchTemplate(avm_bgr, crop_bgr, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val < min_score:
        return None
    return [int(max_loc[0] + cw // 2), int(max_loc[1] + ch // 2)]


def _build_index_from_data_root(
    case_id: str,
    avm_bgr: Any,
    crop_bgr: Any,
    centroids_map: Dict[str, List[int]],
    *,
    crop_match_min_score: float = CROP_MATCH_MIN_SCORE,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """由 manifest 或 crop 模板匹配构造 prompt 用 index。"""
    if case_id in centroids_map:
        pos = [centroids_map[case_id]]
        centroid_source = "manifest"
    else:
        c = _centroid_from_crop_template(
            avm_bgr, crop_bgr, min_score=crop_match_min_score
        )
        if not c:
            return None, "no_centroid"
        pos = [c]
        centroid_source = "crop_template"

    y_cam, y_dir = avm_positions_to_yuyan_camera(pos)
    index = {
        "avm": pos,
        "yellow_freespace": [],
        "yuyan_camera": y_cam,
        "yuyan_direction": y_dir,
        "yuyan_camera_label": YUYAN_CAMERA_LABEL_ZH.get(y_cam or "", y_cam or ""),
        "centroid_source": centroid_source,
    }
    return index, ""


def _fs_car_misdetects(draw_item_dir: str) -> List[List[int]]:
    box_path = os.path.join(draw_item_dir, "box_list_avm.json")
    point_path = os.path.join(draw_item_dir, "point_list_avm.json")
    if not (os.path.isfile(box_path) and os.path.isfile(point_path)):
        return []
    with open(box_path, encoding="utf-8") as f:
        box_list = json.load(f)
    with open(point_path, encoding="utf-8") as f:
        point_list = json.load(f)
    out: List[List[int]] = []
    for segment_points in point_list:
        if is_segment_misdetected(segment_points, box_list, threshold=8.0):
            center = calculate_segment_center(segment_points)
            out.append([center[0], center[1]])
    return out


def _build_index_from_read_data(
    tag_id: int,
    ts: str,
    *,
    read_data_dir: str,
    generate_dir: str,
    chaosheng_pixel_radius: int,
    ignore_fs_types: List[str],
    scan_generate: bool,
) -> Optional[Dict[str, Any]]:
    """由 read_data + generate 推算 index（仅 --fallback-read-data 时使用）。"""
    from tool.crop_read_data_chaosheng import (
        AVM_MATCH_TOLERANCE_US,
        _build_avm_index_from_meta,
        _build_avm_index_scan_generate,
        _match_avm_path,
        render_bev_from_raw_avm,
    )
    from vlm.panoramic_projector import PanoramicProjector

    data_path = os.path.join(read_data_dir, str(tag_id))
    item_path = os.path.join(data_path, ts)
    required = [
        "vehicle2sensing.json",
        "ground.json",
        "car_config.json",
        "chaosheng.json",
        "obstacle.json",
        "pose.json",
    ]
    if not all(os.path.isfile(os.path.join(data_path, f)) for f in required[:3]):
        return None
    if not os.path.isdir(item_path):
        return None
    for f in required[3:]:
        if not os.path.isfile(os.path.join(item_path, f)):
            return None

    with open(os.path.join(data_path, "vehicle2sensing.json"), encoding="utf-8") as f:
        vehicle2sensing = json.load(f)
    with open(os.path.join(data_path, "ground.json"), encoding="utf-8") as f:
        ground = json.load(f)
    with open(os.path.join(data_path, "car_config.json"), encoding="utf-8") as f:
        car_config = json.load(f)
    with open(os.path.join(item_path, "chaosheng.json"), encoding="utf-8") as f:
        chaosheng = json.load(f)
    with open(os.path.join(item_path, "obstacle.json"), encoding="utf-8") as f:
        obstacle = json.load(f)
    with open(os.path.join(item_path, "pose.json"), encoding="utf-8") as f:
        pose = json.load(f)
    plan_path = os.path.join(item_path, "plan.json")
    planning_point = []
    if os.path.isfile(plan_path):
        with open(plan_path, encoding="utf-8") as f:
            planning_point = json.load(f)

    ignore_fs = set(ignore_fs_types or [])
    if ignore_fs:
        chaosheng = [o for o in chaosheng if o.get("freespaceType", "") not in ignore_fs]

    projector = PanoramicProjector()
    projector.apply_chaosheng_z_from_camera_ground_plane(chaosheng, obstacle)
    obstacle = projector.world2vehicle2sensing(obstacle, pose, vehicle2sensing)
    chaosheng = projector.world2vehicle2sensing_chaosheng(chaosheng, pose, vehicle2sensing)

    if scan_generate:
        all_avm = _build_avm_index_scan_generate(generate_dir)
    else:
        all_avm = _build_avm_index_from_meta(tag_id, generate_dir)
        if not all_avm:
            all_avm = _build_avm_index_scan_generate(generate_dir)
    raw_path = _match_avm_path(int(ts), all_avm, tolerance=AVM_MATCH_TOLERANCE_US)
    if not raw_path:
        return None
    raw = cv2.imread(raw_path)
    if raw is None:
        return None

    _img, pos, yellow = render_bev_from_raw_avm(
        projector,
        raw,
        obstacle,
        chaosheng,
        ground,
        planning_point,
        pose,
        vehicle2sensing,
        car_config,
        chaosheng_pixel_radius,
        list(ignore_fs_types) if ignore_fs_types else None,
    )
    if not pos:
        return {"avm": [], "yellow_freespace": yellow or []}

    y_cam, y_dir = avm_positions_to_yuyan_camera(pos)
    return {
        "avm": pos,
        "yellow_freespace": yellow or [],
        "yuyan_camera": y_cam,
        "yuyan_direction": y_dir,
        "yuyan_camera_label": YUYAN_CAMERA_LABEL_ZH.get(y_cam or "", y_cam or ""),
    }


def _resolve_inputs(
    tag_id: int,
    ts: str,
    case_id: str,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """返回 (payload, skip_reason)。payload 含 index, avm_bgr, yuyan_bgr, source, draw_item_dir。"""
    draw_item = os.path.join(args.draw_image_dir, str(tag_id), ts)
    index_path = os.path.join(draw_item, "index_avm.json")
    avm_draw = os.path.join(draw_item, "avm.jpg")

    if os.path.isfile(index_path) and os.path.isfile(avm_draw):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        avm_bgr = cv2.imread(avm_draw)
        if avm_bgr is None:
            return None, "draw_avm_read_failed"
        yuyan_bgr = None
        if args.yuyan:
            ypath = os.path.join(draw_item, "yuyan_draw.jpg")
            if not os.path.isfile(ypath):
                yc = index.get("yuyan_camera")
                if yc:
                    alt = os.path.join(draw_item, f"{yc}.jpg")
                    if os.path.isfile(alt):
                        ypath = alt
            if os.path.isfile(ypath):
                yuyan_bgr = cv2.imread(ypath)
        return {
            "index": index,
            "avm_bgr": avm_bgr,
            "yuyan_bgr": yuyan_bgr,
            "source": "draw_image",
            "draw_item_dir": draw_item,
        }, ""

    avm_path = os.path.join(args.data_root, "images", f"{case_id}.jpg")
    if not os.path.isfile(avm_path):
        return None, "missing_avm_image"
    avm_bgr = cv2.imread(avm_path)
    if avm_bgr is None:
        return None, "avm_image_read_failed"

    crop_bgr = None
    crop_path = os.path.join(args.data_root, "crop", f"{case_id}.jpg")
    if os.path.isfile(crop_path):
        crop_bgr = cv2.imread(crop_path)

    centroids_map = getattr(args, "centroids_map", {}) or {}
    index, centroid_skip = _build_index_from_data_root(
        case_id,
        avm_bgr,
        crop_bgr,
        centroids_map,
        crop_match_min_score=args.crop_match_min_score,
    )
    source = "all_data_v3"

    if index is None and getattr(args, "fallback_read_data", False):
        index = _build_index_from_read_data(
            tag_id,
            ts,
            read_data_dir=args.read_data_dir,
            generate_dir=args.generate_dir,
            chaosheng_pixel_radius=args.chaosheng_pixel_radius,
            ignore_fs_types=args.ignore_fs_types,
            scan_generate=args.scan_generate_dir,
        )
        if index is not None:
            source = "all_data_v3+read_data_fallback"
            index.setdefault("centroid_source", "read_data")

    if index is None:
        return None, centroid_skip or "no_centroid"

    yuyan_bgr = None
    if args.yuyan:
        ypath = os.path.join(args.data_root, "yuyan", f"{case_id}.jpg")
        if os.path.isfile(ypath):
            yuyan_bgr = cv2.imread(ypath)

    return {
        "index": index,
        "avm_bgr": avm_bgr,
        "yuyan_bgr": yuyan_bgr,
        "source": source,
        "draw_item_dir": draw_item if os.path.isdir(draw_item) else "",
    }, ""


def diagnose_one_case(
    tag_id: int,
    ts: str,
    case_id: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t0 = time.time()
    row: Dict[str, Any] = {
        "case_id": case_id,
        "tag_id": tag_id,
        "timestamp_us": ts,
    }
    payload, skip = _resolve_inputs(tag_id, ts, case_id, args)
    if skip:
        row["skip_reason"] = skip
        row["latency_s"] = round(time.time() - t0, 2)
        return row

    index = payload["index"]
    ctx = dict(index)
    ctx["vlm_yuyan_image_included"] = False

    image_list = OrderedDict()
    image_list["avm"] = cv2.cvtColor(payload["avm_bgr"], cv2.COLOR_BGR2RGB)
    if args.yuyan and payload.get("yuyan_bgr") is not None:
        image_list["yuyan_fisheye"] = cv2.cvtColor(payload["yuyan_bgr"], cv2.COLOR_BGR2RGB)
        ctx["vlm_yuyan_image_included"] = True
        if not ctx.get("yuyan_camera"):
            y_cam, y_dir = avm_positions_to_yuyan_camera(index.get("avm") or [])
            ctx["yuyan_camera"] = y_cam
            ctx["yuyan_direction"] = y_dir
            ctx["yuyan_camera_label"] = YUYAN_CAMERA_LABEL_ZH.get(y_cam or "", y_cam or "")

    draw_item_dir = payload.get("draw_item_dir") or ""
    result_fs_car = _fs_car_misdetects(draw_item_dir) if draw_item_dir else []

    analysis_result: Dict[str, Any] = {"positions": []}
    if len(index.get("avm") or []) > 0:
        prompt = prompt_gen(ctx, args.prompt_config)
        if args.debug_thinking:
            prompt += (
                "\n\n#### ⚠️ 调试模式\n"
                "请先**详细输出你对每个检测点的完整分析推理过程**，"
                "然后再输出最终的JSON结果。"
            )
        analysis_result = analyze_scenario_from_images(
            image_list, prompt, args.model
        )
        if "error" in analysis_result:
            row["api_error"] = analysis_result["error"]
            row["source"] = payload["source"]
            row["latency_s"] = round(time.time() - t0, 2)
            return row
    else:
        row["skip_reason"] = "empty_avm_index"

    fs_others = analysis_result.get("positions") or []
    result = {"fs_others": fs_others, "fs_car": result_fs_car}
    misdetected = bool(result_fs_car or fs_others)

    row.update(
        {
            "misdetected": misdetected,
            "fs_others": fs_others,
            "fs_car": result_fs_car,
            "source": payload["source"],
            "n_avm_points": len(index.get("avm") or []),
            "centroid_source": index.get("centroid_source"),
        }
    )

    sub = "misdetected" if misdetected else "normal"
    save_dir = os.path.join(args.output_dir, sub, str(tag_id), ts)
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, "analysis_result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    cv2.imwrite(os.path.join(save_dir, "avm.jpg"), payload["avm_bgr"])
    if payload.get("yuyan_bgr") is not None:
        cv2.imwrite(os.path.join(save_dir, "yuyan.jpg"), payload["yuyan_bgr"])
    row["saved_dir"] = save_dir

    row["latency_s"] = round(time.time() - t0, 2)
    return row


def setup_logging() -> str:
    now = datetime.now()
    log_dir = os.path.join(config.LOG_DIR, now.strftime("%m%d"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"val_misdetect_{now.strftime('%Y%m%d_%H%M%S')}.log"
    )
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    logger.info("日志: %s", log_file)
    return log_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="val 集超声误检诊断（pipeline 同款 VLM，不发飞书）"
    )
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--centroids-manifest",
        default="",
        help=f"超声质心清单，默认 <data-root>/{DEFAULT_CENTROIDS_MANIFEST}",
    )
    parser.add_argument(
        "--crop-match-min-score",
        type=float,
        default=CROP_MATCH_MIN_SCORE,
        help="crop 模板匹配最低置信度（默认 0.5）",
    )
    parser.add_argument(
        "--fallback-read-data",
        action="store_true",
        help="manifest/模板匹配均失败时，回退 read_data+generate 推算 index",
    )
    parser.add_argument("--read-data-dir", default=config.READ_DATA_DIR)
    parser.add_argument("--draw-image-dir", default=config.DRAW_IMAGE_DIR)
    parser.add_argument("--generate-dir", default=config.GENERATE_DIR)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUT,
        help="结果根目录（misdetected/、normal/、predictions.jsonl）",
    )
    parser.add_argument(
        "-o",
        "--predictions-jsonl",
        default="",
        help="汇总 jsonl（默认 <output-dir>/predictions.jsonl）",
    )
    parser.add_argument("--model", nargs="+", default=["auto"])
    parser.add_argument("--prompt-config", default="chaosheng_wujian_avm")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--chaosheng-pixel-radius",
        type=int,
        default=30,
        help="--fallback-read-data 时与 Step5 一致的关联半径",
    )
    parser.add_argument("--ignore-fs-types", nargs="*", default=[])
    parser.add_argument(
        "--scan-generate-dir",
        action="store_true",
        help="不依赖 DR 元数据，扫描 generate 下全部 AVM",
    )
    parser.add_argument("--debug-thinking", action="store_true")
    parser.add_argument(
        "--no-yuyan",
        dest="yuyan",
        action="store_false",
        help="仅 AVM 单图诊断",
    )
    parser.set_defaults(yuyan=True)
    args = parser.parse_args()

    manifest_path = args.centroids_manifest or os.path.join(
        args.data_root, DEFAULT_CENTROIDS_MANIFEST
    )
    args.centroids_map = load_chaosheng_centroids_manifest(manifest_path)

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(
        "centroids manifest: %s (%s 条)",
        manifest_path,
        len(args.centroids_map),
    )
    pred_path = (
        args.predictions_jsonl
        or os.path.join(args.output_dir, "predictions.jsonl")
    )

    cases = load_unique_cases_from_val_jsonl(args.val_jsonl)
    if args.limit > 0:
        cases = cases[: args.limit]

    done = _load_done_case_ids(Path(pred_path)) if args.resume else set()
    total = len(cases)
    logger.info(
        "val=%s unique_cases=%s workers=%s model=%s style=%s",
        args.val_jsonl,
        total,
        args.workers,
        args.model,
        config.VLM_API_STYLE,
    )
    logger.info("输出: %s", pred_path)

    stats = {
        "processed": 0,
        "misdetected": 0,
        "normal": 0,
        "skipped": 0,
        "api_error": 0,
        "resume_skipped": 0,
    }

    mode = "a" if args.resume and os.path.isfile(pred_path) else "w"
    pending = [
        (tag, ts, cid)
        for tag, ts, cid in cases
        if cid not in done
    ]
    stats["resume_skipped"] = total - len(pending)

    def _run_one(item: Tuple[int, str, str]) -> Dict[str, Any]:
        tag, ts, cid = item
        return diagnose_one_case(tag, ts, cid, args)

    with open(pred_path, mode, encoding="utf-8") as out_f:
        if args.workers <= 1:
            for i, item in enumerate(pending, 1):
                row = _run_one(item)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                _tally(stats, row)
                if i % args.log_every == 0 or row.get("api_error"):
                    logger.info(
                        "[%s/%s] %s mis=%s skip=%s err=%s %.1fs",
                        i,
                        len(pending),
                        row["case_id"],
                        row.get("misdetected"),
                        row.get("skip_reason"),
                        bool(row.get("api_error")),
                        row.get("latency_s", 0),
                    )
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_run_one, it): it for it in pending}
                for i, fut in enumerate(as_completed(futures), 1):
                    row = fut.result()
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    _tally(stats, row)
                    if i % args.log_every == 0 or row.get("api_error"):
                        logger.info(
                            "[%s/%s] %s mis=%s",
                            i,
                            len(pending),
                            row["case_id"],
                            row.get("misdetected"),
                        )

    summary = {
        "val_jsonl": args.val_jsonl,
        "data_root": args.data_root,
        "predictions_jsonl": pred_path,
        "unique_cases": total,
        "pending": len(pending),
        **stats,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = pred_path.replace(".jsonl", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("完成 summary=%s", summary_path)
    logger.info(
        "误检=%s 正常=%s 跳过=%s API失败=%s",
        stats["misdetected"],
        stats["normal"],
        stats["skipped"],
        stats["api_error"],
    )
    return 0 if stats["api_error"] == 0 else 1


def _tally(stats: Dict[str, int], row: Dict[str, Any]) -> None:
    stats["processed"] += 1
    if row.get("api_error"):
        stats["api_error"] += 1
    elif row.get("skip_reason"):
        stats["skipped"] += 1
    elif row.get("misdetected"):
        stats["misdetected"] += 1
    else:
        stats["normal"] += 1


if __name__ == "__main__":
    raise SystemExit(main())
