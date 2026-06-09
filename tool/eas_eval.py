#!/usr/bin/env python3
"""EAS 微调模型：val 三分类评测（实体 / 几何 / 类型）。超声误检诊断见 tool/diagnose_val_dataset.py。

python tool/eas_eval.py --eval --limit 10
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as _cfg  # noqa: E402

EAS_BASE = _cfg.EAS_BASE_URL
DEFAULT_TOKEN = _cfg.EAS_TOKEN
DEFAULT_DATA_ROOT = "/mnt/public-data/user/ziroujiang/all_data_v3"
DEFAULT_VAL_JSONL = f"{DEFAULT_DATA_ROOT}/val_dataset_v5.jsonl"


def _eas_auth_headers(token: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": (token or "").strip(),
    }


from prompts_engine.context.object_type_catalog import (  # noqa: E402
    DEPRECATED_OBJECT_TYPE_ALIASES,
    OBJECT_TYPE_ORDER,
    coerce_legacy_object_type,
    normalize_object_type_label,
)

DEFAULT_SAMPLE: Dict[str, Any] = {
    "messages": [
        {
            "role": "system",
            "content": (
                "你是泊车环视场景的超声/视觉联合诊断模型，须结合多图作答。图例：红标=超声地面障碍物"
                "（点、短线或闭合多边形），为分析对象，表示超声在地面上的感知结果。绿线=邻车检测框"
                "投影到地面的多边形，表示邻车可能占用的地面区域。黄线=相机障碍在AVM上的投影轮廓，"
                "用于在鸟瞰中对齐真实可见障碍；判断时对齐黄线与真实障碍本体，比较红标与真实障碍的"
                "关系，勿将红标与黄线本身当作一对匹配目标。中心黑矩形=自车（上为车头、下为车尾），"
                "正在倒车入库。车位中心白箭头=预计倒车方向；若无箭头，默认沿车位中轴线直线倒车。"
                "白矩形框=仅遮挡车牌，与障碍物/标线无关，分析时完全忽略。AVM由鱼眼展开拼接：红标"
                "仅有地面投影、无高度语义；离地越高常渐淡/半透明或与背景融合，属成像与拼接特性，"
                "不等于该处无实物。须结合鱼眼透视理解障碍远近、立面与地面接触，区分竖直方向透视"
                "表现与地面接触位置，避免仅凭AVM上半部发虚误判空间关系。输入按顺序三张：①AVM"
                "鸟瞰：以红标为准；高处虚化不得单独作为无实体依据。②以超声障碍质心为中心的局部"
                "crop，用于聚焦红标。③与AVM主方位一致的单路鱼眼：绿/黄与AVM语义一致，图中不画"
                "红标，红标仍以AVM为准；作透视与尺度参考，减轻仅凭鸟瞰在远近、实体尺度与类型上的"
                "不确定；禁止在鱼眼与AVM之间做像素级距离换算或强行点配对。回答必须严格遵守用户"
                "给出的任务与可选项；只输出要求的标签或词，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                "<image>任务：实体存在性判定。请判断红色超声高亮附近是否存在真实障碍。"
                "可选项：yes, no。"
            ),
        },
        {"role": "assistant", "content": "yes"},
    ],
    "images": [
        "images/119719614_1641037339600000.jpg",
        "crop/119719614_1641037339600000.jpg",
        "yuyan/119719614_1641037339600000.jpg",
    ],
}


def _b64_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _case_id_from_sample(sample: Dict[str, Any]) -> str:
    img0 = sample["images"][0]
    return Path(img0).stem


def _task_from_user(user_content: str) -> str:
    if "实体存在性" in user_content:
        return "entity_existence"
    if "几何一致性" in user_content:
        return "geometry_relation"
    if "障碍物类型" in user_content:
        return "object_type"
    return "unknown"


def _label_from_sample(sample: Dict[str, Any]) -> str:
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            return str(msg["content"]).strip()
    return ""


def _normalize_pred(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # 只取第一行、去掉常见包裹
    text = text.splitlines()[0].strip()
    text = re.sub(r"^[`\"']+|[`\"']+$", "", text)
    text = text.strip().lower()
    n = normalize_object_type_label(text)
    if n in DEPRECATED_OBJECT_TYPE_ALIASES:
        return ""
    if n in OBJECT_TYPE_ORDER:
        return n
    return text


def _build_openai_messages(sample: Dict[str, Any], data_root: Path) -> List[Dict[str, Any]]:
    image_paths = [data_root / p for p in sample["images"]]
    for p in image_paths:
        if not p.is_file():
            raise FileNotFoundError(f"图片不存在: {p}")

    user_text = ""
    system_text = ""
    for msg in sample["messages"]:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]

    user_text_plain = user_text.replace("<image>", "", 1).strip()

    user_content: List[Dict[str, Any]] = []
    for p in image_paths:
        user_content.append(
            {"type": "image_url", "image_url": {"url": _b64_image(p)}}
        )
    user_content.append({"type": "text", "text": user_text_plain})

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]


def _prompt_from_sample(sample: Dict[str, Any]) -> str:
    parts: List[str] = []
    for msg in sample["messages"]:
        if msg["role"] == "system":
            parts.append(str(msg["content"]))
        elif msg["role"] == "user":
            parts.append(str(msg["content"]).replace("<image>", "", 1).strip())
    return "\n\n".join(parts)


def _load_images_for_sample(
    sample: Dict[str, Any], data_root: Path
) -> "OrderedDict[str, Any]":
    from collections import OrderedDict

    if cv2 is None:
        raise RuntimeError("需要 opencv-python (cv2) 加载图片")
    image_list: OrderedDict[str, Any] = OrderedDict()
    for rel in sample["images"]:
        p = data_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"图片不存在: {p}")
        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            raise FileNotFoundError(f"无法读取: {p}")
        key = rel.split("/")[0] if "/" in rel else rel
        image_list[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image_list


def _infer_one_pipeline(
    sample: Dict[str, Any],
    data_root: Path,
    model: str,
) -> Tuple[str, Optional[int], str]:
    """使用 pipeline 同款 VLM_API（config / .env），不发飞书评论。"""
    from vlm.VLM_API import call_qwen_model_with_images

    image_list = _load_images_for_sample(sample, data_root)
    question = _prompt_from_sample(sample)
    pred = call_qwen_model_with_images(image_list, question, model)
    if isinstance(pred, str) and pred.startswith("调用失败"):
        return "", None, pred
    return (pred or "").strip(), 200, ""


def _infer_one(
    sample: Dict[str, Any],
    data_root: Path,
    *,
    url: str,
    headers: Dict[str, str],
    max_tokens: int,
    timeout: int,
    verbose: bool = False,
) -> Tuple[str, Optional[int], str]:
    """返回 (prediction, http_status, error_msg)。"""
    t0 = time.perf_counter()
    if verbose:
        cid = _case_id_from_sample(sample)
        print(f"  [{cid}] 编码 3 张图片为 base64...", flush=True)
    messages = _build_openai_messages(sample, data_root)
    payload = {
        "model": _cfg.EAS_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if verbose:
        payload_bytes = len(json.dumps(payload, ensure_ascii=False))
        print(
            f"  [{cid}] 请求体约 {payload_bytes / 1024 / 1024:.1f} MB，"
            f"提交 EAS 推理（单条常见 20~90s，超时 {timeout}s）...",
            flush=True,
        )
    # connect 超时与读超时分离，避免连不上时傻等满 timeout
    req_timeout: Any = (min(60, timeout), timeout)
    try:
        t_post = time.perf_counter()
        resp = requests.post(
            url, json=payload, headers=headers, timeout=req_timeout
        )
        if verbose:
            print(
                f"  [{cid}] HTTP {resp.status_code}，"
                f"推理+传输耗时 {time.perf_counter() - t_post:.1f}s，"
                f"总计 {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    except requests.exceptions.RequestException as e:
        if verbose:
            print(
                f"  [{cid}] 请求失败（已等待 {time.perf_counter() - t0:.1f}s）: {e}",
                flush=True,
            )
        return "", None, str(e)

    if resp.status_code != 200:
        detail = (resp.text or "").strip()[:500]
        if not detail:
            detail = f"HTTP {resp.status_code}"
        return "", resp.status_code, detail

    try:
        body = resp.json()
        pred = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return "", resp.status_code, f"parse_error: {e}"
    return pred.strip(), resp.status_code, ""


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
                cid = row.get("case_id")
                if cid and not row.get("error"):
                    done.add(str(cid))
            except json.JSONDecodeError:
                continue
    return done


def run_val_eval(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    val_path = Path(args.val_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = Path(args.output) if args.output else out_dir / f"eval_val_{ts}.jsonl"
    summary_path = out_jsonl.with_suffix(".summary.json")

    use_pipeline = getattr(args, "pipeline_vlm", False)
    eas_base = (args.eas_base or EAS_BASE).rstrip("/")
    url = f"{eas_base}/v1/chat/completions"
    token = args.token or DEFAULT_TOKEN
    headers = _eas_auth_headers(token)
    if use_pipeline:

        print(
            f"VLM 后端: pipeline VLM_API style={_cfg.VLM_API_STYLE} "
            f"model={args.model}",
            flush=True,
        )

    samples: List[Dict[str, Any]] = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[: args.limit]

    done_ids = _load_done_case_ids(out_jsonl) if args.resume else set()
    total = len(samples)
    print(f"评测集: {val_path} ({total} 条)", flush=True)
    print(f"输出: {out_jsonl}", flush=True)
    if done_ids:
        print(f"resume: 已有 {len(done_ids)} 条成功结果，将跳过", flush=True)

    stats = {
        "total": 0,
        "ok": 0,
        "match": 0,
        "error": 0,
        "skipped": 0,
        "by_task": {},
    }

    mode = "a" if args.resume and out_jsonl.is_file() else "w"
    with open(out_jsonl, mode, encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples, start=1):
            case_id = _case_id_from_sample(sample)
            if case_id in done_ids:
                stats["skipped"] += 1
                continue

            user_text = next(
                m["content"] for m in sample["messages"] if m["role"] == "user"
            )
            task = _task_from_user(user_text)
            label = _label_from_sample(sample)

            t0 = time.time()
            try:
                if use_pipeline:
                    pred_raw, status, err = _infer_one_pipeline(
                        sample, data_root, args.model
                    )
                else:
                    pred_raw, status, err = _infer_one(
                        sample,
                        data_root,
                        url=url,
                        headers=headers,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
            except FileNotFoundError as e:
                pred_raw, status, err = "", None, str(e)

            elapsed = round(time.time() - t0, 2)
            pred_norm = _normalize_pred(pred_raw) if pred_raw else ""
            label_norm = _normalize_pred(label)
            match = bool(pred_norm and label_norm and pred_norm == label_norm)

            row = {
                "case_id": case_id,
                "task": task,
                "label": label,
                "prediction": pred_raw,
                "prediction_norm": pred_norm,
                "match": match,
                "http_status": status,
                "error": err,
                "latency_s": elapsed,
                "images": sample.get("images", []),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            stats["total"] += 1
            if err:
                stats["error"] += 1
            else:
                stats["ok"] += 1
                if match:
                    stats["match"] += 1

            bt = stats["by_task"].setdefault(
                task, {"total": 0, "ok": 0, "match": 0, "error": 0}
            )
            bt["total"] += 1
            if err:
                bt["error"] += 1
            elif match:
                bt["match"] += 1
                bt["ok"] += 1
            else:
                bt["ok"] += 1

            if idx % args.log_every == 0 or err or not match:
                mark = "OK" if match else ("ERR" if err else "MISMATCH")
                print(
                    f"[{idx}/{total}] {case_id} {task} {mark} "
                    f"pred={pred_norm!r} label={label_norm!r} {elapsed}s",
                    flush=True,
                )

    acc = stats["match"] / stats["ok"] if stats["ok"] else 0.0
    import config as _cfg

    summary = {
        "val_jsonl": str(val_path),
        "data_root": str(data_root),
        "output_jsonl": str(out_jsonl),
        "backend": "pipeline_vlm_api" if use_pipeline else "eas",
        "vlm_api_style": _cfg.VLM_API_STYLE if use_pipeline else "",
        "vlm_model": args.model if use_pipeline else _cfg.EAS_MODEL,
        "eas_base": eas_base,
        "total_samples": total,
        "processed": stats["total"],
        "skipped_resume": stats["skipped"],
        "ok": stats["ok"],
        "error": stats["error"],
        "match": stats["match"],
        "accuracy_on_ok": round(acc, 4),
        "by_task": stats["by_task"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n完成。预测: {out_jsonl}", flush=True)
    print(f"汇总: {summary_path}", flush=True)
    print(
        f"成功 {stats['ok']}/{stats['total']}，"
        f"一致 {stats['match']}，准确率(成功样本) {acc:.2%}，失败 {stats['error']}",
        flush=True,
    )
    return 0 if stats["error"] == 0 else 1


def run_single(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    sample = DEFAULT_SAMPLE
    expected = _label_from_sample(sample)

    eas_base = (args.eas_base or EAS_BASE).rstrip("/")
    url = f"{eas_base}/v1/chat/completions"
    token = args.token or DEFAULT_TOKEN
    headers = _eas_auth_headers(token)

    print(f"数据目录: {data_root}", flush=True)
    print(f"标注期望: {expected!r}", flush=True)
    print("请求中...", url, flush=True)
    print(
        f"（默认超时 {args.timeout}s：先本地编码 3 图，再上传并等 GPU 推理，期间无输出属正常）",
        flush=True,
    )

    pred, status, err = _infer_one(
        sample,
        data_root,
        url=url,
        headers=headers,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        verbose=True,
    )
    if err or status != 200:
        print(f"失败 status={status} err={err or 'unknown'}", file=sys.stderr)
        if status == 404:
            print(
                f"提示: EAS 路径不存在，请确认服务已部署且 --eas-base 正确（当前 {eas_base}）",
                file=sys.stderr,
            )
        elif status == 401:
            print(
                "提示: 401 表示 Token 与该服务不匹配。请在 PAI-EAS 控制台打开 "
                f"「{eas_base.split('/api/predict/')[-1]}」→ 调用信息，"
                "复制 Authorization Token，执行:\n"
                "  export EAS_TOKEN='<新 token>'\n"
                "  python tool/eas_eval.py --token \"$EAS_TOKEN\"",
                file=sys.stderr,
            )
        return 1

    print(f"status: {status}", flush=True)
    print(f"模型输出: {pred!r}", flush=True)
    print(f"期望: {expected!r}", flush=True)
    if _normalize_pred(pred) == _normalize_pred(expected):
        print("与标注一致")
    else:
        print("与标注不一致", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="EAS 微调模型推理 / val 评测")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--eas-base",
        default="",
        help=f"EAS 服务根 URL（默认取 config.EAS_BASE_URL）",
    )
    parser.add_argument(
        "--token",
        default="",
        help="EAS Authorization（默认取 config.EAS_TOKEN / 环境变量 EAS_TOKEN）",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=_cfg.EAS_TIMEOUT)

    parser.add_argument(
        "--eval",
        action="store_true",
        help="评测 val_dataset.jsonl 并写入 tool/（见 --output-dir）",
    )
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL)
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR),
        help="评测结果目录，默认 tool/",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="预测 jsonl 路径；默认 <output-dir>/eval_val_<时间戳>.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅跑前 N 条（0=全部）")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已成功样本")
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument(
        "--single",
        action="store_true",
        help="只跑内置单条样本（默认不加 --eval 时等价）",
    )
    parser.add_argument(
        "--pipeline-vlm",
        action="store_true",
        help="使用 pipeline 同款 VLM_API（.env），评测 val；不发飞书评论",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="--pipeline-vlm 时传给 VLM_API（默认 auto → gemini-3.1-pro-preview 等）",
    )
    args = parser.parse_args()

    if args.eval:
        return run_val_eval(args)
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
