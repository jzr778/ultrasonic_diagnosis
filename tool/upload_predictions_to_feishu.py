#!/usr/bin/env python3
"""
读取预测结果 CSV，按 case_id 聚合多条预测/真实标签，
将 images/crop/yuyan 三张图 + 预测值 + 真实标签 写入飞书电子表格。

CSV 格式（有表头）::

    id,预测,真实标签
    119407453_1772851505200000,yes,yes
    119407453_1772851505200000,aligned,aligned
    119407453_1772851505200000,wheel_stop,wheel_stop

同一 id 可以有多行（对应不同任务的预测），脚本自动按 id 聚合，
在飞书表中只占一行，多条预测/标签以换行分隔。

表格列约定（可通过 --col-* 参数调整）::

    A: case_id
    B: AVM 原图 (images/)
    C: crop 图  (crop/)
    D: 鱼眼图   (yuyan/)
    E: 预测值（多条换行）
    F: 真实标签（多条换行）

用法::

    python tool/upload_predictions_to_feishu.py \\
        --csv /path/to/predictions.csv \\
        --data-dir /mnt/public-data/user/ziroujiang/raw_data_01-04 \\
        --spreadsheet-url "https://rqk9rsooi4.feishu.cn/sheets/XXXX"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import requests

OPEN_API = "https://open.feishu.cn/open-apis"
DEFAULT_REQUEST_TIMEOUT = 15.0
IMAGE_UPLOAD_TIMEOUT = 120

_DEFAULT_FEISHU_APP_ID = "cli_a6e0444aedfbd00b"
_DEFAULT_FEISHU_APP_SECRET = "8W1Art9TRWrV50C7QgITwbYbMMqLKI5x"

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff",
}
_MAX_RANGES_PER_BATCH = 200


# ── 飞书认证 & 工具函数（与 sync_raw_images_to_feishu_sheet.py 一致）────────


def spreadsheet_token_from_url(url: str) -> str:
    m = re.search(r"/sheets/([A-Za-z0-9]+)", url.strip())
    if not m:
        raise ValueError("无法从 URL 解析 spreadsheet token")
    return m.group(1)


def load_app_credentials() -> Tuple[str, str]:
    app_id = (
        os.environ.get("FEISHU_APP_ID", "").strip()
        or os.environ.get("LARK_APP_ID", "").strip()
        or _DEFAULT_FEISHU_APP_ID
    )
    app_secret = (
        os.environ.get("FEISHU_APP_SECRET", "").strip()
        or os.environ.get("LARK_APP_SECRET", "").strip()
        or _DEFAULT_FEISHU_APP_SECRET
    )
    return app_id, app_secret


def get_tenant_access_token(
    app_id: str, app_secret: str, timeout: float = DEFAULT_REQUEST_TIMEOUT
) -> str:
    r = requests.post(
        f"{OPEN_API}/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"获取 tenant_access_token HTTP {r.status_code}: {r.text[:2000]}")
    data = r.json()
    token = data.get("tenant_access_token")
    if not token:
        raise RuntimeError(f"响应中无 tenant_access_token: {data}")
    return str(token)


def get_wiki_node(tenant: str, wiki_token: str, timeout: float) -> Dict[str, Any]:
    resp = requests.get(
        f"{OPEN_API}/wiki/v2/spaces/get_node",
        headers={"Authorization": f"Bearer {tenant}"},
        params={"token": wiki_token},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"get_wiki_node HTTP {resp.status_code}: {resp.text[:2000]}")
    j = resp.json()
    if j.get("code") not in (None, 0):
        raise RuntimeError(f"get_wiki_node 业务错误: {j}")
    node = (j.get("data") or {}).get("node")
    if not node:
        raise RuntimeError(f"get_wiki_node 未返回 node: {j}")
    return node


def spreadsheet_token_from_wiki_node(node: Dict[str, Any]) -> str:
    node_type = str(node.get("obj_type") or node.get("node_type") or "")
    if node_type.lower() not in ("spreadsheet", "sheet"):
        raise RuntimeError(f"Wiki 节点不是 spreadsheet: {node_type!r}")
    t = node.get("obj_token") or node.get("node_token") or node.get("token")
    if not t:
        raise RuntimeError(f"Wiki node 无 token: {node!r}")
    return str(t)


def fetch_sheets(
    spreadsheet_token: str, headers: dict, timeout: float
) -> List[Dict[str, Any]]:
    url_v3 = f"{OPEN_API}/sheets/v3/spreadsheets/{spreadsheet_token}/sheets/query"
    rv3 = requests.get(url_v3, headers=headers, timeout=timeout)
    if rv3.status_code == 200:
        j = rv3.json()
        if j.get("code") in (None, 0):
            sheets = (j.get("data") or {}).get("sheets") or []
            if sheets:
                return sheets
    url_v2 = f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}"
    rv2 = requests.get(url_v2, headers=headers, timeout=timeout)
    if rv2.status_code != 200:
        raise RuntimeError(f"sheets v3/v2 均失败")
    j2 = rv2.json()
    if j2.get("code") not in (None, 0):
        raise RuntimeError(f"sheets v2 业务错误: {j2}")
    raw = (j2.get("data") or {}).get("sheets") or []
    return [
        {
            "sheet_id": s.get("sheet_id") or s.get("sheetId"),
            "title": s.get("title", ""),
            "hidden": bool(s.get("hidden", False)),
            "resource_type": s.get("resource_type", "sheet"),
        }
        for s in raw
    ]


def pick_sheet_id(sheets: list, index: int, title: Optional[str]) -> str:
    visible = [
        s
        for s in sheets
        if not s.get("hidden") and s.get("resource_type", "sheet") == "sheet"
    ]
    if not visible:
        raise RuntimeError("未找到可见 sheet")
    if title:
        for s in visible:
            if s.get("title") == title:
                return str(s["sheet_id"])
        raise RuntimeError(f"未找到标题 {title!r}")
    if index < 0 or index >= len(visible):
        raise RuntimeError(f"sheet 索引越界: {index}")
    return str(visible[index]["sheet_id"])


def _grid_row_count(sheets: list, sheet_id: str) -> int:
    for s in sheets:
        if str(s.get("sheet_id")) == str(sheet_id):
            gp = s.get("grid_properties") or {}
            rc = gp.get("row_count")
            if rc is not None:
                return max(int(rc), 2)
    return 20000


def _cell_to_plain_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, dict):
        for k in ("text", "cellText", "stringValue"):
            v = cell.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return str(cell).strip()


def normalize_case_id(raw: Any) -> str:
    s = _cell_to_plain_text(raw)
    if not s:
        return ""
    lower = s.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".heic", ".tif", ".tiff"):
        if lower.endswith(ext):
            return s[: -len(ext)].strip()
    return s


def values_batch_update(
    spreadsheet_token: str,
    headers: dict,
    value_ranges: List[Dict[str, Any]],
    timeout: float = IMAGE_UPLOAD_TIMEOUT,
) -> None:
    for i in range(0, len(value_ranges), _MAX_RANGES_PER_BATCH):
        chunk = value_ranges[i : i + _MAX_RANGES_PER_BATCH]
        r = requests.post(
            f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_update",
            headers=headers,
            json={"valueRanges": chunk},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(f"values_batch_update 失败: {data}")


def write_image_cell(
    spreadsheet_token: str,
    headers: dict,
    sheet_id: str,
    cell: str,
    image_path: str,
    timeout: float = IMAGE_UPLOAD_TIMEOUT,
) -> None:
    with open(image_path, "rb") as f:
        raw = f.read()
    if not raw:
        raise RuntimeError(f"空图片文件: {image_path!r}")
    name = os.path.basename(image_path)
    if "." not in name:
        name += ".png"
    rng = f"{sheet_id}!{cell}:{cell}"
    url = f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_image"
    body = {"range": rng, "image": list(raw), "name": name}
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"_parse_error": True, "text": r.text[:4000]}
    if r.status_code != 200:
        raise RuntimeError(f"values_image HTTP {r.status_code} cell={cell}: {data}")
    if data.get("code") != 0:
        raise RuntimeError(f"values_image 业务失败 cell={cell}: {data}")


def cell_has_image_or_content(cell: Any) -> bool:
    if cell is None:
        return False
    if isinstance(cell, str):
        return bool(cell.strip())
    if isinstance(cell, dict):
        if cell.get("type") == "embed-image" and cell.get("fileToken"):
            return True
        return bool(_cell_to_plain_text(cell))
    return True


def resolve_image(directory: str, case_id: str) -> Optional[str]:
    """在 directory 下找到与 case_id 同 stem 的图片文件。"""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = os.path.join(directory, case_id + ext)
        if os.path.isfile(p):
            return p
    return None


# ── CSV 解析 & 聚合 ──────────────────────────────────────────────────────


def read_and_group_csv(csv_path: str) -> OrderedDict:
    """读取 CSV，按 id 聚合预测和真实标签。

    返回 OrderedDict: case_id -> {"predictions": [...], "labels": [...]}
    保持首次出现顺序。
    """
    groups: OrderedDict = OrderedDict()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        col_id = fieldnames[0] if fieldnames else "id"
        col_pred = fieldnames[1] if len(fieldnames) > 1 else "预测"
        col_label = fieldnames[2] if len(fieldnames) > 2 else "真实标签"
        for row in reader:
            cid = (row.get(col_id) or "").strip()
            if not cid:
                continue
            lower = cid.lower()
            for ext in (".jpg", ".jpeg", ".png"):
                if lower.endswith(ext):
                    cid = cid[: -len(ext)].strip()
                    break
            pred = (row.get(col_pred) or "").strip()
            label = (row.get(col_label) or "").strip()
            if cid not in groups:
                groups[cid] = {"predictions": [], "labels": []}
            if pred:
                groups[cid]["predictions"].append(pred)
            if label:
                groups[cid]["labels"].append(label)
    return groups


# ── 读已有行 ──────────────────────────────────────────────────────────────


def read_column_a(
    spreadsheet_token: str,
    sheet_id: str,
    headers: dict,
    header_rows: int,
    max_row: int,
    timeout: float,
) -> Tuple[Dict[str, int], int]:
    start = header_rows + 1
    if max_row < start:
        return {}, header_rows
    rng = f"{sheet_id}!A{start}:A{max_row}"
    r = requests.get(
        f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_get",
        headers=headers,
        params={"ranges": rng},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"values_batch_get A 失败: {data}")
    vrs = (data.get("data") or {}).get("valueRanges") or []
    if not vrs:
        return {}, header_rows
    values = vrs[0].get("values") or []
    row_by_case: Dict[str, int] = {}
    last = header_rows
    for i, row in enumerate(values):
        row_1 = start + i
        if not row:
            continue
        cid = normalize_case_id(row[0])
        if not cid:
            continue
        last = max(last, row_1)
        if cid not in row_by_case:
            row_by_case[cid] = row_1
    return row_by_case, last


def read_row_range(
    spreadsheet_token: str,
    sheet_id: str,
    headers: dict,
    col_start: str,
    col_end: str,
    start_row: int,
    end_row: int,
    timeout: float,
) -> List[List[Any]]:
    if end_row < start_row:
        return []
    rng = f"{sheet_id}!{col_start}{start_row}:{col_end}{end_row}"
    r = requests.get(
        f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_get",
        headers=headers,
        params={"ranges": rng},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"values_batch_get {col_start}:{col_end} 失败: {data}")
    vrs = (data.get("data") or {}).get("valueRanges") or []
    if not vrs:
        return []
    return vrs[0].get("values") or []


# ── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将预测结果 CSV 连同 images/crop/yuyan 写入飞书电子表格"
    )
    parser.add_argument("--csv", required=True, help="预测结果 CSV 文件路径")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="数据根目录，下含 images/ crop/ yuyan/ 子目录",
    )
    parser.add_argument("--spreadsheet-url", default="", help="电子表格 URL")
    parser.add_argument("--spreadsheet-token", default="", help="直接指定 token")
    parser.add_argument("--wiki-token", default="")
    parser.add_argument("--sheet-index", type=int, default=0)
    parser.add_argument("--sheet-title", default=None)
    parser.add_argument("--header-rows", type=int, default=1)
    parser.add_argument("--start-row", type=int, default=2)
    parser.add_argument(
        "--col-image", default="B", help="AVM 原图列（默认 B）"
    )
    parser.add_argument("--col-crop", default="C", help="crop 列（默认 C）")
    parser.add_argument("--col-yuyan", default="D", help="yuyan 列（默认 D）")
    parser.add_argument(
        "--col-pred", default="E", help="预测值列（默认 E）"
    )
    parser.add_argument(
        "--col-label", default="F", help="真实标签列（默认 F）"
    )
    parser.add_argument(
        "--image-delay",
        type=float,
        default=0.05,
        help="两次图片写入间隔秒数",
    )
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个 case")
    args = parser.parse_args()

    csv_path = args.csv
    data_dir = args.data_dir.rstrip("/")
    images_dir = os.path.join(data_dir, "images")
    crop_dir = os.path.join(data_dir, "crop")
    yuyan_dir = os.path.join(data_dir, "yuyan")

    if not os.path.isfile(csv_path):
        print(f"CSV 文件不存在: {csv_path}", file=sys.stderr)
        return 1
    for d in (images_dir, crop_dir, yuyan_dir):
        if not os.path.isdir(d):
            print(f"目录不存在: {d}", file=sys.stderr)
            return 1

    groups = read_and_group_csv(csv_path)
    if args.limit:
        limited: OrderedDict = OrderedDict()
        for i, (k, v) in enumerate(groups.items()):
            if i >= args.limit:
                break
            limited[k] = v
        groups = limited

    print(f"CSV 读取: {csv_path}")
    print(f"  唯一 case: {len(groups)}")
    total_preds = sum(len(v["predictions"]) for v in groups.values())
    total_labels = sum(len(v["labels"]) for v in groups.values())
    print(f"  预测条数: {total_preds}, 标签条数: {total_labels}")

    missing_img = 0
    for cid in groups:
        if not resolve_image(images_dir, cid):
            missing_img += 1
    if missing_img:
        print(f"  [WARN] {missing_img} 个 case 在 images/ 下无图片")

    # ── 飞书认证 ──
    app_id, app_secret = load_app_credentials()
    req_to = args.request_timeout

    wiki = (args.wiki_token or "").strip() or os.environ.get("FEISHU_WIKI_TOKEN", "").strip()
    if (args.spreadsheet_token or "").strip():
        token_str = args.spreadsheet_token.strip()
    elif wiki:
        tenant = get_tenant_access_token(app_id, app_secret, req_to)
        node = get_wiki_node(tenant, wiki, req_to)
        token_str = spreadsheet_token_from_wiki_node(node)
    elif (args.spreadsheet_url or "").strip():
        token_str = spreadsheet_token_from_url(args.spreadsheet_url)
    else:
        print("需提供 --spreadsheet-url 或 --spreadsheet-token", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"\n[DRY-RUN] spreadsheet_token={token_str}")
        for i, (cid, info) in enumerate(groups.items()):
            if i >= 5:
                print(f"  ... 共 {len(groups)} 条（略）")
                break
            preds = " | ".join(info["predictions"])
            labels = " | ".join(info["labels"])
            img = resolve_image(images_dir, cid)
            print(f"  {cid}: pred=[{preds}] label=[{labels}] img={'OK' if img else 'MISS'}")
        return 0

    tenant = get_tenant_access_token(app_id, app_secret, req_to)
    headers = {
        "Authorization": f"Bearer {tenant}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept-Encoding": "gzip, deflate",
    }

    sheets = fetch_sheets(token_str, headers, req_to)
    sheet_id = pick_sheet_id(sheets, args.sheet_index, args.sheet_title)
    print(f"spreadsheet_token={token_str}, sheet_id={sheet_id}")

    max_scan = _grid_row_count(sheets, sheet_id)
    row_by_case, last_content = read_column_a(
        token_str, sheet_id, headers, args.header_rows, max_scan, req_to
    )
    print(f"已有 {len(row_by_case)} 行 case_id, A 列最后非空行={last_content}")

    case_ids = list(groups.keys())
    next_new = max(last_content + 1, args.start_row)
    assignments: List[Tuple[str, int, bool]] = []
    for cid in case_ids:
        if cid in row_by_case:
            assignments.append((cid, row_by_case[cid], False))
        else:
            assignments.append((cid, next_new, True))
            row_by_case[cid] = next_new
            next_new += 1

    new_a_rows = [(row, cid) for cid, row, is_new in assignments if is_new]
    if new_a_rows:
        vrs = [
            {"range": f"{sheet_id}!A{r}:A{r}", "values": [[cid]]}
            for r, cid in new_a_rows
        ]
        values_batch_update(token_str, headers, vrs, timeout=max(req_to, IMAGE_UPLOAD_TIMEOUT))
        print(f"写入 A 列: {len(new_a_rows)} 个新行")

    # ── 写文本列（预测 + 真实标签）──
    text_vrs: List[Dict[str, Any]] = []
    col_pred = args.col_pred.upper()
    col_label = args.col_label.upper()
    for cid, row, _ in assignments:
        info = groups[cid]
        pred_text = "\n".join(info["predictions"])
        label_text = "\n".join(info["labels"])
        text_vrs.append({
            "range": f"{sheet_id}!{col_pred}{row}:{col_pred}{row}",
            "values": [[pred_text]],
        })
        text_vrs.append({
            "range": f"{sheet_id}!{col_label}{row}:{col_label}{row}",
            "values": [[label_text]],
        })
    if text_vrs:
        values_batch_update(token_str, headers, text_vrs, timeout=max(req_to, IMAGE_UPLOAD_TIMEOUT))
        print(f"写入预测/标签文本: {len(text_vrs)} 格")

    # ── 写图片列（B/C/D，跳过已有内容）──
    img_cols = [
        (args.col_image.upper(), images_dir, "原图"),
        (args.col_crop.upper(), crop_dir, "crop"),
        (args.col_yuyan.upper(), yuyan_dir, "鱼眼"),
    ]
    wrote = skipped = missed = 0
    for idx, (cid, row, _) in enumerate(assignments):
        for col_letter, directory, label in img_cols:
            img_path = resolve_image(directory, cid)
            if not img_path:
                missed += 1
                continue
            cell = f"{col_letter}{row}"
            try:
                write_image_cell(
                    token_str, headers, sheet_id, cell, img_path,
                    timeout=max(req_to, IMAGE_UPLOAD_TIMEOUT),
                )
                wrote += 1
            except RuntimeError as e:
                if "already" in str(e).lower() or "exist" in str(e).lower():
                    skipped += 1
                else:
                    print(f"  [WARN] 写图失败 {cell} {img_path}: {e}", file=sys.stderr)
                    missed += 1
            if args.image_delay > 0:
                time.sleep(args.image_delay)
        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx + 1}/{len(assignments)} ...")

    print(f"\n图片写入: 成功={wrote}, 跳过已有={skipped}, 缺失/失败={missed}")
    print("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
