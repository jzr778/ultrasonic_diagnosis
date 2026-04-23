#!/usr/bin/env python3
"""
将本地 images / crop / yuyan 目录中的图片同步到飞书电子表格。

你提供的链接形态为「电子表格」：
  https://rqk9rsooi4.feishu.cn/sheets/<spreadsheetToken>
若链接为「多维表格」Base（/base/...），本脚本不适用，需改用 bitable 记录 + 附件字段 API。

表格约定（默认第 1 行为表头，不修改）：
  A 列 case_id：与图片主文件名一致但**不含** .jpg 等后缀。
    新 case_id → 在表尾追加一行并写 A；表中已有 case_id → 复用该行。
  B（AVM 原图）/ C（crop）/ D（鱼眼 yuyan）：**按格判断**，该格已有图或文本则**跳过该格**，
    仅对空格写图（可补全历史上只写了部分列的行）。

凭证（优先级从高到低）：
  环境变量 FEISHU_APP_ID / FEISHU_APP_SECRET 或 LARK_APP_ID / LARK_APP_SECRET
  若未设置，则使用脚本内与 yanshou_sheet_reader 一致的默认自建应用凭证。
  注意：代码内写死密钥不适合公开仓库，勿将本文件 push 到公网。

可选：
  FEISHU_WIKI_TOKEN   知识库节点 token；若表格挂在 Wiki 下，可用 --wiki-token 解析出真实 spreadsheet token

应用需开通：电子表格写权限、调用 Wiki 时需 wiki 相关只读权限；
并在目标表格「…」→「添加文档应用」中授权该应用。

参考：
  写入图片 https://open.feishu.cn/document/server-docs/docs/sheets-v3/data-operation/write-images
  Wiki 节点 https://open.feishu.cn/document/server-docs/docs/wiki-v2/space-node/get_node
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OPEN_API = "https://open.feishu.cn/open-apis"
DEFAULT_REQUEST_TIMEOUT = 15
IMAGE_UPLOAD_TIMEOUT = 120

# 飞书自建应用（tenant_access_token）；与 yanshou_sheet_reader_fixed 默认一致。环境变量可覆盖。
_DEFAULT_FEISHU_APP_ID = "cli_a6e0444aedfbd00b"
_DEFAULT_FEISHU_APP_SECRET = "8W1Art9TRWrV50C7QgITwbYbMMqLKI5x"

DEFAULT_SPREADSHEET_URL = (
    "https://rqk9rsooi4.feishu.cn/sheets/RX35sNkX7hjj4ntsDH2cipV1njd"
)
DEFAULT_IMAGES_DIR = "/mnt/public-data/user/ziroujiang/raw_data_01-03/images"
DEFAULT_CROP_DIR = "/mnt/public-data/user/ziroujiang/raw_data_01-03/crop"
DEFAULT_YUYAN_DIR = "/mnt/public-data/user/ziroujiang/raw_data_01-03/yuyan"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff"}
# values_batch_update 单次 valueRanges 条数上限（保守分块，避免触顶）
_MAX_RANGES_PER_BATCH = 200


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spreadsheet_token_from_url(url: str) -> str:
    m = re.search(r"/sheets/([A-Za-z0-9]+)", url.strip())
    if not m:
        raise ValueError(
            "无法从 URL 解析 spreadsheet token，请确认链接形如 .../sheets/<token>"
        )
    return m.group(1)


def load_app_credentials() -> Tuple[str, str]:
    """环境变量 > 脚本内默认（与 yanshou_sheet_reader_fixed 一致）。"""
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
    """POST /auth/v3/tenant_access_token/internal，兼容仅有 tenant_access_token 无 code 的响应。"""
    r = requests.post(
        f"{OPEN_API}/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"获取 tenant_access_token HTTP {r.status_code}: {r.text[:2000]}"
        )
    data = r.json()
    token = data.get("tenant_access_token")
    if token:
        return str(token)
    raise RuntimeError(f"响应中无 tenant_access_token: {data}")


def get_wiki_node(tenant: str, wiki_token: str, timeout: float) -> Dict[str, Any]:
    """GET wiki/v2/spaces/get_node?token=... → data.node。"""
    url = f"{OPEN_API}/wiki/v2/spaces/get_node"
    headers = {"Authorization": f"Bearer {tenant}"}
    resp = requests.get(
        url, headers=headers, params={"token": wiki_token}, timeout=timeout
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"get_wiki_node HTTP {resp.status_code}: {resp.text[:2000]}"
        )
    j = resp.json()
    if j.get("code") not in (None, 0):
        raise RuntimeError(f"get_wiki_node 业务错误: {j}")
    node = (j.get("data") or {}).get("node")
    if not node:
        raise RuntimeError(f"get_wiki_node 未返回 data.node: {j}")
    return node


def spreadsheet_token_from_wiki_node(node: Dict[str, Any]) -> str:
    """从 Wiki node 解析电子表格 obj_token（与 yanshou_sheet_reader 逻辑一致）。"""
    node_type = node.get("obj_type") or node.get("node_type") or node.get("type")
    if str(node_type).lower() not in ("spreadsheet", "sheet"):
        raise RuntimeError(
            f"Wiki 节点类型为 {node_type!r}，需要 spreadsheet/sheet。node={node!r}"
        )
    t = node.get("obj_token") or node.get("node_token") or node.get("token")
    if not t:
        raise RuntimeError(f"无法从 wiki node 提取 obj_token: {node!r}")
    return str(t)


def fetch_sheets(
    spreadsheet_token: str, headers: dict, timeout: float
) -> List[dict[str, Any]]:
    """优先 sheets v3 query；失败或无列表时回退 v2 GET spreadsheets/{{token}}（与 yanshou_sheet_reader 一致）。"""
    url_v3 = (
        f"{OPEN_API}/sheets/v3/spreadsheets/{spreadsheet_token}/sheets/query"
    )
    resp = requests.get(url_v3, headers=headers, timeout=timeout)
    if resp.status_code == 200:
        j = resp.json()
        if j.get("code") in (None, 0):
            sheets = (j.get("data") or {}).get("sheets") or []
            if sheets:
                return sheets

    url_v2 = f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}"
    resp2 = requests.get(url_v2, headers=headers, timeout=timeout)
    if resp2.status_code != 200:
        raise RuntimeError(
            f"sheets v3/v2 均失败: v3={resp.status_code}, v2={resp2.status_code} {resp2.text[:1500]}"
        )
    j2 = resp2.json()
    if j2.get("code") not in (None, 0):
        raise RuntimeError(f"sheets v2 业务错误: {j2}")
    raw = (j2.get("data") or {}).get("sheets") or []
    if not raw:
        raise RuntimeError(f"sheets v2 无 sheets 列表: {j2}")
    normalized: List[dict[str, Any]] = []
    for s in raw:
        sid = s.get("sheet_id") or s.get("sheetId")
        normalized.append(
            {
                "sheet_id": sid,
                "title": s.get("title", ""),
                "hidden": bool(s.get("hidden", False)),
                "resource_type": s.get("resource_type", "sheet"),
            }
        )
    return normalized


def pick_sheet_id(
    sheets: Iterable[dict[str, Any]], index: int, title: Optional[str]
) -> str:
    # 仅使用「电子表格」网格；若只有 bitable 类型子表，需用多维表格 API
    visible = [
        s
        for s in sheets
        if not s.get("hidden") and s.get("resource_type", "sheet") == "sheet"
    ]
    if not visible:
        types = {s.get("resource_type") for s in sheets if not s.get("hidden")}
        raise RuntimeError(
            "未找到 resource_type=sheet 的网格工作表（当前可见类型: "
            f"{types}）。若该文档实为多维表格，请使用 Base/Bitable API 而非本脚本。"
        )
    if title:
        for s in visible:
            if s.get("title") == title:
                return str(s["sheet_id"])
        titles = [s.get("title") for s in visible]
        raise RuntimeError(f"未找到标题为 {title!r} 的工作表，当前有: {titles}")
    if index < 0 or index >= len(visible):
        raise RuntimeError(f"工作表索引 {index} 越界，共 {len(visible)} 个可见表")
    sid = visible[index].get("sheet_id")
    if not sid:
        raise RuntimeError("工作表缺少 sheet_id")
    return str(sid)


def values_batch_update(
    spreadsheet_token: str,
    headers: dict,
    value_ranges: List[dict[str, Any]],
    timeout: float = IMAGE_UPLOAD_TIMEOUT,
) -> None:
    r = requests.post(
        f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_update",
        headers=headers,
        json={"valueRanges": value_ranges},
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
    """向单个单元格写入图片。

    飞书文档中 ``image`` 类型为 **array（字节 0–255 的 JSON 数组）**；仅用 base64 字符串时
    部分租户会返回 HTTP 400，故优先传 ``list(raw)``。超大图若遇网关体积限制再自行压缩。
    """
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
        raise RuntimeError(
            f"values_image HTTP {r.status_code} range={rng!r} file={image_path!r}: {data}"
        )
    if data.get("code") != 0:
        raise RuntimeError(
            f"values_image 业务失败 range={rng!r} file={image_path!r}: {data}"
        )


def list_image_files(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")
    names = []
    for n in os.listdir(images_dir):
        ext = os.path.splitext(n)[1].lower()
        if ext in IMAGE_SUFFIXES:
            names.append(n)
    return sorted(names)


def resolve_same_stem(directory: str, filename: str) -> Optional[str]:
    direct = os.path.join(directory, filename)
    if os.path.isfile(direct):
        return direct
    if not os.path.isdir(directory):
        return None
    stem = os.path.splitext(filename)[0]
    for n in os.listdir(directory):
        if os.path.splitext(n)[0] == stem and os.path.isfile(os.path.join(directory, n)):
            p = os.path.join(directory, n)
            ext = os.path.splitext(n)[1].lower()
            if ext in IMAGE_SUFFIXES:
                return p
    return None


def col_row_to_a1(col: int, row: int) -> str:
    """col: 0=A, 1=B, ...（本脚本仅用到 A–D）"""
    if col < 0 or col > 25:
        raise ValueError("列索引需在 0–25（A–Z）范围内")
    return f"{chr(ord('A') + col)}{row}"


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
    """A 列 case_id：去掉首尾空白与常见图片后缀（与 images 主文件名 stem 对齐）。"""
    s = _cell_to_plain_text(raw)
    if not s:
        return ""
    lower = s.lower()
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tif", ".tiff"):
        if lower.endswith(ext):
            return s[: -len(ext)].strip()
    return s


def _grid_row_count(sheets: List[dict[str, Any]], sheet_id: str) -> int:
    for s in sheets:
        if str(s.get("sheet_id")) == str(sheet_id):
            gp = s.get("grid_properties") or {}
            rc = gp.get("row_count")
            if rc is not None:
                return max(int(rc), 2)
    return 20000


def read_column_a_existing_cases(
    spreadsheet_token: str,
    sheet_id: str,
    headers: dict,
    header_rows: int,
    max_row: int,
    timeout: float,
) -> Tuple[Dict[str, int], int]:
    """读 A 列已存在的 case_id。

    返回 (case_id -> 首次出现的 1-based 行号, A 列最后一处非空行的行号；若全空则为 header_rows)。
    """
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
        raise RuntimeError(f"values_batch_get A 列失败: {data}")
    vrs = (data.get("data") or {}).get("valueRanges") or []
    if not vrs:
        return {}, header_rows
    values = vrs[0].get("values") or []
    row_by_case: Dict[str, int] = {}
    last_content = header_rows
    for i, row in enumerate(values):
        row_1based = start + i
        if not row:
            continue
        cell = row[0]
        cid = normalize_case_id(cell)
        if not cid:
            continue
        last_content = max(last_content, row_1based)
        if cid not in row_by_case:
            row_by_case[cid] = row_1based
    return row_by_case, last_content


def cell_has_image_or_content(cell: Any) -> bool:
    """单元格是否已有内容（文本或嵌入图，用于 B/C/D 是否跳过写入）。"""
    if cell is None:
        return False
    if isinstance(cell, str):
        return bool(cell.strip())
    if isinstance(cell, dict):
        if cell.get("type") == "embed-image" and cell.get("fileToken"):
            return True
        return bool(_cell_to_plain_text(cell))
    return True


def read_columns_bcd(
    spreadsheet_token: str,
    sheet_id: str,
    headers: dict,
    start_row: int,
    end_row: int,
    timeout: float,
) -> List[List[Any]]:
    """读取 B:D 列，行 start_row..end_row（含）；与 values 下标 i 对应行号 start_row+i。"""
    if end_row < start_row:
        return []
    rng = f"{sheet_id}!B{start_row}:D{end_row}"
    r = requests.get(
        f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_get",
        headers=headers,
        params={"ranges": rng},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"values_batch_get B:D 失败: {data}")
    vrs = (data.get("data") or {}).get("valueRanges") or []
    if not vrs:
        return []
    return vrs[0].get("values") or []


def bcd_cell_at(
    bcd_values: List[List[Any]], start_row: int, row_1based: int, col_idx: int
) -> Any:
    """col_idx: 0=B, 1=C, 2=D。"""
    i = row_1based - start_row
    if i < 0 or i >= len(bcd_values):
        return None
    row = bcd_values[i]
    if col_idx >= len(row):
        return None
    return row[col_idx]


def assign_rows_for_all_files(
    files: List[str],
    row_by_case_sheet: Dict[str, int],
    last_content_row: int,
    min_append_row: int,
) -> Tuple[List[Tuple[str, str, int]], set[int]]:
    """每个文件对应一行：已有 case_id 用表中行号，否则表尾新行。

    返回 ( (文件名, case_id, 行号)... , 需要写入 A 列的新行行号集合 )。
    """
    sheet_rows = dict(row_by_case_sheet)
    next_new = max(last_content_row + 1, min_append_row)
    out: List[Tuple[str, str, int]] = []
    new_rows_for_a: set[int] = set()

    for fn in files:
        case_id = os.path.splitext(fn)[0]
        if case_id in sheet_rows:
            row = sheet_rows[case_id]
        else:
            row = next_new
            next_new += 1
            sheet_rows[case_id] = row
            new_rows_for_a.add(row)
        out.append((fn, case_id, row))
    return out, new_rows_for_a


def write_column_a_scattered(
    spreadsheet_token: str,
    sheet_id: str,
    headers: dict,
    row_case_pairs: List[Tuple[int, str]],
    timeout: float,
) -> None:
    """按行写入 A 列 case_id（无后缀），多分块 values_batch_update。"""
    if not row_case_pairs:
        return
    for i in range(0, len(row_case_pairs), _MAX_RANGES_PER_BATCH):
        chunk = row_case_pairs[i : i + _MAX_RANGES_PER_BATCH]
        vrs = [
            {
                "range": f"{sheet_id}!A{r}:A{r}",
                "values": [[cid]],
            }
            for r, cid in chunk
        ]
        values_batch_update(spreadsheet_token, headers, vrs, timeout=timeout)


def main() -> int:
    if _project_root() not in sys.path:
        sys.path.insert(0, _project_root())

    parser = argparse.ArgumentParser(
        description=(
            "将 images/crop/yuyan 写入飞书电子表格：A 列为无后缀 case_id；"
            "新 case 表尾追加；B/C/D 按格跳过已有内容。"
        )
    )
    parser.add_argument(
        "--spreadsheet-url",
        default=DEFAULT_SPREADSHEET_URL,
        help="电子表格完整 URL（.../sheets/<token>）；与 --wiki-token 二选一优先用显式 token",
    )
    parser.add_argument(
        "--spreadsheet-token",
        default="",
        help="直接指定 spreadsheet token（优先级最高，可免去从 URL 解析）",
    )
    parser.add_argument(
        "--wiki-token",
        default="",
        help=(
            "知识库节点 token（与 yanshou_sheet_reader 一致）；"
            "设置后通过 wiki/v2/spaces/get_node 解析真实表格 token。"
            "也可设环境变量 FEISHU_WIKI_TOKEN。"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=f"常规请求超时秒数（默认 {DEFAULT_REQUEST_TIMEOUT}；写图仍用更长超时）",
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help="原图目录（以此目录下的文件列表为准）",
    )
    parser.add_argument(
        "--crop-dir",
        default=DEFAULT_CROP_DIR,
        help="crop 图目录（默认同级 crop/）",
    )
    parser.add_argument(
        "--yuyan-dir",
        default=DEFAULT_YUYAN_DIR,
        help="yuyan 图目录（默认同级 yuyan/）",
    )
    parser.add_argument(
        "--sheet-index",
        type=int,
        default=0,
        help="使用第几个「可见」工作表，从 0 开始（默认第一个）",
    )
    parser.add_argument(
        "--sheet-title",
        default=None,
        help="若指定则按工作表标题匹配，优先级高于 --sheet-index",
    )
    parser.add_argument(
        "--header-rows",
        type=int,
        default=1,
        help="表头行数（默认 1；A 列从第 header_rows+1 行起读已有 case_id）",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=2,
        help=(
            "新 case 追加时的最小行号下界（默认 2）；"
            "实际追加行 = max(表内 A 列最后非空行的下一行, start-row)"
        ),
    )
    parser.add_argument(
        "--image-delay",
        type=float,
        default=0.02,
        help="两次写入图片之间的间隔秒数，减轻限频（默认 0.02）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将写入的行数与路径，不调用飞书 API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="只处理排序后的前 N 张图（调试用；默认不限制）",
    )
    args = parser.parse_args()

    if args.header_rows < 1:
        print("--header-rows 须 >= 1", file=sys.stderr)
        return 1
    if args.start_row <= args.header_rows:
        print(
            f"--start-row（{args.start_row}）须大于 --header-rows（{args.header_rows}）",
            file=sys.stderr,
        )
        return 1

    app_id, app_secret = load_app_credentials()
    wiki = (args.wiki_token or "").strip() or os.environ.get(
        "FEISHU_WIKI_TOKEN", ""
    ).strip()
    req_to = float(args.request_timeout)

    need_creds = (not args.dry_run) or bool(wiki)
    if need_creds and (not app_id or not app_secret):
        print(
            "需要 FEISHU_APP_ID / FEISHU_APP_SECRET（或 LARK_APP_ID / LARK_APP_SECRET）："
            "写飞书时必填；使用 --wiki-token / FEISHU_WIKI_TOKEN 时 dry-run 也会解析节点故同样需要。",
            file=sys.stderr,
        )
        return 1

    try:
        files = list_image_files(args.images_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    if not files:
        print(f"在 {args.images_dir} 下未发现常见图片后缀文件", file=sys.stderr)
        return 1

    if args.limit is not None:
        if args.limit < 1:
            print("--limit 须为正整数", file=sys.stderr)
            return 1
        files = files[: args.limit]
        print(f">>> 已启用 --limit {args.limit}，仅处理前 {len(files)} 条")

    token_str: str
    tenant: Optional[str] = None
    if (args.spreadsheet_token or "").strip():
        token_str = (args.spreadsheet_token or "").strip()
    elif wiki:
        tenant = get_tenant_access_token(app_id, app_secret, req_to)
        node = get_wiki_node(tenant, wiki, req_to)
        token_str = spreadsheet_token_from_wiki_node(node)
        print(f">>> 已从 Wiki 解析 spreadsheet_token={token_str}")
    else:
        token_str = spreadsheet_token_from_url(args.spreadsheet_url)

    header_rows = args.header_rows
    min_append_row = args.start_row

    print(f"spreadsheet_token={token_str}")
    print(
        f"共 {len(files)} 张图 → A 列 case_id 无后缀；"
        f"表头 {header_rows} 行，新行下界 start-row={min_append_row}"
    )

    if args.dry_run:
        assignments, new_a = assign_rows_for_all_files(
            files, {}, header_rows, min_append_row
        )
        preview = min(
            len(assignments),
            args.limit if args.limit is not None else 5,
        )
        print(
            "  （dry-run 未读表：假定无已有 case；实际会读 A/B:D，"
            "仅对空单元格写图、仅对新行写 A）"
        )
        print(f"  共 {len(assignments)} 条文件；其中需新写 A 的行约 {len(new_a)} 行（无表时=全部新行）")
        for fn, case_id, row in assignments[:preview]:
            mark = " [新行写A]" if row in new_a else " [已有行]"
            print(
                f"  行{row}{mark}: A={case_id!r}  B={os.path.join(args.images_dir, fn)} "
                f"C={resolve_same_stem(args.crop_dir, fn)} "
                f"D={resolve_same_stem(args.yuyan_dir, fn)}"
            )
        if len(assignments) > preview:
            print(
                f"  ... 其余 {len(assignments) - preview} 条略（dry-run 仅展示前 {preview} 条）"
            )
        return 0

    if tenant is None:
        tenant = get_tenant_access_token(app_id, app_secret, req_to)
    headers = {
        "Authorization": f"Bearer {tenant}",
        "Content-Type": "application/json; charset=utf-8",
    }

    sheets = fetch_sheets(token_str, headers, req_to)
    sheet_id = pick_sheet_id(sheets, args.sheet_index, args.sheet_title)
    print(f"使用工作表 sheet_id={sheet_id!r}")

    max_scan = _grid_row_count(sheets, sheet_id)
    row_by_case, last_content = read_column_a_existing_cases(
        token_str,
        sheet_id,
        headers,
        header_rows,
        max_scan,
        req_to,
    )
    print(
        f"已读 A 列：已有 {len(row_by_case)} 个不同 case_id，"
        f"A 列最后非空行={last_content}，扫描上界 A{max_scan}"
    )

    assignments, new_rows_for_a = assign_rows_for_all_files(
        files, row_by_case, last_content, min_append_row
    )
    a_start = header_rows + 1
    max_touch_row = max((r for _, _, r in assignments), default=a_start)
    end_read = min(max_scan, max(max_touch_row, last_content))
    bcd_values = read_columns_bcd(
        token_str, sheet_id, headers, a_start, end_read, req_to
    )
    print(
        f"已读 B:D 行 {a_start}–{end_read}，共 {len(bcd_values)} 行返回值；"
        f"待处理文件 {len(assignments)} 条，新行需写 A 的共 {len(new_rows_for_a)} 行"
    )

    row_to_case_for_a: Dict[int, str] = {}
    for fn, cid, row in assignments:
        if row in new_rows_for_a and row not in row_to_case_for_a:
            row_to_case_for_a[row] = cid
    row_case_pairs = sorted(
        ((r, row_to_case_for_a[r]) for r in new_rows_for_a if r in row_to_case_for_a),
        key=lambda x: x[0],
    )
    if row_case_pairs:
        write_column_a_scattered(
            token_str,
            sheet_id,
            headers,
            list(row_case_pairs),
            timeout=max(req_to, IMAGE_UPLOAD_TIMEOUT),
        )
        print(f"已写入 A 列 case_id（{len(row_case_pairs)} 个新行）")
    else:
        print("无需写 A 列（无新增 case 行）")

    skip_b = skip_c = skip_d = wrote_b = wrote_c = wrote_d = 0
    filled_this_run: set[Tuple[int, str]] = set()

    for idx, (fn, case_id, row) in enumerate(assignments):
        paths: List[Tuple[str, int, str, str]] = [
            ("B", 0, os.path.join(args.images_dir, fn), "原图"),
            ("C", 1, resolve_same_stem(args.crop_dir, fn) or "", "crop"),
            ("D", 2, resolve_same_stem(args.yuyan_dir, fn) or "", "鱼眼"),
        ]
        for col_letter, col_idx, path, label in paths:
            key = (row, col_letter)
            if key in filled_this_run:
                continue
            cell_val = bcd_cell_at(bcd_values, a_start, row, col_idx)
            if cell_has_image_or_content(cell_val):
                if col_letter == "B":
                    skip_b += 1
                elif col_letter == "C":
                    skip_c += 1
                else:
                    skip_d += 1
                continue
            if not path or not os.path.isfile(path):
                if col_letter == "B":
                    print(
                        f"  [WARN] 行{row} case={case_id!r} 缺少原图 {fn}，跳过 B 列",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  [WARN] 行{row} case={case_id!r} 缺 {label} 文件，跳过 {col_letter} 列",
                        file=sys.stderr,
                    )
                continue
            write_image_cell(
                token_str,
                headers,
                sheet_id,
                f"{col_letter}{row}",
                path,
                timeout=max(req_to, IMAGE_UPLOAD_TIMEOUT),
            )
            filled_this_run.add(key)
            if col_letter == "B":
                wrote_b += 1
            elif col_letter == "C":
                wrote_c += 1
            else:
                wrote_d += 1
            if args.image_delay > 0:
                time.sleep(args.image_delay)
        if (idx + 1) % 50 == 0:
            print(f"  已扫描 {idx + 1}/{len(assignments)} 条文件 ...")

    print(
        f"图片写入：B 新写×{wrote_b} 跳过已有×{skip_b}；"
        f"C 新写×{wrote_c} 跳过×{skip_c}；D 新写×{wrote_d} 跳过×{skip_d}"
    )
    print("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
