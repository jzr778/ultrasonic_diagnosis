#!/usr/bin/env python3
"""
将本地 images / crop / yuyan 目录中的图片同步到飞书电子表格。

你提供的链接形态为「电子表格」：
  https://rqk9rsooi4.feishu.cn/sheets/<spreadsheetToken>
若链接为「多维表格」Base（/base/...），本脚本不适用，需改用 bitable 记录 + 附件字段 API。

表格约定（不修改第 1 行表头）：
  从第 start_row 行（默认 2）起：
    A 列：文件名（仅 basename）
    B 列：images 目录对应原图
    C 列：crop 图
    D 列：yuyan 图

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
import base64
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
    """向单个单元格写入图片（二进制经 base64，与开放平台示例一致）。"""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    name = os.path.basename(image_path)
    if "." not in name:
        name += ".png"
    r = requests.post(
        f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_image",
        headers=headers,
        json={
            "range": f"{sheet_id}!{cell}:{cell}",
            "image": b64,
            "name": name,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"values_image 失败 {cell} {image_path!r}: {data}")


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


def main() -> int:
    if _project_root() not in sys.path:
        sys.path.insert(0, _project_root())

    parser = argparse.ArgumentParser(
        description="将 images/crop/yuyan 批量写入飞书电子表格 A–D 列（从第 2 行起，保留第 1 行表头）"
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
        "--start-row",
        type=int,
        default=2,
        help="数据起始行号（默认 2，即保留第 1 行表头）",
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
        default=10,
        metavar="N",
        help="只处理排序后的前 N 张图（调试用；默认不限制）",
    )
    args = parser.parse_args()

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

    start = args.start_row
    end_row = start + len(files) - 1

    print(f"spreadsheet_token={token_str}")
    print(f"共 {len(files)} 个原图，写入行 {start}–{end_row}（列 A–D）")

    if args.dry_run:
        preview = min(len(files), args.limit if args.limit is not None else 5)
        for i, fn in enumerate(files[:preview]):
            row = start + i
            print(
                f"  行{row}: A={fn} B={os.path.join(args.images_dir, fn)} "
                f"C={resolve_same_stem(args.crop_dir, fn)} "
                f"D={resolve_same_stem(args.yuyan_dir, fn)}"
            )
        if len(files) > preview:
            print(f"  ... 其余 {len(files) - preview} 行略（dry-run 仅展示前 {preview} 行）")
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

    # A 列：文件名
    col_a = [[fn] for fn in files]
    a1_start = col_row_to_a1(0, start)
    a1_end = col_row_to_a1(0, end_row)
    values_batch_update(
        token_str,
        headers,
        [
            {
                "range": f"{sheet_id}!{a1_start}:{a1_end}",
                "values": col_a,
            }
        ],
        timeout=max(req_to, IMAGE_UPLOAD_TIMEOUT),
    )
    print("已写入 A 列文件名")

    # B / C / D 列：逐格写图（每格一次 values_image）
    for i, fn in enumerate(files):
        row = start + i
        paths: List[Tuple[str, str, str]] = [
            ("B", os.path.join(args.images_dir, fn), "原图"),
            ("C", resolve_same_stem(args.crop_dir, fn) or "", "crop"),
            ("D", resolve_same_stem(args.yuyan_dir, fn) or "", "yuyan"),
        ]
        for col_letter, path, label in paths:
            cell = f"{col_letter}{row}"
            if not path or not os.path.isfile(path):
                if col_letter == "B":
                    print(f"  [WARN] 行{row} 缺少原图 {fn}，跳过 B 列", file=sys.stderr)
                else:
                    print(f"  [WARN] 行{row} 缺 {label}，跳过 {col_letter} 列", file=sys.stderr)
                continue
            write_image_cell(
                token_str,
                headers,
                sheet_id,
                cell,
                path,
                timeout=max(req_to, IMAGE_UPLOAD_TIMEOUT),
            )
            if args.image_delay > 0:
                time.sleep(args.image_delay)
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(files)} 行 ...")

    print("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
