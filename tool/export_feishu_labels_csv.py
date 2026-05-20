#!/usr/bin/env python3
"""
从飞书电子表格提取 A/E/F/G 列并导出标签 CSV。

python tool/export_feishu_labels_csv.py --generate-jsonl
python tool/export_feishu_labels_csv.py \
  --generate-jsonl \
  --from-csv /mnt/public-data/user/ziroujiang/generate_ground_irregularity/label.csv \
  --jsonl-dir /mnt/public-data/user/ziroujiang/generate_ground_irregularity/

README 规则摘要:
  1) entity_existence ∈ {yes, no}
  2) geometry_relation ∈ {aligned, misaligned}
  3) object_type ∈ {curb_like, wheel_stop, speed_bump, ground_irregularity, other_obstacle}
  且: entity_existence=no 时，geometry_relation/object_type 置空。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OPEN_API = "https://open.feishu.cn/open-apis"
DEFAULT_REQUEST_TIMEOUT = 15.0

# 与 sync_raw_images_to_feishu_sheet.py 保持一致
_DEFAULT_FEISHU_APP_ID = "cli_a6e0444aedfbd00b"
_DEFAULT_FEISHU_APP_SECRET = "8W1Art9TRWrV50C7QgITwbYbMMqLKI5x"
DEFAULT_SPREADSHEET_URL = (
    "https://rqk9rsooi4.feishu.cn/sheets/FyQXsnoiWh13rbtkxWjcRFTYnqf"
)
DEFAULT_OUTPUT_CSV = "/mnt/public-data/user/ziroujiang/generate_ground_irregularity/label.csv"

# README 约定的 object_type 顺序（用于 id 映射）
OBJECT_TYPE_ORDER = [
    "curb_like",
    "wheel_stop",
    "speed_bump",
    "ground_irregularity",
    "other_obstacle",
]
OBJECT_TYPE_ID = {name: i for i, name in enumerate(OBJECT_TYPE_ORDER)}


def spreadsheet_token_from_url(url: str) -> str:
    m = re.search(r"/sheets/([A-Za-z0-9]+)", url.strip())
    if not m:
        raise ValueError("无法从 URL 解析 spreadsheet token（需形如 .../sheets/<token>）")
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
    url = f"{OPEN_API}/wiki/v2/spaces/get_node"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {tenant}"},
        params={"token": wiki_token},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"get_wiki_node HTTP {resp.status_code}: {resp.text[:2000]}")
    payload = resp.json()
    if payload.get("code") not in (None, 0):
        raise RuntimeError(f"get_wiki_node 业务错误: {payload}")
    node = (payload.get("data") or {}).get("node")
    if not node:
        raise RuntimeError(f"get_wiki_node 未返回 node: {payload}")
    return node


def spreadsheet_token_from_wiki_node(node: Dict[str, Any]) -> str:
    node_type = str(node.get("obj_type") or node.get("node_type") or node.get("type") or "")
    if node_type.lower() not in ("spreadsheet", "sheet"):
        raise RuntimeError(f"Wiki 节点不是 spreadsheet/sheet: {node_type!r}")
    token = node.get("obj_token") or node.get("node_token") or node.get("token")
    if not token:
        raise RuntimeError(f"Wiki 节点无可用 token: {node!r}")
    return str(token)


def fetch_sheets(
    spreadsheet_token: str, headers: Dict[str, str], timeout: float
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
        raise RuntimeError(
            f"sheets v3/v2 均失败: v3={rv3.status_code}, v2={rv2.status_code}, resp={rv2.text[:1500]}"
        )
    j2 = rv2.json()
    if j2.get("code") not in (None, 0):
        raise RuntimeError(f"sheets v2 业务错误: {j2}")
    raw = (j2.get("data") or {}).get("sheets") or []
    out: List[Dict[str, Any]] = []
    for s in raw:
        out.append(
            {
                "sheet_id": s.get("sheet_id") or s.get("sheetId"),
                "title": s.get("title", ""),
                "hidden": bool(s.get("hidden", False)),
                "resource_type": s.get("resource_type", "sheet"),
            }
        )
    return out


def pick_sheet_id(
    sheets: Iterable[Dict[str, Any]], index: int, title: Optional[str]
) -> str:
    visible = [
        s
        for s in sheets
        if not s.get("hidden") and s.get("resource_type", "sheet") == "sheet"
    ]
    if not visible:
        raise RuntimeError("未找到可见 sheet 子表（可能是 bitable/base 文档）")
    if title:
        for s in visible:
            if s.get("title") == title:
                return str(s["sheet_id"])
        raise RuntimeError(f"未找到标题为 {title!r} 的 sheet")
    if index < 0 or index >= len(visible):
        raise RuntimeError(f"sheet 索引越界: {index}, 可见数={len(visible)}")
    return str(visible[index]["sheet_id"])


def _cell_text(cell: Any) -> str:
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
    text = _cell_text(raw)
    if not text:
        return ""
    lower = text.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".heic", ".tif", ".tiff"):
        if lower.endswith(ext):
            return text[: -len(ext)].strip()
    return text


def _norm_label(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("：", ":")
    )


ENTITY_ALIASES = {
    "yes": "yes",
    "y": "yes",
    "1": "yes",
    "有": "yes",
    "存在": "yes",
    "是": "yes",
    "no": "no",
    "n": "no",
    "0": "no",
    "无": "no",
    "不存在": "no",
    "否": "no",
}

GEOMETRY_ALIASES = {
    "aligned": "aligned",
    "有效命中": "aligned",
    "命中": "aligned",
    "misaligned": "misaligned",
    "偏移": "misaligned",
    "未命中": "misaligned",
}

OBJECT_TYPE_ALIASES = {
    "路沿/台阶": "curb_like",
    "轮挡": "wheel_stop",
    "减速带": "speed_bump",
    "地面异常": "ground_irregularity",
    "其他障碍": "other_obstacle",
}


def map_entity(raw_text: str) -> Tuple[str, Optional[int]]:
    n = _norm_label(raw_text)
    std = ENTITY_ALIASES.get(n)
    if std == "yes":
        return "yes", 1
    if std == "no":
        return "no", 0
    return "", None


def map_geometry(raw_text: str, entity_id: Optional[int]) -> Tuple[str, Optional[int]]:
    if entity_id == 0:
        return "", None
    n = _norm_label(raw_text)
    std = GEOMETRY_ALIASES.get(n)
    if std == "aligned":
        return "aligned", 1
    if std == "misaligned":
        return "misaligned", 0
    return "", None


def map_object_type(raw_text: str, entity_id: Optional[int]) -> Tuple[str, Optional[int]]:
    if entity_id == 0:
        return "", None
    n = _norm_label(raw_text)
    std = OBJECT_TYPE_ALIASES.get(n)
    if std is None:
        return "", None
    return std, OBJECT_TYPE_ID.get(std)


def read_values_a_to_g(
    spreadsheet_token: str,
    sheet_id: str,
    headers: Dict[str, str],
    start_row: int,
    end_row: int,
    timeout: float,
) -> List[List[Any]]:
    rng = f"{sheet_id}!A{start_row}:G{end_row}"
    r = requests.get(
        f"{OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_get",
        headers=headers,
        params={"ranges": rng},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"读取 A:G 失败: {payload}")
    vrs = (payload.get("data") or {}).get("valueRanges") or []
    if not vrs:
        return []
    return vrs[0].get("values") or []


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "case_id",
        "entity_existence",
        "geometry_relation",
        "object_type",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# JSONL 训练数据生成
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是泊车环视场景的超声/视觉联合诊断模型，须结合多图作答。"
    "图例：红标=超声地面障碍物（点、短线或闭合多边形），为分析对象，表示超声在地面上的感知结果。"
    "绿线=邻车检测框投影到地面的多边形，表示邻车可能占用的地面区域。"
    "黄线=相机障碍在AVM上的投影轮廓，用于在鸟瞰中对齐真实可见障碍；"
    "判断时对齐黄线与真实障碍本体，比较红标与真实障碍的关系，"
    "勿将红标与黄线本身当作一对匹配目标。"
    "中心黑矩形=自车（上为车头、下为车尾），正在倒车入库。"
    "车位中心白箭头=预计倒车方向；若无箭头，默认沿车位中轴线直线倒车。"
    "白矩形框=仅遮挡车牌，与障碍物/标线无关，分析时完全忽略。"
    "AVM由鱼眼展开拼接：红标仅有地面投影、无高度语义；"
    "离地越高常渐淡/半透明或与背景融合，属成像与拼接特性，不等于该处无实物。"
    "须结合鱼眼透视理解障碍远近、立面与地面接触，"
    "区分竖直方向透视表现与地面接触位置，避免仅凭AVM上半部发虚误判空间关系。"
    "输入按顺序三张："
    "①AVM鸟瞰：以红标为准；高处虚化不得单独作为无实体依据。"
    "②以超声障碍质心为中心的局部crop，用于聚焦红标。"
    "③与AVM主方位一致的单路鱼眼：绿/黄与AVM语义一致，图中不画红标，"
    "红标仍以AVM为准；作透视与尺度参考，减轻仅凭鸟瞰在远近、实体尺度与类型上的不确定；"
    "禁止在鱼眼与AVM之间做像素级距离换算或强行点配对。"
    "回答必须严格遵守用户给出的任务与可选项；只输出要求的标签或词，不要解释。"
)

TASK_ENTITY_EXISTENCE = (
    "<image>任务：实体存在性判定。"
    "请判断红色超声高亮附近是否存在真实障碍。可选项：yes, no。"
)

TASK_GEOMETRY_RELATION = (
    "<image>任务：几何一致性判定。"
    "请判断红色超声高亮与附近真实障碍之间的几何关系。可选项：aligned, misaligned。"
)

TASK_OBJECT_TYPE = (
    "<image>任务：障碍物类型判定。"
    "请根据红色超声高亮附近所对应的真实障碍，只输出一个英文标签。"
    "类别含义（帮助理解，回答仍只输出标签本身）："
    "curb_like=路沿/台阶；wheel_stop=轮挡；speed_bump=减速带；"
    "ground_irregularity=地面异常（井盖/轻微凹凸/小坑/纹理突起/地面轨道等）；"
    "other_obstacle=其余障碍（墙/车/柱子/纸箱/人等）。"
    "可选项：curb_like, wheel_stop, speed_bump, ground_irregularity, other_obstacle。"
)


def _make_sample(case_id: str, task_prompt: str, answer: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": [
            f"images/{case_id}.jpg",
            f"crop/{case_id}.jpg",
            f"yuyan/{case_id}.jpg",
        ],
    }


def _write_jsonl(path: str, samples: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def read_label_csv(path: str) -> List[Dict[str, str]]:
    """从已有的 label.csv 读取标注行。"""
    rows: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def generate_training_jsonl(
    rows: List[Dict[str, Any]], out_dir: str
) -> Dict[str, int]:
    """根据标注行生成三个任务 JSONL 和混合 dataset.jsonl。

    Returns:
        各文件的行数 dict。
    """
    entity_samples: List[Dict[str, Any]] = []
    geom_samples: List[Dict[str, Any]] = []
    objtype_samples: List[Dict[str, Any]] = []

    for row in rows:
        cid = str(row.get("case_id", "")).strip()
        entity = str(row.get("entity_existence", "")).strip()
        geom = str(row.get("geometry_relation", "")).strip()
        obj = str(row.get("object_type", "")).strip()
        if not cid or not entity:
            continue

        entity_samples.append(
            _make_sample(cid, TASK_ENTITY_EXISTENCE, entity)
        )

        if entity == "yes" and geom:
            geom_samples.append(
                _make_sample(cid, TASK_GEOMETRY_RELATION, geom)
            )
        if entity == "yes" and obj:
            objtype_samples.append(
                _make_sample(cid, TASK_OBJECT_TYPE, obj)
            )

    _write_jsonl(os.path.join(out_dir, "entity_existence.jsonl"), entity_samples)
    _write_jsonl(os.path.join(out_dir, "geometry_relation.jsonl"), geom_samples)
    _write_jsonl(os.path.join(out_dir, "object_type.jsonl"), objtype_samples)

    dataset = entity_samples + geom_samples + objtype_samples
    _write_jsonl(os.path.join(out_dir, "dataset.jsonl"), dataset)

    counts = {
        "entity_existence": len(entity_samples),
        "geometry_relation": len(geom_samples),
        "object_type": len(objtype_samples),
        "dataset": len(dataset),
    }
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="提取飞书表格 A/E/F/G 列并按 README 规则映射，导出 label.csv"
    )
    parser.add_argument("--spreadsheet-url", default=DEFAULT_SPREADSHEET_URL)
    parser.add_argument("--spreadsheet-token", default="")
    parser.add_argument("--wiki-token", default="")
    parser.add_argument("--sheet-index", type=int, default=0)
    parser.add_argument("--sheet-title", default=None)
    parser.add_argument("--header-rows", type=int, default=1)
    parser.add_argument(
        "--max-read-rows",
        type=int,
        default=20000,
        help="读取上限行号（默认 20000）",
    )
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="默认同 case_id 仅保留最后一条；加此参数则保留重复行",
    )
    parser.add_argument(
        "--generate-jsonl",
        action="store_true",
        help="同时生成 entity_existence / geometry_relation / object_type / dataset 四个 JSONL 训练数据文件",
    )
    parser.add_argument(
        "--jsonl-dir",
        default="",
        help="JSONL 输出目录（默认与 --output-csv 同目录）",
    )
    parser.add_argument(
        "--from-csv",
        default="",
        help="跳过飞书 API，直接从已有 label.csv 生成 JSONL（需配合 --generate-jsonl）",
    )
    args = parser.parse_args()

    from_csv = (args.from_csv or "").strip()
    if from_csv:
        if not args.generate_jsonl:
            print("--from-csv 需配合 --generate-jsonl 使用", file=sys.stderr)
            return 1
        if not os.path.isfile(from_csv):
            print(f"文件不存在: {from_csv}", file=sys.stderr)
            return 1
        rows = read_label_csv(from_csv)
        print(f"从 CSV 读取: {from_csv}, 共 {len(rows)} 行")
        jsonl_dir = (args.jsonl_dir or "").strip() or os.path.dirname(os.path.abspath(from_csv))
        counts = generate_training_jsonl(rows, jsonl_dir)
        for name, cnt in counts.items():
            print(f"  {name}.jsonl: {cnt} 条")
        print(f"JSONL 已写入: {jsonl_dir}/")
        return 0

    if args.header_rows < 1:
        print("--header-rows 需 >= 1", file=sys.stderr)
        return 1
    if args.max_read_rows <= args.header_rows:
        print("--max-read-rows 需 > --header-rows", file=sys.stderr)
        return 1

    app_id, app_secret = load_app_credentials()
    tenant = get_tenant_access_token(app_id, app_secret, args.request_timeout)
    headers = {
        "Authorization": f"Bearer {tenant}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept-Encoding": "gzip, deflate",
    }

    wiki = (args.wiki_token or "").strip() or os.environ.get("FEISHU_WIKI_TOKEN", "").strip()
    if (args.spreadsheet_token or "").strip():
        spreadsheet_token = args.spreadsheet_token.strip()
    elif wiki:
        node = get_wiki_node(tenant, wiki, args.request_timeout)
        spreadsheet_token = spreadsheet_token_from_wiki_node(node)
    else:
        spreadsheet_token = spreadsheet_token_from_url(args.spreadsheet_url)

    sheets = fetch_sheets(spreadsheet_token, headers, args.request_timeout)
    sheet_id = pick_sheet_id(sheets, args.sheet_index, args.sheet_title)

    start_row = args.header_rows + 1
    values = read_values_a_to_g(
        spreadsheet_token,
        sheet_id,
        headers,
        start_row=start_row,
        end_row=args.max_read_rows,
        timeout=args.request_timeout,
    )

    rows: List[Dict[str, Any]] = []
    unknown_entity = unknown_geom = unknown_obj = 0

    for i, row in enumerate(values):
        case_id = normalize_case_id(row[0] if len(row) > 0 else "")
        if not case_id:
            continue

        raw_e = _cell_text(row[4] if len(row) > 4 else "")
        raw_f = _cell_text(row[5] if len(row) > 5 else "")
        raw_g = _cell_text(row[6] if len(row) > 6 else "")

        entity, entity_id = map_entity(raw_e)
        geom, geom_id = map_geometry(raw_f, entity_id)
        obj, obj_id = map_object_type(raw_g, entity_id)

        if raw_e and entity_id is None:
            unknown_entity += 1
        if raw_f and entity_id != 0 and geom_id is None:
            unknown_geom += 1
        if raw_g and entity_id != 0 and obj_id is None:
            unknown_obj += 1

        rows.append(
            {
                "case_id": case_id,
                "entity_existence": entity,
                "geometry_relation": geom,
                "object_type": obj,
            }
        )

    input_count = len(rows)
    if not args.keep_duplicates:
        latest_by_case: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            latest_by_case[str(r["case_id"])] = r
        rows = sorted(latest_by_case.values(), key=lambda x: str(x["case_id"]))

    write_csv(args.output_csv, rows)

    print(f"sheet_id={sheet_id}")
    print(f"读取数据行范围: {start_row}..{args.max_read_rows}")
    print(f"有效 case 行: {input_count}")
    if not args.keep_duplicates:
        print(f"按 case_id 去重后: {len(rows)}")
    print(
        f"未知标签计数: entity={unknown_entity}, geometry={unknown_geom}, object_type={unknown_obj}"
    )
    print(f"已写入: {args.output_csv}")

    if args.generate_jsonl:
        jsonl_dir = (args.jsonl_dir or "").strip() or os.path.dirname(
            os.path.abspath(args.output_csv)
        )
        counts = generate_training_jsonl(rows, jsonl_dir)
        for name, cnt in counts.items():
            print(f"  {name}.jsonl: {cnt} 条")
        print(f"JSONL 已写入: {jsonl_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

