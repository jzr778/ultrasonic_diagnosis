#!/usr/bin/env python3
"""从 v4_eval.json 聚合每个 case 的三类标签，并按规则映射「是否误检」。

规则：
- 地面异常、减速带、轮挡、泊车路沿 → 直接「误检」（不看几何）
- 其他障碍、硬路沿 → 偏移→误检，命中→真实障碍
- 实体不存在 (no) → 误检

用法::

  python tool/build_labels.py
  python tool/build_labels.py -i tool/result/v5_eval.json -o tool/result/v5_eval_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from prompts_engine.context.object_type_catalog import (  # noqa: E402
    normalize_object_type_label,
)

DEFAULT_IN = os.path.join(_script_dir, "v4_eval.json")
DEFAULT_OUT = os.path.join(_script_dir, "result", "v4_eval_labels.csv")

# 直接判误检的障碍物类型（英文 canonical）
AUTO_MISDETECT_OBJECT_TYPES = frozenset(
    {
        "ground_irregularity",
        "speed_bump",
        "wheel_stop",
        "parking_curb",
    }
)

# 需结合几何：偏移→误检，命中→真实障碍
GEOM_RULE_OBJECT_TYPES = frozenset({"other_obstacle", "hard_curb"})

OBJECT_TYPE_ZH = {
    "parking_curb": "泊车路沿",
    "hard_curb": "硬路沿",
    "wheel_stop": "轮挡",
    "speed_bump": "减速带",
    "ground_irregularity": "地面异常",
    "other_obstacle": "其他障碍",
}

GEOM_TO_ZH = {
    "aligned": "命中",
    "misaligned": "偏移",
}


def _extract_label(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s).strip()
    if "\n" in s:
        s = s.splitlines()[-1].strip()
    return s


def _case_id_from_record(obj: Dict[str, Any]) -> str:
    images = obj.get("images") or []
    if not images:
        return ""
    path = images[0].get("path") or images[0]
    if isinstance(path, dict):
        path = path.get("path", "")
    return os.path.splitext(os.path.basename(str(path)))[0]


def _task_kind(user_content: str) -> str:
    if "实体存在性" in user_content:
        return "entity"
    if "几何一致性" in user_content:
        return "geometry"
    if "障碍物类型" in user_content:
        return "object_type"
    return ""


def _norm_entity(label: str) -> str:
    n = normalize_object_type_label(label)
    if n in ("yes", "no"):
        return n
    if label in ("是", "有", "存在"):
        return "yes"
    if label in ("否", "无", "不存在"):
        return "no"
    return n


def _norm_geometry(label: str) -> str:
    n = normalize_object_type_label(label)
    if n in ("aligned", "misaligned"):
        return n
    if label in ("命中", "有效命中"):
        return "aligned"
    if label in ("偏移", "未命中"):
        return "misaligned"
    return n


def _norm_object_type(label: str) -> str:
    n = normalize_object_type_label(label)
    if n in OBJECT_TYPE_ZH:
        return n
    return n


def load_cases_from_v4_eval(path: str) -> Dict[str, Dict[str, str]]:
    cases: Dict[str, Dict[str, str]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {lineno} 行 JSON 无效: {e}") from e
            cid = _case_id_from_record(obj)
            if not cid:
                continue
            user = ""
            for msg in obj.get("messages") or []:
                if msg.get("role") == "user":
                    user = str(msg.get("content", ""))
                    break
            kind = _task_kind(user)
            if not kind:
                continue
            lab = _extract_label(obj.get("labels") if obj.get("labels") is not None else obj.get("response"))
            cases[cid][kind] = lab
    return dict(cases)


def map_ultrasound_misdetect(
    entity: str,
    geometry: str,
    object_type: str,
) -> str:
    """返回「误检」或「真实障碍」，无法判定时返回空串。"""
    if entity == "no":
        return "误检"

    if not entity:
        return ""

    if object_type in AUTO_MISDETECT_OBJECT_TYPES:
        return "误检"

    if object_type in GEOM_RULE_OBJECT_TYPES:
        if geometry == "misaligned":
            return "误检"
        if geometry == "aligned":
            return "真实障碍"
        return ""

    return ""


def build_row(case_id: str, fields: Dict[str, str]) -> Dict[str, str]:
    entity_raw = fields.get("entity", "")
    geom_raw = fields.get("geometry", "")
    obj_raw = fields.get("object_type", "")

    entity = _norm_entity(entity_raw)
    geometry = _norm_geometry(geom_raw) if entity == "yes" else ""
    object_type = _norm_object_type(obj_raw) if entity == "yes" else ""

    geom_zh = GEOM_TO_ZH.get(geometry, "") if geometry else ""
    obj_zh = OBJECT_TYPE_ZH.get(object_type, object_type) if object_type else ""

    misdetect = map_ultrasound_misdetect(entity, geometry, object_type)

    return {
        "case_id": case_id,
        "实体是否存在": entity,
        "超声标记命中或偏移": geom_zh,
        "障碍物类型": obj_zh,
        "是否误检": misdetect,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v4_eval.json → 按 case 聚合标签 CSV")
    parser.add_argument("-i", "--input", default=DEFAULT_IN)
    parser.add_argument("-o", "--output", default=DEFAULT_OUT)
    args = parser.parse_args()

    cases = load_cases_from_v4_eval(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    rows = [build_row(cid, cases[cid]) for cid in sorted(cases.keys())]
    fieldnames = [
        "case_id",
        "实体是否存在",
        "超声标记命中或偏移",
        "障碍物类型",
        "是否误检",
    ]
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter

    c = Counter(r["是否误检"] for r in rows)
    print(f"已写入 {len(rows)} 条 → {args.output}")
    print(f"是否误检分布: {dict(c)}")
    empty = sum(1 for r in rows if not r["是否误检"])
    if empty:
        print(f"警告: {empty} 条无法映射是否误检")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
