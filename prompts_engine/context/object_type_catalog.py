"""
VLM 障碍物类型标签（训练 / 飞书导出 / 评测共用）。

方案 A：原 curb_like 拆为 parking_curb（车位路沿，泊入可压）与 hard_curb（硬路沿/台阶，不可越）。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

# 用于 id 映射与评测合法标签集合
OBJECT_TYPE_ORDER: Tuple[str, ...] = (
    "parking_curb",
    "hard_curb",
    "wheel_stop",
    "speed_bump",
    "ground_irregularity",
    "other_obstacle",
)

OBJECT_TYPE_ID: Dict[str, int] = {name: i for i, name in enumerate(OBJECT_TYPE_ORDER)}

OBJECT_TYPE_ZH: Dict[str, str] = {
    "parking_curb": "车位路沿（泊入可压过）",
    "hard_curb": "硬路沿/台阶（不可压过）",
    "wheel_stop": "轮挡",
    "speed_bump": "减速带",
    "ground_irregularity": "地面异常",
    "other_obstacle": "其他障碍",
}

# 飞书 / 人工标注中文 → 英文标签（_norm_label 之后仍可用中文键）
OBJECT_TYPE_ALIASES: Dict[str, str] = {
    "车位路沿": "parking_curb",
    "车位路沿可压": "parking_curb",
    "可压路沿": "parking_curb",
    "parking_curb": "parking_curb",
    "硬路沿": "hard_curb",
    "硬路沿台阶": "hard_curb",
    "路沿/台阶": "hard_curb",
    "路沿台阶": "hard_curb",
    "不可压路沿": "hard_curb",
    "hard_curb": "hard_curb",
    "轮挡": "wheel_stop",
    "wheel_stop": "wheel_stop",
    "减速带": "speed_bump",
    "speed_bump": "speed_bump",
    "地面异常": "ground_irregularity",
    "ground_irregularity": "ground_irregularity",
    "其他障碍": "other_obstacle",
    "other_obstacle": "other_obstacle",
}

# 已废弃：导入或标注时拒绝，需改为 parking_curb / hard_curb
# 注意：「路沿/台阶」在飞书中仍表示 hard_curb，勿放入本集合（见 OBJECT_TYPE_ALIASES）
DEPRECATED_OBJECT_TYPE_ALIASES: FrozenSet[str] = frozenset(
    {
        "curb_like",
        "路沿",  # 单独「路沿」语义不明，需人工改为 parking_curb / hard_curb
    }
)

OBJECT_TYPE_OPTIONS_CSV = ", ".join(OBJECT_TYPE_ORDER)


def object_type_task_prompt() -> str:
    """障碍物类型判定 user 任务文案（含类别说明与可选项）。"""
    lines = [
        "<image>任务：障碍物类型判定。",
        "请根据红色超声高亮附近所对应的真实障碍，只输出一个英文标签。",
        "类别含义（帮助理解，回答仍只输出标签本身）：",
        f"parking_curb={OBJECT_TYPE_ZH['parking_curb']}：路边/划线车位开口处的边界路沿，"
        "路沿内侧或上方是目标泊入区，泊入时允许骑压；",
        f"hard_curb={OBJECT_TYPE_ZH['hard_curb']}：车道边、人行道边、墙根台阶等，"
        "外侧不可泊、泊入不可压过；",
        f"wheel_stop={OBJECT_TYPE_ZH['wheel_stop']}；",
        f"speed_bump={OBJECT_TYPE_ZH['speed_bump']}；",
        f"ground_irregularity={OBJECT_TYPE_ZH['ground_irregularity']}"
        "（井盖/轻微凹凸/小坑/纹理突起/地面轨道等）；",
        f"other_obstacle={OBJECT_TYPE_ZH['other_obstacle']}（墙/车/柱子/纸箱/人等）。",
        f"可选项：{OBJECT_TYPE_OPTIONS_CSV}。"
    ]
    return "".join(lines)


def normalize_object_type_label(raw: str) -> str:
    """规范化标签文本（小写、去空格），不映射别名。"""
    return (
        raw.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("：", ":")
    )


def map_object_type_label(
    raw_text: str, *, entity_is_no: bool = False
) -> Tuple[str, Optional[int]]:
    """将原始标注映射为 (object_type, id)。entity=no 时返回空。"""
    if entity_is_no:
        return "", None
    n = normalize_object_type_label(raw_text)
    if not n:
        return "", None
    if n in DEPRECATED_OBJECT_TYPE_ALIASES:
        return "", None
    std = OBJECT_TYPE_ALIASES.get(n) or OBJECT_TYPE_ALIASES.get(raw_text.strip())
    if std is None:
        return "", None
    return std, OBJECT_TYPE_ID.get(std)


def coerce_legacy_object_type(label: str, default: str = "hard_curb") -> str:
    """将历史 curb_like 转为 default（仅用于数据迁移，新标注勿用）。"""
    n = normalize_object_type_label(label)
    if n == "curb_like" or n in DEPRECATED_OBJECT_TYPE_ALIASES:
        if default not in OBJECT_TYPE_ID:
            raise ValueError(f"invalid default: {default}")
        return default
    if n in OBJECT_TYPE_ID:
        return n
    return label.strip()
