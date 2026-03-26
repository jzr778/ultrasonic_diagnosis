"""
FreespaceType 整型 / 枚举名 / 中文释义（与 get_data/proto/perception/deeproute_perception_obstacle.proto 一致，0–20）。
"""

from typing import Any, Dict, Tuple

# (整型值, 枚举名, 中文含义)
_FREESPACE_DEFINITIONS: Tuple[Tuple[int, str, str], ...] = (
    (0, "FS_OTHERS_STATIC", "其他静态障碍物"),
    (1, "FS_OTHERS_MOTION", "其他运动障碍物"),
    (2, "FS_WALL", "墙"),
    (3, "FS_CHOCK", "挡车器"),
    (4, "FS_LOCK_ON", "地锁（升起）"),
    (5, "FS_LOCK_OFF", "地锁（放下）"),
    (6, "FS_SPEEDBUMP", "减速带"),
    (7, "FS_FENCE", "围栏"),
    (8, "FS_BIGCAR", "大车"),
    (9, "FS_CAR", "小车"),
    (10, "FS_CONE", "锥桶"),
    (11, "FS_HUMAN", "行人"),
    (12, "FS_BICYCLE", "自行车"),
    (13, "FS_TRICYCLE", "三轮车"),
    (14, "FS_CURB", "路沿"),
    (15, "FS_PILLAR", "柱子"),
    (16, "FS_BUSH", "灌木"),
    (17, "FS_TREE", "树木"),
    (18, "FS_U_TUBE", "U型挡"),
    (19, "FS_ELECTRIC_WIRE", "电线"),
    (20, "FS_DEPTH", "深度区域"),
)

FREESPACE_TYPE_INT_TO_NAME: Dict[int, str] = {v: n for v, n, _ in _FREESPACE_DEFINITIONS}
FREESPACE_TYPE_NAME_ZH: Dict[str, str] = {n: z for _, n, z in _FREESPACE_DEFINITIONS}
FREESPACE_ENUM_NAMES = frozenset(FREESPACE_TYPE_INT_TO_NAME.values())


def normalize_freespace_label(fs_type: Any) -> str:
    """将数据中的 freespaceType 规范为枚举名；缺省、无法解析或整型不在 proto 表内时视为 FS_OTHERS_STATIC。"""
    if fs_type is None:
        return "FS_OTHERS_STATIC"
    if isinstance(fs_type, bool):
        return "FS_OTHERS_STATIC"
    if isinstance(fs_type, int):
        return FREESPACE_TYPE_INT_TO_NAME.get(fs_type, "FS_OTHERS_STATIC")
    if isinstance(fs_type, float) and fs_type == int(fs_type):
        return FREESPACE_TYPE_INT_TO_NAME.get(int(fs_type), "FS_OTHERS_STATIC")
    if isinstance(fs_type, str):
        s = fs_type.strip()
        if not s:
            return "FS_OTHERS_STATIC"
        if s in FREESPACE_ENUM_NAMES:
            return s
        try:
            v = int(s, 10)
            return FREESPACE_TYPE_INT_TO_NAME.get(v, "FS_OTHERS_STATIC")
        except ValueError:
            return "FS_OTHERS_STATIC"
    return "FS_OTHERS_STATIC"


def zh_for_freespace_label(label: str) -> str:
    return FREESPACE_TYPE_NAME_ZH.get(label, FREESPACE_TYPE_NAME_ZH["FS_OTHERS_STATIC"])
