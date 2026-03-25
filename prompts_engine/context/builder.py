"""
Context Builder - 上下文构建器
"""
from typing import Any, Dict

from .freespace_catalog import normalize_freespace_label, zh_for_freespace_label


def _enrich_yellow_freespace_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    d = dict(item)
    label = normalize_freespace_label(d.get("freespaceType"))
    d["freespaceType"] = label
    d["freespaceTypeZh"] = zh_for_freespace_label(label)
    return d


class ContextBuilder:
    """
    Prompt 上下文构建器
    
    将所有数据准备逻辑集中在此，输出纯净的 Context 字典。
    Prompt Engine 不再需要了解任何业务逻辑。
    """
    
    @classmethod
    def build_context(
        cls,
        # 原始数据
        context: Dict,
    ) -> Dict[str, Any]:
        """
        构建完整的 Prompt Context
        
        Returns:
            一个纯字典，包含所有 Prompt 模板需要的变量
        """
        out = dict(context)
        out.setdefault("vlm_yuyan_image_included", False)
        yf = out.get("yellow_freespace")
        if yf:
            out["yellow_freespace"] = [
                _enrich_yellow_freespace_item(x) for x in yf
            ]
        return out
