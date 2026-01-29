"""
Context Builder - 上下文构建器
"""
from typing import Dict, List, Any, Optional

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
        context = context
        
        return context
