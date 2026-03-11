"""
Context - 上下文构建模块

负责将原始数据转换为 Prompt 模板所需的格式化文本和上下文
"""

from .builder import ContextBuilder

__all__ = ["ContextBuilder"]
