"""
Engine - 核心引擎模块

包含：
- BaseAgent: 核心 Agent 类
- AgentFactory: Agent 工厂
- OutputPayload, ImageInput: 输出数据结构
- CoreTasksManager: 核心任务管理器
"""

from .base_agent import BaseAgent
from .agent_factory import AgentFactory
from .core_tasks_manager import CoreTasksManager

__all__ = [
    "BaseAgent",
    "AgentFactory",
    "CoreTasksManager",
]
