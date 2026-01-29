"""
Prompt Engine - 配置驱动的 Prompt 构建引擎

这是一个完全独立的 Prompt 引擎包，基于 LangChain + Jinja2。

核心功能：
1. BaseAgent: 通用 Agent 基类（支持配置驱动的 Prompt 渲染）
2. AgentFactory: Agent 工厂（统一创建接口）
3. CoreTasksManager: 核心任务管理器（动态组合 Schema）
4. ContextBuilder: 数据构建器（将原始数据转换为 Context）

使用示例：
    from prompts_engine import AgentFactory, ContextBuilder
    
    # 1. 构建 Context
    context = ContextBuilder.build_context(
        all_ego_info=ego_data,
        all_critical_info=critical_data,
        frame_labels=["-0.5s", "0.0s"],
        delta_seconds=0.5,
        camera_image_paths=["img1.jpg", "img2.jpg"],
        is_vru=True,
        case_id="case_001",
    )
    
    # 2. 创建 Agent 并渲染 Prompt
    agent = AgentFactory.create("aeb_agent")
    payload = agent.assemble(context)
    
    # 3. 使用生成的 Prompt
    prompt_text = payload['prompt_text']
    images = payload['images']
    pydantic_model = payload['pydantic_model']
"""

from .engine import BaseAgent, AgentFactory, CoreTasksManager
from .context import ContextBuilder

__version__ = "1.0.0"
__all__ = [
    "BaseAgent",
    "AgentFactory",
    "CoreTasksManager",
    "ContextBuilder",
]
