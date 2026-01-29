"""
AgentFactory - Agent 工厂类

负责创建和管理 Agent 实例
"""

from pathlib import Path
from typing import Dict, Optional
from .base_agent import BaseAgent


class AgentFactory:
    """
    Agent 工厂
    
    提供统一的 Agent 创建接口
    """
    
    # Agent 注册表（可以在这里注册自定义 Agent）
    _AGENT_REGISTRY: Dict[str, type] = {}
    
    @classmethod
    def create(cls, agent_name: str = "chaosheng_wujian", custom_config: Optional[Path] = None) -> BaseAgent:
        """
        创建 Agent 实例
        
        Args:
            agent_name: Agent 名称（对应 configs/ 下的 yaml 文件名，不含扩展名）
            custom_config: 自定义配置文件路径（可选）
            
        Returns:
            BaseAgent 实例
        """
        # 如果提供了自定义配置文件，直接使用
        if custom_config:
            config_path = Path(custom_config)
        else:
            # 否则从标准位置加载
            configs_base_dir = Path(__file__).parent.parent / "configs"
            config_path = configs_base_dir / f"{agent_name}.yaml"
        
        if not config_path.exists():
            raise ValueError(f"Agent config file not found: {config_path}")

        # # 检查是否有自定义 Agent 类
        # if agent_name in cls._AGENT_REGISTRY:
        #     agent_class = cls._AGENT_REGISTRY[agent_name]
        #     return agent_class(config_path, agent_name)
        
        # 默认使用 BaseAgent
        return BaseAgent(config_path, agent_name)
    
    @classmethod
    def register_agent(cls, name: str, agent_class: type):
        """
        注册自定义 Agent 类
        
        Args:
            name: Agent 名称
            agent_class: Agent 类（必须继承 BaseAgent）
        """
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"Agent class must inherit from BaseAgent")
        
        cls._AGENT_REGISTRY[name] = agent_class
