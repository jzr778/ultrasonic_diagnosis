"""
Core Tasks Manager - 核心任务管理器

负责：
1. 从 config.yaml 加载任务配置
2. 根据条件激活/禁用任务
3. 动态加载 Schema 类
4. 组合生成最终的 CombinedTasksSchema

这个类复刻了原 prompts/components/layer4_tasks/core_tasks.py 的功能
"""

from typing import Dict, List, Any, Type, Optional
from pydantic import BaseModel, create_model
import importlib
import re


class TaskDefinition:
    """任务定义"""
    
    def __init__(
        self,
        name: str,
        template: str,
        output_field: str,
        schema: str,
        condition: Optional[str] = None
    ):
        self.name = name
        self.template = template
        self.output_field = output_field
        self.schema_name = schema
        self.condition = condition
        self._schema_class = None
    
    def load_schema(self) -> Type[BaseModel]:
        """从 prompts_engine.schemas 统一入口加载 Schema 类"""
        if self._schema_class is None:
            try:
                # 延迟导入 schemas 包，避免循环引用
                from .. import schemas
                
                # 直接从 schemas 模块获取类（__init__.py 已导出所有 Schema）
                if hasattr(schemas, self.schema_name):
                    self._schema_class = getattr(schemas, self.schema_name)
                else:
                    raise AttributeError(f"Schema '{self.schema_name}' not found in prompts_engine.schemas")
                    
            except Exception as e:
                raise ImportError(
                    f"Failed to load schema {self.schema_name} from prompts_engine.schemas: {e}. "
                    f"请确保该类已在 prompts_engine/schemas/__init__.py 中导出。"
                )
        return self._schema_class
    
    def is_active(self, context: Dict[str, Any]) -> bool:
        """根据条件判断任务是否激活"""
        if self.condition is None:
            return True
        
        try:
            # 安全求值：只允许访问 context 中的变量
            return bool(eval(self.condition, {"__builtins__": {}}, context))
        except Exception as e:
            print(f"Warning: Condition evaluation failed for task '{self.name}': {e}")
            return False


class CoreTasksManager:
    """
    核心任务管理器
    
    负责管理所有任务的生命周期：加载、激活、Schema 组合
    
    这个类的设计借鉴了 prompts/components/layer4_tasks/core_tasks.py 的思想，
    但是更加简洁和配置驱动。
    """
    
    def __init__(self, tasks_config: List[Dict[str, Any]]):
        """
        Args:
            tasks_config: 从 config.yaml 加载的 tasks 配置列表
        """
        self.tasks = [
            TaskDefinition(
                name=task.get("name"),
                template=task.get("template"),
                output_field=task.get("output_field"),
                schema=task.get("schema"),
                condition=task.get("condition")
            )
            for task in tasks_config
        ]
    
    def get_active_tasks(self, context: Dict[str, Any]) -> List[TaskDefinition]:
        """
        获取当前激活的任务列表
        
        Args:
            context: 上下文数据（包含所有判断条件需要的变量）
            
        Returns:
            激活的任务列表
        """
        return [task for task in self.tasks if task.is_active(context)]
    
    def create_combined_schema(self, context: Dict[str, Any]) -> Type[BaseModel]:
        """
        根据激活的任务，动态生成组合 Schema
        
        这个方法等价于 core_tasks.py 中的动态 Schema 生成逻辑
        
        Args:
            context: 上下文数据
            
        Returns:
            组合后的 Pydantic Model 类
        """
        active_tasks = self.get_active_tasks(context)
        
        if not active_tasks:
            # 如果没有激活的任务，返回一个空 Schema
            return create_model("EmptySchema")
        
        # 构建 Schema 字段
        schema_fields = {}
        for task in active_tasks:
            schema_class = task.load_schema()
            schema_fields[task.output_field] = (schema_class, ...)
        
        # 动态创建组合 Schema（类似 core_tasks.py 中的 create_model）
        combined_schema = create_model("CombinedTasksSchema", **schema_fields)
        return combined_schema
    
    def get_tasks_metadata(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        获取激活任务的元数据（用于渲染任务描述）
        
        Args:
            context: 上下文数据
            
        Returns:
            任务元数据列表: [{"name": "...", "template": "...", "output_field": "..."}, ...]
        """
        active_tasks = self.get_active_tasks(context)
        return [
            {
                "name": task.name,
                "template": task.template,
                "output_field": task.output_field
            }
            for task in active_tasks
        ]
