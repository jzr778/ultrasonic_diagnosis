"""
BaseAgent - 通用 Agent 基类

这是 Prompt Engine 的核心，负责：
1. 加载 YAML 配置
2. 渲染 Jinja2 模板
3. 动态生成 Pydantic Schema
4. 组装最终 Prompt
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
from jinja2 import Environment, FileSystemLoader
from pydantic import create_model
import yaml
from .core_tasks_manager import CoreTasksManager


class BaseAgent:
    """
    通用 Agent 基类（纯配置驱动，无业务逻辑）
    
    工作流程：
    1. 从 config.yaml 加载配置
    2. 根据 context 渲染模板
    3. 生成动态 Schema
    4. 返回完整的 Prompt Payload
    """
    
    def __init__(self, config_path: Path, agent_name: str = None):
        """
        Args:
            config_path: Agent 配置文件路径（yaml 文件）
            agent_name: Agent 名称（可选，用于标识，如果不提供则从配置文件名推断）
        """
        self.config_path = Path(config_path)
        self.agent_name = agent_name or self.config_path.stem  # 从文件名推断（不含扩展名）
        self.config = self._load_config()
        
        # 初始化 Jinja2 环境（使用顶层的 templates 目录）
        # config_path.parent 是 configs/ 目录，再向上一级 .parent 到达 prompts_engine/
        template_dir = self.config_path.parent.parent / "templates"
        if not template_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {template_dir}")
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        
        # 初始化 CoreTasksManager
        tasks_config = self.config.get('tasks', [])
        self.tasks_manager = CoreTasksManager(tasks_config)
    
    def _load_config(self) -> Dict:
        """加载 Agent 配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def assemble(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        组装完整的 Prompt
        
        Args:
            context: 业务层准备好的完整上下文字典
            
        Returns:
            {
                "prompt_text": str,
                "images": List[ImageInput],
                "pydantic_model": Type[BaseModel]
            }
        """
        # 1. 渲染 System 组件
        system_parts = self._render_components(context)
        
        # 2. 渲染 Inputs（输入数据）
        inputs_text = self._render_inputs(context)
        
        # 3. 渲染任务
        tasks_text = self._render_tasks(context)

        # 4. 渲染输出格式说明
        output_text = self._render_output_format(context)
        
        # 5. 组装最终 Prompt
        prompt_part = [system_parts, inputs_text, tasks_text, output_text]
        prompt_final = "\n".join(prompt_part)
        
        return prompt_final
    
    def _render_components(self, context: Dict[str, Any]) -> List[str]:
        """
        渲染组件列表
        
        Args:
            components: 组件配置列表
            context: 上下文数据
            
        Returns:
            渲染后的文本列表
        """
        results = []
        components = self.config.get('system_components', [])
        
        for comp_config in components:
            # 渲染模板
            template_path = comp_config.get("template")
            if not template_path:
                continue
            
            try:
                template = self.jinja_env.get_template(template_path)
                rendered = template.render(**context)
                if rendered.strip():  # 只添加非空内容
                    results.append(rendered)
            except Exception as e:
                print(f"Warning: Failed to render template {template_path}: {e}")
        
        return "\n".join(results)
    
    def _render_inputs(self, context: Dict[str, Any]) -> str:
        """
        渲染输入数据组件（类似 _render_tasks）
        
        Args:
            context: 上下文数据
            
        Returns:
            渲染后的输入描述文本
        """
        inputs_config = self.config.get('inputs', [])
        if not inputs_config:
            return ""
        
        # 1. 渲染每个 input 的模板（带自动编号）
        input_parts = []
        for i, input_config in enumerate(inputs_config):
            try:
                template = self.jinja_env.get_template(input_config["template"])
                rendered = template.render(**context)
                
                # 添加标题和序号
                if rendered.strip():
                    input_text = f"### {i + 1}. {input_config['name']}\n{rendered}"
                    input_parts.append(input_text)
            except Exception as e:
                print(f"Warning: Failed to render input {input_config.get('name', 'unknown')}: {e}")
        
        if not input_parts:
            return ""
        
        # 2. 组装输入数据部分
        return "## 输入数据\n" + "\n".join(input_parts)
    
    def _render_tasks(self, context: Dict[str, Any]) -> str:
        """
        使用 CoreTasksManager 渲染任务并生成动态 Schema
        
        Args:
            context: 上下文数据
            
        Returns:
            (任务描述文本, 动态 Schema 类)
        """
        # 1. 获取激活的任务元数据
        active_tasks_meta = self.tasks_manager.get_tasks_metadata(context)
        
        if not active_tasks_meta:
            return ""
        
        # 2. 渲染每个任务的模板
        task_descriptions = []
        for i, task_meta in enumerate(active_tasks_meta):
            try:
                template = self.jinja_env.get_template(task_meta["template"])
                rendered = template.render(**context)
                
                task_text = (
                    f"### 任务{i + 1}. {task_meta['name']}流程如下：\n"
                    f"{rendered}"
                )
                task_descriptions.append(task_text)
            except Exception as e:
                print(f"Warning: Failed to render task {task_meta['name']}: {e}")
        
        if not task_descriptions:
            return ""
        
        # 3. 组装任务内容
        tasks_content = "\n".join(task_descriptions)
        
        return tasks_content
    
    def _render_output_format(self, context: Dict[str, Any]) -> str:
        """
        渲染输出格式说明
        
        Args:
            dynamic_schema: 动态生成的 Schema
            context: 上下文数据
            
        Returns:
            渲染后的格式说明文本
        """
        # 检查是否配置了输出格式模板
        output_format_config = self.config.get('output_format')
        if not output_format_config:
            return ""
        
        template_path = output_format_config.get('template')
        if not template_path:
            return ""
        
        try:
            # 渲染模板（传递完整 context，让模板自己处理条件逻辑）
            template = self.jinja_env.get_template(template_path)
            rendered = template.render(**context)
            return rendered
        except Exception as e:
            print(f"Warning: Failed to render output format: {e}")
            return ""
