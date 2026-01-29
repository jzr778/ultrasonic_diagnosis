import os
import sys
from pathlib import Path
from prompts_engine import AgentFactory, ContextBuilder
from prompts_engine import BaseAgent

def prompt_gen(context, prompt_config='chaosheng_wujian_avm'):
    # Agent 创建
    agent = AgentFactory.create(prompt_config)
    # Context 构建
    context = ContextBuilder.build_context(context=context,)
    # Prompt 组装
    prompt = agent.assemble(context)

    return prompt



if __name__ == "__main__":
    context = {
        "panoramic_1": [[300, 200]],
        "panoramic_2": [],
        "panoramic_3": [],
        "panoramic_4": [],
    }
    prompt = prompt_gen(context)