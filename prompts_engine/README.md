# Prompts Engine

一个基于 LangChain 的通用 Prompt 构建引擎，用于多模态 VLM (Vision Language Model) 场景分析。

## 📖 目录

- [核心特性](#核心特性)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [核心机制：Context 驱动的动态构建](#核心机制context-驱动的动态构建)
- [如何创建新的 Agent](#如何创建新的-agent)
- [目录结构](#目录结构)
- [最佳实践](#最佳实践)

## 🌟 核心特性

- **配置驱动**: 所有 Agent 行为通过 YAML 配置定义，无需修改核心代码
- **模板化**: 使用 Jinja2 模板管理所有 Prompt 内容，支持条件渲染和变量注入
- **解耦设计**: 业务逻辑与 Prompt Engine 完全分离
- **Schema 管理**: 统一的 Pydantic Schema 管理，确保输出结构一致性
- **动态组装**: 根据场景上下文动态激活任务和组件
- **可扩展**: 轻松添加新的场景、任务和分析模块

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        Business Layer                        │
│  (准备场景数据、生成可视化、格式化表格)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ Context Dict
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Prompt Engine                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Agent    │  │ Templates  │  │  Schemas   │            │
│  │  Factory   │  │  (Jinja2)  │  │ (Pydantic) │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└──────────────────────┬──────────────────────────────────────┘
                       │ Assembled Prompt + Images + Schema
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      VLM / LLM                               │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

1. **Agent Factory** (`engine/agent_factory.py`): 负责创建和管理 Agent 实例
2. **Base Agent** (`engine/base_agent.py`): Agent 基类，负责 Prompt 组装
3. **Context Builder** (`context/builder.py`): 构建业务层传递给引擎的 Context
4. **Templates**: Jinja2 模板文件，定义 Prompt 内容
5. **Schemas**: Pydantic 模型，定义输出结构
6. **Configs**: YAML 配置文件，定义 Agent 行为

## 🚀 快速开始

### 使用现有 Agent

```python
from prompts_engine.engine.agent_factory import AgentFactory
from prompts_engine.context.builder import ContextBuilder

# 1. 创建 Agent
agent = AgentFactory.create('aeb_agent')  # 或 'takeover_agent'

# 2. 构建 Context
context_builder = ContextBuilder()
context = context_builder.build_context(
    all_ego_info=ego_data,
    all_critical_info=critical_data,
    frame_labels=['Frame-1', 'Frame 0', 'Frame+1'],
    delta_seconds=0.1,
    camera_image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    is_vru=True,
    is_takeover=False,
)

# 3. 组装 Prompt
output = agent.assemble(context)

# 4. 使用输出
prompt_text = output['prompt_text']  # 文本 Prompt
images = output['images']            # 图像列表
```

## 🔧 核心机制：Context 驱动的动态构建

Prompts Engine 的核心优势在于通过 **Context** 驱动 Prompt 和 Schema 的动态构建。理解这一机制对于充分利用引擎的灵活性至关重要。

### 工作流程概览

```
Context (dict)
    ↓
    ├─→ [条件渲染] → Jinja2 Templates → 动态 Prompt 文本
    ├─→ [任务激活] → Task Filtering  → 激活的任务列表
    └─→ [Schema 注入] → Schema Generation → 动态输出结构
```

### 1. Context 驱动的 Prompt 条件渲染

#### 原理说明

Jinja2 模板通过读取 Context 中的标志位，动态决定 Prompt 的内容。这使得同一个模板可以根据不同场景生成不同的提示词。

#### 示例：VRU 意图分析模块

**Context 定义：**
```python
context = {
    "is_vru": True,           # VRU 场景标志
    "is_takeover": False,      # 非接管场景
    "has_trajectory_vis": True, # 有轨迹可视化
}
```

**模板内容：**`templates/layer3_analysis/vru_intent.jinja2`
```jinja2
#### **VRU 意图深度分析**

1. **识别具体物理行为**: 检查VRU是否存在非穿行意图信号...
2. **评估横穿风险与轨迹**: 观察VRU的身体朝向...
3. **判断是否察觉自车**: 评估VRU是否已察觉到自车...

{% if is_takeover %}
4. **轨迹图辅助意图判定**: 轨迹图提供了全局视角...
   - 轨迹连续性
   - 横穿意图确认
   - 规避行为识别
{% endif %}

5. **未来事实驱动的意图判定**: 利用已提供的未来帧...
```

**渲染结果：**
- 当 `is_takeover=False` 时，第 4 步"轨迹图辅助意图判定"不会出现
- 当 `is_takeover=True` 时，完整显示所有步骤

#### 常用条件标志

| Context 字段 | 类型 | 作用 | 示例 |
|-------------|------|------|------|
| `is_vru` | bool | 标识是否为 VRU 场景 | 决定使用 VRU 或 Vehicle 分析指导 |
| `is_takeover` | bool | 标识是否为接管场景 | 决定使用 AEB 或 Takeover 相关术语 |
| `has_trajectory_vis` | bool | 是否有轨迹可视化 | 决定是否显示轨迹分析任务 |
| `has_ego_intent` | bool | 是否启用自车意图分析 | 决定是否包含模块 C |
| `has_aeb_command` | bool | 是否有 AEB 命令数据 | 决定是否显示 AEB 相关提示 |

### 2. Context 驱动的任务动态激活

#### 原理说明

`BaseAgent` 根据 Context 中的标志位，动态筛选需要执行的任务。只有满足条件的任务才会被激活，从而生成对应的输出字段。

#### 配置示例：`configs/aeb_agent.yaml`

```yaml
tasks:
  # 始终激活的任务（无 condition）
  - name: "数据交叉验证"
    template: "layer4_tasks/sub_tasks/data_validation.jinja2"
    output_field: "data_validation"
    schema: "DataValidationSchema"
  
  # 条件激活的任务
  - name: "轨迹可视化图分析"
    template: "layer4_tasks/sub_tasks/trajectory_analysis.jinja2"
    output_field: "trajectory_visualization_analysis"
    schema: "TrajectoryVisualizationAnalysisSchema"
    condition: "has_trajectory_vis"  # 仅当 has_trajectory_vis=True 时激活
```

#### 激活逻辑（`engine/core_tasks_manager.py`）

```python
def get_active_tasks(self, context: Dict[str, Any]) -> List[TaskDefinition]:
    """根据 context 筛选激活的任务"""
    active_tasks = []
    
    for task in self.tasks:
        # 无条件 → 始终激活
        if not task.condition:
            active_tasks.append(task)
            continue
        
        # 检查条件 → 满足才激活
        condition_value = context.get(task.condition, False)
        if condition_value:
            active_tasks.append(task)
    
    return active_tasks
```

#### 实际效果

**场景 1：无轨迹可视化**
```python
context = {"has_trajectory_vis": False}
# → "trajectory_visualization_analysis" 任务不会激活
# → 输出 Schema 中不包含该字段
```

**场景 2：有轨迹可视化**
```python
context = {"has_trajectory_vis": True}
# → "trajectory_visualization_analysis" 任务被激活
# → Prompt 中包含该任务的说明
# → 输出 Schema 中动态添加 "trajectory_visualization_analysis" 字段
```

### 3. Context 驱动的动态 Schema 生成

#### 原理说明

通过 Pydantic v2 的 `Annotated` 和自定义 `ContextualDescription`，Schema 的字段描述可以根据 Context 动态变化。

#### 实现机制（`schemas/base.py`）

```python
from contextvars import ContextVar
from typing import Any, Dict
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue, GetJsonSchemaHandler
from typing_extensions import Annotated

# 全局 ContextVar，存储当前的运行时 Context
schema_context: ContextVar[Dict[str, Any]] = ContextVar('schema_context', default={})

class ContextualDescription:
    """根据 Context 动态选择描述文本"""
    
    def __init__(self, default: str, **conditions: str):
        self.default = default
        self.conditions = conditions  # 条件 → 描述映射
    
    def __get_pydantic_json_schema__(
        self, core_schema: Any, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        
        # 读取当前 Context
        current_context = schema_context.get()
        selected_description = self.default
        
        # 匹配条件，选择对应描述
        for condition_key, description_value in self.conditions.items():
            if current_context.get(condition_key):
                selected_description = description_value
                break
        
        json_schema['description'] = selected_description
        return json_schema
```

#### 使用示例（`schemas/data_validation.py`）

```python
from typing_extensions import Annotated
from .base import ContextualDescription

class DataValidationSchema(BaseModel):
    class OBBDistanceAccuracy(BaseModel):
        has_accuracy_issue: bool
        reported_distance: float
        
        visual_estimated_distance: Annotated[str, ContextualDescription(
            default="根据图像/视频视觉判断的实际间距（如：0.3-0.5m / 1.0-1.5m）",
            is_takeover="视觉/视频判断的间距，作为辅助参考。注意：如果关键目标匹配一致，则以结构化数据距离为准、视觉测距仅作为辅助参考。"
        )]
        
        reasoning: Annotated[str, ContextualDescription(
            default="当obb2obb_distance显示为0.00m或异常小值（<0.5m）时，必须结合图像视觉判断真实间距...",
            is_takeover="对于车辆接管场景，OBB数据通常较准确。即使视觉上感觉距离稍远，若OBB显示近距离，仍应以OBB数据为主要风险判断依据..."
        )]
```

#### Context 注入流程（`engine/base_agent.py`）

```python
def _render_output_format(self, dynamic_schema: Any, context: Dict[str, Any]) -> str:
    """渲染输出格式说明"""
    from ..schemas.base import schema_context
    
    # 1. 将 Context 注入到 ContextVar
    token = schema_context.set(context)
    
    try:
        # 2. 生成 JSON Schema（触发 ContextualDescription 的动态描述选择）
        schema_dict = dynamic_schema.model_json_schema()
    finally:
        # 3. 清理 ContextVar，避免污染
        schema_context.reset(token)
    
    # 4. 格式化 Schema 并渲染成 Prompt
    field_structure_details = SchemaFormatter.format_schema(schema_dict)
    template = self.jinja_env.get_template('layer5_output/format_instructions.jinja2')
    return template.render(field_structure_details=field_structure_details, **context)
```

#### 实际效果对比

**场景 1：AEB 场景 (`is_takeover=False`)**
```json
{
  "visual_estimated_distance": {
    "type": "string",
    "description": "根据图像/视频视觉判断的实际间距（如：0.3-0.5m / 1.0-1.5m）"
  }
}
```

**场景 2：Takeover 场景 (`is_takeover=True`)**
```json
{
  "visual_estimated_distance": {
    "type": "string",
    "description": "视觉/视频判断的间距，作为辅助参考。注意：如果关键目标匹配一致，则以结构化数据距离为准、视觉测距仅作为辅助参考。"
  }
}
```

### 4. 完整示例：从 Context 到输出

#### 输入 Context

```python
context = {
    # 场景标志
    "is_vru": True,
    "is_takeover": False,
    "has_trajectory_vis": True,
    "has_ego_intent": True,
    "has_aeb_command": False,
    
    # 数据内容
    "case_id": "test_001",
    "camera_images": ["img1.jpg", "img2.jpg"],
    "data_table": "... 格式化的数据表格 ...",
    "ego_history_plot": "ego_plot.png",
    # ... 更多数据字段
}
```

#### 引擎处理流程

```python
# 1. BaseAgent 接收 Context
agent = AgentFactory.create('aeb_agent')
output = agent.assemble(context)

# 2. 模板条件渲染
# - VRU 意图分析模块被包含（is_vru=True）
# - 自车意图分析模块被包含（has_ego_intent=True）
# - AEB 命令提示不显示（has_aeb_command=False）

# 3. 任务动态激活
# - 轨迹可视化分析任务被激活（has_trajectory_vis=True）
# - Perfect Time 任务不激活（has_perfect_time_result=False，默认）

# 4. Schema 动态生成
# - OBBDistanceAccuracy.reasoning 使用 AEB 场景描述（is_takeover=False）
# - 输出包含 trajectory_visualization_analysis 字段

# 5. 最终输出
output = {
    'prompt_text': "... 完整的动态生成的 Prompt ...",
    'images': [ImageInput(...), ImageInput(...)],
}
```

#### 输出 Prompt 片段

```
=== System Instructions ===
#### **核心前提 (Core Premise)**
您分析的所有数据均来自一个**AEB影子模式 (Shadow Mode) 记录的潜在风险**事件...

=== Analysis Task ===
#### **可复用分析模块 (Reusable Analysis Modules)**
*   **`模块 A: VRU 意图深度分析 (VRU_INTENT_ANALYSIS_MODULE)`**: ...
*   **`模块 C: 自车意图分析 (EGO_VEHICLE_INTENT_MODULE)`**: ...

#### **核心分析任务 (Core Analysis Tasks)**
**任务 1: 数据交叉验证** ...
**任务 2: 场景分类** ...
**任务 3: 轨迹可视化图分析** ...  ← 动态激活
**任务 4: 完整分析** ...
...

#### **输出格式说明**
{
  "data_validation": { ... },
  "trajectory_visualization_analysis": { ... },  ← 动态字段
  ...
}
```

### 5. 关键设计要点

#### ✅ 优势

1. **灵活性**: 同一套模板和配置支持多种场景组合
2. **可维护性**: 场景差异通过条件标志控制，无需复制代码
3. **类型安全**: Schema 动态生成但类型检查完整
4. **性能**: 按需激活，不生成不需要的内容

#### ⚠️ 注意事项

1. **Context 完整性**: 确保所有模板需要的标志都在 Context 中定义
2. **条件一致性**: 配置文件中的 `condition` 必须与 Context 字段匹配
3. **Schema 注入时机**: `schema_context` 必须在 Schema 生成前设置，之后清理
4. **默认值**: 为可选字段提供合理的默认值（`context.get(key, default)`）

### 6. 调试技巧

#### 查看激活的任务

```python
from prompts_engine.engine.core_tasks_manager import CoreTasksManager

config = agent.config
manager = CoreTasksManager(config['tasks'])
active_tasks = manager.get_active_tasks(context)

print("激活的任务:")
for task in active_tasks:
    print(f"  - {task.name} → {task.output_field}")
```

#### 查看动态 Schema

```python
dynamic_schema = manager.create_combined_schema(context)
schema_json = dynamic_schema.model_json_schema()

import json
print(json.dumps(schema_json, indent=2, ensure_ascii=False))
```

#### 验证条件渲染

```python
# 测试不同 Context
contexts = [
    {"is_vru": True, "is_takeover": False},
    {"is_vru": False, "is_takeover": True},
]

for ctx in contexts:
    output = agent.assemble(ctx)
    print(f"\nContext: {ctx}")
    print(f"Prompt 长度: {len(output['prompt_text'])}")
    print(f"包含 VRU 分析: {'VRU_INTENT_ANALYSIS_MODULE' in output['prompt_text']}")
```

---

## 📝 如何创建新的 Agent

创建一个新的 Agent 需要以下 **5 个步骤**：

### Step 1: 定义配置文件

在 `configs/` 目录下创建新的 YAML 配置文件，例如 `configs/my_new_agent.yaml`：

```yaml
# 基础信息
version: "1.0.0"
description: "我的新 Agent 描述"

# System 消息组件（AI 角色定义）
system_components:
  - template: "layer1_foundation/role_context.jinja2"
    description: "定义 AI 角色和任务"
  
  - template: "layer1_foundation/terminology.jinja2"
    description: "定义术语"
  
  - template: "layer1_foundation/expert_principles.jinja2"
    description: "定义分析原则"

# User 消息组件（输入和分析模块）
user_components:
  - template: "layer2_input/input_description.jinja2"
    description: "描述输入数据"
  
  # 可选组件（带条件激活）
  - template: "layer3_analysis/vru_intent.jinja2"
    description: "VRU 意图分析"
    condition: "is_vru"  # 仅在 is_vru=True 时激活

# 任务列表（定义输出结构）
tasks:
  - name: "数据验证 (Data Validation)"
    template: "layer4_tasks/sub_tasks/data_validation.jinja2"
    output_field: "data_validation"
    schema: "DataValidationSchema"
  
  - name: "场景分类 (Scene Classification)"
    template: "layer4_tasks/sub_tasks/scene_classification.jinja2"
    output_field: "scene_classification"
    schema: "SceneClassificationSchema"
  
  # 更多任务...

# 输出格式
output_format:
  template: "layer5_output/format_instructions.jinja2"
```

### Step 2: 创建模板文件（如需要）

如果需要新的 Prompt 内容，在 `templates/` 目录下创建 Jinja2 模板：

**示例：`templates/layer3_analysis/my_custom_module.jinja2`**

```jinja2
#### **自定义分析模块**

这里是模块的 Prompt 内容。

**分析步骤**:
1. 第一步：观察场景特征
2. 第二步：评估风险等级
3. 第三步：给出结论

{% if is_special_case %}
**特殊场景处理**：
- 针对特殊场景的额外指导
{% endif %}

**输出要求**：
- 必须给出明确的结论
- 提供详细的推理过程
```

### Step 3: 定义输出 Schema（如需要）

如果有新的输出结构，在 `schemas/` 目录下创建 Pydantic 模型：

**示例：`schemas/my_custom_schema.py`**

```python
"""自定义模块的输出 Schema"""

from typing import Literal
from pydantic import Field, BaseModel


class MyCustomSchema(BaseModel):
    """自定义分析结果"""
    
    risk_level: Literal["HIGH", "MEDIUM", "LOW", "NONE"] = Field(
        ...,
        description="风险等级评估"
    )
    
    reasoning: str = Field(
        ...,
        description="详细的分析推理过程"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="置信度（0-1）"
    )
```

然后在 `schemas/__init__.py` 中导出：

```python
from .my_custom_schema import MyCustomSchema

__all__ = [
    # ... 其他 schemas
    "MyCustomSchema",
]
```

### Step 4: 更新 Context Builder（如需要）

如果需要新的 Context 字段，修改 `context/builder.py` 的 `build_context` 方法：

```python
def build_context(
    cls,
    # ... 现有参数
    my_custom_field: bool = False,  # 新增参数
) -> Dict[str, Any]:
    context = {}
    
    # 添加新的标志
    context["has_custom_feature"] = my_custom_field
    
    # ... 其他逻辑
    
    return context
```

### Step 5: 测试新 Agent

创建测试脚本验证 Agent 功能：

```python
from prompts_engine.engine.agent_factory import AgentFactory
from prompts_engine.context.builder import ContextBuilder

# 创建新 Agent
agent = AgentFactory.create('my_new_agent')

# 构建测试 Context
context_builder = ContextBuilder()
context = context_builder.build_context(
    all_ego_info=[...],
    all_critical_info=[...],
    frame_labels=['Frame 0'],
    delta_seconds=0.1,
    camera_image_paths=['test.jpg'],
)

# 组装 Prompt
output = agent.assemble(context)

# 验证输出
assert len(output['prompt_text']) > 0
assert len(output['images']) > 0
print(f"✅ Agent 创建成功！")
print(f"Prompt 长度: {len(output['prompt_text'])} 字符")
```

## 📁 目录结构

```
prompts_engine/
├── README.md                    # 本文件
├── __init__.py                  # 包初始化
├── test_engine.py              # 测试套件
│
├── configs/                     # Agent 配置文件
│   ├── aeb_agent.yaml          # AEB 场景 Agent
│   └── takeover_agent.yaml     # 接管场景 Agent
│
├── engine/                      # 核心引擎
│   ├── base_agent.py           # Agent 基类
│   ├── agent_factory.py        # Agent 工厂
│   ├── core_tasks_manager.py   # 任务管理器
│   ├── schema_formatter.py     # Schema 格式化器
│   └── output_payload.py       # 输出数据结构
│
├── context/                     # Context 构建
│   ├── builder.py              # Context Builder
│   └── formatter.py            # 数据格式化器
│
├── schemas/                     # Pydantic Schemas
│   ├── __init__.py
│   ├── base.py                 # 基础类（动态 Schema）
│   ├── data_validation.py
│   ├── scene_classification.py
│   ├── risk_assessment.py
│   └── ...                     # 其他 schemas
│
└── templates/                   # Jinja2 模板
    ├── layer1_foundation/      # 基础层（角色、术语、原则）
    │   ├── role_context.jinja2
    │   ├── takeover_role_context.jinja2
    │   ├── terminology.jinja2
    │   ├── expert_principles.jinja2
    │   └── takeover_expert_principles.jinja2
    │
    ├── layer2_input/            # 输入层
    │   └── input_description.jinja2
    │
    ├── layer3_analysis/         # 分析模块层
    │   ├── vru_intent.jinja2
    │   ├── ego_vehicle_intent.jinja2
    │   ├── aeb_necessity.jinja2
    │   ├── noasafety_ego_vehicle_intent.jinja2
    │   └── noasafety_necessity.jinja2
    │
    ├── layer4_tasks/            # 任务层
    │   └── sub_tasks/
    │       ├── data_validation.jinja2
    │       ├── scene_classification.jinja2
    │       ├── risk_assessment.jinja2
    │       └── ...             # 其他任务
    │
    └── layer5_output/           # 输出格式层
        └── format_instructions.jinja2
```

## 💡 最佳实践

### 1. 模板设计原则

- **单一职责**: 每个模板只负责一个特定的分析模块或任务
- **条件渲染**: 使用 Jinja2 条件语句处理不同场景
- **变量命名**: 使用清晰、描述性的变量名
- **注释说明**: 在复杂逻辑处添加注释

### 2. Schema 设计原则

- **明确类型**: 使用 Literal 限定枚举值
- **详细描述**: 每个字段都要有清晰的 description
- **合理嵌套**: 使用嵌套类组织复杂结构
- **验证约束**: 使用 Pydantic 验证器（ge、le、regex 等）

### 3. 动态 Schema（根据 Context 变化）

使用 `schemas/base.py` 中的 `ContextualDescription` 实现动态描述：

```python
from typing_extensions import Annotated
from prompts_engine.schemas.base import ContextualDescription

class MySchema(BaseModel):
    distance: Annotated[float, ContextualDescription(
        default="与目标的距离（米）",
        is_takeover="接管时刻与目标的距离（米）",
        is_vru="VRU 与自车的距离（米）"
    )]
```

### 4. 配置文件最佳实践

- **描述清晰**: 每个组件添加 description 说明其作用
- **合理分层**: System 组件（角色定义）→ User 组件（输入+分析）→ Tasks（输出）
- **条件激活**: 使用 condition 字段实现动态组件加载
- **版本管理**: 记录 version 字段便于追踪变更

### 5. Context 设计原则

- **布尔标志**: 使用 `is_*`、`has_*` 命名布尔变量
- **数据分离**: 原始数据（列表）与格式化文本（字符串）分开
- **完整性**: 确保所有模板需要的变量都在 Context 中
- **可选字段**: 对于可选数据，提供默认值或 None

### 6. 测试策略

- **单元测试**: 测试每个 Schema 的验证逻辑
- **集成测试**: 测试完整的 Agent 创建和 Prompt 组装
- **场景测试**: 测试不同 Context 下的条件渲染
- **回归测试**: 确保修改不影响现有功能

## 🔍 调试技巧

### 查看生成的 Prompt

```python
output = agent.assemble(context)
with open("debug_prompt.txt", "w") as f:
    f.write(output['prompt_text'])
```

### 检查 Context 内容

```python
import json
print(json.dumps(context, indent=2, default=str))
```

### 验证 Schema 结构

```python
from prompts_engine.engine.core_tasks_manager import CoreTasksManager

manager = CoreTasksManager(config['tasks'])
dynamic_schema = manager.create_combined_schema(context)
print(dynamic_schema.model_json_schema())
```

## 📚 参考资料

- [LangChain 文档](https://python.langchain.com/)
- [Jinja2 模板语法](https://jinja.palletsprojects.com/)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [REFACTOR_DOCUMENTATION.md](./REFACTOR_DOCUMENTATION.md) - 详细的重构文档

## 🤝 贡献指南

1. 保持代码风格一致
2. 添加充分的注释和文档
3. 新增功能需要添加测试
4. 提交前运行 `test_engine.py` 确保所有测试通过

## 📝 更新日志

### v1.0.0 (2025-01)
- ✅ 初始版本
- ✅ 支持 AEB 和 Takeover 两种场景
- ✅ 完整的模板系统
- ✅ 动态 Schema 支持
- ✅ 条件渲染功能
