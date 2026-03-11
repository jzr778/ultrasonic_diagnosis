import base64
import json
from PIL import Image
import io
import re
import os
import sys
from openai import OpenAI
from typing import Dict, List, Any
from collections import Counter

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"图像编码错误: {e}")
        return None

def encode_image_array_to_base64(image_array):
    """将numpy图像数组转换为base64编码"""
    try:
        image_rgb = image_array
        # 转换为PIL图像
        img = Image.fromarray(image_rgb.astype('uint8'))
        # 转换为base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=100)
        image_bytes = buffer.getvalue()

        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"图像数组编码错误: {e}")
        return None

def call_qwen_model_with_images(image_list, question, model):
    """
    调用Qwen模型进行多图像问答，支持重试机制

    Args:
        image_inputs: 可以是图像文件路径列表或numpy数组列表
        question: 问题文本
        max_retries: 最大重试次数
    """
    # 配置客户端 - 使用阿里云百炼的端点
    client = OpenAI(
        api_key=config.VLM_API_KEY,
        base_url=config.VLM_BASE_URL,
    )

    # 构建消息内容
    content = [{"type": "text", "text": question}]

    for key, value in image_list.items():
        base64_image = encode_image_array_to_base64(value)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # 构建请求数据
    messages = [{
        "role": "user", "content": content
    }]
    # 调用
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用失败: {e}"

def extract_json_from_text(text):
    """从文本中提取JSON内容"""
    # 首先尝试直接解析
    return json.loads(text)


def majority_coordinate_voting_with_empty(results: Dict[str, Dict[str, List]]) -> List[List[int]]:
    """
    多数坐标投票算法（考虑空列表作为特殊坐标）
    """
    total_models = len(results)
    all_coordinates = []
    empty_count = 0

    # 收集所有模型的输出
    for model_name, model_output in results.items():
        positions = model_output.get('positions', [])

        if not positions:  # 空列表
            empty_count += 1
        else:
            # 处理非空坐标
            for position in positions:
                coordinates = position.get('pixel_coordinates', [])
                if isinstance(coordinates, list) and len(coordinates) == 2:
                    coord = tuple(coordinates)
                    all_coordinates.append(coord)

    # 情况1：空列表构成多数
    if empty_count > total_models / 2:
        return {'positions': []}

    # 情况2：统计非空坐标的出现次数
    coord_counter = Counter(all_coordinates)

    # 找出所有出现次数超过一半的坐标
    majority_coords = []
    for coord, count in coord_counter.items():
        if count > total_models / 2:
            majority_coords.append(list(coord))

    return {'positions': majority_coords}

def analyze_scenario_from_images(image_list, prompt, model_list):
    results = {}
    for model in model_list:
        result = call_qwen_model_with_images(image_list, prompt, model)
        # 尝试解析JSON结果
        json_result = None
        try:
            # 尝试直接解析
            json_result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            # 如果直接解析失败，尝试从代码块中提取
            if isinstance(result, str):
                # 尝试匹配 ```json ... ``` 格式
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', result, re.DOTALL)
                if match:
                    try:
                        json_result = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass
                else:
                    # 尝试匹配纯JSON内容（没有代码块）
                    try:
                        # 查找可能的JSON开头和结尾
                        json_pattern = r'\{.*\}|\[.*\]'
                        matches = re.findall(json_pattern, result, re.DOTALL)
                        for match in matches:
                            try:
                                json_result = json.loads(match)
                                break
                            except json.JSONDecodeError:
                                continue
                    except:
                        pass

        # 存储结果
        if json_result:
            results[model] = json_result
        else:
            results[model] = None
    results = {k: v for k, v in results.items() if v is not None}
    if len(results) == 0:
        return None

    results = majority_coordinate_voting_with_empty(results)

    return results


if __name__ == "__main__":
    results = {
        "model1": {
            "positions": [
                {"pixel_coordinates": [100, 200]},
                {"pixel_coordinates": [300, 400]}
            ]
        },
        "model2": {
            "positions": [
                {"pixel_coordinates": [100, 200]},
                {"pixel_coordinates": [300, 400]}
            ]
        },
        "model3": {
            "positions": [
                {"pixel_coordinates": [100, 200]},
                {"pixel_coordinates": [300, 400]},
                {"pixel_coordinates": [500, 600]}
            ]
        }
    }

    a = majority_coordinate_voting_with_empty(results)
    print(a)