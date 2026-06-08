import base64
import json
from PIL import Image
import io
import re
import os
import sys
from typing import Any, Dict, List
from collections import Counter

import requests
from openai import OpenAI

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

_auto_model_cache = {}


def _use_vertex_api() -> bool:
    style = getattr(config, "VLM_API_STYLE", "openai")
    if style == "vertex":
        return True
    return "vertex" in (config.VLM_BASE_URL or "").lower()


def _resolve_model_name(model, client=None):
    """model 为 auto 时：vertex 用 VLM_MODEL；openai 则 list 首个可用模型。"""
    if model != "auto":
        return model
    if _use_vertex_api():
        return getattr(config, "VLM_MODEL", None) or "gemini-3.1-pro-preview"
    cache_key = (config.VLM_API_KEY, config.VLM_BASE_URL, "openai")
    if cache_key not in _auto_model_cache:
        if client is None:
            client = OpenAI(
                api_key=config.VLM_API_KEY,
                base_url=config.VLM_BASE_URL,
            )
        models = client.models.list()
        _auto_model_cache[cache_key] = models.data[0].id
    return _auto_model_cache[cache_key]


def _vertex_generate_urls(model: str) -> List[str]:
    bases = [config.VLM_BASE_URL]
    alt = getattr(config, "VLM_BASE_URL_ALT", "") or ""
    if alt and alt.rstrip("/") != (config.VLM_BASE_URL or "").rstrip("/"):
        bases.append(alt)
    urls = []
    for base in bases:
        if not base:
            continue
        b = base.rstrip("/")
        urls.append(f"{b}/models/{model}:generateContent")
    return urls


def _vertex_parts_from_images(image_list: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"text": question}]
    for _key, value in image_list.items():
        base64_image = encode_image_array_to_base64(value)
        if not base64_image:
            continue
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image,
                }
            }
        )
    return parts


def _parse_vertex_response(payload: Dict[str, Any]) -> str:
    if payload.get("error"):
        err = payload["error"]
        if isinstance(err, dict):
            return f"调用失败: {err.get('message', err)}"
        return f"调用失败: {err}"
    texts: List[str] = []
    for cand in payload.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            t = part.get("text")
            if t:
                texts.append(t)
    return "\n".join(texts).strip()


def call_vertex_model_with_images(
    image_list: Dict[str, Any],
    question: str,
    model: str,
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 8192,
    timeout: float = 180.0,
) -> str:
    """七牛 bypass Vertex generateContent（Bearer Token）。"""
    resolved_model = _resolve_model_name(model)
    body = {
        "contents": [
            {
                "role": "user",
                "parts": _vertex_parts_from_images(image_list, question),
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    headers = {
        "Authorization": f"Bearer {config.VLM_API_KEY}",
        "Content-Type": "application/json",
    }
    last_err = ""
    for url in _vertex_generate_urls(resolved_model):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            try:
                payload = resp.json()
            except ValueError:
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                continue
            if resp.status_code >= 400:
                last_err = _parse_vertex_response(payload) or f"HTTP {resp.status_code}"
                continue
            text = _parse_vertex_response(payload)
            if text.startswith("调用失败:"):
                last_err = text
                continue
            return text
        except requests.RequestException as e:
            last_err = str(e)
    return f"调用失败: {last_err or 'vertex generateContent 无可用响应'}"


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
    多图 VLM 调用。VLM_API_STYLE=vertex 时走 generateContent，否则 OpenAI 兼容接口。

    Args:
        image_list: 图像名 -> numpy 数组
        question: 问题文本
        model: 模型名或 auto
    """
    if _use_vertex_api():
        return call_vertex_model_with_images(image_list, question, model)

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
        resolved_model = _resolve_model_name(model, client)
        response = client.chat.completions.create(
            model=resolved_model,
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
    """调用多个模型分析图像，返回投票结果。

    Returns:
        成功时返回 {"positions": [...]};
        全部失败时返回 {"error": "...", "raw_responses": {...}} 供调用方记录日志。
    """
    results = {}
    raw_responses = {}
    for model in model_list:
        result = call_qwen_model_with_images(image_list, prompt, model)
        raw_responses[model] = result
        json_result = None
        try:
            json_result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            if isinstance(result, str):
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', result, re.DOTALL)
                if match:
                    try:
                        json_result = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass
                else:
                    try:
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

        if json_result:
            results[model] = json_result
        else:
            results[model] = None
    results = {k: v for k, v in results.items() if v is not None}
    if len(results) == 0:
        failed_details = "; ".join(
            f"[{m}] {str(r)[:200]}" for m, r in raw_responses.items()
        )
        return {"error": failed_details, "raw_responses": raw_responses}

    voted = majority_coordinate_voting_with_empty(results)
    voted["_raw_responses"] = raw_responses

    return voted


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