#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞书项目 (Feishu Project) 视图数据导出工具

本脚本用于提取飞书项目中特定视图 (View) 下的所有工作项数据，
并自动将底层字段 ID 转换为易读的字段名称，最终导出为 JSON 文件。

使用前请确保环境变量中配置了以下飞书插件凭证：
- FEISHU_PLUGIN_ID
- FEISHU_PLUGIN_SECRET
- FEISHU_USER_KEY
"""

import os
import sys
import json
import argparse
import urllib.request
from urllib.error import HTTPError

# 飞书 API 基础配置
BASE_URL = "https://project.feishu.cn/open_api"

def http_json(method, url, headers=None, body=None, timeout=30):
    """通用的 HTTP JSON 请求辅助函数"""
    headers = headers or {}
    headers.setdefault("Content-Type", "application/json")
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
    except HTTPError as err:
        payload = err.read().decode("utf-8")
        
    try:
        res = json.loads(payload)
        if res.get("err_code") not in (0, "0", None):
            print(f"API 警告: {res.get('err_msg', res)}", file=sys.stderr)
        return res
    except json.JSONDecodeError:
        raise RuntimeError(f"API 返回非 JSON：{payload[:200]}")


def get_plugin_token(plugin_id, plugin_secret):
    """获取飞书项目插件 Token"""
    url = f"{BASE_URL}/authen/plugin_token"
    body = {"plugin_id": plugin_id, "plugin_secret": plugin_secret, "type": 0}
    payload = http_json("POST", url, headers=None, body=body)
    token = payload.get("data", {}).get("token")
    if not token:
        raise RuntimeError(f"无法获取插件 token: {payload}")
    return token


def fetch_fix_view_ids(headers, project_key, view_id, page_size=200, max_pages=50):
    """获取指定视图下的所有工作项 ID"""
    ids = []
    page_num = 1
    while page_num <= max_pages:
        url = f"{BASE_URL}/{project_key}/fix_view/{view_id}?page_size={page_size}&page_num={page_num}"
        payload = http_json("GET", url, headers)
        data = payload.get("data", {})
        
        raw_ids = []
        if isinstance(data, dict):
            raw_ids = data.get("work_item_id_list", []) or []
            
        ids.extend([int(i) for i in raw_ids if isinstance(i, (int, str))])
        
        pagination = payload.get("pagination", {})
        total = pagination.get("total") if isinstance(pagination, dict) else None
        
        if isinstance(total, int):
            if page_num * page_size >= total:
                break
        else:
            if not raw_ids or len(raw_ids) < page_size:
                break
        page_num += 1
    return ids


def fetch_work_item_types(headers, project_key):
    """获取项目下所有工作项类型"""
    payload = http_json("GET", f"{BASE_URL}/{project_key}/work_item/all-types", headers)
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [t.get("type_key") for t in data if isinstance(t, dict) and t.get("type_key")]


def resolve_work_item_type_key(headers, project_key, sample_id, candidate_types):
    """通过探测首个 ID，确定该视图下工作项的具体类型"""
    for type_key in candidate_types:
        items = query_work_items_by_ids(headers, project_key, type_key, [sample_id])
        if items:
            return type_key
    return None


def query_work_items_by_ids(headers, project_key, type_key, item_ids):
    """批量查询工作项详情"""
    if not item_ids:
        return []
    body = {"work_item_ids": item_ids}
    url = f"{BASE_URL}/{project_key}/work_item/{type_key}/query"
    payload = http_json("POST", url, headers, body)
    data = payload.get("data")
    if isinstance(data, list):
        return [i for i in data if isinstance(i, dict)]
    return []


def fetch_field_meta(headers, project_key):
    """获取项目中所有字段的映射元数据 (field_key -> field_name)"""
    payload = http_json("POST", f"{BASE_URL}/{project_key}/field/all", headers, {})
    data = payload.get("data", [])
    mapping = {}
    if isinstance(data, list):
        for field in data:
            if not isinstance(field, dict):
                continue
            key = field.get("field_key") or field.get("key") or field.get("fieldKey")
            name = field.get("name") or field.get("field_name") or field.get("fieldName")
            if key and name:
                mapping[str(key)] = str(name)
    return mapping


def normalize_value(value):
    """规范化字段值，将复杂的嵌套字典结构展平为字符串或列表"""
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("plain_text", "text", "label", "name", "display_name", "title", "value"):
            v = value.get(key)
            if isinstance(v, (str, int, float)):
                return str(v)
        for key in ("users", "items", "members"):
            v = value.get(key)
            if isinstance(v, list):
                return [normalize_value(i) for i in v if i is not None]
        if "url" in value and isinstance(value.get("url"), str):
            return value.get("url")
        return value
    if isinstance(value, list):
        out = []
        for v in value:
            nv = normalize_value(v)
            if isinstance(nv, list):
                out.extend(nv)
            elif nv is not None and nv != "":
                out.append(nv)
        return out
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


def format_item_details(detail, field_name_map):
    """格式化工作项，组装易读的命名属性"""
    fields = detail.get("fields", [])
    named_fields = {}
    
    for f in fields:
        if not isinstance(f, dict):
            continue
            
        key = f.get("field_key")
        name = field_name_map.get(key, key)
        
        val = normalize_value(f.get("field_value"))
        if val in (None, "", [], {}):
            continue
            
        if name in named_fields:
            if isinstance(named_fields[name], list):
                if isinstance(val, list):
                    named_fields[name].extend(val)
                else:
                    named_fields[name].append(val)
            else:
                named_fields[name] = [named_fields[name], val]
        else:
            named_fields[name] = val
            
    return named_fields


def main():
    parser = argparse.ArgumentParser(description="导出飞书项目特定视图下的所有工作项数据。")
    parser.add_argument("-p", "--project-key", required=True, help="飞书项目的 Project Key (例如: iffcom)")
    parser.add_argument("-v", "--view-id", required=True, help="飞书视图的 View ID (可以从 URL 中获取)")
    parser.add_argument("-o", "--output", default="feishu_id_list.json", help="导出的 JSON 文件路径")
    parser.add_argument("--feishu-id-only", action="store_true", 
                        help="只输出飞书 ID 列表（从 name 字段中提取的数字 ID）")
    
    args = parser.parse_args()

    # 硬编码的飞书凭证
    plugin_id = "MII_64EDCCED5EC38003"
    plugin_secret = "F0B574D7270754A7A4BF4EB60FEBD5C4"
    user_key = "7343719510022553604"


    print("1. 正在获取插件 Token...")
    try:
        plugin_token = get_plugin_token(plugin_id, plugin_secret)
    except Exception as e:
        print(f"获取 Token 失败: {e}")
        sys.exit(1)

    headers = {
        "Content-Type": "application/json",
        "X-PLUGIN-TOKEN": plugin_token,
        "X-USER-KEY": user_key
    }

    print(f"2. 正在获取视图 [{args.view_id}] 下的工作项列表...")
    try:
        item_ids = fetch_fix_view_ids(headers, args.project_key, args.view_id)
        print(f"   -> 发现 {len(item_ids)} 个工作项。")
    except Exception as e:
        print(f"获取视图列表失败: {e}")
        sys.exit(1)

    if not item_ids:
        print("视图为空，退出执行。")
        sys.exit(0)

    if args.feishu_id_only:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(item_ids, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 导出成功！共 {len(item_ids)} 个 ID 已保存至: {args.output}")
        return

    print("3. 正在解析工作项类型与字段元数据...")
    candidate_types = fetch_work_item_types(headers, args.project_key)
    resolved_type = resolve_work_item_type_key(headers, args.project_key, item_ids[0], candidate_types)
    
    if not resolved_type:
        print("无法解析该视图下的工作项类型，请检查项目及视图权限。")
        sys.exit(1)
        
    field_name_map = fetch_field_meta(headers, args.project_key)

    print(f"4. 正在分批拉取工作项详情 (类型: {resolved_type})...")
    all_details = []
    chunk_size = 50
    for i in range(0, len(item_ids), chunk_size):
        chunk = item_ids[i:i+chunk_size]
        res = query_work_items_by_ids(headers, args.project_key, resolved_type, chunk)
        all_details.extend(res)
        print(f"   -> 已拉取 {min(i + chunk_size, len(item_ids))} / {len(item_ids)}")

    print("5. 正在清洗和格式化数据...")
    results = []
    for d in all_details:
        named_fields = format_item_details(d, field_name_map)
        results.append({
            "id": d.get("id"),
            "name": d.get("name"),
            "named_fields": named_fields
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 导出成功！共 {len(results)} 条数据已保存至: {args.output}")

if __name__ == "__main__":
    main()
