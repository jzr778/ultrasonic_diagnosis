#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞书项目 ID 映射导出工具

从飞书项目视图中提取 tag_id 与 feishu_id 的对应关系，
输出为 {tag_id: feishu_id} 字典形式的 JSON 文件。

- tag_id:   工作项 name 字段中提取的数字 ID (如 LPA05-120394123-xxx → 120394123)
- feishu_id: 飞书项目的工作项 ID (work_item_id)
"""

import os
import sys
import json
import argparse
import re
import urllib.request
from urllib.error import HTTPError

BASE_URL = "https://project.feishu.cn/open_api"


def http_json(method, url, headers=None, body=None, timeout=30):
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
    url = f"{BASE_URL}/authen/plugin_token"
    body = {"plugin_id": plugin_id, "plugin_secret": plugin_secret, "type": 0}
    payload = http_json("POST", url, headers=None, body=body)
    token = payload.get("data", {}).get("token")
    if not token:
        raise RuntimeError(f"无法获取插件 token: {payload}")
    return token


def fetch_fix_view_ids(headers, project_key, view_id, page_size=200, max_pages=50):
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
    payload = http_json("GET", f"{BASE_URL}/{project_key}/work_item/all-types", headers)
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [t.get("type_key") for t in data if isinstance(t, dict) and t.get("type_key")]


def resolve_work_item_type_key(headers, project_key, sample_id, candidate_types):
    for type_key in candidate_types:
        items = query_work_items_by_ids(headers, project_key, type_key, [sample_id])
        if items:
            return type_key
    return None


def query_work_items_by_ids(headers, project_key, type_key, item_ids):
    if not item_ids:
        return []
    body = {"work_item_ids": item_ids}
    url = f"{BASE_URL}/{project_key}/work_item/{type_key}/query"
    payload = http_json("POST", url, headers, body)
    data = payload.get("data")
    if isinstance(data, list):
        return [i for i in data if isinstance(i, dict)]
    return []


def extract_tag_id(name):
    """从 name 字段中提取 tag_id (例如：LPA05-120003819-xxx → 120003819)"""
    if not name:
        return None
    match = re.search(r'-(\d+)-', name)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="导出飞书项目视图中 tag_id 与 feishu_id 的映射关系。"
    )
    parser.add_argument("-p", "--project-key", required=True,
                        help="飞书项目的 Project Key (例如: iffcom)")
    parser.add_argument("-v", "--view-id", required=True,
                        help="飞书视图的 View ID")
    parser.add_argument("-o", "--output", default="id_mapping.json",
                        help="输出的 JSON 文件路径 (默认: id_mapping.json)")

    args = parser.parse_args()

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
        "X-USER-KEY": user_key,
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

    print("3. 正在解析工作项类型...")
    candidate_types = fetch_work_item_types(headers, args.project_key)
    resolved_type = resolve_work_item_type_key(
        headers, args.project_key, item_ids[0], candidate_types
    )
    if not resolved_type:
        print("无法解析该视图下的工作项类型，请检查项目及视图权限。")
        sys.exit(1)

    print(f"4. 正在分批拉取工作项详情 (类型: {resolved_type})...")
    all_details = []
    chunk_size = 50
    for i in range(0, len(item_ids), chunk_size):
        chunk = item_ids[i:i + chunk_size]
        res = query_work_items_by_ids(headers, args.project_key, resolved_type, chunk)
        all_details.extend(res)
        print(f"   -> 已拉取 {min(i + chunk_size, len(item_ids))} / {len(item_ids)}")

    print("5. 正在构建 tag_id → feishu_id 映射...")
    mapping = {}
    skipped = 0
    for d in all_details:
        feishu_id = d.get("id")
        tag_id = extract_tag_id(d.get("name"))
        if tag_id and feishu_id:
            mapping[str(tag_id)] = int(feishu_id)
        else:
            skipped += 1

    if skipped:
        print(f"   -> 警告: {skipped} 个工作项无法提取 tag_id，已跳过")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n导出成功！共 {len(mapping)} 条映射已保存至: {args.output}")


if __name__ == "__main__":
    main()
