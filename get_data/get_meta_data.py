import requests
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json;charset=UTF-8'
}

def get_meta_data(tag_id):
    request_body = {"condition" : {"id": {"eq" : tag_id,},}}
    response = requests.post(config.DR_TAG_QUERY_URL, headers=headers, data=json.dumps(request_body))
    if response.status_code == 200:
        meta_data = response.json()
        return meta_data
    else:
        print(f"\n❌ 请求失败:")
        print(f"  状态码: {response.status_code}")
        print(f"  原因: {response.reason}")
        print(f"  响应内容: {response.text}")
        return None

if __name__ == '__main__':
    tag_id = 90294380
    meta_data = get_meta_data(tag_id=tag_id)
    print(meta_data)
    output_path = os.path.join(os.path.dirname(__file__), "meta_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
