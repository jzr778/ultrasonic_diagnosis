import requests
import json
import os

url_query_tag_instance = "https://drplatform-backend.deeproute.cn/scene/tag/instance/query"

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json;charset=UTF-8'
}

def get_meta_data(tag_id):
    request_body = {"condition" : {"id": {"eq" : tag_id,},}}
    # 发送请求
    response = requests.post(url_query_tag_instance, headers=headers, data=json.dumps(request_body))
    # 打印状态信息
    if response.status_code == 200:
        # 解析JSON
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
    # 保存
    output_path = "meta_data.json"
    output_path = os.path.join(os.path.dirname(__file__), output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)

# 90294405, 90294401, 90294397, 90294393, 90294380,
# 90294374, 90294370, 90294359, 90294347, 90294334,
# 90294333, 90294329, 90294293, 90294286, 90293836

