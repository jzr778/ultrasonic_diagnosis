import requests
import json
import argparse
from typing import List, Dict

def query_bag_names_by_tag_instance_ids(tag_instance_ids: List[str]) -> List[Dict]:
    url = "https://drplatform-backend.deeproute.cn/scene/tag/instance/query/highLevel"  # 确保接口地址正确
    results = []
    step = 100

    for i in range(0, len(tag_instance_ids), step):
        batch_ids = tag_instance_ids[i:i+step]
        print(f"查询批次 {i//step + 1}/{(len(tag_instance_ids) + step -1)//step}...")

        payload = {
            "condition": {"id": {"in": batch_ids}},
            "orderBys": [],
            "page": 0,
            "size": step
        }

        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            data = response.json()
            print("调试：接口返回内容", data)  # 临时打印响应，确认结构

            if data.get("status") == "SUCCESS" and "body" in data and len(data["body"]) > 0:
                for item in data["body"]:
                    results.append({
                        "tag_instance_id": item.get("id"),
                        "bag_names": item.get("bagsName", [])
                    })
            else:
                print(f"查询结果为空或接口异常: {data}")

        except Exception as e:
            print(f"查询错误: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='通过tagInstanceId查询对应的bag name。')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--id', type=str, help='单个tag_instance_id')
    group.add_argument('--file', type=str, help='包含tag_instance_id的文件路径，每行一个')
    parser.add_argument('--output', type=str, help='输出结果的JSON文件路径，默认不输出')

    args = parser.parse_args()

    tag_instance_ids = []
    if args.id:
        tag_instance_ids = [args.id]
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                tag_instance_ids = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return

    print(f"共查询 {len(tag_instance_ids)} 个tag_instance_id...")
    results = query_bag_names_by_tag_instance_ids(tag_instance_ids)

    # 打印结果
    for result in results:
        print(f"tag_instance_id: {result['tag_instance_id']}")
        print(f"bag_names: {', '.join(result['bag_names'])}")
        print("-" * 50)

    # 保存结果到文件
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"结果已保存至 {args.output}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    main()
