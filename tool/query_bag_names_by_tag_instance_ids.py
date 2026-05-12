import requests
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Iterable, Set

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ID_MAPPING = _REPO_ROOT / "get_data" / "id_mapping.json"
_DEFAULT_BAG_LIST = _REPO_ROOT / "offline_avm_generate_release" / "bag_list.txt"


def _load_tag_ids_from_id_mapping(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"id_mapping 预期为 JSON 对象，得到: {type(data)}")
    return [str(k) for k in data.keys()]


def _iter_heavy_topic_group_bags(bag_names: Iterable[str]) -> List[str]:
    return [b for b in bag_names if isinstance(b, str) and b.endswith(".Heavy_Topic_Group.bag")]


def _merge_heavy_bags_into_bag_list(bag_list_path: Path, new_heavy: List[str]) -> tuple[int, int]:
    """将 new_heavy 追加到 bag_list_path（已存在的行保留顺序；新 bag 去重追加）。返回 (合并前行数, 新增行数)。"""
    existing: List[str] = []
    if bag_list_path.exists():
        existing = [
            ln.strip()
            for ln in bag_list_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    before = len(existing)
    seen: Set[str] = set(existing)
    added = 0
    for b in new_heavy:
        if b not in seen:
            existing.append(b)
            seen.add(b)
            added += 1
    bag_list_path.parent.mkdir(parents=True, exist_ok=True)
    bag_list_path.write_text("\n".join(existing) + ("\n" if existing else ""), encoding="utf-8")
    return before, added


def query_bag_names_by_tag_instance_ids(
    tag_instance_ids: List[str],
    *,
    verbose: bool = False,
) -> List[Dict]:
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
            if verbose:
                print("调试：接口返回内容", data)

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
    parser = argparse.ArgumentParser(
        description="通过 tagInstanceId 查询对应的 bag 名称；可将 Heavy_Topic_Group.bag 合并写入 bag_list.txt。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=str, help="单个 tag_instance_id")
    group.add_argument("--file", type=str, help="包含 tag_instance_id 的文件路径，每行一个")
    group.add_argument(
        "--from-id-mapping",
        nargs="?",
        const="__default__",
        metavar="PATH",
        help=(
            "从 id_mapping.json 读取键作为 tag_instance_id 列表；"
            f"省略 PATH 时默认 { _DEFAULT_ID_MAPPING }"
        ),
    )
    parser.add_argument("--output", type=str, help="输出结果的 JSON 文件路径，默认不输出")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="打印接口原始返回（调试用）"
    )
    parser.add_argument(
        "--bag-list",
        type=str,
        default=str(_DEFAULT_BAG_LIST),
        help=f"合并写入 .Heavy_Topic_Group.bag 的路径，默认 { _DEFAULT_BAG_LIST }",
    )
    parser.add_argument(
        "--no-write-bag-list",
        action="store_true",
        help="不把 Heavy bag 写入 bag_list.txt",
    )

    args = parser.parse_args()

    tag_instance_ids: List[str] = []
    if args.id:
        tag_instance_ids = [args.id]
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                tag_instance_ids = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return
    elif args.from_id_mapping is not None:
        map_path = (
            _DEFAULT_ID_MAPPING
            if args.from_id_mapping == "__default__"
            else Path(args.from_id_mapping).expanduser()
        )
        try:
            tag_instance_ids = _load_tag_ids_from_id_mapping(map_path)
        except Exception as e:
            print(f"读取 id_mapping 时出错 ({map_path}): {e}")
            return

    print(f"共查询 {len(tag_instance_ids)} 个tag_instance_id...")
    results = query_bag_names_by_tag_instance_ids(
        tag_instance_ids, verbose=bool(args.verbose)
    )

    # 打印结果
    for result in results:
        print(f"tag_instance_id: {result['tag_instance_id']}")
        print(f"bag_names: {', '.join(result['bag_names'])}")
        print("-" * 50)

    # 合并 Heavy bag 到 bag_list.txt
    if not args.no_write_bag_list and results:
        collected: List[str] = []
        seen_run: Set[str] = set()
        for result in results:
            for b in _iter_heavy_topic_group_bags(result.get("bag_names") or []):
                if b not in seen_run:
                    collected.append(b)
                    seen_run.add(b)
        bag_list_path = Path(args.bag_list).expanduser()
        if not os.path.isabs(str(bag_list_path)):
            bag_list_path = (_REPO_ROOT / bag_list_path).resolve()
        before, added = _merge_heavy_bags_into_bag_list(bag_list_path, collected)
        print(
            f"已合并 Heavy_Topic_Group.bag 至 {bag_list_path} "
            f"(本轮接口共 {len(collected)} 个 Heavy；合并前文件 {before} 行；实际新增 {added} 行)"
        )

    # 保存结果到文件
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"结果已保存至 {args.output}")
        except Exception as e:
            print(f"保存文件时出错: {e}")


if __name__ == "__main__":
    main()
