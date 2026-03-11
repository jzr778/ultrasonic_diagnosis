import os
import sys
import re
import json
from typing import Dict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from get_data.dr_client import download_trip_config


def parse_protobuf_data(data: bytes) -> Dict[str, float]:
    """解析 ground.cfg，提取 ground_in_sensing 参数。"""
    text = data.decode('utf-8')
    match = re.search(r'ground_in_sensing\s*{([^}]+)}', text, re.DOTALL)
    if not match:
        return {}
    result = {}
    for param_name, param_value in re.findall(r'(\w+)\s*:\s*([\d.-]+)', match.group(1)):
        result[param_name] = float(param_value)
    return result


def get_ground(trip_id):
    raw = download_trip_config(trip_id, "ground.cfg")
    return parse_protobuf_data(raw)


if __name__ == '__main__':
    trip_id = 80617951
    ground = get_ground(trip_id=trip_id)
    print(ground)
    output_path = os.path.join(os.path.dirname(__file__), "ground.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground, f, indent=2)
