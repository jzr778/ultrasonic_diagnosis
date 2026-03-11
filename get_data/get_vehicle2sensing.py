import os
import sys
import re
import json
from typing import Dict, List

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from get_data.dr_client import download_trip_config


def parse_protobuf_data(data: bytes) -> Dict[str, Dict[str, List[float]]]:
    """解析 lidars.cfg，提取 position 和 orientation 信息。"""
    text = data.decode('utf-8')
    results = {}
    pattern = (
        r'(\w+)\s*{\s*position\s*{\s*x:\s*([\d.-]+)\s*y:\s*([\d.-]+)\s*z:\s*([\d.-]+)\s*}'
        r'\s*orientation\s*{\s*qx:\s*([\d.-]+)\s*qy:\s*([\d.-]+)\s*qz:\s*([\d.-]+)\s*qw:\s*([\d.-]+)'
    )
    for match in re.findall(pattern, text, re.DOTALL):
        results[match[0]] = {
            'position': [float(match[1]), float(match[2]), float(match[3])],
            'orientation': [float(match[4]), float(match[5]), float(match[6]), float(match[7])],
        }
    return results


def get_vehicle2sensing(trip_id):
    raw = download_trip_config(trip_id, "lidars.cfg")
    parsed = parse_protobuf_data(raw)
    return parsed['vehicle_to_sensing']


if __name__ == '__main__':
    trip_id = 80617951
    vehicle2sensing = get_vehicle2sensing(trip_id=trip_id)
    print(vehicle2sensing)
    output_path = os.path.join(os.path.dirname(__file__), "vehicle2sensing.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vehicle2sensing, f, indent=2)
