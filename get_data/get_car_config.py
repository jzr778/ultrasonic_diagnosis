import os
import sys
import re
import json

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from get_data.dr_client import download_trip_config


def extract_params(byte_data):
    """从 car_config.cfg 字节中提取车身尺寸参数。"""
    text = byte_data.decode('utf-8', errors='ignore')

    def get_value(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    return {
        'front_edge_to_center': get_value(r'front_edge_to_center:\s*([\d\.-]+)'),
        'back_edge_to_center': get_value(r'back_edge_to_center:\s*([\d\.-]+)'),
        'left_edge_to_center': get_value(r'left_edge_to_center:\s*([\d\.-]+)'),
        'right_edge_to_center': get_value(r'right_edge_to_center:\s*([\d\.-]+)'),
    }


def get_car_config(trip_id):
    raw = download_trip_config(trip_id, "car_config.cfg")
    return extract_params(raw)


if __name__ == '__main__':
    trip_id = 51300511
    car_config = get_car_config(trip_id=trip_id)
    print(car_config)
    output_path = os.path.join(os.path.dirname(__file__), "car_config.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(car_config, f, indent=2)
