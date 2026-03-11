import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from get_data.dr_client import download_trip_config


def get_camera_engine_parameters(trip_id):
    """下载 cameras.cfg 原始字节。"""
    return download_trip_config(trip_id, "cameras.cfg")


if __name__ == '__main__':
    trip_id = 51300511
    cameras_parameters = get_camera_engine_parameters(trip_id=trip_id)
    print(cameras_parameters)
    output_path = os.path.join(os.path.dirname(__file__), f"cameras_parameters_{trip_id}.json")
    with open(output_path, 'wb') as f:
        f.write(cameras_parameters)
