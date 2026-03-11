"""
从 bag 中提取 VLM 诊断所需的结构化 JSON 数据。

输出目录结构:
  read_data/{tag_id}/
  ├── meta_data.json
  ├── vehicle2sensing.json
  ├── car_config.json
  ├── ground.json
  ├── cameras_parameters.json
  └── {timestamp}/
      ├── chaosheng.json
      ├── obstacle.json
      ├── pose.json
      └── plan.json
"""

import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from bag_reader import BagReader
from get_camera_parameters import get_camera_engine_parameters
from get_vehicle2sensing import get_vehicle2sensing
from get_car_config import get_car_config
from get_ground import get_ground


def save_data(tag_id, output_root=None):
    if output_root is None:
        output_root = config.READ_DATA_DIR

    data_path = os.path.join(output_root, str(tag_id))
    os.makedirs(data_path, exist_ok=True)

    # ── 读取 bag ──
    reader = BagReader(tag_id=tag_id)

    with open(os.path.join(data_path, 'meta_data.json'), 'w', encoding='utf-8') as f:
        json.dump(reader.meta_data, f, ensure_ascii=False, indent=2)
    print("meta_data")

    reader.scan_ultrasonic_events()

    if not reader.perception_time_list:
        print(f"  tag_id={tag_id} 无超声波事件，跳过 bag 数据提取")
        return

    for t in reader.perception_time_list:
        os.makedirs(os.path.join(data_path, str(int(t))), exist_ok=True)

    for t, chaosheng_data in reader.chaosheng_results.items():
        path = os.path.join(data_path, str(int(t)), 'chaosheng.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chaosheng_data, f, indent=2, ensure_ascii=False)
    print("chaosheng_data")

    obstacle_results = reader.extract_obstacles()
    for t, obstacle_data in obstacle_results.items():
        if not obstacle_data:
            continue
        path = os.path.join(data_path, str(int(t)), 'obstacle.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obstacle_data['obstacle'], f, indent=2, ensure_ascii=False)
    print("obstacle_data")

    pose_results = reader.extract_poses()
    for t, pose_data in pose_results.items():
        if not pose_data:
            continue
        path = os.path.join(data_path, str(int(t)), 'pose.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)
    print("pose_data")

    plan_results = reader.extract_planning()
    for t, plan_data in plan_results.items():
        if not plan_data:
            continue
        path = os.path.join(data_path, str(int(t)), 'plan.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)
    print("plan_data")

    # ── 通过 API 获取配置数据 ──
    trip_id = reader.trip_id

    vehicle2sensing = get_vehicle2sensing(trip_id)
    with open(os.path.join(data_path, 'vehicle2sensing.json'), 'w', encoding='utf-8') as f:
        json.dump(vehicle2sensing, f, ensure_ascii=False, indent=2)
    print("vehicle2sensing")

    car_cfg = get_car_config(trip_id)
    with open(os.path.join(data_path, 'car_config.json'), 'w', encoding='utf-8') as f:
        json.dump(car_cfg, f, ensure_ascii=False, indent=2)
    print("car_config")

    ground = get_ground(trip_id=trip_id)
    with open(os.path.join(data_path, 'ground.json'), 'w', encoding='utf-8') as f:
        json.dump(ground, f, ensure_ascii=False, indent=2)
    print("ground")

    cameras_parameters = get_camera_engine_parameters(trip_id=trip_id)
    cameras_parameters = cameras_parameters.decode('utf-8').splitlines()
    with open(os.path.join(data_path, 'cameras_parameters.json'), 'w', encoding='utf-8') as f:
        json.dump(cameras_parameters, f, ensure_ascii=False, indent=2)
    print("cameras_parameters")


if __name__ == "__main__":
    tag_id_list = [
        97020543,
    ]
    for tag_id in tag_id_list:
        try:
            save_data(tag_id=tag_id)
            print(f"{tag_id} saved")
        except Exception as e:
            print(f"Error saving tag_id {tag_id}: {e}")
