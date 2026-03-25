"""
从 bag 中提取 VLM 诊断所需的结构化 JSON 数据。

输出目录结构:
  read_data/{tag_id}/
  ├── meta_data.json
  ├── vehicle2sensing.json
  ├── car_config.json
  ├── ground.json
  ├── cameras_parameters.json
  └── {timestamp}/          （仅当该目录下 JSON 齐全，且在 extract_fisheye 时四路 jpg 齐全时才保留）
      ├── chaosheng.json
      ├── obstacle.json
      ├── pose.json
      ├── plan.json
      └── panoramic_*.jpg    （extract_fisheye=True 时）
"""

import json
import os
import shutil
import sys

import cv2

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from get_data.bag_reader import BagReader
from get_data.get_camera_parameters import get_camera_engine_parameters
from get_data.get_vehicle2sensing import get_vehicle2sensing
from get_data.get_car_config import get_car_config
from get_data.get_ground import get_ground

# 时间戳子目录须同时具备（绘图 / VLM 缺一不可则整目录不落盘）
_REQUIRED_JSON = (
    "chaosheng.json",
    "obstacle.json",
    "pose.json",
    "plan.json",
)


def _timestamp_payload_complete(ts_dir, require_fisheye):
    """返回 (是否完整, 首个缺失或空文件名)。"""
    for name in _REQUIRED_JSON:
        p = os.path.join(ts_dir, name)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False, name
    if require_fisheye:
        for cam in config.CAMERA_NAMES:
            p = os.path.join(ts_dir, f"{cam}.jpg")
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                return False, f"{cam}.jpg"
    return True, None


def save_data(tag_id, output_root=None, extract_fisheye=True):
    if output_root is None:
        output_root = config.READ_DATA_DIR

    data_path = os.path.join(output_root, str(tag_id))
    os.makedirs(data_path, exist_ok=True)

    # ── 读取 bag ──
    reader = BagReader(tag_id=tag_id)

    with open(os.path.join(data_path, 'meta_data.json'), 'w', encoding='utf-8') as f:
        json.dump(reader.meta_data, f, ensure_ascii=False, indent=2)
    print("meta_data")

    # ── 通过 API 获取配置数据（不依赖超声波事件，始终下载） ──
    trip_id = reader.trip_id

    config_tasks = [
        ("vehicle2sensing.json", lambda: get_vehicle2sensing(trip_id)),
        ("car_config.json",      lambda: get_car_config(trip_id)),
        ("ground.json",          lambda: get_ground(trip_id=trip_id)),
    ]
    for filename, fetch_fn in config_tasks:
        try:
            data = fetch_fn()
            with open(os.path.join(data_path, filename), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  {filename}")
        except Exception as e:
            print(f"  [WARN] 下载 {filename} 失败: {e}")

    try:
        raw = get_camera_engine_parameters(trip_id=trip_id)
        cameras_parameters = raw.decode('utf-8').splitlines()
        with open(os.path.join(data_path, 'cameras_parameters.json'), 'w', encoding='utf-8') as f:
            json.dump(cameras_parameters, f, ensure_ascii=False, indent=2)
        print("  cameras_parameters.json")
    except Exception as e:
        print(f"  [WARN] 下载 cameras_parameters.json 失败: {e}")

    # ── 提取超声波相关 bag 数据 ──
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

    if extract_fisheye:
        print("  提取鱼眼最近邻帧（Heavy bag）...")
        image_results = reader.extract_nearest_images()
        n_img = 0
        for t in reader.perception_time_list:
            sub = image_results.get(t) or {}
            ts_dir = os.path.join(data_path, str(int(t)))
            for cam in config.CAMERA_NAMES:
                ent = sub.get(cam)
                if ent and ent.get("image") is not None:
                    fp = os.path.join(ts_dir, f"{cam}.jpg")
                    cv2.imwrite(fp, ent["image"])
                    n_img += 1
        print(f"  鱼眼图写入完成（共 {n_img} 张）")

    removed = []
    for t in reader.perception_time_list:
        ts_dir = os.path.join(data_path, str(int(t)))
        if not os.path.isdir(ts_dir):
            continue
        ok, missing = _timestamp_payload_complete(ts_dir, extract_fisheye)
        if not ok:
            shutil.rmtree(ts_dir, ignore_errors=True)
            removed.append((int(t), missing))
    if removed:
        for ts_val, missing in removed:
            print(f"  [SKIP] ts={ts_val} 缺少或为空: {missing}，已删除不完整目录")
        print(
            f"  时间戳目录: 保留 {len(reader.perception_time_list) - len(removed)} 个 / "
            f"删除不完整 {len(removed)} 个"
        )

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
