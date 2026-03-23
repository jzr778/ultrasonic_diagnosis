"""
公共 bag 读取层。

统一封装对远端 DpBag 的读取逻辑，避免多个脚本重复打开 / 解析同一份 bag。
提供以下能力：
  1. scan_ultrasonic_events  — 扫描 Light bag 中的超声波停车事件
  2. extract_nearest_images  — 从 Heavy bag 提取与事件最近邻的 4 路鱼眼图（内存）
  3. extract_obstacles       — 从 Light bag 提取 /perception/objects
  4. extract_poses           — 从 Light bag 提取 /localization/pose
  5. extract_planning        — 从 Light bag 提取 /planner/trajectory
"""

import os
import re
import sys

import cv2
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

sys.path.insert(0, config.PROTO_LOCAL_DIR)

from dpbag import strip_header
from dpbag.bag.bag import DpBag

try:
    from drivers.sensor_image_pb2 import CompressedImage
    from perception.deeproute_perception_obstacle_pb2 import PerceptionObstacles
    from drivers.gnss.ins_pb2 import Ins
    from planning.planning_pb2 import ADCTrajectory
except ImportError as e:
    print(f"Proto import error: {e}")
    sys.exit(1)

from google.protobuf.json_format import MessageToDict
from get_data.get_meta_data import get_meta_data


class BagReader:
    """对一个 tag_id 对应的所有 bag 进行统一读取。"""

    def __init__(self, tag_id=None, meta_data=None):
        if meta_data is None:
            meta_data = get_meta_data(tag_id=tag_id)
        self.meta_data = meta_data
        self.trip_id = meta_data['body'][0]['tripId']

        bags = meta_data['body'][0]['bagsName']
        self.all_light_bags = sorted(b for b in bags if 'Light' in b)
        self.all_heavy_bags = sorted(b for b in bags if 'Heavy' in b)

        self.perception_time_list = []
        self.chaosheng_results = {}
        self.event_light_bags = []
        self.event_heavy_bags = []

    # ──────────────────────────────────────────────────────────
    #  超声波事件扫描（Light bag）
    # ──────────────────────────────────────────────────────────

    def scan_ultrasonic_events(self):
        """扫描所有 Light bag，筛选 MODEL_PARKING + PLANNING_STOP_OBSTACLE + ULTRASONIC。

        返回 (perception_time_list, chaosheng_results)
        同时填充 self.event_light_bags / self.event_heavy_bags。
        """
        for bag_name in self.all_light_bags:
            if is_p01t_vehicle_bag(bag_name):
                print(f"    [SKIP] P01T 车型 bag，跳过: {os.path.basename(bag_name)}")
                continue
            try:
                with DpBag(bag=bag_name) as bag:
                    for _, msg, _ in bag.read_messages(
                        topics=[config.CHAOSHENG_TOPIC],
                        dpbag_name=bag_name,
                        force_get_data_by_raw=True,
                    ):
                        obj = PerceptionObstacles()
                        raw_msg = strip_header(msg.data)
                        obj.ParseFromString(raw_msg)
                        per_t = obj.time_measurement

                        data_list = []
                        for item in obj.perception_obstacle:
                            data = MessageToDict(item) if hasattr(item, 'DESCRIPTOR') else item
                            if (data.get("modelType") == 'MODEL_PARKING'
                                    and data.get("type") == 'PLANNING_STOP_OBSTACLE'
                                    and data.get("sensorType") == 'ULTRASONIC'):
                                data_list.append(data)

                        if data_list:
                            if bag_name not in self.event_light_bags:
                                self.event_light_bags.append(bag_name)
                            self.perception_time_list.append(per_t)
                            self.chaosheng_results[per_t] = data_list
            except Exception as e:
                print(f"    [WARN] 跳过 {bag_name}: {e}")

        self.event_heavy_bags = sorted(
            b.replace("Light", "Heavy") for b in self.event_light_bags
        )
        return self.perception_time_list, self.chaosheng_results

    # ──────────────────────────────────────────────────────────
    #  鱼眼图最近邻提取（Heavy bag）
    # ──────────────────────────────────────────────────────────

    def extract_nearest_images(self):
        """从 Heavy bag 中为每个超声波事件提取全局最近邻 4 路鱼眼图。

        返回:
          {per_t: {cam_name: {'image': ndarray, 'timestamp_us': int, 'source_bag': str}}}
        """
        image_results = {t: {} for t in self.perception_time_list}
        min_diffs = {
            t: {cam: float('inf') for cam in config.CAMERA_NAMES}
            for t in self.perception_time_list
        }

        for topic, cam_name in zip(config.CAMERA_TOPICS, config.CAMERA_NAMES):
            print(f"    提取 {cam_name} ...")
            count = 0

            for heavy_bag in self.event_heavy_bags:
                try:
                    with DpBag(bag=heavy_bag) as bag:
                        for _, msg, _ in bag.read_messages(
                            topics=[topic],
                            dpbag_name=heavy_bag,
                            force_get_data_by_raw=True,
                        ):
                            obj = CompressedImage()
                            raw_msg = strip_header(msg.data)
                            obj.ParseFromString(raw_msg)
                            ts_us = int(obj.header.timestamp_sec * 1e6)

                            for evt_t in self.perception_time_list:
                                diff = abs(ts_us - evt_t)
                                if diff < min_diffs[evt_t][cam_name]:
                                    min_diffs[evt_t][cam_name] = diff
                                    img = cv2.imdecode(
                                        np.frombuffer(obj.data, np.uint8),
                                        cv2.IMREAD_COLOR,
                                    )
                                    image_results[evt_t][cam_name] = {
                                        'image': img,
                                        'timestamp_us': ts_us,
                                        'source_bag': heavy_bag,
                                    }
                                    count += 1
                except Exception as e:
                    print(f"      [WARN] 跳过 {heavy_bag}: {e}")

            print(f"      -> 更新 {count} 次匹配")

        return image_results

    # ──────────────────────────────────────────────────────────
    #  障碍物提取（Light bag）
    # ──────────────────────────────────────────────────────────

    def extract_obstacles(self):
        """从 Light bag 读取 /perception/objects，按最近邻时间戳匹配。

        返回: {per_t: {'obstacle': [...], 'time_diff': float}}
        """
        results = {t: {} for t in self.perception_time_list}
        min_diffs = {t: float('inf') for t in self.perception_time_list}

        for bag_name in self.event_light_bags:
            try:
                with DpBag(bag=bag_name) as bag:
                    for _, msg, _ in bag.read_messages(
                        topics=[config.OBSTACLE_TOPIC],
                        dpbag_name=bag_name,
                        force_get_data_by_raw=True,
                    ):
                        obj = PerceptionObstacles()
                        raw_msg = strip_header(msg.data)
                        obj.ParseFromString(raw_msg)
                        t = obj.time_measurement

                        for evt_t in self.perception_time_list:
                            diff = abs(t - evt_t)
                            if diff < min_diffs[evt_t]:
                                min_diffs[evt_t] = diff
                                data_list = [
                                    MessageToDict(item) if hasattr(item, 'DESCRIPTOR') else item
                                    for item in obj.perception_obstacle
                                ]
                                results[evt_t] = {'obstacle': data_list, 'time_diff': diff}
            except Exception as e:
                print(f"    [WARN] 跳过 {bag_name} (obstacle): {e}")

        return results

    # ──────────────────────────────────────────────────────────
    #  位姿提取（Light bag）
    # ──────────────────────────────────────────────────────────

    def extract_poses(self):
        """从 Light bag 读取 /localization/pose，按最近邻时间戳匹配。

        返回: {per_t: {'position': [x,y,z], 'euler_angles': [x,y,z], 'time_diff': float}}
        """
        results = {t: {} for t in self.perception_time_list}
        min_diffs = {t: float('inf') for t in self.perception_time_list}

        for bag_name in self.event_light_bags:
            try:
                with DpBag(bag=bag_name) as bag:
                    for _, msg, _ in bag.read_messages(
                        topics=[config.POSE_TOPIC],
                        dpbag_name=bag_name,
                        force_get_data_by_raw=True,
                    ):
                        obj = Ins()
                        raw_msg = strip_header(msg.data)
                        obj.ParseFromString(raw_msg)
                        t = obj.measurement_time

                        for evt_t in self.perception_time_list:
                            diff = abs(t - evt_t)
                            if diff < min_diffs[evt_t]:
                                min_diffs[evt_t] = diff
                                pos = obj.position
                                euler = obj.euler_angles
                                results[evt_t] = {
                                    'position': [pos.x, pos.y, pos.z],
                                    'euler_angles': [euler.x, euler.y, euler.z],
                                    'time_diff': diff,
                                }
            except Exception as e:
                print(f"    [WARN] 跳过 {bag_name} (pose): {e}")

        return results

    # ──────────────────────────────────────────────────────────
    #  规划轨迹提取（Light bag）
    # ──────────────────────────────────────────────────────────

    def extract_planning(self):
        """从 Light bag 读取 /planner/trajectory，按最近邻时间戳匹配。

        返回: {per_t: [{'relative_time': ..., 'x': ..., 'y': ...}, ...]}
        """
        results = {t: {} for t in self.perception_time_list}
        min_diffs = {t: float('inf') for t in self.perception_time_list}

        for bag_name in self.event_light_bags:
            try:
                with DpBag(bag=bag_name) as bag:
                    for _, msg, _ in bag.read_messages(
                        topics=[config.PLANNING_TOPIC],
                        dpbag_name=bag_name,
                        force_get_data_by_raw=True,
                    ):
                        obj = ADCTrajectory()
                        raw_msg = strip_header(msg.data)
                        obj.ParseFromString(raw_msg)
                        t = obj.header.timestamp_sec * 1e6

                        for evt_t in self.perception_time_list:
                            diff = abs(t - evt_t)
                            if diff < min_diffs[evt_t]:
                                min_diffs[evt_t] = diff
                                results[evt_t] = [
                                    {
                                        'relative_time': pt.relative_time,
                                        'x': pt.path_point.x,
                                        'y': pt.path_point.y,
                                    }
                                    for pt in obj.trajectory_point
                                ]
            except Exception as e:
                print(f"    [WARN] 跳过 {bag_name} (planning): {e}")

        return results
