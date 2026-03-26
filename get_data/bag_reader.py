"""
公共 bag 读取层。

统一封装对远端 DpBag 的读取逻辑，避免多个脚本重复打开 / 解析同一份 bag。
提供以下能力：
  1. scan_ultrasonic_events  — 扫描 Light bag 中的超声波停车事件
  2. extract_nearest_images  — 从 Heavy bag 提取与事件最近邻的 4 路鱼眼图（内存）；丢弃
     ``/canbus/car_state`` 中 misc.rear_view_mirror 表示折叠的候选帧（不参与最近邻替换）
  3. extract_obstacles       — 从 Light bag 提取 /perception/objects
  4. extract_poses           — 从 Light bag 提取 /localization/pose
  5. extract_planning        — 从 Light bag 提取 /planner/trajectory
"""

import bisect
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

# 截断的 deeproute_perception_obstacle_pb2 缺少 PerceptionObstacles，需先注册
import get_data.perception_obstacles_compat  # noqa: F401

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

try:
    from canbus.car_info_pb2 import CarInfo
except ImportError:
    CarInfo = None

from google.protobuf.json_format import MessageToDict
from get_data.get_meta_data import get_meta_data

# 鱼眼帧与 car_state 对齐容差（微秒）；超出则不作为「折叠」依据（避免误杀）
_CAR_STATE_MIRROR_MAX_SKEW_US = 300_000


def _bag_meta_timestamp_us(meta):
    """从 dpbag read_messages 第三项解析墙钟时间（微秒），若无法解析则 None。"""
    if meta is None:
        return None
    if isinstance(meta, (int, float)):
        v = int(meta)
        if v > 10**15:
            return v // 1000
        if v > 10**12:
            return v
        if v > 10**9:
            return v * 1000
        return None
    if isinstance(meta, dict):
        for key in ("timestamp_ns", "time_ns", "t_ns"):
            v = meta.get(key)
            if v is not None:
                return int(v) // 1000
        for key in ("timestamp_us", "time_us", "t_us"):
            v = meta.get(key)
            if v is not None:
                return int(v)
        for key in ("timestamp_sec", "time_sec"):
            v = meta.get(key)
            if v is not None:
                return int(float(v) * 1e6)
    return None


def _car_info_timestamp_us(car_info_pb, meta=None):
    """CarInfo → 微秒时间戳；优先 meta，其次 time_system / time_meas 等常见字段。"""
    t_meta = _bag_meta_timestamp_us(meta)
    if t_meta is not None:
        return t_meta
    ts = int(getattr(car_info_pb, "time_system", 0) or 0)
    if ts > 10**15:
        return ts // 1000
    if ts > 10**12:
        return ts
    if ts > 10**9:
        return ts * 1000
    tm = int(getattr(car_info_pb, "time_meas", 0) or 0)
    if tm > 10**12:
        return tm
    if tm > 10**9:
        return tm * 1000
    if tm > 0:
        return tm * 1000
    tmg = int(getattr(car_info_pb, "time_mgmt_plane", 0) or 0)
    if tmg > 10**12:
        return tmg
    if tmg > 10**9:
        return tmg * 1000
    return None


def _misc_rear_view_mirror_all_open(car_info_pb):
    """misc.rear_view_mirror：全 True/1 视为展开；任一 False/0 为折叠；无字段则 None（不据此筛）。"""
    m = car_info_pb.misc
    if not m.rear_view_mirror:
        return None
    for x in m.rear_view_mirror:
        if not bool(x):
            return False
    return True


def _collect_rear_view_mirror_timeline(heavy_bags, topic):
    """[(ts_us, mirrors_all_open), ...]，已按 ts_us 排序。"""
    if CarInfo is None or not topic:
        return [], []
    samples = []
    for heavy_bag in heavy_bags:
        try:
            with DpBag(bag=heavy_bag) as bag:
                for _, msg, meta in bag.read_messages(
                    topics=[topic],
                    dpbag_name=heavy_bag,
                    force_get_data_by_raw=True,
                ):
                    obj = CarInfo()
                    raw_msg = strip_header(msg.data)
                    try:
                        obj.ParseFromString(raw_msg)
                    except Exception:
                        continue
                    ts_us = _car_info_timestamp_us(obj, meta)
                    if ts_us is None:
                        continue
                    open_ok = _misc_rear_view_mirror_all_open(obj)
                    if open_ok is None:
                        continue
                    samples.append((ts_us, open_ok))
        except Exception as e:
            print(f"      [WARN] car_state 读取跳过 {heavy_bag}: {e}")
    if not samples:
        return [], []
    samples.sort(key=lambda x: x[0])
    tss = [s[0] for s in samples]
    oks = [s[1] for s in samples]
    return tss, oks


def _mirror_open_at_image_ts(timeline_ts, timeline_open, image_ts_us, max_skew_us):
    """在 image_ts_us 邻近的 car_state 上判断是否全部展开；无可靠样本则 None。"""
    if not timeline_ts:
        return None
    idx = bisect.bisect_left(timeline_ts, image_ts_us)
    cand = []
    if idx < len(timeline_ts):
        cand.append(idx)
    if idx > 0:
        cand.append(idx - 1)
    best = min(cand, key=lambda i: abs(timeline_ts[i] - image_ts_us))
    if abs(timeline_ts[best] - image_ts_us) > max_skew_us:
        return None
    return timeline_open[best]


def is_p01t_vehicle_bag(bag_name):
    """P01T 车型 bag：名称中含 ``-P01T-<车号>``，如 ``YR-P01T-4_...``。"""
    base = os.path.basename(bag_name)
    return re.search(r"-P01T-", base, re.IGNORECASE) is not None


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

        mirror_ts, mirror_open = _collect_rear_view_mirror_timeline(
            self.event_heavy_bags, getattr(config, "CAR_STATE_TOPIC", "") or ""
        )
        if mirror_ts:
            print(
                f"    car_state 后视镜样本 {len(mirror_ts)} 条，"
                f"折叠帧不参与鱼眼最近邻匹配（topic={config.CAR_STATE_TOPIC}）"
            )
        elif getattr(config, "CAR_STATE_TOPIC", ""):
            print(
                f"    [INFO] 未从 Heavy bag 读到 {config.CAR_STATE_TOPIC} 有效后视镜字段，"
                f"鱼眼提取不叠加后视镜折叠过滤"
            )

        for topic, cam_name in zip(config.CAMERA_TOPICS, config.CAMERA_NAMES):
            print(f"    提取 {cam_name} ...")
            count = 0
            mirror_fold_skips = 0

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
                                    mir = _mirror_open_at_image_ts(
                                        mirror_ts,
                                        mirror_open,
                                        ts_us,
                                        _CAR_STATE_MIRROR_MAX_SKEW_US,
                                    )
                                    if mir is False:
                                        mirror_fold_skips += 1
                                        continue
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
            if mirror_fold_skips:
                print(
                    f"      -> 后视镜折叠: 跳过 {mirror_fold_skips} 个鱼眼候选帧"
                    f"（不参与该路最近邻；topic={config.CAR_STATE_TOPIC}）"
                )

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
