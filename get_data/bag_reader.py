"""
公共 bag 读取层。

统一封装对远端 DpBag 的读取逻辑，避免多个脚本重复打开 / 解析同一份 bag。
提供以下能力：
  1. scan_ultrasonic_events  — 扫描 Light bag 中的超声波停车事件
  2. extract_nearest_images  — 从 Heavy bag 提取与事件最近邻的 4 路鱼眼图（内存）；丢弃
     Light bag 内 ``config.CAR_STATE_TOPIC``（CarInfo）中 misc.rear_view_mirror 表示折叠的候选帧（不参与最近邻替换）
  3. extract_obstacles       — 从 Light bag 提取 /perception/objects
  4. extract_poses           — 从 Light bag 提取 /localization/pose
  5. extract_planning        — 从 Light bag 提取 /planner/trajectory
"""

import bisect
import logging
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

sys.path.insert(0, config.PROTO_LOCAL_DIR)

# 确保本地 proto 优先于 conda site-packages 中的同名包
_PROTO_PKGS = ("drivers", "perception", "planning", "canbus",
               "common", "localization", "calibration")
for _pkg in _PROTO_PKGS:
    if _pkg in sys.modules:
        del sys.modules[_pkg]

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

# 鱼眼帧与车身 CarInfo 时间线对齐容差（微秒）；超出则不作为「折叠」依据（避免误杀）
_CAR_STATE_MIRROR_MAX_SKEW_US = 50_000


def _perception_time_measurement_us(evt_t):
    """``PerceptionObstacles.time_measurement`` 按纳秒存时，``// 1000`` 即为微秒，与 ``ts_car`` 同源比较。"""
    return int(evt_t) // 1000


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
    # dpbag 等可能返回带属性的对象而非 dict
    for key in ("timestamp_ns", "time_ns", "t_ns"):
        v = getattr(meta, key, None)
        if v is not None:
            return int(v) // 1000
    for key in ("timestamp_us", "time_us", "t_us"):
        v = getattr(meta, key, None)
        if v is not None:
            return int(v)
    for key in ("timestamp_sec", "time_sec", "timestamp"):
        v = getattr(meta, key, None)
        if v is not None:
            if isinstance(v, float):
                return int(v * 1e6)
            return int(v) * 1_000_000 if int(v) < 10**12 else int(v)
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
    """misc.rear_view_mirror（repeated bool）：全为 True 视为展开；任一为 False 为折叠；无元素则 None（不据此筛）。"""
    m = car_info_pb.misc
    if not m.rear_view_mirror:
        return None
    for x in m.rear_view_mirror:
        if not x:
            return False
    return True


def _nearest_mirror_open_sorted(candidates, t_us, max_skew_us):
    """candidates: 已按 ts_car 排序的 [(ts_car, open_ok), ...]。返回距 t_us 最近且在容差内的 open_ok，否则 None。"""
    if not candidates:
        return None
    ts_list = [c[0] for c in candidates]
    idx = bisect.bisect_left(ts_list, t_us)
    best_i = None
    best_d = None
    for j in (idx, idx - 1):
        if 0 <= j < len(candidates):
            d = abs(candidates[j][0] - t_us)
            if best_d is None or d < best_d:
                best_d = d
                best_i = j
    if best_i is None or best_d > max_skew_us:
        return None
    return candidates[best_i][1]


def _rear_view_mirror_timeline_from_candidates(
    candidates,
    topic,
    perception_time_list=None,
    *,
    n_msg=0,
    n_parse_err=0,
    n_no_ts=0,
    n_mirror_empty=0,
):
    """将 CarInfo 候选 ``[(ts_us, open_ok), ...]`` 对齐为后视镜时间线。"""
    if not candidates:
        if n_msg == 0:
            print(
                f"    [INFO] Light bag 无 {topic} 消息（或未能枚举），"
                f"跳过后视镜折叠过滤"
            )
        else:
            print(
                f"    [INFO] {topic} 在 Light bag 共 {n_msg} 条，"
                f"但无「有效后视镜」候选：解析失败 {n_parse_err}，"
                f"无时间戳 {n_no_ts}，misc.rear_view_mirror 为空 {n_mirror_empty} "
                f"（schema 为 repeated bool，至少需 1 个元素）"
            )
        return [], []
    candidates.sort(key=lambda x: x[0])
    if perception_time_list:
        samples = []
        n_no_align = 0
        for evt_t in perception_time_list:
            evt_us = _perception_time_measurement_us(evt_t)
            open_ok = _nearest_mirror_open_sorted(
                candidates, evt_us, _CAR_STATE_MIRROR_MAX_SKEW_US
            )
            if open_ok is None:
                n_no_align += 1
                continue
            samples.append((int(evt_t), open_ok))
        samples.sort(key=lambda x: x[0])
        if not samples:
            print(
                f"    [INFO] {topic} 有 {len(candidates)} 条带后视镜的 CarInfo 样本，"
                f"但 {len(perception_time_list)} 个超声时刻均在 "
                f"±{_CAR_STATE_MIRROR_MAX_SKEW_US}us 内对不上，跳过后视镜过滤"
            )
            return [], []
        if n_no_align:
            print(
                f"    [INFO] 超声事件中 {n_no_align}/{len(perception_time_list)} 个"
                f" 在 ±{_CAR_STATE_MIRROR_MAX_SKEW_US}us 内未匹配 CarInfo；"
                f"时间线按超声时刻 {len(samples)} 点"
            )
    else:
        samples = list(candidates)
    tss = [s[0] for s in samples]
    oks = [s[1] for s in samples]
    return tss, oks


def bag_name_to_config_prefix(bag_name):
    """与 offline_avm config 目录名一致：basename 去扩展名段（取首段）。"""
    return os.path.basename(bag_name).split(".")[0]


def avm_skip_mirror_fold_info(tag_ids):
    """用于 Pipeline：哪些 tag 在至少一个超声波事件时刻被 CarInfo（CAR_STATE_TOPIC）判为后视镜折叠。"""
    folded_tags = set()
    folded_prefixes = set()
    for tag_id in tag_ids:
        tid = int(tag_id)
        reader = BagReader(tag_id=tid)
        reader.scan_ultrasonic_events()
        if not reader.perception_time_list:
            continue
        summary = reader.get_mirror_fold_summary()
        if summary.get("folded"):
            folded_tags.add(tid)
            folded_prefixes |= set(summary.get("folded_heavy_prefixes") or [])
    return folded_tags, folded_prefixes


def is_p01t_vehicle_bag(bag_name):
    """P01T 车型 bag：名称中含 ``-P01T-<车号>``，如 ``YR-P01T-4_...``。"""
    base = os.path.basename(bag_name)
    return re.search(r"-P01T-", base, re.IGNORECASE) is not None


def _is_remote_read_timeout_error(exc: BaseException) -> bool:
    """DpBag / TOS 读超时、STORAGE_ACCESS_ERROR 等（便于日志里关联 tag_id）。"""
    s = (str(exc) + " " + repr(exc)).lower()
    if "timed out" in s or "timeout" in s:
        return True
    if "storage_access" in s or "30201" in s:
        return True
    if "network error occurred while reading" in s:
        return True
    return False


def _emit_pipeline_log(message: str, level: int = logging.WARNING) -> None:
    """写入 pipeline 日志：主进程走 logging；子进程无 handler 时追加 PIPELINE_LOG_FILE。"""
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        logger.log(level, message)
        return
    log_path = os.environ.get("PIPELINE_LOG_FILE")
    if log_path:
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}]   {message}\n")
            # 子进程无 pipeline logger：dpbag 的 E 行只打 stderr；这里补一行带 tag_id 便于终端与日志对齐
            if "tag_id=" in message:
                one = message.replace("\n", " ").strip()
                if len(one) > 400:
                    one = one[:397] + "..."
                print(one, file=sys.stderr, flush=True)
        except OSError:
            print(message, file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


class BagReader:
    """对一个 tag_id 对应的所有 bag 进行统一读取。"""

    def __init__(self, tag_id=None, meta_data=None):
        if meta_data is None:
            meta_data = get_meta_data(tag_id=tag_id)
        self.meta_data = meta_data
        self.trip_id = meta_data['body'][0]['tripId']

        # 与 id_mapping 的键一致，由调用方传入（unpack_tag / save_data / BagReader(tag_id=...)）
        self.tag_id = tag_id

        bags = meta_data['body'][0]['bagsName']
        self.all_light_bags = sorted(b for b in bags if 'Light' in b)
        self.all_heavy_bags = sorted(b for b in bags if 'Heavy' in b)

        self.perception_time_list = []
        self.chaosheng_results = {}
        self.event_light_bags = []
        self.event_heavy_bags = []
        self._ultrasonic_scanned = False
        self._light_payloads_scanned = False
        self._nearest_images_cached = None
        self._obstacle_results_cached = None
        self._pose_results_cached = None
        self._planning_results_cached = None
        self._mirror_timeline_cache = ([], [])
        self._event_mirror_open_cache = {}

    def _warn_bag_read_failed(self, bag_name: str, exc: BaseException, *, phase: str = "") -> None:
        label = f" ({phase})" if phase else ""
        tid = self.tag_id
        prefix = f"tag_id={tid} " if tid is not None else ""
        if tid is not None and _is_remote_read_timeout_error(exc):
            _emit_pipeline_log(
                f"[WARN] {prefix}远端存储/网络超时 bag={bag_name}{label}: {exc}"
            )
        else:
            _emit_pipeline_log(f"[WARN] {prefix}跳过 bag={bag_name}{label}: {exc}")

    # ──────────────────────────────────────────────────────────
    #  超声波事件扫描（Light bag）
    # ──────────────────────────────────────────────────────────

    def scan_ultrasonic_events(self):
        """扫描所有 Light bag，筛选 MODEL_PARKING + PLANNING_STOP_OBSTACLE + ULTRASONIC。

        返回 (perception_time_list, chaosheng_results)
        同时填充 self.event_light_bags / self.event_heavy_bags。
        """
        if self._ultrasonic_scanned:
            return self.perception_time_list, self.chaosheng_results
        for bag_name in self.all_light_bags:
            if is_p01t_vehicle_bag(bag_name):
                tid = self.tag_id
                prefix = f"tag_id={tid} " if tid is not None else ""
                _emit_pipeline_log(
                    f"[WARN] {prefix}[SKIP] P01T 车型 bag，跳过: {os.path.basename(bag_name)}"
                )
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
                self._warn_bag_read_failed(bag_name, e)

        self.event_heavy_bags = sorted(
            b.replace("Light", "Heavy") for b in self.event_light_bags
        )
        self._ultrasonic_scanned = True
        return self.perception_time_list, self.chaosheng_results

    def _ensure_light_payloads_scanned(self):
        """对 event_light_bags 做一次多 topic 合并扫描，复用 obstacle/pose/planning/mirror。"""
        if self._light_payloads_scanned:
            return
        self.scan_ultrasonic_events()

        obstacle_results = {t: {} for t in self.perception_time_list}
        obstacle_min_diffs = {t: float('inf') for t in self.perception_time_list}
        pose_results = {t: {} for t in self.perception_time_list}
        pose_min_diffs = {t: float('inf') for t in self.perception_time_list}
        planning_results = {t: {} for t in self.perception_time_list}
        planning_min_diffs = {t: float('inf') for t in self.perception_time_list}

        mirror_topic = getattr(config, "CAR_STATE_TOPIC", "") or ""
        topics = [config.OBSTACLE_TOPIC, config.POSE_TOPIC, config.PLANNING_TOPIC]
        if CarInfo is not None and mirror_topic:
            topics.append(mirror_topic)
        mirror_candidates = []
        n_msg = n_parse_err = n_no_ts = n_mirror_empty = 0

        for bag_name in self.event_light_bags:
            try:
                with DpBag(bag=bag_name) as bag:
                    for topic, msg, meta in bag.read_messages(
                        topics=topics,
                        dpbag_name=bag_name,
                        force_get_data_by_raw=True,
                    ):
                        raw_msg = strip_header(msg.data)

                        if topic == config.OBSTACLE_TOPIC:
                            obj = PerceptionObstacles()
                            obj.ParseFromString(raw_msg)
                            t = obj.time_measurement
                            data_list = None
                            for evt_t in self.perception_time_list:
                                diff = abs(t - evt_t)
                                if diff < obstacle_min_diffs[evt_t]:
                                    if data_list is None:
                                        data_list = [
                                            MessageToDict(item) if hasattr(item, 'DESCRIPTOR') else item
                                            for item in obj.perception_obstacle
                                        ]
                                    obstacle_min_diffs[evt_t] = diff
                                    obstacle_results[evt_t] = {
                                        'obstacle': data_list,
                                        'time_diff': diff,
                                    }
                            continue

                        if topic == config.POSE_TOPIC:
                            obj = Ins()
                            obj.ParseFromString(raw_msg)
                            t = obj.measurement_time
                            pos = euler = None
                            for evt_t in self.perception_time_list:
                                diff = abs(t - evt_t)
                                if diff < pose_min_diffs[evt_t]:
                                    if pos is None:
                                        pos = obj.position
                                        euler = obj.euler_angles
                                    pose_min_diffs[evt_t] = diff
                                    pose_results[evt_t] = {
                                        'position': [pos.x, pos.y, pos.z],
                                        'euler_angles': [euler.x, euler.y, euler.z],
                                        'time_diff': diff,
                                    }
                            continue

                        if topic == config.PLANNING_TOPIC:
                            obj = ADCTrajectory()
                            obj.ParseFromString(raw_msg)
                            t = obj.header.timestamp_sec * 1e6
                            traj_points = None
                            for evt_t in self.perception_time_list:
                                diff = abs(t - evt_t)
                                if diff < planning_min_diffs[evt_t]:
                                    if traj_points is None:
                                        traj_points = [
                                            {
                                                'relative_time': pt.relative_time,
                                                'x': pt.path_point.x,
                                                'y': pt.path_point.y,
                                            }
                                            for pt in obj.trajectory_point
                                        ]
                                    planning_min_diffs[evt_t] = diff
                                    planning_results[evt_t] = traj_points
                            continue

                        if topic == mirror_topic and CarInfo is not None:
                            n_msg += 1
                            obj = CarInfo()
                            try:
                                obj.ParseFromString(raw_msg)
                            except Exception:
                                n_parse_err += 1
                                continue
                            ts_car = _car_info_timestamp_us(obj, meta)
                            if ts_car is None:
                                n_no_ts += 1
                                continue
                            open_ok = _misc_rear_view_mirror_all_open(obj)
                            if open_ok is None:
                                n_mirror_empty += 1
                                continue
                            mirror_candidates.append((ts_car, open_ok))
            except Exception as e:
                self._warn_bag_read_failed(bag_name, e, phase="light payloads")

        mirror_ts, mirror_open = _rear_view_mirror_timeline_from_candidates(
            mirror_candidates,
            mirror_topic,
            self.perception_time_list,
            n_msg=n_msg,
            n_parse_err=n_parse_err,
            n_no_ts=n_no_ts,
            n_mirror_empty=n_mirror_empty,
        )
        self._mirror_timeline_cache = (mirror_ts, mirror_open)
        self._event_mirror_open_cache = {
            int(t): ok for t, ok in zip(mirror_ts, mirror_open)
        }
        self._obstacle_results_cached = obstacle_results
        self._pose_results_cached = pose_results
        self._planning_results_cached = planning_results
        self._light_payloads_scanned = True

    def get_mirror_fold_summary(self):
        """返回当前 tag 的后视镜折叠摘要，供落盘缓存 / pipeline 复用。"""
        self._ensure_light_payloads_scanned()
        prefixes = sorted({bag_name_to_config_prefix(b) for b in self.event_heavy_bags})
        aligned = [
            {"timestamp": int(t), "all_open": bool(ok)}
            for t, ok in zip(*self._mirror_timeline_cache)
        ]
        folded = any(not item["all_open"] for item in aligned)
        folded_prefixes = prefixes if folded else []
        return {
            "car_state_topic": getattr(config, "CAR_STATE_TOPIC", "") or "",
            "perception_time_list": [int(t) for t in self.perception_time_list],
            "aligned_event_mirror": aligned,
            "folded": folded,
            "folded_heavy_prefixes": folded_prefixes,
            "event_heavy_prefixes": prefixes,
        }

    # ──────────────────────────────────────────────────────────
    #  鱼眼图最近邻提取（Heavy bag）
    # ──────────────────────────────────────────────────────────

    def extract_nearest_images(self):
        """从 Heavy bag 中为每个超声波事件提取全局最近邻 4 路鱼眼图。

        返回:
          {per_t: {cam_name: {'image': ndarray, 'timestamp_us': int, 'source_bag': str}}}
        """
        if self._nearest_images_cached is not None:
            return self._nearest_images_cached
        self._ensure_light_payloads_scanned()
        image_results = {t: {} for t in self.perception_time_list}
        min_diffs = {
            t: {cam: float('inf') for cam in config.CAMERA_NAMES}
            for t in self.perception_time_list
        }

        evt_mirror = self._event_mirror_open_cache
        topic_to_camera = dict(zip(config.CAMERA_TOPICS, config.CAMERA_NAMES))
        update_counts = {cam: 0 for cam in config.CAMERA_NAMES}
        mirror_fold_skips = {cam: 0 for cam in config.CAMERA_NAMES}

        print("    提取 4 路相机最近邻帧（每个 Heavy bag 仅打开一次）...")
        if self._mirror_timeline_cache[0]:
            print(
                f"    CarInfo 后视镜时间线 {len(self._mirror_timeline_cache[0])} 点（超声时刻 + Light bag {config.CAR_STATE_TOPIC}），"
                f"折叠帧不参与鱼眼最近邻匹配"
            )

        for heavy_bag in self.event_heavy_bags:
            try:
                with DpBag(bag=heavy_bag) as bag:
                    for topic, msg, _ in bag.read_messages(
                        topics=config.CAMERA_TOPICS,
                        dpbag_name=heavy_bag,
                        force_get_data_by_raw=True,
                    ):
                        cam_name = topic_to_camera.get(topic)
                        if cam_name is None:
                            continue
                        obj = CompressedImage()
                        raw_msg = strip_header(msg.data)
                        obj.ParseFromString(raw_msg)
                        ts_us = int(obj.header.timestamp_sec * 1e6)

                        update_targets = []
                        for evt_t in self.perception_time_list:
                            diff = abs(ts_us - evt_t)
                            if diff < min_diffs[evt_t][cam_name]:
                                mir = evt_mirror.get(int(evt_t))
                                if mir is False:
                                    mirror_fold_skips[cam_name] += 1
                                    continue
                                update_targets.append((evt_t, diff))
                        if not update_targets:
                            continue

                        img = cv2.imdecode(
                            np.frombuffer(obj.data, np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        for evt_t, diff in update_targets:
                            min_diffs[evt_t][cam_name] = diff
                            image_results[evt_t][cam_name] = {
                                'image': img,
                                'timestamp_us': ts_us,
                                'source_bag': heavy_bag,
                            }
                            update_counts[cam_name] += 1
            except Exception as e:
                self._warn_bag_read_failed(heavy_bag, e, phase="heavy fisheye")

        for cam_name in config.CAMERA_NAMES:
            print(f"      -> {cam_name} 更新 {update_counts[cam_name]} 次匹配")
            if mirror_fold_skips[cam_name]:
                print(
                    f"      -> {cam_name} 后视镜折叠: 跳过 {mirror_fold_skips[cam_name]} 个鱼眼候选帧"
                    f"（不参与该路最近邻；topic={config.CAR_STATE_TOPIC}）"
                )

        self._nearest_images_cached = image_results
        return image_results

    # ──────────────────────────────────────────────────────────
    #  障碍物提取（Light bag）
    # ──────────────────────────────────────────────────────────

    def extract_obstacles(self):
        """从 Light bag 读取 /perception/objects，按最近邻时间戳匹配。

        返回: {per_t: {'obstacle': [...], 'time_diff': float}}
        """
        self._ensure_light_payloads_scanned()
        return self._obstacle_results_cached

    # ──────────────────────────────────────────────────────────
    #  位姿提取（Light bag）
    # ──────────────────────────────────────────────────────────

    def extract_poses(self):
        """从 Light bag 读取 /localization/pose，按最近邻时间戳匹配。

        返回: {per_t: {'position': [x,y,z], 'euler_angles': [x,y,z], 'time_diff': float}}
        """
        self._ensure_light_payloads_scanned()
        return self._pose_results_cached

    # ──────────────────────────────────────────────────────────
    #  规划轨迹提取（Light bag）
    # ──────────────────────────────────────────────────────────

    def extract_planning(self):
        """从 Light bag 读取 /planner/trajectory，按最近邻时间戳匹配。

        返回: {per_t: [{'relative_time': ..., 'x': ..., 'y': ...}, ...]}
        """
        self._ensure_light_payloads_scanned()
        return self._planning_results_cached
