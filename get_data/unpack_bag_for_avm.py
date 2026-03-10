"""
将远端 bag 解包为 offline_avm_generate 工具所需的输入目录结构。
仅解包超声波触发停车事件(MODEL_PARKING + PLANNING_STOP_OBSTACLE + ULTRASONIC)
对应时刻的最近邻图像帧。

输出目录结构（兼容 /mnt/public-data/training/avp/mighty/samples 格式）:
  output_root/
  ├── config/YYYYMM/{bag_name}/
  │   ├── data_index.csv          # TIMESTAMP, panoramic_1..4, Data_dir
  │   ├── ground.cfg
  │   └── cameras.cfg
  ├── panoramic_1/YYYYMM/{bag_name}/{sec}_{usec}.jpg
  ├── panoramic_2/YYYYMM/{bag_name}/{sec}_{usec}.jpg
  ├── panoramic_3/YYYYMM/{bag_name}/{sec}_{usec}.jpg
  └── panoramic_4/YYYYMM/{bag_name}/{sec}_{usec}.jpg

用法:
  python unpack_bag_for_avm.py
  修改底部 tag_id_list 即可指定要处理的 tag。
"""

import json
import os
import sys
import re
import bisect
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.join(current_dir, "proto")
sys.path.insert(0, proto_dir)

from dpbag import strip_header
from dpbag.bag.bag import DpBag

os.environ['DPBAG_DP_USERNAME'] = 'perceptionteam'
os.environ['DPBAG_DP_PASSWORD'] = 'r6zR86V4*+=*'

try:
    from drivers.sensor_image_pb2 import CompressedImage
    from perception.deeproute_perception_obstacle_pb2 import PerceptionObstacles
except ImportError as e:
    print(f"Proto import error: {e}")
    sys.exit(1)

from google.protobuf.json_format import MessageToDict

try:
    from drfile.drfile_client import DrFileClient, ClientConfiguration
    from drfile.modules.sdk.model.request.file_transfer_request import GetFileRequest
    from dplib.env import EnvConfig
    from dplib import DpEngine
    DRFILE_AVAILABLE = True
except ImportError:
    print("警告: drfile 模块未安装，config 文件下载功能将不可用")
    DRFILE_AVAILABLE = False

from get_meta_data import get_meta_data

# ============ 配置 ============
OUTPUT_ROOT = "/mnt/public-data/user/ziroujiang/avp/samples"

PERCEPTIONTEAM_USERNAME = "perceptionteam"
PERCEPTIONTEAM_PASSWORD = "r6zR86V4*+=*"
DR_ENDPOINT = os.getenv("DR_ENDPOINT", "https://drplatform-backend.deeproute.cn")

CAMERA_TOPICS = [
    "/sensors/camera/panoramic_1_raw_data/compressed_proto",
    "/sensors/camera/panoramic_2_raw_data/compressed_proto",
    "/sensors/camera/panoramic_3_raw_data/compressed_proto",
    "/sensors/camera/panoramic_4_raw_data/compressed_proto",
]
CAMERA_NAMES = ["panoramic_1", "panoramic_2", "panoramic_3", "panoramic_4"]

CHAOSHENG_TOPIC = "/planner/stop_objects"

# ============ DrFile 客户端（单例） ============
_dr_client = None
_dr_engine = None


def _get_env_config():
    cfg = EnvConfig()
    cfg.retry_times = 5
    cfg.dplib_retry_sleep_time = 20
    return cfg


def get_dr_client():
    global _dr_client
    if not DRFILE_AVAILABLE:
        return None
    if _dr_client is None:
        _dr_client = DrFileClient(
            ClientConfiguration(
                username=PERCEPTIONTEAM_USERNAME,
                password=PERCEPTIONTEAM_PASSWORD,
                endpoint=DR_ENDPOINT,
                token_expired_time=-1,
                show_config=False,
                show_progress=False,
                show_summary=False,
            )
        )
    return _dr_client


def get_dr_engine():
    global _dr_engine
    if not DRFILE_AVAILABLE:
        return None
    if _dr_engine is None:
        _dr_engine = DpEngine(
            username=PERCEPTIONTEAM_USERNAME,
            password=PERCEPTIONTEAM_PASSWORD,
            env_config=_get_env_config(),
        )
    return _dr_engine


def download_config_file(trip_id, config_filename):
    """从 DrFile 下载配置文件的原始字节。"""
    dr_engine = get_dr_engine()
    trip_meta = dr_engine.get_trip_meta(trip_id=int(trip_id))
    trip_name = trip_meta.trip_name
    dr_client = get_dr_client()
    return dr_client.download_bytes(
        GetFileRequest(
            namespace='trip',
            path=f"/{trip_name}/configs/{config_filename}",
        )
    )


# ============ 工具函数 ============

def extract_bag_prefix(bag_name):
    """从 bag 名中去掉 .Heavy_Topic_Group.bag 等后缀，只保留前缀。
    例: YR-C01-35_20260120_062850.Heavy_Topic_Group.bag -> YR-C01-35_20260120_062850
    """
    return os.path.basename(bag_name).split('.')[0]


def extract_yyyymm(bag_name):
    """从 bag 名(如 YR-C01-35_20260120_062850.Heavy_Topic_Group.bag)中提取 YYYYMM。"""
    match = re.search(r'_(\d{8})_', bag_name)
    if match:
        return match.group(1)[:6]
    return None


def bisect_closest(sorted_ts_list, target):
    """在已排序的时间戳列表中找到最接近 target 的索引。"""
    idx = bisect.bisect_left(sorted_ts_list, target)
    if idx == 0:
        return 0
    if idx == len(sorted_ts_list):
        return len(sorted_ts_list) - 1
    if abs(sorted_ts_list[idx] - target) < abs(sorted_ts_list[idx - 1] - target):
        return idx
    return idx - 1


def ts_us_to_filename(ts_us):
    """微秒时间戳 → 文件名，如 1708430665466796 → '1708430665_466796.jpg'"""
    sec = ts_us // 1_000_000
    usec = ts_us % 1_000_000
    return f"{sec}_{usec:06d}.jpg"


# ============ 核心逻辑 ============

def scan_ultrasonic_events(light_bags):
    """
    从 Light bag 中扫描超声波触发的停车事件，返回:
      - perception_time_list: 事件时间戳列表
      - event_light_bags: 包含事件的 Light bag 列表
    """
    perception_time_list = []
    event_light_bags = []

    for bag_name in light_bags:
        try:
            with DpBag(bag=bag_name) as bag:
                for topic, msg, _ in bag.read_messages(
                    topics=[CHAOSHENG_TOPIC],
                    dpbag_name=bag_name,
                    force_get_data_by_raw=True,
                ):
                    obj = PerceptionObstacles()
                    raw_msg = strip_header(msg.data)
                    obj.ParseFromString(raw_msg)
                    per_t = obj.time_measurement

                    has_ultrasonic = False
                    for item in obj.perception_obstacle:
                        if hasattr(item, 'DESCRIPTOR'):
                            data = MessageToDict(item)
                        else:
                            data = item
                        model_type = data.get("modelType", "")
                        obs_type = data.get("type", "")
                        sensor_type = data.get("sensorType", "")
                        if (model_type == 'MODEL_PARKING'
                                and obs_type == 'PLANNING_STOP_OBSTACLE'
                                and sensor_type == 'ULTRASONIC'):
                            has_ultrasonic = True
                            break

                    if has_ultrasonic:
                        if bag_name not in event_light_bags:
                            event_light_bags.append(bag_name)
                        perception_time_list.append(per_t)
        except Exception as e:
            print(f"    [WARN] 跳过 {bag_name}: {e}")

    return perception_time_list, event_light_bags


def extract_nearest_frames(heavy_bags, perception_time_list, output_root, yyyymm_map):
    """
    跨所有 Heavy bag 全局匹配：为每个超声波事件时间戳找到全局最近邻的 4 路相机图像。
    返回 per-bag 结构: {heavy_bag: {cam_name: [(timestamp_us, filename), ...]}}

    yyyymm_map: {heavy_bag_name: (bag_prefix, yyyymm)}
    """
    per_bag_frames = {hb: {cam: [] for cam in CAMERA_NAMES} for hb in heavy_bags}

    for topic, cam_name in zip(CAMERA_TOPICS, CAMERA_NAMES):
        print(f"    提取 {cam_name} ...")

        min_diffs = {t: float('inf') for t in perception_time_list}
        best_data = {t: None for t in perception_time_list}

        for heavy_bag in heavy_bags:
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

                        for evt_time in perception_time_list:
                            diff = abs(ts_us - evt_time)
                            if diff < min_diffs[evt_time]:
                                min_diffs[evt_time] = diff
                                best_data[evt_time] = (ts_us, obj.data, heavy_bag)
            except Exception as e:
                print(f"      [WARN] 跳过 {heavy_bag}: {e}")

        saved = set()
        count = 0
        for evt_time in perception_time_list:
            if best_data[evt_time] is None:
                continue
            ts_us, img_bytes, src_bag = best_data[evt_time]
            if ts_us in saved:
                continue
            saved.add(ts_us)

            img = cv2.imdecode(
                np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                continue

            bag_prefix, yyyymm = yyyymm_map[src_bag]
            cam_dir = os.path.join(output_root, cam_name, yyyymm, bag_prefix)
            os.makedirs(cam_dir, exist_ok=True)

            fname = ts_us_to_filename(ts_us)
            cv2.imwrite(os.path.join(cam_dir, fname), img)
            per_bag_frames[src_bag][cam_name].append((ts_us, fname))
            count += 1

        for hb in heavy_bags:
            per_bag_frames[hb][cam_name].sort(key=lambda x: x[0])
        print(f"      -> {count} 帧")

    return per_bag_frames


def save_data_index(config_dir, camera_frames, bag_prefix):
    """生成 data_index.csv（以 panoramic_1 为参考，最近邻匹配其余相机）。
    格式兼容 mighty/samples: TIMESTAMP, panoramic_1..4, Data_dir，使用 ', ' 分隔。
    """
    ref_frames = camera_frames[CAMERA_NAMES[0]]
    if not ref_frames:
        print("    [WARN] 未提取到任何帧，跳过 data_index.csv")
        return

    other_ts = {}
    other_fnames = {}
    for cam_name in CAMERA_NAMES[1:]:
        frames = camera_frames[cam_name]
        other_ts[cam_name] = [f[0] for f in frames]
        other_fnames[cam_name] = [f[1] for f in frames]

    csv_path = os.path.join(config_dir, "data_index.csv")
    with open(csv_path, "w", newline="") as f:
        header = ", ".join(["TIMESTAMP"] + CAMERA_NAMES + ["Data_dir"])
        f.write(header + "\n")
        for ref_ts, ref_fname in ref_frames:
            parts = [str(ref_ts), ref_fname]
            for cam_name in CAMERA_NAMES[1:]:
                ts_list = other_ts[cam_name]
                if not ts_list:
                    parts.append("")
                    continue
                idx = bisect_closest(ts_list, ref_ts)
                parts.append(other_fnames[cam_name][idx])
            parts.append(bag_prefix)
            f.write(", ".join(parts) + "\n")

    print(f"    data_index.csv: {len(ref_frames)} 条记录")


def unpack_tag(tag_id, output_root=OUTPUT_ROOT):
    """根据 tag_id 扫描超声波事件并解包对应的最近邻图像帧。"""
    print(f"\n{'=' * 60}")
    print(f"Tag ID: {tag_id}")
    print(f"{'=' * 60}")

    meta_data = get_meta_data(tag_id=tag_id)
    if not meta_data:
        print(f"[ERROR] 获取 meta_data 失败: tag_id={tag_id}")
        return

    trip_id = meta_data['body'][0]['tripId']
    bag_name_list = meta_data['body'][0]['bagsName']
    light_bags = sorted([b for b in bag_name_list if 'Light' in b])
    print(f"Trip ID: {trip_id}")

    # 第一步：从 Light bag 扫描超声波停车事件
    print(f"\n扫描超声波停车事件 (Light bags: {len(light_bags)}) ...")
    perception_time_list, event_light_bags = scan_ultrasonic_events(light_bags)
    print(f"  发现 {len(perception_time_list)} 个超声波停车事件")

    if not perception_time_list:
        print("  无超声波事件，跳过")
        return

    # 将有事件的 Light bag 名转为对应的 Heavy bag 名
    heavy_bags = sorted([b.replace("Light", "Heavy") for b in event_light_bags])
    print(f"  关联 Heavy bags ({len(heavy_bags)}):")

    # 构建 bag -> (prefix, yyyymm) 映射，并过滤无效 bag
    yyyymm_map = {}
    valid_heavy_bags = []
    for heavy_bag in heavy_bags:
        bag_prefix = extract_bag_prefix(heavy_bag)
        yyyymm = extract_yyyymm(bag_prefix)
        if not yyyymm:
            print(f"    [SKIP] 无法从 bag 名提取 YYYYMM: {bag_prefix}")
            continue
        yyyymm_map[heavy_bag] = (bag_prefix, yyyymm)
        valid_heavy_bags.append(heavy_bag)
        print(f"    - {bag_prefix} (YYYYMM={yyyymm})")

    if not valid_heavy_bags:
        print("  无有效 Heavy bag，跳过")
        return

    # 第二步：下载配置文件（同一 trip 的 config 相同，每个 bag 目录各存一份）
    if DRFILE_AVAILABLE:
        for heavy_bag in valid_heavy_bags:
            bag_prefix, yyyymm = yyyymm_map[heavy_bag]
            config_dir = os.path.join(output_root, "config", yyyymm, bag_prefix)
            os.makedirs(config_dir, exist_ok=True)
            for cfg_name in ["cameras.cfg", "ground.cfg"]:
                cfg_path = os.path.join(config_dir, cfg_name)
                if os.path.exists(cfg_path):
                    print(f"    {bag_prefix}/{cfg_name} 已存在，跳过下载")
                    continue
                try:
                    cfg_bytes = download_config_file(trip_id, cfg_name)
                    with open(cfg_path, "wb") as f:
                        f.write(cfg_bytes)
                    print(f"    已下载 {bag_prefix}/{cfg_name}")
                except Exception as e:
                    print(f"    [WARN] 下载 {bag_prefix}/{cfg_name} 失败: {e}")

    # 第三步：跨所有 Heavy bag 全局匹配，提取最近邻图像帧
    print(f"\n  跨 {len(valid_heavy_bags)} 个 Heavy bag 全局匹配图像帧 ...")
    per_bag_frames = extract_nearest_frames(
        valid_heavy_bags, perception_time_list, output_root, yyyymm_map
    )

    # 第四步：为每个 bag 目录生成各自的 data_index.csv
    for heavy_bag in valid_heavy_bags:
        bag_prefix, yyyymm = yyyymm_map[heavy_bag]
        config_dir = os.path.join(output_root, "config", yyyymm, bag_prefix)
        save_data_index(config_dir, per_bag_frames[heavy_bag], bag_prefix)

    print(f"\n完成! 输出目录: {output_root}")


if __name__ == "__main__":
    tag_id_list = [
        97020543,
    ]
    for tag_id in tag_id_list:
        unpack_tag(tag_id)
