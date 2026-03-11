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

import bisect
import os
import re

import cv2

from bag_reader import BagReader, CAMERA_NAMES

try:
    from drfile.drfile_client import DrFileClient, ClientConfiguration
    from drfile.modules.sdk.model.request.file_transfer_request import GetFileRequest
    from dplib.env import EnvConfig
    from dplib import DpEngine
    DRFILE_AVAILABLE = True
except ImportError:
    print("警告: drfile 模块未安装，config 文件下载功能将不可用")
    DRFILE_AVAILABLE = False

# ============ 配置 ============
OUTPUT_ROOT = "/mnt/public-data/user/ziroujiang/avp/samples"

PERCEPTIONTEAM_USERNAME = "perceptionteam"
PERCEPTIONTEAM_PASSWORD = "r6zR86V4*+=*"
DR_ENDPOINT = os.getenv("DR_ENDPOINT", "https://drplatform-backend.deeproute.cn")

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


# ============ 保存逻辑 ============

def save_images_to_disk(image_results, output_root):
    """将 BagReader 返回的内存图像按 AVM 目录结构保存到磁盘。

    返回 per_bag_frames: {heavy_bag: {cam_name: [(timestamp_us, filename), ...]}}
    """
    per_bag_frames = {}

    for evt_t, cam_dict in image_results.items():
        for cam_name, info in cam_dict.items():
            img = info['image']
            ts_us = info['timestamp_us']
            src_bag = info['source_bag']

            if img is None:
                continue

            bag_prefix = extract_bag_prefix(src_bag)
            yyyymm = extract_yyyymm(bag_prefix)
            if not yyyymm:
                continue

            cam_dir = os.path.join(output_root, cam_name, yyyymm, bag_prefix)
            os.makedirs(cam_dir, exist_ok=True)

            fname = ts_us_to_filename(ts_us)
            cv2.imwrite(os.path.join(cam_dir, fname), img)

            per_bag_frames.setdefault(src_bag, {cam: [] for cam in CAMERA_NAMES})
            per_bag_frames[src_bag][cam_name].append((ts_us, fname))

    for bag_data in per_bag_frames.values():
        for cam_name in CAMERA_NAMES:
            bag_data[cam_name].sort(key=lambda x: x[0])

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


# ============ 核心入口 ============

def unpack_tag(tag_id, output_root=OUTPUT_ROOT):
    """根据 tag_id 扫描超声波事件并解包对应的最近邻图像帧。"""
    print(f"\n{'=' * 60}")
    print(f"Tag ID: {tag_id}")
    print(f"{'=' * 60}")

    reader = BagReader(tag_id=tag_id)
    print(f"Trip ID: {reader.trip_id}")

    # 第一步：扫描超声波停车事件
    print(f"\n扫描超声波停车事件 (Light bags: {len(reader.all_light_bags)}) ...")
    perception_time_list, _ = reader.scan_ultrasonic_events()
    print(f"  发现 {len(perception_time_list)} 个超声波停车事件")

    if not perception_time_list:
        print("  无超声波事件，跳过")
        return

    print(f"  关联 Heavy bags ({len(reader.event_heavy_bags)}):")
    valid_heavy_bags = []
    yyyymm_map = {}
    for heavy_bag in reader.event_heavy_bags:
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

    # 第二步：下载配置文件
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
                    cfg_bytes = download_config_file(reader.trip_id, cfg_name)
                    with open(cfg_path, "wb") as f:
                        f.write(cfg_bytes)
                    print(f"    已下载 {bag_prefix}/{cfg_name}")
                except Exception as e:
                    print(f"    [WARN] 下载 {bag_prefix}/{cfg_name} 失败: {e}")

    # 第三步：提取最近邻图像帧并保存
    print(f"\n  跨 {len(valid_heavy_bags)} 个 Heavy bag 全局匹配图像帧 ...")
    image_results = reader.extract_nearest_images()
    per_bag_frames = save_images_to_disk(image_results, output_root)

    # 第四步：为每个 bag 目录生成各自的 data_index.csv
    for heavy_bag in valid_heavy_bags:
        if heavy_bag not in per_bag_frames:
            continue
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
