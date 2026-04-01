"""
将远端 bag 解包为 offline_avm_generate 工具所需的输入目录结构。
仅解包超声波触发停车事件(MODEL_PARKING + PLANNING_STOP_OBSTACLE + ULTRASONIC)
对应时刻的最近邻图像帧。

P01T 车型 bag（名称中含 ``-P01T-<车号>``，如 ``YR-P01T-4_...``）在
``BagReader.scan_ultrasonic_events`` 中会被跳过，不参与解包与事件关联。

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
import sys

import cv2

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from get_data.bag_reader import BagReader
from get_data.dr_client import DRFILE_AVAILABLE, download_trip_config


# ============ 工具函数 ============

def extract_bag_prefix(bag_name):
    """从 bag 名中去掉 .Heavy_Topic_Group.bag 等后缀，只保留前缀。"""
    return os.path.basename(bag_name).split('.')[0]


def extract_yyyymm(bag_name):
    """从 bag 名中提取 YYYYMM。"""
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

            per_bag_frames.setdefault(src_bag, {cam: [] for cam in config.CAMERA_NAMES})
            per_bag_frames[src_bag][cam_name].append((ts_us, fname))

    for bag_data in per_bag_frames.values():
        for cam_name in config.CAMERA_NAMES:
            bag_data[cam_name].sort(key=lambda x: x[0])

    return per_bag_frames


def save_data_index(config_dir, camera_frames, bag_prefix):
    """生成 data_index.csv（以 panoramic_1 为参考，最近邻匹配其余相机）。"""
    ref_frames = camera_frames[config.CAMERA_NAMES[0]]
    if not ref_frames:
        print("    [WARN] 未提取到任何帧，跳过 data_index.csv")
        return

    other_ts = {}
    other_fnames = {}
    for cam_name in config.CAMERA_NAMES[1:]:
        frames = camera_frames[cam_name]
        other_ts[cam_name] = [f[0] for f in frames]
        other_fnames[cam_name] = [f[1] for f in frames]

    csv_path = os.path.join(config_dir, "data_index.csv")
    with open(csv_path, "w", newline="") as f:
        header = ", ".join(["TIMESTAMP"] + config.CAMERA_NAMES + ["Data_dir"])
        f.write(header + "\n")
        for ref_ts, ref_fname in ref_frames:
            parts = [str(ref_ts), ref_fname]
            for cam_name in config.CAMERA_NAMES[1:]:
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

def unpack_tag(tag_id, output_root=None, reader=None, return_reader=False):
    """根据 tag_id 扫描超声波事件并解包对应的最近邻图像帧。"""
    if output_root is None:
        output_root = config.SAMPLES_DIR

    print(f"\n{'=' * 60}")
    print(f"Tag ID: {tag_id}")
    print(f"{'=' * 60}")

    if reader is None:
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
            cfg_dir = os.path.join(output_root, "config", yyyymm, bag_prefix)
            os.makedirs(cfg_dir, exist_ok=True)
            for cfg_name in ["cameras.cfg", "ground.cfg"]:
                cfg_path = os.path.join(cfg_dir, cfg_name)
                if os.path.exists(cfg_path):
                    print(f"    {bag_prefix}/{cfg_name} 已存在，跳过下载")
                    continue
                try:
                    cfg_bytes = download_trip_config(reader.trip_id, cfg_name)
                    with open(cfg_path, "wb") as f:
                        f.write(cfg_bytes)
                    print(f"    已下载 {bag_prefix}/{cfg_name}")
                except Exception as e:
                    print(f"    [WARN] 下载 {bag_prefix}/{cfg_name} 失败: {e}")

    # 第三步：从 Heavy bag 提取最近邻鱼眼帧并保存；后视镜折叠过滤在 extract_nearest_images 内从 Light bag 读 CAR_STATE_TOPIC（CarInfo）
    print(
        f"\n  [鱼眼/图像] 跨 {len(valid_heavy_bags)} 个 Heavy bag 全局匹配 4 路帧 "
        f"（车身 CarInfo 仅从 Light bag 读 {config.CAR_STATE_TOPIC}，见下方日志） ..."
    )
    image_results = reader.extract_nearest_images()
    per_bag_frames = save_images_to_disk(image_results, output_root)

    # 第四步：为每个 bag 目录生成各自的 data_index.csv
    for heavy_bag in valid_heavy_bags:
        if heavy_bag not in per_bag_frames:
            continue
        bag_prefix, yyyymm = yyyymm_map[heavy_bag]
        cfg_dir = os.path.join(output_root, "config", yyyymm, bag_prefix)
        save_data_index(cfg_dir, per_bag_frames[heavy_bag], bag_prefix)

    print(f"\n完成! 输出目录: {output_root}")
    if return_reader:
        return reader


if __name__ == "__main__":
    tag_id_list = [
        120281390,
    ]
    for tag_id in tag_id_list:
        unpack_tag(tag_id)
