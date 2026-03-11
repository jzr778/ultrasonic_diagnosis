"""
⚠️ 已废弃 — 此模块的功能已合并到 bag_reader.py (BagReader 类)。
保留此文件仅供参考，新代码请直接使用 bag_reader.BagReader。
"""

import json
import os
import sys
import cv2
import numpy as np


# 修改1: 将当前目录下的proto路径添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.join(current_dir, "proto")
sys.path.insert(0, proto_dir)


from dpbag import strip_header
from dpbag.bag.bag import DpBag

# 设置环境变量
os.environ['DPBAG_DP_USERNAME'] = 'perceptionteam'
os.environ['DPBAG_DP_PASSWORD'] = 'r6zR86V4*+=*'
# proto
# proto - 修改2: 使用相对路径导入
try:
    from drivers.sensor_image_pb2 import CompressedImage
    from perception.deeproute_perception_obstacle_pb2 import PerceptionObstacles
    from drivers.gnss.ins_pb2 import Ins
    from planning.planning_pb2 import ADCTrajectory
except ImportError as e:
    print(f"Proto import error: {e}")
    print("Please ensure proto files are in the correct location")
    sys.exit(1)
# get data
from get_meta_data import get_meta_data
from google.protobuf.json_format import MessageToDict

chaosheng_obstacle_topic_map = {
    "/planner/stop_objects": PerceptionObstacles,
}

camera_topic_map = {
    "/sensors/camera/panoramic_1_raw_data/compressed_proto": CompressedImage,
    "/sensors/camera/panoramic_2_raw_data/compressed_proto": CompressedImage,
    "/sensors/camera/panoramic_3_raw_data/compressed_proto": CompressedImage,
    "/sensors/camera/panoramic_4_raw_data/compressed_proto": CompressedImage
}
obstacle_topic_map = {
    "/perception/objects": PerceptionObstacles,
}
pose_topic_map = {
    "/localization/pose": Ins,
}
plan_topic_map = {
    "/planner/trajectory": ADCTrajectory,
}


def get_image_obstacle_pose(meta_data):

    bag_name_list = meta_data['body'][0]['bagsName']

    # 分别提取包名
    heavy_bags = sorted([bag_name for bag_name in bag_name_list if 'Heavy' in bag_name])
    light_bags = sorted([bag_name for bag_name in bag_name_list if 'Light' in bag_name])
    new_light_bags = []

    # 从light_bag读取超声的障碍物信息和时间
    perception_time_list = []
    chaosheng_results = {}
    for i in range(len(light_bags)):
        bag_name = light_bags[i]
        topic_list = list(chaosheng_obstacle_topic_map.keys())
        with DpBag(bag=bag_name) as bag:
            for topic, msg, _ in bag.read_messages(
                    topics=topic_list,
                    dpbag_name=bag_name,
                    force_get_data_by_raw=True,
            ):
                # Parse proto
                obj = chaosheng_obstacle_topic_map[topic]()
                raw_msg = strip_header(msg.data)
                obj.ParseFromString(raw_msg)
                obstacle = obj.perception_obstacle
                per_t = obj.time_measurement
                data_list = []
                for item in obstacle:
                    # 如果元素是protobuf消息，先转换为字典
                    if hasattr(item, 'DESCRIPTOR'):  # 是protobuf消息
                        data = MessageToDict(item)
                    else:
                        data = item
                    modelType = data.get("modelType", "")
                    type = data.get("type", "")
                    sensorType = data.get("sensorType", "")
                    # print(modelType + "***" + type + "***" + sensorType)
                    if modelType == 'MODEL_PARKING' and type == 'PLANNING_STOP_OBSTACLE' and sensorType == 'ULTRASONIC':
                        data_list.append(data)
                if data_list:
                    if bag_name not in new_light_bags:
                        new_light_bags.append(bag_name)
                    perception_time_list.append(per_t)
                    chaosheng_results[per_t] = data_list

    # 更新bag
    light_bags = new_light_bags
    heavy_bags = [bag.replace("Light", "Heavy") for bag in light_bags]

    # 从heavy_bag读取图像
    image_results = {}
    # 初始化最小差异字典，用于跟踪每个时间戳的最小差异
    min_diffs = {}
    for time in perception_time_list:
        image_results[time] = {}
        min_diffs[time] = {}
        for i in range(len(camera_topic_map)):
            topic_list = [key for j, key in enumerate(camera_topic_map) if j == i]
            camera_name = topic_list[0].split('/')[-2]
            # 初始化每个相机的差异为无穷大
            min_diffs[time][camera_name] = float('inf')
    # 遍历所有bag文件
    for i in range(len(heavy_bags)):
        bag_name = heavy_bags[i]
        for i in range(len(camera_topic_map)):
            topic_list= [key for j, key in enumerate(camera_topic_map) if j == i]
            camera_name = topic_list[0].split('/')[-2]
            # print(camera_name)
            with DpBag(bag=bag_name) as bag:
                for topic, msg, _ in bag.read_messages(
                        topics=topic_list,
                        dpbag_name=bag_name,
                        force_get_data_by_raw=True,
                ):
                    # Parse camera proto
                    obj = camera_topic_map[topic]()
                    raw_msg = strip_header(msg.data)
                    obj.ParseFromString(raw_msg)
                    t = obj.header.timestamp_sec * 1e6
                    for time in perception_time_list:
                        # 计算时间差的绝对值
                        time_diff = abs(t - time)
                        if time_diff < min_diffs[time][camera_name]:
                            # 更新最小差异
                            min_diffs[time][camera_name] = time_diff
                            img = cv2.imdecode(
                                np.frombuffer(obj.data, np.uint8), cv2.IMREAD_COLOR
                            )
                            # 存储图像数据
                            image_results[time][camera_name] = {
                                'image': img,
                                'time_diff': time_diff
                            }

    # 从light_bag读取障碍物信息
    obstacle_results = {}
    # 初始化最小差异字典，用于跟踪每个时间戳的最小差异
    min_diffs = {}
    for time in perception_time_list:
        obstacle_results[time] = {}
        min_diffs[time] = float('inf')
    for i in range(len(light_bags)):
        bag_name = light_bags[i]
        topic_list = list(obstacle_topic_map.keys())
        with DpBag(bag=bag_name) as bag:
            for topic, msg, _ in bag.read_messages(
                    topics=topic_list,
                    dpbag_name=bag_name,
                    force_get_data_by_raw=True,
            ):
                # Parse camera proto
                obj = obstacle_topic_map[topic]()
                raw_msg = strip_header(msg.data)
                obj.ParseFromString(raw_msg)
                obstacle = obj.perception_obstacle
                t = obj.time_measurement
                for time in perception_time_list:
                    # 计算时间差的绝对值
                    time_diff = abs(t - time)
                    if time_diff < min_diffs[time]:
                        # 更新最小差异
                        min_diffs[time] = time_diff
                        # 转换为普通列表
                        data_list = []
                        for item in obstacle:
                            # 如果元素是protobuf消息，先转换为字典
                            if hasattr(item, 'DESCRIPTOR'):  # 是protobuf消息
                                data_list.append(MessageToDict(item))
                            else:
                                data_list.append(item)
                        obstacle_results[time] = {
                            'obstacle': data_list,
                            'time_diff': time_diff
                        }

    # 从light_bag读取pose信息
    pose_results = {}
    # 初始化最小差异字典，用于跟踪每个时间戳的最小差异
    min_diffs = {}
    for time in perception_time_list:
        pose_results[time] = {}
        min_diffs[time] = float('inf')
    for i in range(len(light_bags)):
        bag_name = light_bags[i]
        topic_list = list(pose_topic_map.keys())
        with DpBag(bag=bag_name) as bag:
            for topic, msg, _ in bag.read_messages(
                    topics=topic_list,
                    dpbag_name=bag_name,
                    force_get_data_by_raw=True,
            ):
                # Parse camera proto
                obj = pose_topic_map[topic]()
                raw_msg = strip_header(msg.data)
                obj.ParseFromString(raw_msg)
                t = obj.measurement_time
                for time in perception_time_list:
                    # 计算时间差的绝对值
                    time_diff = abs(t - time)
                    if time_diff < min_diffs[time]:
                        # 更新最小差异
                        min_diffs[time] = time_diff
                        pos = obj.position
                        pos_list = [pos.x, pos.y, pos.z]
                        euler = obj.euler_angles
                        euler_list = [euler.x, euler.y, euler.z]
                        pose_results[time] = {
                            'position': pos_list,
                            'euler_angles': euler_list,
                            'time_diff': time_diff
                        }

    # 从light_bag读取plan信息
    plan_results = {}
    # 初始化最小差异字典，用于跟踪每个时间戳的最小差异
    min_diffs = {}
    for time in perception_time_list:
        plan_results[time] = {}
        min_diffs[time] = float('inf')
    for i in range(len(light_bags)):
        bag_name = light_bags[i]
        topic_list = list(plan_topic_map.keys())
        with DpBag(bag=bag_name) as bag:
            for topic, msg, _ in bag.read_messages(
                    topics=topic_list,
                    dpbag_name=bag_name,
                    force_get_data_by_raw=True,
            ):
                # Parse camera proto
                obj = plan_topic_map[topic]()
                raw_msg = strip_header(msg.data)
                obj.ParseFromString(raw_msg)
                t = obj.header.timestamp_sec * 1e6
                for time in perception_time_list:
                    # 计算时间差的绝对值
                    time_diff = abs(t - time)
                    if time_diff < min_diffs[time]:
                        # 更新最小差异
                        min_diffs[time] = time_diff
                        planning = []
                        for trajectory_point in obj.trajectory_point:
                            point_dict = {
                                'relative_time': trajectory_point.relative_time,
                                'x': trajectory_point.path_point.x,
                                'y': trajectory_point.path_point.y,
                            }
                            planning.append(point_dict)
                        plan_results[time] = planning


    return chaosheng_results, image_results, obstacle_results, pose_results, plan_results

if __name__ == "__main__":
    tag_id = 97020543
    meta_data = get_meta_data(tag_id=tag_id)
    chaosheng_results, image_results, obstacle_results, pose_results, plan_results = get_image_obstacle_pose(meta_data=meta_data)
    # 保存
    base_path = os.path.join(os.path.dirname(__file__), 'read_data')
    data_path = os.path.join(base_path, str(tag_id))
    os.makedirs(data_path, exist_ok=True)
    for time, camera_data in image_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        os.makedirs(time_path, exist_ok=True)
        img_1 = camera_data["panoramic_1_raw_data"]["image"]
        img_2 = camera_data["panoramic_2_raw_data"]["image"]
        img_3 = camera_data["panoramic_3_raw_data"]["image"]
        img_4 = camera_data["panoramic_4_raw_data"]["image"]
        path_1 = os.path.join(time_path, 'panoramic_1.jpg')
        path_2 = os.path.join(time_path, 'panoramic_2.jpg')
        path_3 = os.path.join(time_path, 'panoramic_3.jpg')
        path_4 = os.path.join(time_path, 'panoramic_4.jpg')
        cv2.imwrite(path_1, img_1)
        cv2.imwrite(path_2, img_2)
        cv2.imwrite(path_3, img_3)
        cv2.imwrite(path_4, img_4)

    for time, chaosheng_data in chaosheng_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'chaosheng.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chaosheng_data, f, indent=2, ensure_ascii=False)

    for time, obstacle_data in obstacle_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'obstacle.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obstacle_data['obstacle'], f, indent=2, ensure_ascii=False)

    for time, pose_data in pose_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'pose.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)

    for time, plan_data in plan_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'plan.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)

    print("debug")



