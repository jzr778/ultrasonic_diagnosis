import requests
import json
import os
import sys
import argparse
import cv2
import re
import numpy as np
sys.path.append("/mnt/pubic-data/shared/public/trajcaching_v3/debs/proto")
sys.path.append("/mnt/pubic-data/shared/public/trajcaching_v3/debs/scenariohouse")
# Add proto directory to Python path
# sys.path.append('/opt/deeproute/common/common-protocol/include/proto/')
sys.path.insert(0, "/mnt/pubic-data/shared/public/trajcaching_v3/debs/proto")
sys.path.insert(0, "/mnt/pubic-data/shared/public/trajcaching_v3/debs/scenariohouse")
from dpbag import strip_header
from collections import defaultdict
from dpbag.bag.bag import DpBag

# 设置环境变量
os.environ['DPBAG_DP_USERNAME'] = 'perceptionteam'
os.environ['DPBAG_DP_PASSWORD'] = 'r6zR86V4*+=*'
# proto
import drivers
import perception
from drivers.sensor_image_pb2 import CompressedImage
from perception.deeproute_perception_obstacle_pb2 import PerceptionObstacles
from drivers.gnss.ins_pb2 import Ins
from planning.planning_pb2 import ADCTrajectory
# get data
from get_meta_data import get_meta_data
from google.protobuf.json_format import MessageToDict

plan_topic_map = {
    "/planner/trajectory": ADCTrajectory,
}

def get_planning(meta_data, tag_time=0):
    bag_name_list = meta_data['body'][0]['bagsName']
    if tag_time == 0:
        tag_time = meta_data['body'][0]['ntpTime']

    # 分别提取包名
    heavy_bags = sorted([bag_name for bag_name in bag_name_list if 'Heavy' in bag_name])[1:]
    light_bags = sorted([bag_name for bag_name in bag_name_list if 'Light' in bag_name])[1:]

    # 从light_bag读取planning信息
    planning = []
    for i in range(len(light_bags)):
        bag_name = light_bags[i]
        topic_list = list(plan_topic_map.keys())
        flag = False
        with DpBag(bag=bag_name) as bag:
            for topic, msg, t in bag.read_messages(
                    topics=topic_list,
                    dpbag_name=bag_name,
                    force_get_data_by_raw=True,
            ):
                t = t.to_sec() * 1e6
                if tag_time - t <= 0:
                    # Parse camera proto
                    obj = plan_topic_map[topic]()
                    raw_msg = strip_header(msg.data)
                    obj.ParseFromString(raw_msg)
                    for trajectory_point in obj.trajectory_point:
                        point_dict = {
                            'relative_time': trajectory_point.relative_time,
                            'x': trajectory_point.path_point.x,
                            'y': trajectory_point.path_point.y,
                        }
                        planning.append(point_dict)
                    flag = True
                    break
        if flag:
            break

    return planning

if __name__ == "__main__":
    tag_id = 63787346
    tag_time = 1762138214000000
    meta_data = get_meta_data(tag_id=tag_id)
    planning = get_planning(meta_data=meta_data, tag_time=tag_time)
    path = os.path.join(os.path.dirname(__file__), 'planning.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(planning, f, indent=2, ensure_ascii=False)
    print("debug")



