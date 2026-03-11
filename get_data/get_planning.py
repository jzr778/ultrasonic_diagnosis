"""
⚠️ 已废弃 — 此模块的 planning 提取功能已合并到 bag_reader.py (BagReader.extract_planning)。
保留此文件仅供独立调用/调试使用。
"""

import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

sys.path.insert(0, config.PROTO_LOCAL_DIR)

from dpbag import strip_header
from dpbag.bag.bag import DpBag
from planning.planning_pb2 import ADCTrajectory
from get_meta_data import get_meta_data

plan_topic_map = {
    config.PLANNING_TOPIC: ADCTrajectory,
}


def get_planning(meta_data, tag_time=0):
    bag_name_list = meta_data['body'][0]['bagsName']
    if tag_time == 0:
        tag_time = meta_data['body'][0]['ntpTime']

    light_bags = sorted([b for b in bag_name_list if 'Light' in b])[1:]

    planning = []
    for bag_name in light_bags:
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
                    obj = plan_topic_map[topic]()
                    raw_msg = strip_header(msg.data)
                    obj.ParseFromString(raw_msg)
                    for trajectory_point in obj.trajectory_point:
                        planning.append({
                            'relative_time': trajectory_point.relative_time,
                            'x': trajectory_point.path_point.x,
                            'y': trajectory_point.path_point.y,
                        })
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
