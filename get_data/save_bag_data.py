from get_meta_data import get_meta_data
from get_camera_parameters import get_camera_engine_parameters
from get_chaosheng_image_obstacle_pose import get_image_obstacle_pose
from get_vehicle2sensing import get_vehicle2sensing
from get_planning import get_planning
from get_car_config import get_car_config
from get_ground import get_ground
import os
import json

def save_data(tag_id):
    # 保存路径
    data_path = os.path.join('/mnt/public-data/user/ziroujiang/avp/read_data', str(tag_id))
    os.makedirs(data_path, exist_ok=True)
    meta_data = get_meta_data(tag_id=tag_id)
    with open(data_path + '/meta_data.json', 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    print("meta_data")
    chaosheng_results, image_results, obstacle_results, pose_results, plan_results = get_image_obstacle_pose(meta_data=meta_data)
    for time in image_results.keys():
        time_path = os.path.join(data_path, str(int(time)))
        os.makedirs(time_path, exist_ok=True)
    for time, chaosheng_data in chaosheng_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'chaosheng.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chaosheng_data, f, indent=2, ensure_ascii=False)
    print("chaosheng_data")
    for time, obstacle_data in obstacle_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'obstacle.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obstacle_data['obstacle'], f, indent=2, ensure_ascii=False)
    print("obstacle_data")
    for time, pose_data in pose_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'pose.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)
    print("pose_data")
    for time, plan_data in plan_results.items():
        time_path = os.path.join(data_path, str(int(time)))
        path = os.path.join(time_path, 'plan.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)
    print("plan_data")
    trip_id = meta_data['body'][0]['tripId']
    vehicle2sensing = get_vehicle2sensing(trip_id)
    with open(data_path + '/vehicle2sensing.json', 'w', encoding='utf-8') as f:
        json.dump(vehicle2sensing, f, ensure_ascii=False, indent=2)
    print("vehicle2sensing")
    car_config = get_car_config(trip_id)
    with open(data_path + '/car_config.json', 'w', encoding='utf-8') as f:
        json.dump(car_config, f, ensure_ascii=False, indent=2)
    print("car_config")
    ground = get_ground(trip_id=trip_id)
    with open(data_path + '/ground.json', 'w', encoding='utf-8') as f:
        json.dump(ground, f, ensure_ascii=False, indent=2)
    print("ground")
    cameras_parameters = get_camera_engine_parameters(trip_id=trip_id)
    cameras_parameters = cameras_parameters.decode('utf-8').splitlines()
    with open(data_path + '/cameras_parameters.json', 'w', encoding='utf-8') as f:
        json.dump(cameras_parameters, f, ensure_ascii=False, indent=2)
    print("cameras_parameters")


if __name__ == "__main__":

    tag_id_list = [
        # 97020556, 97020543, 97020546, 97020561, 97020563,
        # 97020525, 97020564, 97020559, 97020554, 97020567,
        # 97020550, 97020541
        97020543
    ]
    target_id_list = []
    for i in range(len(tag_id_list)):
        tag_id = tag_id_list[i]
        try:
            target_id = save_data(tag_id=tag_id)
            print(str(tag_id) + " saved")
            if target_id != -1:
                target_id_list.append(target_id)
        except Exception as e:
            print(f"Error saving tag_id {tag_id}: {e}")
            continue
    print(target_id_list)



