from get_meta_data import get_meta_data
from get_camera_parameters import get_camera_engine_parameters
from get_chaosheng_image_obstacle_pose import get_image_obstacle_pose
from get_vehicle2sensing import get_vehicle2sensing
from panoramic_projector import PanoramicProjector
from get_planning import get_planning
from get_car_config import get_car_config
from get_ground import get_ground
import cv2
import math
import numpy as np
import os
import json
import pandas as pd

def save_data(tag_id):

    meta_data = get_meta_data(tag_id=tag_id)
    chaosheng_results, image_results, obstacle_results, pose_results, plan_results = get_image_obstacle_pose(meta_data=meta_data)
    if len(chaosheng_results) == 0:
        return -1
    # 保存路径
    data_path = os.path.join('read_data', str(tag_id))
    os.makedirs(data_path, exist_ok=True)
    with open(data_path + '/meta_data.json', 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    print("meta_data")
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
    print("panoramic_image")
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
    return tag_id


if __name__ == "__main__":

    tag_id_list = [
        # 98728126, 98724502, 98679616, 98659678, 98635248, 98603631, 98600166, 98589942, 98520646, 98519237, 98508706,
        # 98502291, 98392892, 98389446, 98387572, 98384402, 98383844, 98370438, 98366683, 98363590, 98346751, 98332195,
        # 98331448, 98321419, 98320210, 98258742, 98197434, 98194251, 98176724, 98171873,
        # 98137519, 98109136, 98108088, 98105777, 98099917, 98087711, 98084796, 98079185, 98063883, 98051294, 98042121,
        # 98034789, 98028883, 98025395, 98216867, 98205761, 98205733, 98194533, 98196500, 98173996, 98170551, 98163052,
        # 98162178, 98162652, 98152344, 98145987, 98139983, 98085892, 98081681, 98070814,
        # 98070312, 98051394, 98032573, 98023371, 98022338, 98021115, 98021113, 98202594, 98182014, 98152996, 98150450,
        # 98139494, 98089489, 98089354, 98089372, 98079154, 98058551, 98055569, 98046191, 98046160, 98039173, 98033188,
        # 98030160, 98024149, 98023687, 97938019, 97937995, 97937879, 97937501, 97937127, 97921340, 97908022,

        # 99634306, 99624582, 99622577, 99619771, 99618544, 99617574, 99615631, 99614143, 99611943, 99608222,
        # 99600935, 99589655, 99585961, 99582352, 99571260, 99566777, 99563479, 99552470, 99548826, 99542047,
        # 99538757, 99491932, 99477298, 99475457, 99467644, 99460115, 99458486, 99440319,
        # 99430481, 99430717, 99330124, 99319820, 99319179, 99317128, 99309055, 99308270, 99307070, 99298743, 99275700,
        # 99270955, 98672923, 98663332, 98606530, 98661023, 98500436, 98476526, 98472867, 98472528, 98466533, 98464549,

        # 99940155, 99923934, 99907365, 99901870, 99889523, 99887481, 99887767, 99864201, 99862722, 99859129, 99843879,
        # 99842520, 99837341, 99768348, 99762100, 99742692, 99736577, 99905921, 99903406, 99898064, 99893525, 99891070,
        # 99887597, 99874193, 99871068, 99863416, 99856521, 99853011, 99837641, 99777448,
        # 99749260, 99740464, 99737888, 99731035, 99724552, 99723326, 99722714, 99718601, 99718309, 99455627, 99454409,
        # 99448954, 99445564, 99440656, 99438271, 99432368, 99418189, 99416784, 99415760, 99413967, 99376605

        100077430, 100071914, 100071054, 100070260, 100067023, 100063499, 100062320, 100037137, 100035390, 100020285,
        100027865, 99997866
    ]
    target_id_list = []
    for i in range(len(tag_id_list)):
        tag_id = tag_id_list[i]
        target_id = save_data(tag_id=tag_id)
        print(str(tag_id) + " saved")
        if target_id != -1:
            target_id_list.append(target_id)

    print(target_id_list)



