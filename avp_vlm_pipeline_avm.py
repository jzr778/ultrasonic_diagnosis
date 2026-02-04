"""
AEB LLM标注管道 - 直接从PKL文件提取图像版本

这个脚本直接从record PKL文件中提取camera图像数据，无需预先保存图像文件。
主要功能：
1. 从CSV文件中获取case信息和PKL文件路径
2. 直接从PKL文件中提取5帧摄像头图像（center ± 2×0.5s）
3. 提取agent结构化数据（自车和关键目标信息）
4. 调用LLM进行场景分析
5. 保存分析结果

"""
from panoramic_projector import PanoramicProjector
from get_data.get_meta_data import get_meta_data
from point2box_mindistance_avm import get_max_distance_for_segment, calculate_segment_center
from prompts_engine.prompt_gen import prompt_gen
from VLM_API import analyze_scenario_from_images
from comment.add_comment import FeishuCommentTester
import json
import os
import argparse
import pandas as pd
import pickle
import sys
import cv2
from typing import Tuple
import math

sys.path.append("/mnt/public-data/shared/public/trajcaching_v3/debs/proto")
sys.path.append("/mnt/public-data/shared/public/trajcaching_v3/debs/scenariohouse")


def get_direction_from_position(target_x: int, target_y: int,
                                img_width: int = 640, img_height: int = 800) -> str:
    """
    判断目标点相对于车辆的方位（车头朝图像上方）
    坐标系：原点在左上角，x向右，y向下

    Args:
        target_x: 目标点x坐标 (0-640)
        target_y: 目标点y坐标 (0-800)
        img_width: 图像宽度 (默认640)
        img_height: 图像高度 (默认800)

    Returns:
        方位描述字符串
    """
    # 车辆中心点（图像中心）
    car_center_x = img_width // 2  # 320
    car_center_y = img_height // 2  # 400

    # 计算相对坐标（以车辆为中心）
    dx = target_x - car_center_x  # 向右为正
    dy = target_y - car_center_y  # 向下为正

    # 关键理解：
    # 1. 图像坐标系：原点左上角，x向右增加，y向下增加
    # 2. 车头朝向：图像上方（y减小的方向）
    # 3. 所以：
    #    - 正前方：目标在车辆上方（y < 400）
    #    - 正后方：目标在车辆下方（y > 400）
    #    - 正右方：目标在车辆右侧（x > 320）
    #    - 正左方：目标在车辆左侧（x < 320）

    # 计算角度（以车头方向为0度）
    # 车头方向：图像上方 = 方向向量 (0, -1)
    # 目标方向：向量 (dx, dy)
    # 但需要转换为数学坐标系：y向上为正

    # 在数学坐标系中：
    # 车头方向向量 = (0, -1) [向上]
    # 目标方向向量 = (dx, -dy) [因为图像y向下，数学y向上]

    # 计算目标向量与车头方向的角度
    # atan2(y, x) 返回点(x,y)与x轴正方向的夹角
    angle_rad = math.atan2(dx, -dy)  # 注意参数顺序：atan2(y, x)

    # 转换为0-360度的角度
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    # 8方位判断
    # 调整角度，使0°对应正前方
    # 当目标在正上方时：dx=0, dy<0 => angle_deg=0°

    if 337.5 <= angle_deg or angle_deg < 22.5:
        return "正前方"
    elif 22.5 <= angle_deg < 67.5:
        return "右前方"
    elif 67.5 <= angle_deg < 112.5:
        return "正右方"
    elif 112.5 <= angle_deg < 157.5:
        return "右后方"
    elif 157.5 <= angle_deg < 202.5:
        return "正后方"
    elif 202.5 <= angle_deg < 247.5:
        return "左后方"
    elif 247.5 <= angle_deg < 292.5:
        return "正左方"
    elif 292.5 <= angle_deg < 337.5:
        return "左前方"


def main():
    parser = argparse.ArgumentParser(description="使用VLLM分析AVP场景图像")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_avm",
        help="保存分析结果JSON文件的输出目录"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="get_data/read_data",
        help="数据路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="gemini-3-pro-preview",
        # default='gemini-3-flash-preview',
        default=["gemini-3-pro-preview", 'gemini-3-flash-preview', 'claude-sonnet-4-5-20250929', 'gpt-5.2-2025-12-11'],
        # default=["gemini-3-pro-preview", ],
        help="指定的QWen版本"
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default="chaosheng_wujian_avm",
        help="指定的prompt配置文件"
    )
    parser.add_argument(
        "--tag-id-list",
        nargs="+",
        type=int,
        default=[
            97020556, 97020543, 97020546, 97020561, 97020563,
            97020525, 97020564, 97020559, 97020554, 97020567,
            97020550, 97020541
        ],
        help="指定要分析的tag"
    )
    parser.add_argument(
        "--feishu-id-list",
        nargs="+",
        type=int,
        default=[
            6644231273
        ],
        help="指定要分析的tag"
    )
    args = parser.parse_args()

    for i in range(len(args.tag_id_list)):
        tag_id = args.tag_id_list[i]
        # feishu_id = args.feishu_id_list[i]
        # pre_comment_record = '大模型诊断结果：\n'
        # comment_record = ''
        # 准备数据 ##########################################################################################
        data_path = os.path.join(args.data_path, str(tag_id))
        with open(data_path + '/vehicle2sensing.json', 'r', encoding='utf-8') as f:
            vehicle2sensing = json.load(f)
        with open(data_path + '/ground.json', 'r', encoding='utf-8') as f:
            ground = json.load(f)
        with open(data_path + '/cameras_parameters.json', 'r', encoding='utf-8') as f:
            cameras_parameters = json.load(f)
        with open(data_path + '/car_config.json', 'r', encoding='utf-8') as f:
            car_config = json.load(f)
        focal_length = 162.6
        camera_height = 3.44
        projector = PanoramicProjector()
        # 时间戳list
        all_items = os.listdir(data_path)
        # 过滤出文件夹
        folders = []
        for item in all_items:
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                folders.append(item)
        all_items = sorted(folders, key=lambda x: int(x))
        # 为每一个时间戳匹配avm
        avm_path_list = {}
        meta_data = get_meta_data(tag_id=tag_id)
        bag_list = meta_data['body'][0]['bagsName']
        # 提取包名
        bag_list = sorted([bag_name for bag_name in bag_list if 'Heavy' in bag_name])
        bag_list = [item.split('.')[0] for item in bag_list]
        for ts in all_items:
            ts_tmp = ts[:10] + "_" + ts[10:]
            prefix_12 = ts_tmp[:12]  # 取前12位
            matched_file = None  # 存储匹配到的文件路径
            # 遍历所有文件夹
            for bag in bag_list:
                bag_path = os.path.join("/mnt/public-data/deelooper/csi-data-aly/shared/public/yiliu03", bag)
                if not os.path.exists(bag_path):
                    continue
                # 遍历当前文件夹中的所有文件
                for fname in os.listdir(bag_path):
                    name_without_ext = os.path.splitext(fname)[0]
                    if name_without_ext[:12] == prefix_12:
                        # 找到匹配，记录完整路径
                        matched_file = os.path.join(bag_path, fname)
                        break  # 跳出文件循环
                if matched_file:
                    break  # 跳出文件夹循环
            # 记录结果
            avm_path_list[ts] = matched_file
        # 图像保存路径
        image_save_path = os.path.join('get_data/draw_image', str(tag_id))
        os.makedirs(image_save_path, exist_ok=True)
        for item in all_items:
            print(str(tag_id) + '******' + item)
            # 在avm上画检测点 ##########################################################################################
            item_path = os.path.join(data_path, item)
            item_save_path = os.path.join(image_save_path, item)
            os.makedirs(item_save_path, exist_ok=True)
            with open(item_path + '/chaosheng.json', 'r', encoding='utf-8') as f:
                chaosheng = json.load(f)
            with open(item_path + '/obstacle.json', 'r', encoding='utf-8') as f:
                obstacle = json.load(f)
            with open(item_path + '/pose.json', 'r', encoding='utf-8') as f:
                pose = json.load(f)
            with open(item_path + '/plan.json', 'r', encoding='utf-8') as f:
                planning_point = json.load(f)
            # 坐标变换
            obstacle, ULTRASONIC_z = projector.world2vehicle2sensing(obstacle, pose, vehicle2sensing)
            chaosheng = projector.world2vehicle2sensing_chaosheng(chaosheng, pose, vehicle2sensing, ULTRASONIC_z)
            avm_path = avm_path_list[item]
            if avm_path:
                avm_image = cv2.imread(avm_path)
                planning_point = projector.world2vehicle2sensing_planning(planning_point, pose, vehicle2sensing)
                to_tail = car_config["back_edge_to_center"]
                for point in planning_point:
                    point[0] -= to_tail
                # 转换为DataFrame
                planning_point_df = pd.DataFrame(planning_point, columns=['x', 'y', 'z'])
                # 去除重复行
                planning_point_df = planning_point_df.drop_duplicates()
                planning_point = planning_point_df.values.tolist()
                # 在AVM上绘制检测信息
                index = {
                    "avm": None,
                }
                bev_img_with_obstacles, pos = projector.draw_obstacles_on_bev(
                    avm_image, obstacle, chaosheng, ground, focal_length, camera_height, planning_point
                )
                index["avm"] = pos
                cv2.imwrite(item_save_path + '/avm.jpg', bev_img_with_obstacles)
                with open(item_save_path + "/index_avm.json", 'w', encoding='utf-8') as f:
                    json.dump(index, f, indent=2)
                # 在AVM上画FS_CAR点 ###########################################################################################
                # 在AVM上绘制检测信息
                bev_img_with_fs_car, box_list, point_list = projector.draw_fs_car_on_bev(
                    avm_image, obstacle, chaosheng, ground, focal_length, camera_height, planning_point
                )
                cv2.imwrite(item_save_path + '/avm_fs_car.jpg', bev_img_with_fs_car)
                with open(item_save_path + "/box_list_avm.json", 'w', encoding='utf-8') as f:
                    json.dump(box_list, f, indent=2)
                with open(item_save_path + "/point_list_avm.json", 'w', encoding='utf-8') as f:
                    json.dump(point_list, f, indent=2)
                # 在AVM上判断FS_CAR #####################################################################################
                # 结果
                result_fs_car = []
                # 计算每个点到所有边框的最小距离
                for segment_points in point_list:
                    # 计算线段的最大距离
                    max_distance = get_max_distance_for_segment(segment_points, box_list)
                    # 计算线段中心点
                    center_point = calculate_segment_center(segment_points)
                    # 判断是否误检（距离 > 8）
                    if max_distance > 8:
                        result_fs_car.append([center_point[0], center_point[1]])

                if result_fs_car:
                    print(result_fs_car)

                # AI诊断 ###########################################################################################
                # index['avm'] = []
                if len(index['avm']) == 0:
                    analysis_result = {}
                    analysis_result['positions'] = []
                else:
                    # 读取图像数据
                    image_list = {}
                    panoramic_1 = cv2.cvtColor(bev_img_with_obstacles, cv2.COLOR_BGR2RGB)
                    image_list['panoramic_1'] = panoramic_1
                    # VLM###################################################################################################
                    # 进行分析（传入图像数组而不是文件路径）
                    # 生成prompt
                    prompt_config = args.prompt_config
                    prompt = prompt_gen(index, prompt_config)
                    analysis_result = analyze_scenario_from_images(image_list, prompt, args.model)
                    # 打印并保存结果
                    if analysis_result is None:
                        print(f"tag {tag_id}，时间戳 {item}: 未从API获取到有效结果")
                        continue
                # # 保存单个case的结果到valid目录
                # result = {
                #     "fs_others": analysis_result['positions'],
                #     "fs_car": result_fs_car,
                # }
                # # 结果保存路径
                # save_path = os.path.join(args.output_dir, str(tag_id))
                # os.makedirs(save_path, exist_ok=True)
                # save_path = os.path.join(save_path, item)
                # os.makedirs(save_path, exist_ok=True)
                # if result_fs_car or analysis_result['positions']:
                #     print(result)
                #     analysis_json_path = os.path.join(f"{save_path}/analysis_result.json")
                #     with open(analysis_json_path, 'w', encoding='utf-8') as f:
                #         json.dump(result, f, ensure_ascii=False, indent=2)
                #     print(f"tag {tag_id}，时间戳 {item}：分析结果已保存到 {analysis_json_path}")
                #
                #     direction_text = ""
                #     direction = []
                #     for coor in result_fs_car:
                #         d = get_direction_from_position(int(coor[0]), int(coor[1]))
                #         direction.append(d)
                #     if direction:
                #         direction_text = direction_text + 'FS_CAR误检点相对于车的位置：' + ', '.join(direction) + " "
                #
                #     direction = []
                #     for coor in analysis_result['positions']:
                #         d = get_direction_from_position(int(coor[0]), int(coor[1]))
                #         direction.append(d)
                #     if direction:
                #         direction_text = direction_text + 'FS_OTHERS_STATIC误检点相对于车的位置：' + ', '.join(direction)
                #
                #     comment_record = comment_record + '时间戳' + str(item) + ': ' + direction_text + "\n"
        # if comment_record:
        #     comment_record = pre_comment_record + comment_record
        #     print(comment_record)
        #     tester = FeishuCommentTester()
        #     test_url = "https://project.feishu.cn/iffcom/case/detail/" + str(feishu_id)
        #     tester.test_comment(test_url, comment_record)
        #     # 后面自带 (From plugin-飞书项目OPENAPI)
        # print("")


if __name__ == "__main__":
    main()
