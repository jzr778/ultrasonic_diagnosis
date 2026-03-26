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
import json
import os
import shutil
import argparse
import pandas as pd
import pickle
import sys
import cv2
from typing import Tuple
import math
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from collections import OrderedDict

import config
from vlm.panoramic_projector import PanoramicProjector

from vlm.VLM_API import analyze_scenario_from_images
from vlm.point2box_mindistance_avm import is_segment_misdetected, calculate_segment_center
from get_data.get_meta_data import get_meta_data
from prompts_engine.prompt_gen import prompt_gen
from comment.add_comment import FeishuCommentTester

sys.path.append(os.path.join(config.PROTO_DEBS_DIR, "proto"))
sys.path.append(os.path.join(config.PROTO_DEBS_DIR, "scenariohouse"))

logger = logging.getLogger(__name__)


def setup_logging():
    now = datetime.now()
    log_dir = os.path.join(str(config.PROJECT_ROOT), "logs", now.strftime("%m%d"))
    os.makedirs(log_dir, exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"vlm_avm_{timestamp}.log")

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_file}")
    return log_file


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


# AVM 八方位 → 与超声主证据最相关的单路鱼眼（前/右/后/左）
_AVM_DIRECTION_TO_YUYAN_CAM = {
    "正前方": "panoramic_1",
    "左前方": "panoramic_1",
    "右前方": "panoramic_1",
    "正右方": "panoramic_2",
    "右后方": "panoramic_2",
    "正后方": "panoramic_3",
    "左后方": "panoramic_4",
    "正左方": "panoramic_4",
}

YUYAN_CAMERA_LABEL_ZH = {
    "panoramic_1": "前视鱼眼（车辆正前）",
    "panoramic_2": "右视鱼眼（车辆右侧）",
    "panoramic_3": "后视鱼眼（车辆正后）",
    "panoramic_4": "左视鱼眼（车辆左侧）",
}


def avm_positions_to_yuyan_camera(positions):
    """由 AVM 上红色超声质心集合估计主方位并选取一路鱼眼。"""
    if not positions:
        return None, None
    mx = int(round(sum(p[0] for p in positions) / len(positions)))
    my = int(round(sum(p[1] for p in positions) / len(positions)))
    direction = get_direction_from_position(mx, my)
    cam = _AVM_DIRECTION_TO_YUYAN_CAM.get(direction, "panoramic_1")
    return cam, direction


def select_yuyan_camera_from_fisheye_markers(index_fisheye, bev_chaosheng_positions):
    """在 plot_fisheye_polygon 标红之后，选一路鱼眼送进 VLM。

    - 仅一路有红色超声标记 → 用该路。
    - 多路有标记：若 BEV 主方位对应相机也在其中 → 与 AVM 一致；否则取标记点最多的那路。
    - 无路有标记：回退 BEV 主方位相机；无 BEV 点时回退 panoramic_1。
    """
    order = ["panoramic_1", "panoramic_2", "panoramic_3", "panoramic_4"]
    non_empty = [c for c in order if index_fisheye.get(c)]
    if len(non_empty) == 1:
        cam = non_empty[0]
        _, direction = avm_positions_to_yuyan_camera(bev_chaosheng_positions or [])
        return cam, direction
    if len(non_empty) > 1:
        bev_cam, direction = avm_positions_to_yuyan_camera(bev_chaosheng_positions or [])
        if bev_cam and bev_cam in non_empty:
            return bev_cam, direction
        best = max(non_empty, key=lambda c: len(index_fisheye[c]))
        _, direction = avm_positions_to_yuyan_camera(bev_chaosheng_positions or [])
        return best, direction
    bev_cam, direction = avm_positions_to_yuyan_camera(bev_chaosheng_positions or [])
    if bev_cam:
        return bev_cam, direction
    return "panoramic_1", None


def draw_single_tag(tag_id, args):
    """处理单个 tag_id 的绘图流程：生成 AVM 标注图像及中间数据。

    Returns:
        dict: 包含以下可能的 key（互斥，每次只出现一组）:
            - missing_files:  [(tag_id, [缺失文件列表])]
            - no_ultrasonic:  [tag_id]
            - no_avm_match:   [(tag_id, ts_total, ts_matched)]
            - draw_success:   [(tag_id, ts_total, ts_matched)]
    """
    data_path = os.path.join(args.data_path, str(tag_id))
    required_files = ['vehicle2sensing.json', 'ground.json', 'cameras_parameters.json', 'car_config.json']
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]
    if missing:
        logger.warning(f"[绘图] tag={tag_id} 缺少配置文件 {missing}，跳过")
        return {"missing_files": [(tag_id, missing)]}
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
    camera_name_to_trans_mat = projector.get_transform(cameras_parameters, projector.cameras)
    all_items = os.listdir(data_path)
    folders = []
    for item in all_items:
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    all_items = sorted(folders, key=lambda x: int(x))
    if not all_items:
        logger.info(f"[绘图] tag={tag_id} 无时间戳目录（无超声波事件），跳过")
        return {"no_ultrasonic": [tag_id]}
    AVM_MATCH_TOLERANCE = 50000  # 50ms，单位微秒
    avm_path_list = {}
    meta_data = get_meta_data(tag_id=tag_id)
    bag_list = meta_data['body'][0]['bagsName']
    bag_list = sorted([bag_name for bag_name in bag_list if 'Heavy' in bag_name])
    bag_list = [item.split('.')[0] for item in bag_list]
    all_avm_files = []
    for bag in bag_list:
        bag_path = os.path.join(config.GENERATE_DIR, bag)
        if not os.path.exists(bag_path):
            continue
        for fname in os.listdir(bag_path):
            name_without_ext = os.path.splitext(fname)[0]
            try:
                avm_ts = int(name_without_ext)
            except ValueError:
                continue
            all_avm_files.append((avm_ts, os.path.join(bag_path, fname)))
    for ts in all_items:
        ts_val = int(ts)
        matched_file = None
        best_diff = AVM_MATCH_TOLERANCE + 1
        for avm_ts, avm_fpath in all_avm_files:
            diff = abs(avm_ts - ts_val)
            if diff < best_diff:
                best_diff = diff
                matched_file = avm_fpath
        avm_path_list[ts] = matched_file
    image_save_path = os.path.join(config.DRAW_IMAGE_DIR, str(tag_id))
    drawn_count = 0
    for item in all_items:
        logger.info(f"[绘图] tag={tag_id}, ts={item}")
        item_path = os.path.join(data_path, item)
        item_save_path = os.path.join(image_save_path, item)
        with open(item_path + '/chaosheng.json', 'r', encoding='utf-8') as f:
            chaosheng = json.load(f)
        ignore_fs = set(getattr(args, 'ignore_fs_types', []) or [])
        if ignore_fs:
            chaosheng = [o for o in chaosheng if o.get("freespaceType", "") not in ignore_fs]
        with open(item_path + '/obstacle.json', 'r', encoding='utf-8') as f:
            obstacle = json.load(f)
        with open(item_path + '/pose.json', 'r', encoding='utf-8') as f:
            pose = json.load(f)
        with open(item_path + '/plan.json', 'r', encoding='utf-8') as f:
            planning_point = json.load(f)
        projector.apply_chaosheng_z_from_camera_ground_plane(chaosheng, obstacle)
        obstacle = projector.world2vehicle2sensing(obstacle, pose, vehicle2sensing)
        chaosheng = projector.world2vehicle2sensing_chaosheng(chaosheng, pose, vehicle2sensing)
        avm_path = avm_path_list[item]
        if avm_path:
            os.makedirs(item_save_path, exist_ok=True)
            avm_image = cv2.imread(avm_path)
            planning_point = projector.world2vehicle2sensing_planning(planning_point, pose, vehicle2sensing)
            to_tail = car_config["back_edge_to_center"]
            for point in planning_point:
                point[0] -= to_tail
            planning_point_df = pd.DataFrame(planning_point, columns=['x', 'y', 'z'])
            planning_point_df = planning_point_df.drop_duplicates()
            planning_point = planning_point_df.values.tolist()
            has_non_fs_car = any(o.get("freespaceType", "") != "FS_CAR" for o in chaosheng)
            has_fs_car = any(o.get("freespaceType", "") == "FS_CAR" for o in chaosheng)
            if has_non_fs_car:
                bev_img_with_obstacles, pos, yellow_fs = projector.draw_obstacles_on_bev(
                    avm_image, obstacle, chaosheng, ground, focal_length, camera_height, planning_point,
                    chaosheng_pixel_radius=30,
                    ignore_camera_freespace_types=ignore_fs if ignore_fs else None,
                )
                index = {"avm": pos, "yellow_freespace": yellow_fs}
                if getattr(args, "yuyan", True):
                    index_fisheye = {}
                    for cam in projector.cameras:
                        fish_src = os.path.join(item_path, f"{cam}.jpg")
                        trans = camera_name_to_trans_mat.get(cam)
                        if not os.path.isfile(fish_src):
                            logger.warning(
                                f"[绘图] tag={tag_id}, ts={item} 无鱼眼原图 {fish_src}，"
                                f"该路跳过"
                            )
                            index_fisheye[cam] = []
                            continue
                        if trans is None:
                            logger.warning(
                                f"[绘图] tag={tag_id}, ts={item} 无相机标定 {cam}，该路跳过"
                            )
                            index_fisheye[cam] = []
                            continue
                        raw_fish = cv2.imread(fish_src)
                        if raw_fish is None:
                            logger.warning(
                                f"[绘图] tag={tag_id}, ts={item} 鱼眼读图失败 {fish_src}"
                            )
                            index_fisheye[cam] = []
                            continue
                        fish_drawn, fish_pts = projector.plot_fisheye_polygon(
                            raw_fish.copy(),
                            obstacle,
                            chaosheng,
                            trans["extrinsics"],
                            trans["distortion_coeff"],
                            trans["intrinsics"],
                            cam,
                            ground,
                            focal_length,
                            camera_height,
                            chaosheng_pixel_radius=30,
                            ignore_camera_freespace_types=ignore_fs
                            if ignore_fs
                            else None,
                            bev_height=800,
                            bev_width=640,
                            resize=True,
                        )
                        out_fish = os.path.join(item_save_path, f"{cam}.jpg")
                        cv2.imwrite(out_fish, fish_drawn)
                        index_fisheye[cam] = fish_pts
                    with open(
                        os.path.join(item_save_path, "index_fisheye.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(index_fisheye, f, indent=2)
                    y_cam, y_dir = select_yuyan_camera_from_fisheye_markers(
                        index_fisheye, pos
                    )
                    index["yuyan_camera"] = y_cam
                    index["yuyan_direction"] = y_dir
                    index["yuyan_fish_points"] = index_fisheye.get(y_cam) or []
                    index["yuyan_camera_label"] = YUYAN_CAMERA_LABEL_ZH.get(
                        y_cam, y_cam
                    )
                    candidates = [c for c in index_fisheye if index_fisheye[c]]
                    index["yuyan_marker_cameras"] = candidates
                    selected_drawn = os.path.join(item_save_path, f"{y_cam}.jpg")
                    if os.path.isfile(selected_drawn):
                        shutil.copy2(
                            selected_drawn,
                            os.path.join(item_save_path, "yuyan_draw.jpg"),
                        )
                    else:
                        logger.warning(
                            f"[绘图] tag={tag_id}, ts={item} 选中 {y_cam} 但无输出图，"
                            f"未生成 yuyan_draw.jpg"
                        )
                cv2.imwrite(item_save_path + '/avm.jpg', bev_img_with_obstacles)
                with open(item_save_path + "/index_avm.json", 'w', encoding='utf-8') as f:
                    json.dump(index, f, indent=2)
            if has_fs_car:
                bev_img_with_fs_car, box_list, point_list = projector.draw_fs_car_on_bev(
                    avm_image, obstacle, chaosheng, ground, focal_length, camera_height, planning_point,
                    chaosheng_pixel_radius=30
                )
                cv2.imwrite(item_save_path + '/avm_fs_car.jpg', bev_img_with_fs_car)
                with open(item_save_path + "/box_list_avm.json", 'w', encoding='utf-8') as f:
                    json.dump(box_list, f, indent=2)
                with open(item_save_path + "/point_list_avm.json", 'w', encoding='utf-8') as f:
                    json.dump(point_list, f, indent=2)
            drawn_count += 1
        else:
            logger.warning(f"[绘图] tag={tag_id}, ts={item} 未匹配到 AVM 图像文件（容差{AVM_MATCH_TOLERANCE}μs），跳过")
    ts_total = len(all_items)
    if drawn_count == 0:
        logger.warning(f"[绘图] tag={tag_id} 共 {ts_total} 个时间戳均未匹配到 AVM 图像，无绘图输出")
        return {"no_avm_match": [(tag_id, ts_total, 0)]}
    logger.info(f"[绘图] tag={tag_id} 完成，共 {ts_total} 个时间戳，成功绘制 {drawn_count} 个")
    return {"draw_success": [(tag_id, ts_total, drawn_count)]}


def diagnose_single_tag(tag_id, feishu_id, args):
    """处理单个 tag_id 的大模型诊断流程：读取已绘制图像，调用 VLM 分析

    Returns:
        dict: 包含以下 key:
            - no_draw_output: [tag_id]          （无绘图结果时）
            - misdetected:    [(tag_id, ts)]     （检测到误检的时间戳）
            - normal:         [(tag_id, ts)]     （正常的时间戳）
            - api_error:      [(tag_id, ts)]     （API 调用失败的时间戳）
    """
    stats = {"misdetected": [], "normal": [], "api_error": []}
    pre_comment_record = '大模型诊断结果：\n'
    comment_record = ''
    debug_thinking = getattr(args, 'debug_thinking', False)
    debug_thinking_path = getattr(args, 'debug_thinking_path', None)
    image_save_path = os.path.join(config.DRAW_IMAGE_DIR, str(tag_id))
    if not os.path.isdir(image_save_path):
        logger.warning(f"[诊断] tag={tag_id} 无绘图结果目录，跳过")
        stats["no_draw_output"] = [tag_id]
        return stats
    all_items = sorted(
        [d for d in os.listdir(image_save_path)
         if os.path.isdir(os.path.join(image_save_path, d))],
        key=lambda x: int(x)
    )
    for item in all_items:
        item_save_path = os.path.join(image_save_path, item)
        index_path = os.path.join(item_save_path, "index_avm.json")
        avm_img_path = os.path.join(item_save_path, "avm.jpg")
        box_list_path = os.path.join(item_save_path, "box_list_avm.json")
        point_list_path = os.path.join(item_save_path, "point_list_avm.json")
        has_avm = os.path.isfile(index_path) and os.path.isfile(avm_img_path)
        has_fs_car_data = os.path.isfile(box_list_path) and os.path.isfile(point_list_path)
        if not has_avm and not has_fs_car_data:
            logger.warning(f"[诊断] tag={tag_id}, ts={item} 缺少绘图输出，跳过")
            continue
        logger.info(f"[诊断] tag={tag_id}, ts={item}")
        result_fs_car = []
        if has_fs_car_data:
            with open(box_list_path, 'r', encoding='utf-8') as f:
                box_list = json.load(f)
            with open(point_list_path, 'r', encoding='utf-8') as f:
                point_list = json.load(f)
            for segment_points in point_list:
                if is_segment_misdetected(segment_points, box_list, threshold=8.0):
                    center_point = calculate_segment_center(segment_points)
                    result_fs_car.append([center_point[0], center_point[1]])
        if result_fs_car:
            logger.info(f"[诊断] tag={tag_id}, ts={item} fs_car规则校验误检: {result_fs_car}")
        analysis_result = {'positions': []}
        if has_avm:
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            if len(index.get('avm', [])) > 0:
                bev_img = cv2.imread(avm_img_path)
                image_list = OrderedDict()
                image_list["avm"] = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
                yuyan_draw_path = os.path.join(item_save_path, "yuyan_draw.jpg")
                if getattr(args, "yuyan", True) and not os.path.isfile(yuyan_draw_path):
                    yc = index.get("yuyan_camera")
                    if yc:
                        alt = os.path.join(item_save_path, f"{yc}.jpg")
                        if os.path.isfile(alt):
                            yuyan_draw_path = alt
                ctx = dict(index)
                ctx["vlm_yuyan_image_included"] = False
                if getattr(args, "yuyan", True) and os.path.isfile(yuyan_draw_path):
                    yb = cv2.imread(yuyan_draw_path)
                    if yb is not None:
                        image_list["yuyan_fisheye"] = cv2.cvtColor(yb, cv2.COLOR_BGR2RGB)
                        ctx["vlm_yuyan_image_included"] = True
                prompt_config = args.prompt_config
                prompt = prompt_gen(ctx, prompt_config)
                if getattr(args, 'debug_thinking', False):
                    prompt += ("\n\n#### ⚠️ 调试模式\n"
                               "请先**详细输出你对每个检测点的完整分析推理过程**"
                               "（包括定位、关联性筛选、真伪判断的每一步思考），"
                               "然后再输出最终的JSON结果。")
                analysis_result = analyze_scenario_from_images(image_list, prompt, args.model)
                raw_responses = analysis_result.pop("_raw_responses", {})
                if debug_thinking_path and raw_responses:
                    with open(debug_thinking_path, 'a', encoding='utf-8') as df:
                        df.write(f"\n{'=' * 60}\n")
                        df.write(f"tag={tag_id}, ts={item}\n")
                        df.write(f"{'=' * 60}\n")
                        for model_name, resp in raw_responses.items():
                            df.write(f"\n--- model: {model_name} ---\n")
                            df.write(str(resp) + "\n")
                if "error" in analysis_result:
                    logger.warning(f"[诊断] tag={tag_id}, ts={item} API 返回异常: {analysis_result['error']}")
                    stats["api_error"].append((tag_id, item))
                    continue
        result = {
            "fs_others": analysis_result['positions'],
            "fs_car": result_fs_car,
        }
        if result_fs_car or analysis_result['positions']:
            stats["misdetected"].append((tag_id, item))
            save_path = os.path.join(args.output_dir, "misdetected", str(tag_id), item)
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"[诊断] tag={tag_id}, ts={item} 误检结果: {result}")
            analysis_json_path = os.path.join(save_path, "analysis_result.json")
            with open(analysis_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            for jpg in os.listdir(item_save_path):
                if jpg.endswith(".jpg"):
                    shutil.copy2(os.path.join(item_save_path, jpg), save_path)
            logger.info(f"[诊断] tag={tag_id}, ts={item} 结果已保存 → {save_path}")

            direction_text = ""
            direction = []
            for coor in result_fs_car:
                d = get_direction_from_position(int(coor[0]), int(coor[1]))
                direction.append(d)
            if direction:
                direction_text = direction_text + 'FS_CAR误检点相对于车的位置：' + ', '.join(direction) + " "

            direction = []
            for coor in analysis_result['positions']:
                d = get_direction_from_position(int(coor[0]), int(coor[1]))
                direction.append(d)
            if direction:
                direction_text = direction_text + 'FS_OTHERS误检点相对于车的位置：' + ', '.join(direction)

            comment_record = comment_record + '时间戳' + str(item) + ': ' + direction_text + "\n"
        else:
            stats["normal"].append((tag_id, item))
            save_path = os.path.join(args.output_dir, "normal", str(tag_id), item)
            os.makedirs(save_path, exist_ok=True)
            for jpg in os.listdir(item_save_path):
                if jpg.endswith(".jpg"):
                    shutil.copy2(os.path.join(item_save_path, jpg), save_path)
    if comment_record:
        comment_record = pre_comment_record + comment_record
        logger.info(f"[诊断] tag={tag_id} 飞书评论:\n{comment_record}")
        # tester = FeishuCommentTester()
        # test_url = f"https://project.feishu.cn/{config.FEISHU_PROJECT_KEY}/case/detail/{feishu_id}"
        # tester.test_comment(test_url, comment_record)
    logger.info(f"[诊断] tag={tag_id} 完成 (误检={len(stats['misdetected'])}, 正常={len(stats['normal'])}, API异常={len(stats['api_error'])})")
    return stats


def process_single_tag(tag_id, feishu_id, args):
    """处理单个 tag_id 的完整流程（绘图 + 按需诊断）"""
    draw_result = draw_single_tag(tag_id, args)
    if "draw_success" not in draw_result:
        return draw_result
    diagnose_result = diagnose_single_tag(tag_id, feishu_id, args)
    return {**draw_result, **diagnose_result}


def main():
    parser = argparse.ArgumentParser(description="使用VLLM分析AVP场景图像")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.RESULT_DIR,
        help="保存分析结果JSON文件的输出目录"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=config.READ_DATA_DIR,
        help="数据路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["auto"],
        help="模型名称列表，'auto' 表示自动从 API 获取"
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default="chaosheng_wujian_avm",
        help="指定的prompt配置文件"
    )
    parser.add_argument(
        "--id-mapping",
        type=str,
        default=os.path.join(str(config.PROJECT_ROOT), "get_data", "id_mapping.json"),
        help="tag_id → feishu_id 映射文件路径 (JSON dict)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行线程数 (默认8)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "draw", "diagnose"],
        default="all",
        help="运行模式: all=全流程, draw=仅绘图, diagnose=仅诊断 (默认: all)"
    )
    parser.add_argument(
        "--ignore-fs-types",
        type=str,
        nargs="*",
        default=[],
        help="绘图时忽略的超声 freespaceType 列表，如 --ignore-fs-types FS_CURB FS_CHOCK"
    )
    parser.add_argument(
        "--debug-thinking",
        action="store_true",
        default=False,
        help="记录 VLM 模型的原始回复（含思考过程）到 logs/debug_thinking/"
    )
    parser.add_argument(
        "--no-yuyan",
        dest="yuyan",
        action="store_false",
        help="关闭鱼眼抽帧保存、鱼眼标注与 VLM 双图输入（默认开启鱼眼链路）",
    )
    parser.set_defaults(yuyan=True)
    args = parser.parse_args()

    log_file = setup_logging()
    if args.debug_thinking:
        debug_thinking_path = log_file.replace("vlm_avm_", "debug_thinking_").replace(".log", ".txt")
        args.debug_thinking_path = debug_thinking_path
        open(debug_thinking_path, 'w', encoding='utf-8').close()  # 清空/创建文件
        logger.info(f"思考过程将记录到: {debug_thinking_path}")
    else:
        args.debug_thinking_path = None

    with open(args.id_mapping, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)

    total_tags = len(id_mapping)
    logger.info(f"参数: mode={args.mode}, workers={args.workers}, model={args.model}, tag数量={total_tags}")

    all_stats = {
        "missing_files":   [],   # [(tag_id, [缺失文件])]
        "no_ultrasonic":   [],   # [tag_id]
        "no_avm_match":    [],   # [tag_id]
        "draw_success":    [],   # [tag_id]
        "no_draw_output":  [],   # [tag_id]
        "misdetected":     [],   # [(tag_id, ts)]
        "normal":          [],   # [(tag_id, ts)]
        "api_error":       [],   # [(tag_id, ts)]
        "exception":       [],   # [(tag_id, error_msg)]
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for tag_id_str, feishu_id in id_mapping.items():
            tag_id = int(tag_id_str)
            if args.mode == "draw":
                future = executor.submit(draw_single_tag, tag_id, args)
            elif args.mode == "diagnose":
                future = executor.submit(diagnose_single_tag, tag_id, feishu_id, args)
            else:
                future = executor.submit(process_single_tag, tag_id, feishu_id, args)
            futures[future] = tag_id

        for future in as_completed(futures):
            tag_id = futures[future]
            try:
                result = future.result()
                if isinstance(result, dict):
                    for key in all_stats:
                        all_stats[key].extend(result.get(key, []))
            except Exception as e:
                logger.error(f"[异常] tag={tag_id} 处理失败: {e}", exc_info=True)
                all_stats["exception"].append((tag_id, str(e)))

    _print_summary(logger, args.mode, total_tags, all_stats)


def _print_summary(logger, mode, total_tags, stats):
    """输出格式化的结果汇总。"""
    W = 64
    SEP = "=" * W
    THIN = "-" * W

    logger.info("")
    logger.info(SEP)
    logger.info("  VLM Pipeline 结果汇总")
    logger.info(SEP)
    logger.info(f"  输入 tag 总数: {total_tags}")
    logger.info("")

    # ── 绘图阶段 ──
    if mode in ("draw", "all"):
        logger.info(f"  {'── 绘图阶段 ──':─<{W - 4}}")
        logger.info("")

        n_missing = len(stats["missing_files"])
        n_no_us = len(stats["no_ultrasonic"])
        n_no_avm = len(stats["no_avm_match"])
        n_draw_ok = len(stats["draw_success"])
        n_fail = n_missing + n_no_avm

        ok_ts_total = sum(t for _, t, _ in stats["draw_success"])
        ok_ts_drawn = sum(d for _, _, d in stats["draw_success"])
        no_avm_ts = sum(t for _, t, _ in stats["no_avm_match"])

        logger.info(f"  绘图成功:       {n_draw_ok} 个 tag ({ok_ts_drawn}/{ok_ts_total} 个时间戳)")
        for tag_id, ts_total, ts_drawn in stats["draw_success"]:
            logger.info(f"    tag={tag_id}  ({ts_drawn}/{ts_total} 个时间戳)")

        logger.info(f"  绘图失败:       {n_fail} 个 tag")
        if n_missing:
            logger.info(f"    缺少配置文件: {n_missing} 个")
            for tag_id, missing in stats["missing_files"]:
                logger.info(f"      tag={tag_id}  缺少: {', '.join(missing)}")
        if n_no_avm:
            logger.info(f"    无 AVM 匹配:  {n_no_avm} 个 ({no_avm_ts} 个时间戳)")
            for tag_id, ts_total, _ in stats["no_avm_match"]:
                logger.info(f"      tag={tag_id}  ({ts_total} 个时间戳均未匹配)")

        logger.info(f"  无超声波事件:   {n_no_us} 个 tag")
        for tag_id in stats["no_ultrasonic"]:
            logger.info(f"    tag={tag_id}")
        logger.info("")

    # ── 诊断阶段 ──
    if mode in ("diagnose", "all"):
        logger.info(f"  {'── 诊断阶段 ──':─<{W - 4}}")
        logger.info("")

        n_no_draw = len(stats["no_draw_output"])
        if n_no_draw:
            logger.info(f"  无绘图结果:     {n_no_draw} 个 tag")
            for tag_id in stats["no_draw_output"]:
                logger.info(f"    tag={tag_id}")

        n_misdet = len(stats["misdetected"])
        n_normal = len(stats["normal"])
        n_api_err = len(stats["api_error"])
        tags_misdet = sorted(set(t for t, _ in stats["misdetected"]))
        tags_normal = sorted(set(t for t, _ in stats["normal"]))

        logger.info(f"  检测到误检:     {n_misdet} 条 ({len(tags_misdet)} 个 tag)")
        for tag_id, ts in stats["misdetected"]:
            logger.info(f"    tag={tag_id}, ts={ts}")
        logger.info(f"  检测正常:       {n_normal} 条 ({len(tags_normal)} 个 tag)")
        for tag_id, ts in stats["normal"]:
            logger.info(f"    tag={tag_id}, ts={ts}")
        logger.info(f"  API 异常:       {n_api_err} 条")
        for tag_id, ts in stats["api_error"]:
            logger.info(f"    tag={tag_id}, ts={ts}")
        logger.info("")

    # ── 异常 ──
    n_exc = len(stats["exception"])
    if n_exc:
        logger.info(f"  {'── 运行异常 ──':─<{W - 4}}")
        logger.info("")
        logger.info(f"  处理异常:       {n_exc} 个 tag")
        for tag_id, err in stats["exception"]:
            logger.error(f"    tag={tag_id}  错误: {err}")
        logger.info("")

    # ── 总计 ──
    logger.info(THIN)
    if mode in ("draw", "all"):
        n_missing = len(stats["missing_files"])
        n_no_us = len(stats["no_ultrasonic"])
        n_no_avm = len(stats["no_avm_match"])
        n_draw_ok = len(stats["draw_success"])
        n_fail = n_missing + n_no_avm
        ok_ts_total = sum(t for _, t, _ in stats["draw_success"])
        ok_ts_drawn = sum(d for _, _, d in stats["draw_success"])
        no_avm_ts = sum(t for _, t, _ in stats["no_avm_match"])
        logger.info(f"  绘图: 成功 {ok_ts_drawn}/{ok_ts_total} 个时间戳 ({n_draw_ok} tag) / 失败 {n_fail} tag (缺文件 {n_missing} tag + 无AVM {no_avm_ts} 个时间戳 {n_no_avm} tag) / 无超声 {n_no_us} tag")
    if mode in ("diagnose", "all"):
        n_misdet = len(stats["misdetected"])
        n_normal = len(stats["normal"])
        n_api_err = len(stats["api_error"])
        n_ts_total = n_misdet + n_normal + n_api_err
        logger.info(f"  诊断: 共 {n_ts_total} 条 (误检 {n_misdet} / 正常 {n_normal} / API异常 {n_api_err})")
    n_exc = len(stats["exception"])
    if n_exc:
        logger.info(f"  异常: {n_exc} 个 tag")
    logger.info(SEP)


if __name__ == "__main__":
    import sys
    sys.argv = [
        "avp_vlm_pipeline_avm.py",
        "--id-mapping", "/tmp/test_debug.json",
        "--mode", "draw",  # 调试鱼眼/绘图时先只跑 draw，避免走 VLM
        # "--model", "gemini-3-pro-preview",
    ]
    main()
