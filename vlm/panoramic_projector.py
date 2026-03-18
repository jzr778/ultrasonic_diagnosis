"""
全景相机障碍物投影可视化工具
支持将障碍物投影到panoramic_1/2/3/4四个全景相机图像上并拼接保存
"""
import os
import sys
import cv2
import numpy as np
import warnings
import json
import math
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Optional, Tuple


CAMERA_NAME_TO_INDEX = {
    "panoramic_1": 0,
    "panoramic_2": 1,
    "panoramic_3": 2,
    "panoramic_4": 3,
}


class PanoramicProjector:
    """
    全景相机障碍物投影器
    用于将障碍物投影到四个全景相机图像上并拼接保存
    """
    
    def __init__(self):

        # 初始化全景投影器
        self.cameras = [
            "panoramic_1",
            "panoramic_2",
            "panoramic_3",
            "panoramic_4",
        ]

    def world2vehicle2sensing(self, obstacle, pose, vehicle2sensing):
        translation_m2v, roll_pitch_yaw = pose['position'], pose['euler_angles']
        vehicle_to_sensing = vehicle2sensing['position']
        translation_m2v = np.asarray(translation_m2v).reshape(1, -1)
        rotation_m2v = self.get_rotation_matrix(roll_pitch_yaw)
        rotation_m2v_inv = np.linalg.inv(rotation_m2v)
        vehicle_to_sensing = np.asarray(vehicle_to_sensing).reshape(1, -1)
        ULTRASONIC_z = 0.0
        # 障碍物坐标变换
        for item in obstacle:
            sensor_type = item.get('sensorType', 0)
            if sensor_type in ['ULTRASONIC', 'CAMERA']:
                polygon_points_sensing = []
                for poly_point in item['polygonArea']['point']:
                    if sensor_type == 'ULTRASONIC':
                        ULTRASONIC_z = poly_point['z']
                    point_poly_w = np.array([[poly_point['x'], poly_point['y'], poly_point['z']]])
                    point_poly_s = (
                            rotation_m2v_inv.dot((point_poly_w - translation_m2v).T).T - vehicle_to_sensing
                    )
                    polygon_points_sensing.append({
                        "x": point_poly_s[0][0],
                        "y": point_poly_s[0][1],
                        "z": point_poly_s[0][2]
                    })
                item['polygonArea']['point'] = polygon_points_sensing
        return obstacle, ULTRASONIC_z

    def world2vehicle2sensing_chaosheng(self, obstacle, pose, vehicle2sensing, ULTRASONIC_z):
        translation_m2v, roll_pitch_yaw = pose['position'], pose['euler_angles']
        vehicle_to_sensing = vehicle2sensing['position']
        translation_m2v = np.asarray(translation_m2v).reshape(1, -1)
        rotation_m2v = self.get_rotation_matrix(roll_pitch_yaw)
        rotation_m2v_inv = np.linalg.inv(rotation_m2v)
        vehicle_to_sensing = np.asarray(vehicle_to_sensing).reshape(1, -1)
        # 障碍物坐标变换
        for item in obstacle:
            sensor_type = item.get('sensorType', 0)
            if sensor_type == 'ULTRASONIC':
                polygon_points_sensing = []
                for poly_point in item['polygonArea']['point']:
                    if ULTRASONIC_z == 0.0:
                        point_poly_w = np.array([[poly_point['x'], poly_point['y'], poly_point['z']]])
                    else:
                        point_poly_w = np.array([[poly_point['x'], poly_point['y'], ULTRASONIC_z]])
                    point_poly_s = (
                            rotation_m2v_inv.dot((point_poly_w - translation_m2v).T).T - vehicle_to_sensing
                    )
                    polygon_points_sensing.append({
                        "x": point_poly_s[0][0],
                        "y": point_poly_s[0][1],
                        "z": point_poly_s[0][2]
                    })
                item['polygonArea']['point'] = polygon_points_sensing
        return obstacle

    def world2vehicle2sensing_planning(self, planning_points, pose, vehicle2sensing):
        translation_m2v, roll_pitch_yaw = pose['position'], pose['euler_angles']
        vehicle_to_sensing = vehicle2sensing['position']
        translation_m2v = np.asarray(translation_m2v).reshape(1, -1)
        rotation_m2v = self.get_rotation_matrix(roll_pitch_yaw)
        rotation_m2v_inv = np.linalg.inv(rotation_m2v)
        vehicle_to_sensing = np.asarray(vehicle_to_sensing).reshape(1, -1)
        # 坐标变换
        new_planning = []
        index = 0
        for point in planning_points:
            point_poly_w = np.array([[point['x'], point['y'], 0]])
            point_poly_s = (
                    rotation_m2v_inv.dot((point_poly_w - translation_m2v).T).T - vehicle_to_sensing
            )
            index = index + 1
            new_planning.append([
                point_poly_s[0][0],
                point_poly_s[0][1],
                point_poly_s[0][2]
            ])
        return new_planning

    def get_obstacle_color(self, obstacle):
        """
        统一的颜色获取函数，与BEV保持一致

        Args:
            obstacle: 障碍物对象，包含 type 和可选的 freespace_type

        Returns:
            BGR颜色值
        """
        obj_type = obstacle.get('type', 0)

        sensor_type = obstacle.get('sensorType', 0)
        if sensor_type == 'ULTRASONIC':
            return ULTRASONIC_COLOR

        # 如果是PARK_FREESPACE，根据freespace_type确定颜色
        if obj_type == TYPE_PARK_FREESPACE:
            freespace_type = obstacle.get('freespaceType', None)
            if freespace_type is not None:
                return FREESPACE_TYPE_COLORS.get(freespace_type, TYPE_COLORS.get(TYPE_PARK_FREESPACE, DEFAULT_COLOR))
            else:
                return TYPE_COLORS.get(TYPE_PARK_FREESPACE, DEFAULT_COLOR)
        else:
            return TYPE_COLORS.get(obj_type, DEFAULT_COLOR)

    def get_rotation_matrix(self, roll_pitch_yaw):
        sr = math.sin(roll_pitch_yaw[0])
        sp = math.sin(roll_pitch_yaw[1])
        sy = math.sin(roll_pitch_yaw[2])
        cr = math.cos(roll_pitch_yaw[0])
        cp = math.cos(roll_pitch_yaw[1])
        cy = math.cos(roll_pitch_yaw[2])
        rot = np.array(
            [
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
                -sp,
                cp * sr,
                cp * cr,
            ]
        ).reshape((3, 3))
        return rot
        
    def get_transform(self, config_file: str, camera_names: List[str]) -> Dict:
        """
        获取相机变换矩阵（简化版，只包含必要的投影参数）
        从 .cfg 文件读取相机参数
        
        Args:
            config_file: 相机配置文件路径 (.cfg格式)
            camera_names: 相机名称列表
        
        Returns:
            相机名称到变换矩阵的映射
        """
        camera_name_to_trans_mat = {}

        camera_config = self._read_camera_cfg(config_file)
        
        for camera_name in camera_names:
            # 在cfg中查找对应的相机配置
            cam_cfg = None
            for cam in camera_config.get('camera', []):
                if cam.get('camera_dev') == camera_name:
                    cam_cfg = cam
                    break
            
            if cam_cfg is None:
                print(f"Warning: 未找到相机 {camera_name} 的配置")
                continue
            
            # 从嵌套结构中提取参数
            try:
                params = cam_cfg.get('parameters', {})
                extrinsic = params.get('extrinsic', {})
                sensor_to_cam = extrinsic.get('sensor_to_cam', {})
                intrinsic = params.get('intrinsic', {})
                
                # 获取外参：位置和方向
                pos = sensor_to_cam.get('position', {})
                ori = sensor_to_cam.get('orientation', {})
                
                # 使用四元数构建旋转矩阵
                from scipy.spatial.transform import Rotation as R
                quat = [ori['qx'], ori['qy'], ori['qz'], ori['qw']]  # scipy使用 [x,y,z,w] 顺序
                rotation_matrix = R.from_quat(quat).as_matrix()
                
                # 构建 4x4 外参矩阵（lidar到相机的变换）
                camera2lidar = np.eye(4, dtype=np.float32)
                camera2lidar[:3, :3] = rotation_matrix
                camera2lidar[:3, 3] = [pos['x'], pos['y'], pos['z']]
                lidar2camera = np.linalg.inv(camera2lidar)
                
                # 构建内参矩阵
                camera2image = np.eye(4, dtype=np.float32)
                camera2image[0, 0] = intrinsic['f_x']  # fx
                camera2image[1, 1] = intrinsic['f_y']  # fy
                camera2image[0, 2] = intrinsic['o_x']  # cx
                camera2image[1, 2] = intrinsic['o_y']  # cy
                
                # 畸变系数
                distortion_coeff = np.array([
                    intrinsic.get('k_1', 0),
                    intrinsic.get('k_2', 0),
                    intrinsic.get('k_3', 0),
                    intrinsic.get('k_4', 0)
                ], dtype=np.float32)
                
                trans_Mat = {
                    "distortion_coeff": distortion_coeff,
                    "intrinsics": camera2image,
                    "extrinsics": lidar2camera,  # 直接的 lidar 到相机的变换
                }
                camera_name_to_trans_mat[camera_name] = trans_Mat
                
            except KeyError as e:
                print(f"Error: 相机 {camera_name} 配置缺少必要字段: {e}")
                continue
            
        return camera_name_to_trans_mat
    
    def _read_camera_cfg(self, lines: str) -> Dict:
        """
        读取 .cfg 格式的相机配置文件（protobuf text format）
        
        Args:
            cfg_file: 配置文件路径
        
        Returns:
            配置字典，格式为 {'camera': [cam1_dict, cam2_dict, ...]}
        """
        def parse_value(value_str):
            """解析配置值"""
            value_str = value_str.strip()
            
            # 如果是字符串（带引号）
            if value_str.startswith('"') and value_str.endswith('"'):
                return value_str[1:-1]
            
            # 尝试解析为数字
            try:
                if '.' in value_str or 'e' in value_str.lower():
                    return float(value_str)
                else:
                    return int(value_str)
            except ValueError:
                return value_str
        
        def read_block(lines, start_idx):
            """递归读取配置块"""
            result = {}
            i = start_idx
            
            while i < len(lines):
                line = lines[i].strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    i += 1
                    continue
                
                # 如果遇到闭合大括号，返回
                if line == '}':
                    return result, i
                
                # 解析键值对或嵌套块
                if ':' in line and '{' not in line:
                    # 简单的键值对
                    key, value = line.split(':', 1)
                    result[key.strip()] = parse_value(value.strip())
                    i += 1
                elif '{' in line:
                    # 嵌套块
                    key = line.split('{')[0].strip()
                    sub_block, end_idx = read_block(lines, i + 1)
                    result[key] = sub_block
                    i = end_idx + 1
                else:
                    i += 1
            
            return result, i
        
        cameras = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # 查找 config 块
            if line == 'config {' or line.startswith('config {'):
                cam_dict, end_idx = read_block(lines, i + 1)
                cameras.append(cam_dict)
                i = end_idx + 1
            else:
                i += 1
        
        return {'camera': cameras}

    @staticmethod
    def _point_to_segment_dist_2d(px, py, x1, y1, x2, y2):
        """点 (px,py) 到线段 (x1,y1)-(x2,y2) 的 2D 最短距离。"""
        dx, dy = x2 - x1, y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def _precompute_chaosheng_img_points(self, chaosheng, ground_param,
                                         focal_length, camera_height,
                                         image_height, image_width):
        """将所有超声障碍物顶点投影到图像坐标系，返回 np.ndarray (M, 2)。"""
        all_pts = []
        for obs in chaosheng:
            polygon = obs.get("polygonArea", {}).get("point", [])
            if not polygon:
                continue
            pts = np.array(
                [[p.get('x', 0), p.get('y', 0), p.get('z', 0)] for p in polygon],
                dtype=np.float32)
            try:
                pts_2d = self.transform_sensor_to_avm_image(
                    pts, ground_param, focal_length, camera_height,
                    image_height, image_width)
                all_pts.append(pts_2d)
            except Exception:
                continue
        if all_pts:
            return np.vstack(all_pts)
        return np.empty((0, 2), dtype=np.float32)

    def _obstacle_near_chaosheng_pixels(self, obstacle, chaosheng_img_pts,
                                        pixel_threshold, ground_param,
                                        focal_length, camera_height,
                                        image_height, image_width):
        """判断相机障碍物任一顶点到任一超声顶点的最小像素距离 <= pixel_threshold。"""
        polygon_area = obstacle.get("polygonArea", {}).get("point", [])
        if not polygon_area:
            return False
        pts_3d = np.array(
            [[p.get('x', 0), p.get('y', 0), 0.0] for p in polygon_area],
            dtype=np.float32)
        try:
            pts_2d = self.transform_sensor_to_avm_image(
                pts_3d, ground_param, focal_length, camera_height,
                image_height, image_width)
        except Exception:
            return False
        for op in pts_2d:
            dists = np.sqrt(np.sum((chaosheng_img_pts - op) ** 2, axis=1))
            if np.min(dists) <= pixel_threshold:
                return True
        return False

    def transform_sensor_to_avm_image(self, sensing_points, ground_param, virtual_camera_focal_length, virtual_camera_height,
                                      image_height, image_width):
        """
        将lidar坐标系下的点转换到AVM图像坐标系

        Args:
            sensing_points: (N, 3) lidar坐标系下的点 [x, y, z]
            ground_param: (4,) 地面参数 [a, b, c, d]
            virtual_camera_focal_length: 虚拟相机焦距
            virtual_camera_height: 虚拟相机高度
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            img_points: (N, 2) 图像坐标系下的点 [u, v]
        """
        keys_order = ['a', 'b', 'c', 'd']
        ground_param = np.array([ground_param[key] for key in keys_order])
        assert sensing_points.shape[0] != 0, "points size should not be 0"
        assert sensing_points.shape[1] >= 3, "points should have at least 3 dimensions (x, y, z)"

        # 虚拟相机内参矩阵
        img2cam = np.array([[virtual_camera_focal_length, 0, image_width / 2.0],
                            [0, virtual_camera_focal_length, image_height / 2.0],
                            [0, 0, 1]], dtype=np.float32)

        # 地面到BEV坐标系的变换
        iso_ground_bev = np.eye(4)
        r_ground_bev = np.array([[0, -1, 0],
                                 [-1, 0, 0],
                                 [0, 0, -1]], dtype=np.float32)
        iso_ground_bev[:3, :3] = r_ground_bev
        iso_ground_bev[2, 3] = virtual_camera_height

        # Sensing到地面的对齐
        iso_sensing_ground = np.eye(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iso_sensing_ground[:3, :3] = R.align_vectors([[0, 0, 1]], [ground_param[:3]])[0].as_matrix()
        iso_sensing_ground[:3, 3] = np.array([0, 0, -ground_param[3]], dtype=np.float32)

        iso_sensing_bev = np.dot(iso_sensing_ground, iso_ground_bev)
        iso_bev_sensing = np.linalg.inv(iso_sensing_bev)

        # 将点投影到地面高度
        sensing_points_z = sensing_points[:, :3].copy()
        sensing_points_z[:, 2] = -ground_param[3]
        sensing_points_z = np.vstack((sensing_points_z.T, np.ones(
            (1, sensing_points_z.shape[0])).astype(np.float32)))

        # 转换到虚拟相机坐标系
        cam_points = np.dot(iso_bev_sensing, sensing_points_z)

        # 透视除法
        cam_points = cam_points[:3, :] / virtual_camera_height

        # 投影到图像
        img_points = np.dot(img2cam, cam_points).T
        img_points = img_points[:, :2]

        return img_points


    def draw_obstacles_on_bev(self, image, obstacles, chaosheng, ground_param, virtual_camera_focal_length, virtual_camera_height,
                              planning_point, chaosheng_pixel_radius=None):
        """
        在BEV图像上绘制障碍物的polygon

        Args:
            image: 输入BEV图像 (H, W, 3)
            obstacles: 障碍物列表
            ground_param: 地面参数
            virtual_camera_focal_length: 虚拟相机焦距
            virtual_camera_height: 虚拟相机高度
            chaosheng_pixel_radius: 若非 None，仅绘制质心在超声障碍图像中心 <= 该像素距离内的相机障碍物

        Returns:
            绘制了障碍物的图像
        """
        image_out = image.copy()
        image_height, image_width = image.shape[:2]
        # 画planing
        if len(planning_point) > 0:
            planning_point = np.array(planning_point, dtype=np.float32)
            # 转换到图像坐标系
            planning_point = self.transform_sensor_to_avm_image(
                planning_point,
                ground_param,
                virtual_camera_focal_length=virtual_camera_focal_length,
                virtual_camera_height=virtual_camera_height,
                image_height=image_height,
                image_width=image_width
            )
            # 初始化
            transformed_tail_points = planning_point
            start = transformed_tail_points[0]
            end = transformed_tail_points[-1]
            if end[1] - start[1] >= 5.0:
                # 使用抗锯齿线绘制
                for i in range(len(transformed_tail_points) - 1):
                    p1 = transformed_tail_points[i]
                    p2 = transformed_tail_points[i + 1]
                    cv2.line(image_out,
                             (int(p1[0]), int(p1[1])),
                             (int(p2[0]), int(p2[1])),
                             color=(255, 255, 255),
                             thickness=2)
                # 获取最后两个点
                p1 = transformed_tail_points[-2]  # 倒数第二个点
                p2 = transformed_tail_points[-1]  # 最后一个点（终点）

                # 计算方向向量
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                # 计算方向角度
                angle = math.atan2(dy, dx)

                # 计算箭头端点位置
                arrow_tip = (int(p2[0]), int(p2[1]))

                # 箭头左侧点
                arrow_length = 5
                left_angle = angle + math.pi * 0.75  # 135度
                left_point = (
                    int(p2[0] + arrow_length * math.cos(left_angle)),
                    int(p2[1] + arrow_length * math.sin(left_angle))
                )

                # 箭头右侧点
                right_angle = angle - math.pi * 0.75  # -135度
                right_point = (
                    int(p2[0] + arrow_length * math.cos(right_angle)),
                    int(p2[1] + arrow_length * math.sin(right_angle))
                )

                # 绘制箭头（填充三角形）
                arrow_points = np.array([arrow_tip, left_point, right_point], dtype=np.int32)
                cv2.fillPoly(image_out, [arrow_points], color=(255, 255, 255))

        chaosheng_img_pts = np.empty((0, 2), dtype=np.float32)
        if chaosheng_pixel_radius is not None:
            chaosheng_img_pts = self._precompute_chaosheng_img_points(
                chaosheng, ground_param, virtual_camera_focal_length,
                virtual_camera_height, image_height, image_width)

        for obstacle in obstacles:
            sensor_type = obstacle.get('sensorType', 0)
            if sensor_type == 'CAMERA':
                if chaosheng_pixel_radius is not None and \
                        (len(chaosheng_img_pts) == 0 or
                         not self._obstacle_near_chaosheng_pixels(
                             obstacle, chaosheng_img_pts, chaosheng_pixel_radius,
                             ground_param, virtual_camera_focal_length,
                             virtual_camera_height, image_height, image_width)):
                    continue
                obj_type = obstacle.get('type', 0)
                model_type = obstacle.get('modelType', 0)
                fs_type = obstacle.get('freespaceType', 0)
                if (obj_type == 'VEHICLE' and model_type == 'MODEL_PARKING') or (
                        obj_type == 'TRUCK' and model_type == 'MODEL_PARKING'):
                    color = (0, 255, 0)
                    polygon_area = obstacle.get("polygonArea", {}).get("point", [])
                    if polygon_area is None or len(polygon_area) == 0:
                        continue

                    # 提取所有点的坐标
                    points_3d = []
                    for point in polygon_area:
                        x = point.get('x', 0)
                        y = point.get('y', 0)
                        z = 0.0
                        points_3d.append([x, y, z])

                    points_3d = np.array(points_3d, dtype=np.float32)

                    if len(points_3d) == 0:
                        continue

                    if len(points_3d) <= 5:
                        continue

                    # 转换到图像坐标系
                    try:
                        points_2d = self.transform_sensor_to_avm_image(
                            points_3d,
                            ground_param,
                            virtual_camera_focal_length=virtual_camera_focal_length,
                            virtual_camera_height=virtual_camera_height,
                            image_height=image_height,
                            image_width=image_width
                        )
                    except Exception as e:
                        print(f"Warning: Failed to transform points for obstacle {obstacle.get('id', 'unknown')}: {e}")
                        continue

                    # 绘制连线（不连接首尾）
                    for i in range(len(points_2d) - 1):
                        pt1 = points_2d[i]
                        pt2 = points_2d[i + 1]
                        u1, v1 = int(pt1[0]), int(pt1[1])
                        u2, v2 = int(pt2[0]), int(pt2[1])
                        # 画线
                        cv2.line(image_out, (u1, v1), (u2, v2), color, 2)

                if obj_type == 'PARK_FREESPACE':
                    color = (0, 255, 255)  # 黄色 (BGR)

                    polygon_area = obstacle.get("polygonArea", {}).get("point", [])
                    if polygon_area is None or len(polygon_area) == 0:
                        continue

                    points_3d = []
                    for point in polygon_area:
                        x = point.get('x', 0)
                        y = point.get('y', 0)
                        z = 0.0
                        points_3d.append([x, y, z])

                    points_3d = np.array(points_3d, dtype=np.float32)

                    if len(points_3d) == 0:
                        continue

                    try:
                        points_2d = self.transform_sensor_to_avm_image(
                            points_3d,
                            ground_param,
                            virtual_camera_focal_length=virtual_camera_focal_length,
                            virtual_camera_height=virtual_camera_height,
                            image_height=image_height,
                            image_width=image_width
                        )
                    except Exception as e:
                        print(f"Warning: Failed to transform points for obstacle {obstacle.get('id', 'unknown')}: {e}")
                        continue

                    for i in range(len(points_2d) - 1):
                        pt1 = points_2d[i]
                        pt2 = points_2d[i + 1]
                        u1, v1 = int(pt1[0]), int(pt1[1])
                        u2, v2 = int(pt2[0]), int(pt2[1])
                        cv2.line(image_out, (u1, v1), (u2, v2), color, 2)

        pos = []
        for obstacle in chaosheng:
            if obstacle.get("freespaceType", "") == "FS_CAR":
                continue
            color = (0, 0, 255)
            polygon_area = obstacle.get("polygonArea", {}).get("point", [])
            if polygon_area is None or len(polygon_area) == 0:
                continue

            points_3d = []
            for point in polygon_area:
                x = point.get('x', 0)
                y = point.get('y', 0)
                z = point.get('z', 0)
                points_3d.append([x, y, z])

            points_3d = np.array(points_3d, dtype=np.float32)
            if len(points_3d) == 0:
                continue

            try:
                points_2d = self.transform_sensor_to_avm_image(
                    points_3d,
                    ground_param,
                    virtual_camera_focal_length=virtual_camera_focal_length,
                    virtual_camera_height=virtual_camera_height,
                    image_height=image_height,
                    image_width=image_width
                )
            except Exception as e:
                print(f"Warning: Failed to transform points for obstacle {obstacle.get('id', 'unknown')}: {e}")
                continue
            points_2d_int = points_2d.astype(np.int32)
            valid_points = points_2d_int[:-1] if len(points_2d_int) > 1 and np.array_equal(points_2d_int[0], points_2d_int[-1]) else points_2d_int

            if len(valid_points) >= 2:
                for i in range(len(valid_points) - 1):
                    pt1 = tuple(valid_points[i])
                    pt2 = tuple(valid_points[i + 1])
                    u1, v1 = int(pt1[0]), int(pt1[1])
                    u2, v2 = int(pt2[0]), int(pt2[1])
                    if 0 <= u1 < image_width and 0 <= v1 < image_height:
                        cv2.circle(image_out, (u1, v1), 2, color, -1)
                    if 0 <= u2 < image_width and 0 <= v2 < image_height:
                        cv2.circle(image_out, (u2, v2), 2, color, -1)
                    cv2.line(image_out, (u1, v1), (u2, v2), color, 2)

                center = np.mean(valid_points, axis=0)
                pos.append([int(center[0]), int(center[1])])

        return image_out, pos

    def compute_fs_car_data(self, obstacles, chaosheng, ground_param, virtual_camera_focal_length,
                            virtual_camera_height, image_height, image_width):
        """计算 FS_CAR 规则校验所需数据（box_list, point_list），不绘图。

        Returns:
            box_list:  相机端 FS_CAR/FS_BIGCAR 及 VEHICLE/TRUCK+MODEL_PARKING 的投影多边形顶点
            point_list: 超声端 FS_CAR 的投影多边形顶点
        """
        box_list = []
        point_list = []

        for obstacle in obstacles:
            if obstacle.get('sensorType', 0) != 'CAMERA':
                continue
            obj_type = obstacle.get('type', 0)
            model_type = obstacle.get('modelType', 0)
            fs_type = obstacle.get('freespaceType', 0)

            need_box = False
            if (obj_type == 'VEHICLE' and model_type == 'MODEL_PARKING') or \
               (obj_type == 'TRUCK' and model_type == 'MODEL_PARKING'):
                need_box = True
            elif obj_type == 'PARK_FREESPACE' and fs_type in ('FS_CAR', 'FS_BIGCAR'):
                need_box = True

            if not need_box:
                continue

            polygon_area = obstacle.get("polygonArea", {}).get("point", [])
            if not polygon_area:
                continue

            z_default = 0.0 if obj_type == 'PARK_FREESPACE' else None
            points_3d = []
            for point in polygon_area:
                x = point.get('x', 0)
                y = point.get('y', 0)
                z = z_default if z_default is not None else point.get('z', 0)
                points_3d.append([x, y, z])
            points_3d = np.array(points_3d, dtype=np.float32)
            if len(points_3d) == 0:
                continue
            if obj_type != 'PARK_FREESPACE' and len(points_3d) <= 5:
                continue

            try:
                points_2d = self.transform_sensor_to_avm_image(
                    points_3d, ground_param,
                    virtual_camera_focal_length=virtual_camera_focal_length,
                    virtual_camera_height=virtual_camera_height,
                    image_height=image_height, image_width=image_width)
            except Exception:
                continue

            box = [[int(pt[0]), int(pt[1])] for pt in points_2d]
            box_list.append(box)

        for obstacle in chaosheng:
            if obstacle.get("freespaceType", "") != "FS_CAR":
                continue
            polygon_area = obstacle.get("polygonArea", {}).get("point", [])
            if not polygon_area:
                continue
            points_3d = np.array(
                [[p.get('x', 0), p.get('y', 0), p.get('z', 0)] for p in polygon_area],
                dtype=np.float32)
            if len(points_3d) == 0:
                continue
            try:
                points_2d = self.transform_sensor_to_avm_image(
                    points_3d, ground_param,
                    virtual_camera_focal_length=virtual_camera_focal_length,
                    virtual_camera_height=virtual_camera_height,
                    image_height=image_height, image_width=image_width)
            except Exception:
                continue
            p_list = []
            for pt in points_2d:
                u, v = int(pt[0]), int(pt[1])
                if 0 <= u < image_width and 0 <= v < image_height:
                    p_list.append([u, v])
            point_list.append(p_list)

        return box_list, point_list






