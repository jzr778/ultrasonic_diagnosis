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
from typing import Dict, List, Tuple

from prompts_engine.context.freespace_catalog import normalize_freespace_label


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

    @staticmethod
    def _fit_ground_plane_z_from_xy_points(xs, ys, zs):
        """在世界系下用 (x,y,z) 点拟合 z = a*x + b*y + c。

        - 点数 >= 3：最小二乘平面；
        - 点数 1～2：退化为常数 z = median(z)，即 a=b=0, c=median；
        - 0 点：返回 None。
        """
        xs = np.asarray(xs, dtype=np.float64).ravel()
        ys = np.asarray(ys, dtype=np.float64).ravel()
        zs = np.asarray(zs, dtype=np.float64).ravel()
        n = xs.size
        if n == 0:
            return None
        if n < 3:
            return 0.0, 0.0, float(np.median(zs))
        A = np.column_stack([xs, ys, np.ones(n)])
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        return float(coef[0]), float(coef[1]), float(coef[2])

    def apply_chaosheng_z_from_camera_ground_plane(self, chaosheng, obstacle):
        """世界系下：用 ``obstacle`` 中 CAMERA 多边形顶点拟合地面 z(x,y)，写回 ``chaosheng`` 里 ULTRASONIC 各点 z。

        须在 ``world2vehicle2sensing*`` 之前调用；``obstacle`` / ``chaosheng`` 须与定位同一坐标系。
        """
        xs, ys, zs = [], [], []
        for item in obstacle or []:
            if item.get("sensorType") != "CAMERA":
                continue
            for pt in (item.get("polygonArea") or {}).get("point") or []:
                xs.append(float(pt.get("x", 0.0)))
                ys.append(float(pt.get("y", 0.0)))
                zs.append(float(pt.get("z", 0.0)))
        plane = self._fit_ground_plane_z_from_xy_points(xs, ys, zs)
        if plane is None:
            return chaosheng
        a, b, c = plane
        for item in chaosheng or []:
            if item.get("sensorType") != "ULTRASONIC":
                continue
            for pt in (item.get("polygonArea") or {}).get("point") or []:
                x = float(pt.get("x", 0.0))
                y = float(pt.get("y", 0.0))
                pt["z"] = a * x + b * y + c
        return chaosheng

    def world2vehicle2sensing(self, obstacle, pose, vehicle2sensing):
        translation_m2v, roll_pitch_yaw = pose['position'], pose['euler_angles']
        vehicle_to_sensing = vehicle2sensing['position']
        translation_m2v = np.asarray(translation_m2v).reshape(1, -1)
        rotation_m2v = self.get_rotation_matrix(roll_pitch_yaw)
        rotation_m2v_inv = np.linalg.inv(rotation_m2v)
        vehicle_to_sensing = np.asarray(vehicle_to_sensing).reshape(1, -1)
        for item in obstacle:
            if item.get('sensorType', 0) != 'CAMERA':
                continue
            polygon_points_sensing = []
            for poly_point in item['polygonArea']['point']:
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
        return obstacle

    def world2vehicle2sensing_chaosheng(self, obstacle, pose, vehicle2sensing):
        translation_m2v, roll_pitch_yaw = pose['position'], pose['euler_angles']
        vehicle_to_sensing = vehicle2sensing['position']
        translation_m2v = np.asarray(translation_m2v).reshape(1, -1)
        rotation_m2v = self.get_rotation_matrix(roll_pitch_yaw)
        rotation_m2v_inv = np.linalg.inv(rotation_m2v)
        vehicle_to_sensing = np.asarray(vehicle_to_sensing).reshape(1, -1)
        for item in obstacle:
            if item.get('sensorType', 0) != 'ULTRASONIC':
                continue
            polygon_points_sensing = []
            for poly_point in item['polygonArea']['point']:
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
        
        if isinstance(lines, (list, tuple)):
            lines = [str(x) for x in lines]
        elif isinstance(lines, str):
            lines = lines.splitlines()
        else:
            lines = []

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

    def _fisheye_draw_polyline_edges(
        self,
        img: np.ndarray,
        points_3d: List[List[float]],
        color: Tuple[int, int, int],
        thickness: int,
        rvec,
        tvec,
        K,
        D,
        cam: str,
        p_world: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """将 sensing 系 3D 折线投影到鱼眼并绘制边（与 BEV 一致的不闭合折线）。"""
        if len(points_3d) < 2:
            return
        pts_np = np.ascontiguousarray(np.array(points_3d, dtype=np.float32).reshape(-1, 1, 3))
        if cam == "panoramic_1":
            mask_xy = pts_np[:, 0, 0] > p_world[0]
        elif cam == "panoramic_2":
            mask_xy = pts_np[:, 0, 1] < p_world[1]
        elif cam == "panoramic_3":
            mask_xy = pts_np[:, 0, 0] < p_world[0]
        elif cam == "panoramic_4":
            mask_xy = pts_np[:, 0, 1] > p_world[1]
        else:
            mask_xy = np.ones(len(pts_np), dtype=bool)
        pts_np = pts_np[mask_xy]
        if pts_np.shape[0] == 0:
            return
        proj, _ = cv2.fisheye.projectPoints(pts_np, rvec=rvec, tvec=tvec, K=K, D=D)
        points_2d = proj.reshape(-1, 2)
        u = points_2d[:, 0]
        v = points_2d[:, 1]
        mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        points_2d_int = np.round(points_2d[mask]).astype(np.int32)
        if len(points_2d_int) < 2:
            return
        for i in range(len(points_2d_int) - 1):
            cv2.line(
                img,
                tuple(points_2d_int[i]),
                tuple(points_2d_int[i + 1]),
                color=color,
                thickness=thickness,
            )

    def plot_fisheye_polygon(
        self,
        img: np.ndarray,
        obstacles: List,
        chaosheng: List,
        extrinsics: np.ndarray,
        distortion_coeff: np.ndarray,
        intrinsics: np.ndarray,
        cam: str,
        ground_param,
        virtual_camera_focal_length: float,
        virtual_camera_height: float,
        chaosheng_pixel_radius=None,
        ignore_camera_freespace_types=None,
        bev_height: int = 800,
        bev_width: int = 640,
        resize: bool = True,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """鱼眼上对齐 draw_obstacles_on_bev：绿=泊车车、黄=PARK_FREESPACE（含 ignore_fs）、红=超声非 FS_CAR。

        resize 为 True 时将图缩至 1920×1440，并按比例缩放内参 K；相机多边形顶点 z 取自 json；超声顶点 z 由
        当前 sensing 系下全部 CAMERA 多边形顶点拟合 z=a*x+b*y+c（无相机顶点时退化为 json 的 z）。
        chaosheng_pixel_radius / ground_param 为 None 时不做 BEV 距离过滤。
        """
        orig_h, orig_w = int(img.shape[0]), int(img.shape[1])
        K = intrinsics[:3, :3].astype(np.float64)
        D = np.asarray(distortion_coeff[:4], dtype=np.float64)
        if resize:
            target_w, target_h = 1920, 1440
            sx = target_w / float(orig_w)
            sy = target_h / float(orig_h)
            K = K.copy()
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
            img = cv2.resize(img, (target_w, target_h))

        rvec = R.from_matrix(extrinsics[:3, :3]).as_rotvec()
        tvec = extrinsics[:3, 3].astype(np.float64)

        height, width = int(img.shape[0]), int(img.shape[1])

        p_cam = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        extrinsics_inv = np.linalg.inv(extrinsics)
        p_world = np.matmul(extrinsics_inv, p_cam)

        xs_fit, ys_fit, zs_fit = [], [], []
        for obstacle in obstacles:
            if obstacle.get("sensorType", 0) != "CAMERA":
                continue
            for p in (obstacle.get("polygonArea") or {}).get("point") or []:
                xs_fit.append(float(p.get("x", 0.0)))
                ys_fit.append(float(p.get("y", 0.0)))
                zs_fit.append(float(p.get("z", 0.0)))
        plane_abc = (
            self._fit_ground_plane_z_from_xy_points(xs_fit, ys_fit, zs_fit)
            if xs_fit
            else None
        )

        skip_fs = set(ignore_camera_freespace_types or [])
        chaosheng_img_pts = np.empty((0, 2), dtype=np.float32)
        if chaosheng_pixel_radius is not None and ground_param is not None:
            chaosheng_img_pts = self._precompute_chaosheng_img_points(
                chaosheng, ground_param, virtual_camera_focal_length,
                virtual_camera_height, bev_height, bev_width,
                plane_abc=plane_abc,
            )

        for obstacle in obstacles:
            sensor_type = obstacle.get('sensorType', 0)
            if sensor_type != 'CAMERA':
                continue
            if chaosheng_pixel_radius is not None and ground_param is not None and (
                    len(chaosheng_img_pts) == 0 or
                    not self._obstacle_near_chaosheng_pixels(
                        obstacle, chaosheng_img_pts, chaosheng_pixel_radius,
                        ground_param, virtual_camera_focal_length,
                        virtual_camera_height, bev_height, bev_width)):
                continue
            obj_type = obstacle.get('type', 0)
            model_type = obstacle.get('modelType', 0)
            if (obj_type == 'VEHICLE' and model_type == 'MODEL_PARKING') or (
                    obj_type == 'TRUCK' and model_type == 'MODEL_PARKING'):
                polygon_area = obstacle.get("polygonArea", {}).get("point", [])
                if not polygon_area:
                    continue
                points_3d = [
                    [p.get('x', 0), p.get('y', 0), float(p.get('z', 0))]
                    for p in polygon_area
                ]
                if len(points_3d) <= 5:
                    continue
                self._fisheye_draw_polyline_edges(
                    img, points_3d, (0, 255, 0), 2,
                    rvec, tvec, K, D, cam, p_world, width, height)
            if obj_type == 'PARK_FREESPACE':
                fs_label = normalize_freespace_label(obstacle.get("freespaceType"))
                if fs_label in skip_fs:
                    continue
                polygon_area = obstacle.get("polygonArea", {}).get("point", [])
                if not polygon_area:
                    continue
                points_3d = [
                    [p.get('x', 0), p.get('y', 0), float(p.get('z', 0))]
                    for p in polygon_area
                ]
                self._fisheye_draw_polyline_edges(
                    img, points_3d, (0, 255, 255), 2,
                    rvec, tvec, K, D, cam, p_world, width, height)

        pos: List[List[int]] = []
        for obstacle in chaosheng:
            if normalize_freespace_label(obstacle.get("freespaceType")) == "FS_CAR":
                continue
            color = (0, 0, 255)
            polygon_area = obstacle.get("polygonArea", {}).get("point", [])
            if not polygon_area:
                continue
            if plane_abc is not None:
                a, b, c = plane_abc
                points_3d = [
                    [
                        float(p.get("x", 0)),
                        float(p.get("y", 0)),
                        a * float(p.get("x", 0)) + b * float(p.get("y", 0)) + c,
                    ]
                    for p in polygon_area
                ]
            else:
                points_3d = [
                    [p.get('x', 0), p.get('y', 0), float(p.get('z', 0))]
                    for p in polygon_area
                ]
            pts_np = np.ascontiguousarray(np.array(points_3d, dtype=np.float32).reshape(-1, 1, 3))
            if cam == "panoramic_1":
                mask_xy = pts_np[:, 0, 0] > p_world[0]
            elif cam == "panoramic_2":
                mask_xy = pts_np[:, 0, 1] < p_world[1]
            elif cam == "panoramic_3":
                mask_xy = pts_np[:, 0, 0] < p_world[0]
            elif cam == "panoramic_4":
                mask_xy = pts_np[:, 0, 1] > p_world[1]
            else:
                mask_xy = np.ones(len(pts_np), dtype=bool)
            pts_np = pts_np[mask_xy]
            if pts_np.shape[0] == 0:
                continue
            proj, _ = cv2.fisheye.projectPoints(pts_np, rvec=rvec, tvec=tvec, K=K, D=D)
            points_2d = proj.reshape(-1, 2)
            u = points_2d[:, 0]
            v = points_2d[:, 1]
            mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            points_2d_int = np.round(points_2d[mask]).astype(np.int32)
            if len(points_2d_int) == 0:
                continue
            x_max, x_min = np.max(points_2d_int[:, 0]), np.min(points_2d_int[:, 0])
            y_max, y_min = np.max(points_2d_int[:, 1]), np.min(points_2d_int[:, 1])
            if (x_max - x_min) > width / 2.0 and (y_max - y_min) > height / 2.0:
                continue
            valid_points = points_2d_int[:-1] if len(points_2d_int) > 1 and np.array_equal(
                points_2d_int[0], points_2d_int[-1]) else points_2d_int
            if len(valid_points) < 2:
                continue
            for i in range(len(valid_points) - 1):
                pt1, pt2p = tuple(valid_points[i]), tuple(valid_points[i + 1])
                u1, v1 = int(pt1[0]), int(pt1[1])
                u2, v2 = int(pt2p[0]), int(pt2p[1])
                if 0 <= u1 < width and 0 <= v1 < height:
                    cv2.circle(img, (u1, v1), 4, color, -1)
                if 0 <= u2 < width and 0 <= v2 < height:
                    cv2.circle(img, (u2, v2), 4, color, -1)
                cv2.line(img, pt1, pt2p, color=color, thickness=4)
            center = np.mean(valid_points, axis=0)
            pos.append([int(center[0]), int(center[1])])

        return img, pos

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
                                         image_height, image_width,
                                         plane_abc=None):
        """将所有超声障碍物顶点投影到图像坐标系，返回 np.ndarray (M, 2)。

        plane_abc 为 (a, b, c) 时，各顶点 z 采用拟合平面 z = a*x + b*y + c（与 sensing 系下
        相机多边形顶点最小二乘一致）；为 None 时用 json 中的 z。
        """
        all_pts = []
        for obs in chaosheng:
            polygon = obs.get("polygonArea", {}).get("point", [])
            if not polygon:
                continue
            rows = []
            for p in polygon:
                x = float(p.get("x", 0))
                y = float(p.get("y", 0))
                if plane_abc is not None:
                    a, b, c = plane_abc
                    z = a * x + b * y + c
                else:
                    z = float(p.get("z", 0))
                rows.append([x, y, z])
            pts = np.array(rows, dtype=np.float32)
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
                              planning_point, chaosheng_pixel_radius=None, ignore_camera_freespace_types=None):
        """
        在BEV图像上绘制障碍物的polygon

        Args:
            image: 输入BEV图像 (H, W, 3)
            obstacles: 障碍物列表
            ground_param: 地面参数
            virtual_camera_focal_length: 虚拟相机焦距
            virtual_camera_height: 虚拟相机高度
            chaosheng_pixel_radius: 若非 None，仅保留「相机障碍多边形任一顶点到任一超声障碍多边形顶点」的
                欧氏像素距离最小值 <= 该阈值的条目后再绘制；超声侧点为各 chaosheng 多边形全部顶点（非质心）。
            ignore_camera_freespace_types: 若为 set/list，规范化后的枚举名（如 FS_CURB）命中的条目不绘制、不写入 yellow 元数据。
                freespaceType 缺省、无法解析或整型不在 0–16 时规范为 FS_OTHERS_STATIC。

        Returns:
            tuple: (绘制后的图像, 红色超声质心列表, 黄色 PARK_FREESPACE 元数据列表)
            黄色列表每项: {"freespaceType": str, "centroid": [u, v]}（组 prompt 时由 ContextBuilder 补全 freespaceTypeZh）
        """
        image_out = image.copy()
        image_height, image_width = image.shape[:2]
        yellow_freespace_meta = []
        skip_fs = set(ignore_camera_freespace_types or [])
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
                    fs_raw = obstacle.get("freespaceType")
                    fs_label = normalize_freespace_label(fs_raw)
                    if fs_label in skip_fs:
                        continue
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

                    c = np.mean(points_2d, axis=0)
                    yellow_freespace_meta.append({
                        "freespaceType": fs_label,
                        "centroid": [int(round(c[0])), int(round(c[1]))],
                    })

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

        return image_out, pos, yellow_freespace_meta

    def draw_fs_car_on_bev(self, image, obstacles, chaosheng, ground_param, virtual_camera_focal_length,
                           virtual_camera_height, planning_point, chaosheng_pixel_radius=None):
        """在 BEV 图像上绘制 FS_CAR 相关障碍物，同时返回规则校验所需数据。

        Returns:
            image_out: 绘制后的图像
            box_list:  相机端 FS_CAR/FS_BIGCAR 及 VEHICLE/TRUCK+MODEL_PARKING 的投影多边形顶点
            point_list: 超声端 FS_CAR 的投影多边形顶点
        """
        image_out = image.copy()
        image_height, image_width = image.shape[:2]

        if len(planning_point) > 0:
            planning_point = np.array(planning_point, dtype=np.float32)
            planning_point = self.transform_sensor_to_avm_image(
                planning_point, ground_param,
                virtual_camera_focal_length=virtual_camera_focal_length,
                virtual_camera_height=virtual_camera_height,
                image_height=image_height, image_width=image_width)
            transformed_tail_points = planning_point
            start = transformed_tail_points[0]
            end = transformed_tail_points[-1]
            if end[1] - start[1] >= 5.0:
                for i in range(len(transformed_tail_points) - 1):
                    p1 = transformed_tail_points[i]
                    p2 = transformed_tail_points[i + 1]
                    cv2.line(image_out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                             color=(255, 255, 255), thickness=2)
                p1 = transformed_tail_points[-2]
                p2 = transformed_tail_points[-1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle = math.atan2(dy, dx)
                arrow_tip = (int(p2[0]), int(p2[1]))
                arrow_length = 5
                left_point = (int(p2[0] + arrow_length * math.cos(angle + math.pi * 0.75)),
                              int(p2[1] + arrow_length * math.sin(angle + math.pi * 0.75)))
                right_point = (int(p2[0] + arrow_length * math.cos(angle - math.pi * 0.75)),
                               int(p2[1] + arrow_length * math.sin(angle - math.pi * 0.75)))
                arrow_points = np.array([arrow_tip, left_point, right_point], dtype=np.int32)
                cv2.fillPoly(image_out, [arrow_points], color=(255, 255, 255))

        chaosheng_img_pts = np.empty((0, 2), dtype=np.float32)
        if chaosheng_pixel_radius is not None:
            chaosheng_img_pts = self._precompute_chaosheng_img_points(
                chaosheng, ground_param, virtual_camera_focal_length,
                virtual_camera_height, image_height, image_width)

        box_list = []
        point_list = []

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
                if (obj_type == 'VEHICLE' and model_type == 'MODEL_PARKING') or \
                   (obj_type == 'TRUCK' and model_type == 'MODEL_PARKING'):
                    color = (0, 255, 0)
                    polygon_area = obstacle.get("polygonArea", {}).get("point", [])
                    if not polygon_area:
                        continue
                    points_3d = []
                    for point in polygon_area:
                        points_3d.append([point.get('x', 0), point.get('y', 0), point.get('z', 0)])
                    points_3d = np.array(points_3d, dtype=np.float32)
                    if len(points_3d) <= 5:
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
                    for i in range(len(points_2d) - 1):
                        pt1, pt2 = points_2d[i], points_2d[i + 1]
                        cv2.line(image_out, (int(pt1[0]), int(pt1[1])),
                                 (int(pt2[0]), int(pt2[1])), color, 2)

                if obj_type == 'PARK_FREESPACE':
                    fs_label = normalize_freespace_label(obstacle.get("freespaceType"))
                    color = (0, 255, 255)
                    is_fs_car = fs_label in ('FS_CAR', 'FS_BIGCAR')
                    polygon_area = obstacle.get("polygonArea", {}).get("point", [])
                    if not polygon_area:
                        continue
                    points_3d = np.array(
                        [[p.get('x', 0), p.get('y', 0), 0.0] for p in polygon_area],
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
                    if is_fs_car:
                        box = [[int(pt[0]), int(pt[1])] for pt in points_2d]
                        box_list.append(box)
                    for i in range(len(points_2d) - 1):
                        pt1, pt2 = points_2d[i], points_2d[i + 1]
                        cv2.line(image_out, (int(pt1[0]), int(pt1[1])),
                                 (int(pt2[0]), int(pt2[1])), color, 2)

        for obstacle in chaosheng:
            if obstacle.get("freespaceType", "") != "FS_CAR":
                continue
            color = (0, 0, 255)
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
            for i in range(len(points_2d) - 1):
                pt1, pt2 = points_2d[i], points_2d[i + 1]
                u1, v1 = int(pt1[0]), int(pt1[1])
                u2, v2 = int(pt2[0]), int(pt2[1])
                if 0 <= u1 < image_width and 0 <= v1 < image_height:
                    cv2.circle(image_out, (u1, v1), 2, color, -1)
                if 0 <= u2 < image_width and 0 <= v2 < image_height:
                    cv2.circle(image_out, (u2, v2), 2, color, -1)
                cv2.line(image_out, (u1, v1), (u2, v2), color, 2)

        return image_out, box_list, point_list






