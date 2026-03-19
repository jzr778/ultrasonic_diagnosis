import os
import math
import json
from typing import List, Tuple


def is_point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """
    判断点是否在多边形内部（使用射线法）
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1 = polygon[0]
    for i in range(1, n + 1):
        p2 = polygon[i % n]

        # 检查点是否在边线上
        if min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]):
            if min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]):
                if abs((x - p1[0]) * (p2[1] - p1[1]) - (y - p1[1]) * (p2[0] - p1[0])) < 1e-9:
                    return True

        # 射线法判断
        if y > min(p1[1], p2[1]) and y <= max(p1[1], p2[1]):
            if x <= max(p1[0], p2[0]):
                if p1[1] != p2[1]:
                    xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if p1[0] == p2[0] or x <= xinters:
                    inside = not inside

        p1 = p2

    return inside


def distance_point_to_line(point: List[float],
                           line_start: List[float],
                           line_end: List[float]) -> float:
    """
    计算点到线段的最短距离
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if line_length_sq == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))

    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)

    return math.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)


def min_distance_point_to_box(point: List[float], box: List[List[float]]) -> float:
    """
    计算一个点到单个边框的最小距离（如果在框内则返回0）
    """
    # 如果点在框内，距离为0
    if is_point_in_polygon(point, box):
        return 0.0

    # 否则计算点到边框的最短距离
    min_distance = float('inf')
    n = len(box)

    for i in range(n):
        dist = distance_point_to_line(point, box[i], box[(i + 1) % n])
        if dist < min_distance:
            min_distance = dist

    return min_distance


def min_distance_point_to_all_boxes(point: List[float], boxes: List[List[List[float]]]) -> float:
    """
    计算一个点到所有边框的最小距离
    """
    min_distance = float('inf')

    for box in boxes:
        if not box:  # 跳过空边框
            continue

        # 计算点到当前边框的距离
        distance = min_distance_point_to_box(point, box)

        # 如果距离为0（在框内），直接返回0
        if distance == 0:
            return 0.0

        # 更新最小距离
        if distance < min_distance:
            min_distance = distance

    return min_distance if min_distance != float('inf') else 0.0


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[List[int]]:
    """
    使用Bresenham算法获取两点之间所有整数像素位置
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append([x0, y0])
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def calculate_segment_center(segment_points: List[List[int]]) -> List[float]:
    """
    计算线段中心点坐标
    """
    if not segment_points:
        return [0.0, 0.0]

    # 计算所有点的平均值作为中心点
    sum_x = sum(p[0] for p in segment_points)
    sum_y = sum(p[1] for p in segment_points)
    count = len(segment_points)

    return [int(sum_x / count), int(sum_y / count)]


def is_segment_misdetected(segment_points: List[List[int]], boxes: List[List[List[float]]],
                           threshold: float = 8.0, max_threshold: float = 20.0) -> bool:
    """判断超声 FS_CAR 线段是否为误检。
    规则1: 如果 >= 一半的顶点到最近相机框的距离 <= threshold，则认为不是误检。
    规则2: 即使满足规则1，若任意顶点距离 > max_threshold，仍算误检。
    多边形闭合时首尾顶点相同，去重后再判断。
    """
    if len(segment_points) == 0:
        return False
    seen = set()
    unique_pts = []
    for pt in segment_points:
        key = (pt[0], pt[1])
        if key not in seen:
            seen.add(key)
            unique_pts.append(pt)
    distances = [
        min_distance_point_to_all_boxes([float(pt[0]), float(pt[1])], boxes)
        for pt in unique_pts
    ]
    if max(distances) > max_threshold:
        return True
    within = sum(1 for d in distances if d <= threshold)
    return within < len(unique_pts) / 2


def get_max_distance_for_segment(segment_points: List[List[int]], boxes: List[List[List[float]]]) -> float:
    """
    计算单个线段上所有像素点到边框的最大最小距离
    当线段只有一个点时，计算该点的距离
    当有多个点时，遍历每对相邻点(i和i+1)，计算它们之间所有像素点到边框的距离
    返回所有距离中的最大值
    """
    if len(segment_points) == 0:
        return 0.0

    # 情况1: 只有一个点
    if len(segment_points) == 1:
        point = segment_points[0]
        point_float = [float(point[0]), float(point[1])]
        return min_distance_point_to_all_boxes(point_float, boxes)

    max_distance = 0.0

    # 情况2: 有多个点，遍历每对相邻点
    for i in range(len(segment_points) - 1):
        p1 = segment_points[i]
        p2 = segment_points[i + 1]

        # 转换为整数坐标（四舍五入）
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))

        # 获取两点之间所有整数像素点
        pixel_points = bresenham_line(x1, y1, x2, y2)

        # 计算每个像素点到边框的距离
        for pixel_point in pixel_points:
            distance = min_distance_point_to_all_boxes([float(pixel_point[0]), float(pixel_point[1])], boxes)

            # 更新最大距离
            if distance > max_distance:
                max_distance = distance

    return max_distance


# 使用示例
if __name__ == "__main__":
    tag_id_list = [87220060, 87220045, 87220054, 87220040, 87220022,]
    for i in range(len(tag_id_list)):
        tag_id = tag_id_list[i]
        print(tag_id)
        data_path = os.path.join('get_data/draw_image', str(tag_id))
        with open(data_path + '/point_list_avm.json', 'r', encoding='utf-8') as f:
            point_list = json.load(f)
        with open(data_path + '/box_list_avm.json', 'r', encoding='utf-8') as f:
            box_list = json.load(f)

        # 遍历所有线段（无论是一个还是多个）
        for segment_points in point_list:
            # 计算线段的最大距离
            max_distance = get_max_distance_for_segment(segment_points, box_list)

            # 计算线段中心点
            center_point = calculate_segment_center(segment_points)

            # 判断是否误检（距离 > 8）
            if max_distance > 8:
                print(f"  线段{idx + 1}: 中心点({center_point[0]:.1f}, {center_point[1]:.1f}), "
                      f"最大距离={max_distance:.2f}, 是误检")
            else:
                print(f"  线段{idx + 1}: 中心点({center_point[0]:.1f}, {center_point[1]:.1f}), "
                      f"最大距离={max_distance:.2f}, 正确检测")