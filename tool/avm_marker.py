#!/usr/bin/env python3
"""在 AVM（或任意）图片窗口内用鼠标绘制红色标记。

OpenCV 无内置“画笔”控件，采用 ``imshow`` + ``setMouseCallback`` 在图像副本上 ``line`` 实现。

用法::

    python tool/avm_marker.py /path/to/avm.jpg
    python tool/avm_marker.py /path/to/avm.jpg -o /path/out_marked.jpg

交互：

- 按住左键拖动：红色笔划（BGR ``(0,0,255)``）
- ``s``：保存到 ``-o`` 指定路径（未指定则用原图旁 ``*_marked.jpg``）
- ``z`` / ``r``：撤销一笔 / 清空
- ``q`` 或 ``ESC``：退出
- 独立工具无 ``n``；由 ``generate_avm_from_case --mark-avm`` 调用时支持 ``n`` 下一张
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

# 绘制窗口初次打开时的外边尺寸（可按图像同比缩放）；小图拉大、大图避免超出常见屏幕
WINDOW_LONG_SIDE_DEFAULT_MIN = 1280
WINDOW_LONG_SIDE_DEFAULT_MAX = 2048


def compute_default_imshow_window_size(
    img_h: int,
    img_w: int,
    *,
    min_long_side: int = WINDOW_LONG_SIDE_DEFAULT_MIN,
    max_long_side: int = WINDOW_LONG_SIDE_DEFAULT_MAX,
) -> Tuple[int, int]:
    """返回 (宽度, 高度) 供 ``cv2.resizeWindow``；保持图像宽高比，长边落在 [min, max]。"""
    hi, wi = int(img_h), int(img_w)
    if hi <= 0 or wi <= 0:
        return (min_long_side, int(round(min_long_side * 9 / 16)))
    long_side = max(hi, wi)
    target = min(max(long_side, min_long_side), max_long_side)
    scale = target / float(long_side)
    out_w = max(1, int(round(wi * scale)))
    out_h = max(1, int(round(hi * scale)))
    return (out_w, out_h)


def normalize_line_thickness_px(value: float | int) -> int:
    """逻辑线宽（可为小数）→ ``cv2.line`` 整数像素。``0 < x < 1`` 为最细 1px。"""
    x = float(value)
    if x <= 0:
        return 1
    if x < 1:
        return 1
    return max(1, int(round(x)))


@dataclass
class MarkerState:
    win: str
    base: np.ndarray  # 原始图，BGR
    canvas: np.ndarray  # 当前显示/编辑
    drawing: bool = False
    last: Tuple[int, int] = (-1, -1)
    color: Tuple[int, int, int] = (0, 0, 255)
    thickness: int = 1
    undo_stack: List[np.ndarray] = field(default_factory=list)
    stroke_points: List[Tuple[int, int]] = field(default_factory=list)
    all_stroke_points: List[Tuple[int, int]] = field(default_factory=list)

    def snapshot_for_undo(self) -> None:
        self.undo_stack.append(self.canvas.copy())

    def undo(self) -> None:
        if not self.undo_stack:
            self.canvas = self.base.copy()
            return
        self.canvas = self.undo_stack.pop()
        self.all_stroke_points.clear()


def _on_mouse(event: int, x: int, y: int, flags: int, param: Optional[MarkerState]) -> None:
    if param is None:
        return
    st = param

    if event == cv2.EVENT_LBUTTONDOWN:
        st.drawing = True
        st.last = (x, y)
        st.stroke_points = [(x, y)]
        st.snapshot_for_undo()

    elif event == cv2.EVENT_MOUSEMOVE:
        if st.drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.line(st.canvas, st.last, (x, y), st.color, st.thickness, lineType=cv2.LINE_AA)
            st.last = (x, y)
            st.stroke_points.append((x, y))
            cv2.imshow(st.win, st.canvas)

    elif event == cv2.EVENT_LBUTTONUP:
        if st.drawing and st.stroke_points:
            st.all_stroke_points.extend(st.stroke_points)
            st.stroke_points = []
        st.drawing = False


def _stroke_center(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """返回所有笔划点的质心 (x, y)；无点则返回 None。"""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys))))


def _update_crop_id_json(
    path: str, case_id: str, center: Tuple[int, int]
) -> None:
    """追加/更新 crop_id.json：``{ "case_id": "[x,y]", ... }``。"""
    data: Dict[str, str] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    data[case_id] = f"[{center[0]},{center[1]}]"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def interactive_red_mark_session(
    image_bgr: np.ndarray,
    *,
    save_path: str,
    window_title: str = "AVM Marker",
    thickness: float = 1,
    allow_next: bool = False,
    case_id: str = "",
    crop_id_path: str = "",
) -> Literal["quit", "next"]:
    """阻塞 OpenCV 窗口：鼠标画红色标记。``s`` 覆盖 ``save_path``；``q``/ESC 返回 ``quit``；``n`` 需 ``allow_next``。

    当 ``case_id`` 和 ``crop_id_path`` 都非空时，保存操作会同时把本次标记笔划的质心坐标写入 ``crop_id_path`` JSON。
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr 为空")

    win = window_title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    base = image_bgr.copy()
    canvas = base.copy()
    dw, dh = compute_default_imshow_window_size(base.shape[0], base.shape[1])
    try:
        cv2.resizeWindow(win, dw, dh)
    except cv2.error:
        pass
    state = MarkerState(
        win=win,
        base=base,
        canvas=canvas,
        thickness=normalize_line_thickness_px(thickness),
    )
    cv2.setMouseCallback(win, _on_mouse, state)

    msg = "左键拖动红色 | s 保存 | z 撤销 | r 清空"
    if allow_next:
        msg += " | n 下一张"
    msg += " | q/ESC 退出"
    print(msg)

    cv2.imshow(win, state.canvas)
    result: Literal["quit", "next"] = "quit"

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            result = "quit"
            break
        if allow_next and key == ord("n"):
            if case_id and crop_id_path and state.all_stroke_points:
                center = _stroke_center(state.all_stroke_points)
                if center is not None:
                    _update_crop_id_json(crop_id_path, case_id, center)
                    print(f"标记质心 [{center[0]},{center[1]}] → {crop_id_path}  (case={case_id})")
            result = "next"
            break
        if key == ord("s"):
            ok = cv2.imwrite(save_path, state.canvas)
            if ok:
                print(f"已保存: {save_path}")
            else:
                print(f"保存失败: {save_path}", file=sys.stderr)
            if case_id and crop_id_path:
                center = _stroke_center(state.all_stroke_points)
                if center is not None:
                    _update_crop_id_json(crop_id_path, case_id, center)
                    print(f"标记质心 [{center[0]},{center[1]}] → {crop_id_path}  (case={case_id})")
                else:
                    print("[mark] 未检测到绘制笔划，跳过坐标写入")
        elif key == ord("r"):
            state.canvas = state.base.copy()
            state.undo_stack.clear()
            state.all_stroke_points.clear()
            cv2.imshow(win, state.canvas)
            print("已重置为原图")
        elif key == ord("z"):
            state.undo()
            cv2.imshow(win, state.canvas)
            print("已撤销")

    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在图片上用鼠标画红色标记（OpenCV 窗口）")
    p.add_argument("image", help="输入图片路径")
    p.add_argument(
        "-o",
        "--out",
        default="",
        help="保存路径；默认在输入文件同目录生成 <stem>_marked<ext>",
    )
    p.add_argument(
        "-t",
        "--thickness",
        type=float,
        default=1,
        help="线宽（像素，可为小数；<1 为最细 1px），默认 1",
    )
    p.add_argument(
        "-w",
        "--window",
        default="AVM Marker",
        help="窗口标题",
    )
    return p.parse_args(argv)


def default_out_path(image_path: str) -> str:
    root, ext = os.path.splitext(image_path)
    if not ext:
        ext = ".jpg"
    return f"{root}_marked{ext}"


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    path = os.path.abspath(args.image)
    if not os.path.isfile(path):
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图片: {path}", file=sys.stderr)
        return 1

    out = args.out.strip() or default_out_path(path)
    interactive_red_mark_session(
        img,
        save_path=out,
        window_title=args.window,
        thickness=args.thickness,
        allow_next=False,
    )
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
