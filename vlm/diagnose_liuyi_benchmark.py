"""
针对 liuyi_benchmark 数据集的 VLM 大模型诊断脚本。

数据目录结构（与 draw_image 一致）:
  liuyi_benchmark/{tag_id}/{timestamp}/
    ├── avm.jpg
    ├── index_avm.json
    ├── box_list_avm.json   (可选，FS_CAR规则校验用)
    └── point_list_avm.json (可选，FS_CAR规则校验用)

用法:
  python vlm/diagnose_liuyi_benchmark.py
  python vlm/diagnose_liuyi_benchmark.py --model gemini-3-pro-preview
  python vlm/diagnose_liuyi_benchmark.py --workers 4 --prompt-config chaosheng_wujian_avm
"""

import json
import os
import sys
import shutil
import argparse
import cv2
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from vlm.VLM_API import analyze_scenario_from_images
from vlm.avp_vlm_pipeline_avm import get_direction_from_position
from vlm.point2box_mindistance_avm import is_segment_misdetected, calculate_segment_center
from prompts_engine.prompt_gen import prompt_gen

logger = logging.getLogger(__name__)

BENCHMARK_DIR = "/mnt/public-data/user/ziroujiang/avp/liuyi_benchmark"
RESULT_DIR = "/mnt/public-data/user/ziroujiang/avp/result_liuyi_benchmark"


def setup_logging():
    now = datetime.now()
    log_dir = os.path.join(str(config.PROJECT_ROOT), "logs", now.strftime("%m%d"))
    os.makedirs(log_dir, exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"liuyi_benchmark_{timestamp}.log")

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


def diagnose_single_tag(tag_id, args):
    """对单个 tag_id 运行 VLM 诊断。"""
    stats = {"misdetected": [], "normal": [], "api_error": []}
    tag_dir = os.path.join(args.data_dir, str(tag_id))
    if not os.path.isdir(tag_dir):
        logger.warning(f"[诊断] tag={tag_id} 目录不存在，跳过")
        stats["no_draw_output"] = [tag_id]
        return stats

    all_items = sorted(
        [d for d in os.listdir(tag_dir) if os.path.isdir(os.path.join(tag_dir, d))],
        key=lambda x: int(x)
    )
    if not all_items:
        logger.warning(f"[诊断] tag={tag_id} 无时间戳子目录，跳过")
        stats["no_draw_output"] = [tag_id]
        return stats

    comment_record = ""

    for item in all_items:
        item_path = os.path.join(tag_dir, item)
        index_path = os.path.join(item_path, "index_avm.json")
        avm_img_path = os.path.join(item_path, "avm.jpg")

        if not os.path.isfile(avm_img_path):
            logger.warning(f"[诊断] tag={tag_id}, ts={item} 缺少 avm.jpg，跳过")
            continue
        if not os.path.isfile(index_path):
            logger.warning(f"[诊断] tag={tag_id}, ts={item} 缺少 index_avm.json，跳过")
            continue

        logger.info(f"[诊断] tag={tag_id}, ts={item}")

        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        box_list_path = os.path.join(item_path, "box_list_avm.json")
        point_list_path = os.path.join(item_path, "point_list_avm.json")
        result_fs_car = []
        if os.path.isfile(box_list_path) and os.path.isfile(point_list_path):
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

        if len(index.get('avm', [])) == 0:
            analysis_result = {'positions': []}
        else:
            bev_img = cv2.imread(avm_img_path)
            panoramic_1 = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
            image_list = {'panoramic_1': panoramic_1}
            prompt = prompt_gen(index, args.prompt_config)
            analysis_result = analyze_scenario_from_images(image_list, prompt, args.model)
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

            with open(os.path.join(save_path, "analysis_result.json"), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            for jpg in os.listdir(item_path):
                if jpg.endswith(".jpg"):
                    shutil.copy2(os.path.join(item_path, jpg), save_path)
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
                direction_text = direction_text + 'FS_OTHERS_STATIC误检点相对于车的位置：' + ', '.join(direction)

            comment_record += f"  ts={item}: {direction_text}\n"
        else:
            stats["normal"].append((tag_id, item))
            save_path = os.path.join(args.output_dir, "normal", str(tag_id), item)
            os.makedirs(save_path, exist_ok=True)
            for jpg in os.listdir(item_path):
                if jpg.endswith(".jpg"):
                    shutil.copy2(os.path.join(item_path, jpg), save_path)

    if comment_record:
        logger.info(f"[诊断] tag={tag_id} 误检方位:\n{comment_record}")

    logger.info(f"[诊断] tag={tag_id} 完成 (误检={len(stats['misdetected'])}, 正常={len(stats['normal'])}, API异常={len(stats['api_error'])})")
    return stats


def main():
    parser = argparse.ArgumentParser(description="liuyi_benchmark VLM 诊断")
    parser.add_argument("--data-dir", default=BENCHMARK_DIR,
                        help=f"数据目录 (默认: {BENCHMARK_DIR})")
    parser.add_argument("--output-dir", default=RESULT_DIR,
                        help=f"结果输出目录 (默认: {RESULT_DIR})")
    parser.add_argument("--model", nargs="+", default=["auto"],
                        help="VLM 模型名称列表 (默认: auto)")
    parser.add_argument("--prompt-config", default="chaosheng_wujian_avm",
                        help="prompt 配置文件名 (默认: chaosheng_wujian_avm)")
    parser.add_argument("--workers", type=int, default=8,
                        help="并行线程数 (默认: 8)")
    args = parser.parse_args()

    setup_logging()

    tag_ids = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ], key=lambda x: int(x))

    total_tags = len(tag_ids)
    logger.info(f"参数: data_dir={args.data_dir}, workers={args.workers}, model={args.model}")
    logger.info(f"发现 {total_tags} 个 tag")

    all_stats = {
        "no_draw_output": [],
        "misdetected":    [],
        "normal":         [],
        "api_error":      [],
        "exception":      [],
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for tag_id_str in tag_ids:
            tag_id = int(tag_id_str)
            future = executor.submit(diagnose_single_tag, tag_id, args)
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

    # ── 汇总 ──
    W = 64
    SEP = "=" * W
    THIN = "-" * W

    logger.info("")
    logger.info(SEP)
    logger.info("  liuyi_benchmark 诊断结果汇总")
    logger.info(SEP)
    logger.info(f"  输入 tag 总数: {total_tags}")
    logger.info("")

    n_no_draw = len(all_stats["no_draw_output"])
    if n_no_draw:
        logger.info(f"  无数据/目录:    {n_no_draw} 个 tag")
        for tag_id in all_stats["no_draw_output"]:
            logger.info(f"    tag={tag_id}")

    n_misdet = len(all_stats["misdetected"])
    n_normal = len(all_stats["normal"])
    n_api_err = len(all_stats["api_error"])
    tags_misdet = sorted(set(t for t, _ in all_stats["misdetected"]))
    tags_normal = sorted(set(t for t, _ in all_stats["normal"]))

    logger.info(f"  检测到误检:     {n_misdet} 条 ({len(tags_misdet)} 个 tag)")
    for tag_id, ts in all_stats["misdetected"]:
        logger.info(f"    tag={tag_id}, ts={ts}")
    logger.info(f"  检测正常:       {n_normal} 条 ({len(tags_normal)} 个 tag)")
    for tag_id, ts in all_stats["normal"]:
        logger.info(f"    tag={tag_id}, ts={ts}")
    logger.info(f"  API 异常:       {n_api_err} 条")
    for tag_id, ts in all_stats["api_error"]:
        logger.info(f"    tag={tag_id}, ts={ts}")

    n_exc = len(all_stats["exception"])
    if n_exc:
        logger.info(f"  运行异常:       {n_exc} 个 tag")
        for tag_id, err in all_stats["exception"]:
            logger.error(f"    tag={tag_id}  错误: {err}")

    logger.info("")
    logger.info(THIN)
    n_ts_total = n_misdet + n_normal + n_api_err
    logger.info(f"  总计: {n_ts_total} 条 (误检 {n_misdet} / 正常 {n_normal} / API异常 {n_api_err})")
    if n_exc:
        logger.info(f"  异常: {n_exc} 个 tag")
    logger.info(SEP)


if __name__ == "__main__":
    main()
