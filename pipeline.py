#!/usr/bin/env python3
"""
AVP 全流程 Pipeline

步骤:
  1. get_id_mapping.py         → get_data/id_mapping.json  ({tag_id: feishu_id})
  2. bag.py                    → offline_avm_generate_release/bag_list.txt
  3. unpack_bag_for_avm.py     解包 bag 为鱼眼图输入
  4. save_bag_data.py          准备 read_data (vehicle2sensing / obstacle / pose 等)
  5. run_standalone.sh          拼接鱼眼图
  6. avp_vlm_pipeline_avm.py   绘制 AVM 标注图像
  7. avp_vlm_pipeline_avm.py   大模型诊断

日志自动保存到 logs/pipeline_<时间戳>.log，同时在终端实时输出。

用法:
  python pipeline.py -p iffcom -v U9zPLpFvR --model gemini-3-pro-preview
  python pipeline.py -p iffcom -v U9zPLpFvR --skip-steps 1 2
  python pipeline.py -p iffcom -v U9zPLpFvR --log-dir my_logs
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

import config

PROJECT_ROOT = str(config.PROJECT_ROOT)
PYTHON = sys.executable
TOTAL_STEPS = 7


def setup_logging(log_dir):
    """配置日志：同时输出到文件和终端"""
    now = datetime.now()
    date_dir = os.path.join(log_dir, now.strftime("%m%d"))
    os.makedirs(date_dir, exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(date_dir, f"pipeline_{timestamp}.log")

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_file}")
    return logger, log_file


log = logging.getLogger("pipeline")


def banner(step_num, desc):
    log.info("")
    log.info("=" * 60)
    log.info(f"  Step {step_num}/{TOTAL_STEPS}: {desc}")
    log.info("=" * 60)


def run(cmd, cwd=None, check=True, **kwargs):
    """运行子进程，stdout/stderr 实时输出并写入日志"""
    cmd_str = " ".join(cmd)
    log.info(f"  $ {cmd_str}")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )
    for line in proc.stdout:
        line = line.rstrip("\n")
        log.info(f"    {line}")
    proc.wait()

    if check and proc.returncode != 0:
        log.error(f"  命令失败 (exit code: {proc.returncode}): {cmd_str}")
        sys.exit(proc.returncode)
    return proc


# ── Step 1 ──────────────────────────────────────────────────
def step1_get_id_mapping(project_key, view_id, output_path):
    banner(1, "获取 tag_id ↔ feishu_id 映射")
    run([
        PYTHON, os.path.join(PROJECT_ROOT, "get_data", "get_id_mapping.py"),
        "-p", project_key, "-v", view_id,
        "-o", output_path,
    ])
    with open(output_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    tag_ids = [int(k) for k in mapping.keys()]
    feishu_ids = list(mapping.values())
    log.info(f"  ✅ 获取到 {len(mapping)} 条映射 → {output_path}")
    return tag_ids, feishu_ids


# ── Step 2 ──────────────────────────────────────────────────
def step2_get_bags(tag_ids, bag_list_path):
    banner(2, "获取 bag 列表")

    from get_data.get_meta_data import get_meta_data

    bags = []
    for tag_id in tag_ids:
        meta_data = get_meta_data(tag_id=tag_id)
        if not meta_data or not meta_data.get("body"):
            log.warning(f"  ⚠️  tag_id {tag_id} 无 meta_data，跳过")
            continue
        bag_name_list = meta_data["body"][0].get("bagsName", [])
        heavy_bags = sorted(b for b in bag_name_list if "Heavy" in b)
        bags.extend(heavy_bags)

    os.makedirs(os.path.dirname(os.path.abspath(bag_list_path)), exist_ok=True)
    with open(bag_list_path, "w", encoding="utf-8") as f:
        for bag in bags:
            f.write(bag + "\n")
    log.info(f"  ✅ 共 {len(bags)} 个 bag → {bag_list_path}")
    return bags


# ── Step 3 ──────────────────────────────────────────────────
def step3_unpack_bags(tag_ids, samples_dir):
    banner(3, "解包 bag (unpack_bag_for_avm)")

    from get_data.unpack_bag_for_avm import unpack_tag

    for tag_id in tag_ids:
        log.info(f"  解包 tag_id={tag_id} ...")
        unpack_tag(tag_id, output_root=samples_dir)

    log.info(f"  ✅ 解包完成，共处理 {len(tag_ids)} 个 tag")


# ── Step 4 ──────────────────────────────────────────────────
def step4_save_bag_data(tag_ids, read_data_dir):
    banner(4, "准备 read_data (save_bag_data)")

    from get_data.save_bag_data import save_data

    success = 0
    for tag_id in tag_ids:
        data_path = os.path.join(read_data_dir, str(tag_id))
        if os.path.isdir(data_path) and os.path.isfile(
            os.path.join(data_path, "vehicle2sensing.json")
        ):
            log.info(f"  tag_id={tag_id} 已存在，跳过")
            success += 1
            continue
        log.info(f"  保存 tag_id={tag_id} 数据 ...")
        try:
            save_data(tag_id, output_root=read_data_dir)
            success += 1
            log.info(f"  tag_id={tag_id} ✅")
        except Exception as e:
            log.warning(f"  tag_id={tag_id} 失败: {e}")

    log.info(f"  ✅ read_data 准备完成 ({success}/{len(tag_ids)})")


# ── Step 5 ──────────────────────────────────────────────────
def _find_unpacked_bags(samples_dir):
    """扫描 config 目录，返回实际已解包的 bag 名及其 YYYYMM 映射"""
    config_root = os.path.join(samples_dir, "config")
    if not os.path.isdir(config_root):
        return {}
    result = {}
    for yyyymm in sorted(os.listdir(config_root)):
        yyyymm_dir = os.path.join(config_root, yyyymm)
        if not os.path.isdir(yyyymm_dir):
            continue
        for bag_prefix in os.listdir(yyyymm_dir):
            cfg_path = os.path.join(yyyymm_dir, bag_prefix, "ground.cfg")
            if os.path.isfile(cfg_path):
                result[bag_prefix] = yyyymm
    return result


def step5_generate_avm(samples_dir, generate_dir):
    banner(5, "拼接鱼眼图 (offline_avm_generate_release)")

    avm_dir = os.path.join(PROJECT_ROOT, "offline_avm_generate_release")
    run_sh = os.path.join(avm_dir, "run_standalone.sh")
    if not os.path.isfile(run_sh):
        log.error(f"  ❌ 未找到 {run_sh}，跳过")
        return

    unpacked = _find_unpacked_bags(samples_dir)
    if not unpacked:
        log.warning(f"  ⚠️  config 目录下未找到已解包的 bag，跳过")
        return

    log.info(f"  发现 {len(unpacked)} 个已解包 bag，开始拼接")
    for bag_name, yyyymm in sorted(unpacked.items()):
        out_path = os.path.join(generate_dir, bag_name)
        if os.path.isdir(out_path) and os.listdir(out_path):
            log.info(f"  {bag_name} 已生成，跳过")
            continue
        log.info(f"  ===== Processing: {bag_name} (YYYYMM={yyyymm}) =====")
        run(
            ["bash", run_sh,
             "--interval", "1",
             "-i", samples_dir,
             "-o", generate_dir,
             "-b", bag_name],
            cwd=avm_dir,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    log.info(f"  ✅ 鱼眼图拼接完成")


# ── Step 6 ──────────────────────────────────────────────────
def step6_draw_images(id_mapping_path, read_data_dir):
    banner(6, "绘制 AVM 标注图像")

    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, "vlm", "avp_vlm_pipeline_avm.py"),
        "--id-mapping", id_mapping_path,
        "--data-path", read_data_dir,
        "--mode", "draw",
    ]
    run(cmd)
    log.info(f"  ✅ 绘图完成")


# ── Step 7 ──────────────────────────────────────────────────
def step7_run_vlm(id_mapping_path, read_data_dir, model=None):
    banner(7, "运行 VLM 大模型诊断")

    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, "vlm", "avp_vlm_pipeline_avm.py"),
        "--id-mapping", id_mapping_path,
        "--data-path", read_data_dir,
        "--mode", "diagnose",
    ]
    if model:
        cmd.extend(["--model"] + model)
    run(cmd)
    log.info(f"  ✅ VLM 诊断完成")


# ── main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AVP 全流程 Pipeline")
    parser.add_argument("-p", "--project-key", default=config.FEISHU_PROJECT_KEY,
                        help="飞书项目 Key")
    parser.add_argument("-v", "--view-id", default="U9zPLpFvR",
                        help="飞书视图 ID (默认: U9zPLpFvR)")
    parser.add_argument("--samples-dir",
                        default=config.SAMPLES_DIR,
                        help="unpack 输出 / AVM 输入目录")
    parser.add_argument("--generate-dir",
                        default=config.GENERATE_DIR,
                        help="AVM 鱼眼图输出目录")
    parser.add_argument("--read-data-dir",
                        default=config.READ_DATA_DIR,
                        help="save_bag_data 输出 / VLM 读取目录")
    parser.add_argument("--skip-steps", nargs="*", type=int, default=[],
                        help="跳过指定步骤编号 (1-7)，如 --skip-steps 1 2")
    parser.add_argument("--model", nargs="+", default=["auto"],
                        help="VLM 模型名称列表，透传给 step7 (默认: auto)")
    parser.add_argument("--list-models", action="store_true",
                        help="查询并列出所有可用的 VLM 模型，然后退出")
    parser.add_argument("--id-mapping",
                        default=os.path.join(PROJECT_ROOT, "get_data", "id_mapping.json"),
                        help="tag_id → feishu_id 映射文件 (默认: get_data/id_mapping.json)")
    parser.add_argument("--log-dir", default=os.path.join(PROJECT_ROOT, "logs"),
                        help="日志输出目录 (默认: logs/)")
    args = parser.parse_args()

    if args.list_models:
        from openai import OpenAI
        client = OpenAI(api_key=config.VLM_API_KEY, base_url=config.VLM_BASE_URL)
        models = client.models.list()
        print("可用模型列表：")
        for i, m in enumerate(models.data, 1):
            print(f"  {i}. {m.id}")
        sys.exit(0)

    _, log_file = setup_logging(args.log_dir)

    id_mapping_path = args.id_mapping
    bag_list_path = os.path.join(PROJECT_ROOT, "offline_avm_generate_release", "bag_list.txt")

    skip = set(args.skip_steps)

    log.info(f"项目根目录: {PROJECT_ROOT}")
    log.info(f"参数: project={args.project_key}, view={args.view_id}, "
             f"skip={args.skip_steps or '无'}")

    # Step 1
    if 1 not in skip:
        tag_ids, feishu_ids = step1_get_id_mapping(
            args.project_key, args.view_id, id_mapping_path
        )
    else:
        with open(id_mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        tag_ids = [int(k) for k in mapping.keys()]
        feishu_ids = list(mapping.values())
        log.info(f"[跳过 Step 1] 从文件加载 {len(mapping)} 条 tag_id ↔ feishu_id 映射")

    # Step 2
    if 2 not in skip:
        step2_get_bags(tag_ids, bag_list_path)
    else:
        log.info(f"[跳过 Step 2]")

    # Step 3
    if 3 not in skip:
        step3_unpack_bags(tag_ids, args.samples_dir)
    else:
        log.info(f"[跳过 Step 3]")

    # Step 4
    if 4 not in skip:
        step4_save_bag_data(tag_ids, args.read_data_dir)
    else:
        log.info(f"[跳过 Step 4]")

    # Step 5
    if 5 not in skip:
        step5_generate_avm(args.samples_dir, args.generate_dir)
    else:
        log.info(f"[跳过 Step 5]")

    # Step 6
    if 6 not in skip:
        step6_draw_images(id_mapping_path, args.read_data_dir)
    else:
        log.info(f"[跳过 Step 6]")

    # Step 7
    if 7 not in skip:
        step7_run_vlm(id_mapping_path, args.read_data_dir, model=args.model)
    else:
        log.info(f"[跳过 Step 7]")

    log.info("")
    log.info("=" * 60)
    log.info(f"  🎉 Pipeline 全部完成!")
    log.info(f"  日志已保存: {log_file}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
