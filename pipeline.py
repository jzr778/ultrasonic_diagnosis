#!/usr/bin/env python3
"""
AVP 全流程 Pipeline

步骤:
  1. get_id_mapping.py         → get_data/id_mapping.json  ({tag_id: feishu_id})
  2. bag.py                    → offline_avm_generate_release/bag_list.txt
  3. unpack_bag_for_avm + save_bag_data   解包 samples 并准备 read_data（每 tag 独立 BagReader；默认多 tag 并行，--unpack-workers 可调）
  4. run_standalone.sh          拼接鱼眼图（若车身 CarInfo 在超声事件时刻判后视镜折叠则跳过对应 bag）
  5. avp_vlm_pipeline_avm.py   绘制 AVM 标注图像（折叠 tag 从映射中剔除）
  6. avp_vlm_pipeline_avm.py   大模型诊断（同上）

日志自动保存到 logs/pipeline_<时间戳>.log，同时在终端实时输出。

用法:
  python pipeline.py -p iffcom -v U9zPLpFvR --model gemini-3-pro-preview
  python pipeline.py -p iffcom -v U9zPLpFvR --skip-steps 1 2
  python pipeline.py -p iffcom -v U9zPLpFvR --log-dir my_logs
  python pipeline.py ... --no-yuyan   # 关闭鱼眼抽帧与双图 VLM
  python pipeline.py ... --chaosheng-pixel-radius 40   # Step5 BEV 超声-相机关联半径（默认 30）
  python pipeline.py ... --unpack-workers 1   # Step3 强制串行（默认按 CPU 并行，上限 4）

原 7 步中曾单独跳过 Step 3 或 4 的场景，现合并为 Step 3（跳过 3 即不解包也不写 read_data）。
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import config

PROJECT_ROOT = str(config.PROJECT_ROOT)
PYTHON = sys.executable
TOTAL_STEPS = 6
# Step3 默认并行度：上限 4 减轻远端限流风险；单核机器为 1。
DEFAULT_UNPACK_WORKERS = max(1, min(4, (os.cpu_count() or 4)))
_MIRROR_FOLD_CACHE = "mirror_fold_cache.json"


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
def _read_data_has_ultrasonic_timestamps(data_path):
    """是否存在超声波事件对应的时间戳子目录（与 save_bag_data 写入的 chaosheng 一致）。"""
    if not os.path.isdir(data_path):
        return False
    try:
        for name in os.listdir(data_path):
            if not str(name).isdigit():
                continue
            ts_dir = os.path.join(data_path, name)
            if os.path.isdir(ts_dir) and os.path.isfile(
                os.path.join(ts_dir, "chaosheng.json")
            ):
                return True
    except OSError:
        return False
    return False


def _collect_bag_prefixes_for_tags(tag_ids, samples_dir):
    """根据 tag_ids 反查 meta_data，收集本批 tag 涉及的所有 Heavy bag_prefix + YYYYMM。"""
    from get_data.unpack_bag_for_avm import extract_bag_prefix, extract_yyyymm

    result = {}
    try:
        from get_data.get_meta_data import get_meta_data
    except ImportError:
        return result
    for tag_id in tag_ids:
        try:
            meta = get_meta_data(tag_id=tag_id)
        except Exception:
            continue
        if not meta or not meta.get("body"):
            continue
        for bag_name in meta["body"][0].get("bagsName") or []:
            if "Heavy" not in bag_name:
                continue
            prefix = extract_bag_prefix(bag_name)
            yyyymm = extract_yyyymm(prefix)
            if prefix and yyyymm:
                result[prefix] = yyyymm
    return result


def step3_unpack_and_save_bag_data(
    tag_ids,
    samples_dir,
    read_data_dir,
    extract_fisheye=True,
    unpack_workers=DEFAULT_UNPACK_WORKERS,
):
    """返回本次涉及的 bag_prefix → yyyymm 映射（含已跳过和新解包的），供 Step 4 过滤。"""
    banner(
        3,
        "解包 samples 并准备 read_data（unpack_bag_for_avm + save_bag_data；多 tag 可并行）",
    )

    from get_data.save_bag_data import save_data
    from get_data.step3_unpack_worker import unpack_one_tag
    from get_data.unpack_bag_for_avm import unpack_tag

    success = 0
    pending = []
    for tag_id in tag_ids:
        data_path = os.path.join(read_data_dir, str(tag_id))
        v2s = os.path.join(data_path, "vehicle2sensing.json")
        if os.path.isdir(data_path) and os.path.isfile(v2s):
            log.info(f"  tag_id={tag_id} read_data 已存在，跳过解包与 save_data")
            success += 1
            continue
        pending.append(tag_id)

    workers = max(1, int(unpack_workers or 1))
    if not pending:
        log.info(f"  ✅ Step 3 完成（解包 + read_data）({success}/{len(tag_ids)})")
        bag_prefixes = _collect_bag_prefixes_for_tags(tag_ids, samples_dir)
        return bag_prefixes

    if workers == 1 or len(pending) == 1:
        for tag_id in pending:
            log.info(f"  tag_id={tag_id} 解包 + save_data ...")
            try:
                reader = unpack_tag(
                    tag_id, output_root=samples_dir, return_reader=True
                )
                save_data(
                    tag_id,
                    output_root=read_data_dir,
                    extract_fisheye=extract_fisheye,
                    reader=reader,
                )
                success += 1
                log.info(f"  tag_id={tag_id} ✅")
            except Exception as e:
                log.warning(f"  tag_id={tag_id} 失败: {e}")
    else:
        n = min(workers, len(pending))
        log.info(
            f"  Step3 并行解包: {len(pending)} 个 tag，进程数={n}（--unpack-workers={workers}）"
        )
        payloads = [
            (tid, samples_dir, read_data_dir, extract_fisheye) for tid in pending
        ]
        with ProcessPoolExecutor(max_workers=n) as ex:
            futures = {
                ex.submit(unpack_one_tag, p): p[0] for p in payloads
            }
            for fut in as_completed(futures):
                tag_id, ok, err = fut.result()
                if ok:
                    success += 1
                    log.info(f"  tag_id={tag_id} ✅")
                else:
                    log.warning(f"  tag_id={tag_id} 失败: {err}")

    log.info(f"  ✅ Step 3 完成（解包 + read_data）({success}/{len(tag_ids)})")
    bag_prefixes = _collect_bag_prefixes_for_tags(tag_ids, samples_dir)
    return bag_prefixes


# ── Step 4 ──────────────────────────────────────────────────
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


def step4_generate_avm(
    samples_dir,
    generate_dir,
    skip_bag_prefixes=None,
    only_bag_prefixes=None,
):
    """拼接鱼眼图。

    only_bag_prefixes: 若非空，只处理该集合内的 bag（来自 Step 3 的返回值），
                       不再全量扫描 samples/config/。
    """
    banner(4, "拼接鱼眼图 (offline_avm_generate_release)")

    skip_bag_prefixes = skip_bag_prefixes or set()

    avm_dir = os.path.join(PROJECT_ROOT, "offline_avm_generate_release")
    run_sh = os.path.join(avm_dir, "run_standalone.sh")
    if not os.path.isfile(run_sh):
        log.error(f"  ❌ 未找到 {run_sh}，跳过")
        return

    if only_bag_prefixes is not None:
        targets = {k: v for k, v in only_bag_prefixes.items() if k not in skip_bag_prefixes}
        if not targets:
            log.info("  本次 tag 无需拼接的 bag（全部被跳过或无 Heavy bag），结束")
            return
        log.info(f"  本次 tag 涉及 {len(only_bag_prefixes)} 个 bag，过滤后待处理 {len(targets)} 个")
    else:
        unpacked = _find_unpacked_bags(samples_dir)
        if not unpacked:
            log.warning(f"  ⚠️  config 目录下未找到已解包的 bag，跳过")
            return
        targets = {k: v for k, v in unpacked.items() if k not in skip_bag_prefixes}
        log.info(f"  发现 {len(unpacked)} 个已解包 bag，过滤后待处理 {len(targets)} 个")

    for bag_name, yyyymm in sorted(targets.items()):
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


# ── Step 5 ──────────────────────────────────────────────────
def step5_draw_images(id_mapping_path, read_data_dir, ignore_fs_types=None, yuyan=True,
                      chaosheng_pixel_radius=30):
    banner(5, "绘制 AVM 标注图像")

    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, "vlm", "avp_vlm_pipeline_avm.py"),
        "--id-mapping", id_mapping_path,
        "--data-path", read_data_dir,
        "--mode", "draw",
        "--chaosheng-pixel-radius", str(chaosheng_pixel_radius),
    ]
    if ignore_fs_types:
        cmd.extend(["--ignore-fs-types"] + ignore_fs_types)
    if not yuyan:
        cmd.append("--no-yuyan")
    run(cmd)
    log.info(f"  ✅ 绘图完成")


# ── Step 6 ──────────────────────────────────────────────────
def step6_run_vlm(id_mapping_path, read_data_dir, model=None, ignore_fs_types=None,
                  debug_thinking=False, yuyan=True):
    banner(6, "运行 VLM 大模型诊断")

    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, "vlm", "avp_vlm_pipeline_avm.py"),
        "--id-mapping", id_mapping_path,
        "--data-path", read_data_dir,
        "--mode", "diagnose",
    ]
    if model:
        cmd.extend(["--model"] + model)
    if ignore_fs_types:
        cmd.extend(["--ignore-fs-types"] + ignore_fs_types)
    if debug_thinking:
        cmd.append("--debug-thinking")
    if not yuyan:
        cmd.append("--no-yuyan")
    run(cmd)
    log.info(f"  ✅ VLM 诊断完成")


def _load_mirror_skip_info_from_read_data(tag_ids, read_data_dir):
    """优先复用 read_data 内缓存的后视镜折叠结果，避免重复远端读 bag。"""
    folded_tags = set()
    folded_prefixes = set()
    missing_tags = []
    for tag_id in tag_ids:
        cache_path = os.path.join(read_data_dir, str(tag_id), _MIRROR_FOLD_CACHE)
        if not os.path.isfile(cache_path):
            missing_tags.append(int(tag_id))
            continue
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            log.warning(f"  读取后视镜缓存失败 tag_id={tag_id}: {e}")
            missing_tags.append(int(tag_id))
            continue
        if payload.get("folded"):
            folded_tags.add(int(tag_id))
            folded_prefixes.update(payload.get("folded_heavy_prefixes") or [])
    return folded_tags, folded_prefixes, missing_tags


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
                        help="跳过指定步骤编号 (1-6)，如 --skip-steps 1 2")
    parser.add_argument("--model", nargs="+", default=["auto"],
                        help="VLM 模型名称列表，透传给 step6 (默认: auto)")
    parser.add_argument("--list-models", action="store_true",
                        help="查询并列出所有可用的 VLM 模型，然后退出")
    parser.add_argument("--id-mapping",
                        default=os.path.join(PROJECT_ROOT, "get_data", "id_mapping.json"),
                        help="tag_id → feishu_id 映射文件 (默认: get_data/id_mapping.json)")
    parser.add_argument("--ignore-fs-types", nargs="*", default=[],
                        help="绘图/诊断时忽略的超声 freespaceType，如 --ignore-fs-types FS_CURB FS_CHOCK")
    parser.add_argument("--debug-thinking", action="store_true",
                        help="Step6 记录 VLM 原始回复到 logs/MMDD/debug_thinking_*.txt")
    parser.add_argument("--log-dir", default=os.path.join(PROJECT_ROOT, "logs"),
                        help="日志输出目录 (默认: logs/)")
    parser.add_argument(
        "--no-yuyan",
        dest="yuyan",
        action="store_false",
        help="关闭鱼眼解包与 VLM 鱼眼辅助（默认开启）",
    )
    parser.set_defaults(yuyan=True)
    parser.add_argument(
        "--chaosheng-pixel-radius",
        type=int,
        default=30,
        help="Step5 BEV 超声与相机障碍关联像素半径（默认 30）",
    )
    parser.add_argument(
        "--unpack-workers",
        type=int,
        default=DEFAULT_UNPACK_WORKERS,
        metavar="N",
        help=(
            "Step3 并行解包进程数（默认 min(CPU 核数, 4)；多 tag 并行；远端易限流可改小，"
            "强制串行用 1）"
        ),
    )
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
             f"yuyan={args.yuyan}, chaosheng_pixel_radius={args.chaosheng_pixel_radius}, "
             f"unpack_workers={args.unpack_workers}, "
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
    tag_bag_prefixes = None
    if 3 not in skip:
        tag_bag_prefixes = step3_unpack_and_save_bag_data(
            tag_ids,
            args.samples_dir,
            args.read_data_dir,
            extract_fisheye=args.yuyan,
            unpack_workers=args.unpack_workers,
        )
    else:
        log.info(f"[跳过 Step 3]")

    # 与 bag_reader 一致：Light bag CAR_STATE_TOPIC（CarInfo）在超声事件时刻若判后视镜折叠 → 不跑 Step4–6。
    # 若 Step 3–4 均已跳过，则不再访问远端 bag（仅跑 5/6 时使用已有 read_data，跳过后视镜预检）。
    mirror_skip_tags = set()
    mirror_skip_prefixes = set()
    mirror_filtered_mapping_path = None
    prep_skipped = {3, 4}.issubset(skip)
    need_mirror_check = (
        not prep_skipped
        and (4 not in skip or 5 not in skip or 6 not in skip)
    )
    cache_tags, cache_prefixes, missing_cache_tags = _load_mirror_skip_info_from_read_data(
        tag_ids, args.read_data_dir
    )
    if cache_tags or not missing_cache_tags:
        mirror_skip_tags |= cache_tags
        mirror_skip_prefixes |= cache_prefixes
        log.info(
            f"  从 read_data 缓存读取后视镜折叠结果：folded_tags={sorted(cache_tags)}，"
            f"缓存缺失={len(missing_cache_tags)}"
        )
    if prep_skipped and (5 not in skip or 6 not in skip) and missing_cache_tags:
        log.info(
            "  [跳过后视镜折叠远端预检] Step 3–4 已跳过，且部分 tag 无 read_data 缓存；"
            "Step 5/6 仅按已有缓存剔除，未命中缓存的 tag 保留"
        )
    if need_mirror_check:
        from get_data.bag_reader import avm_skip_mirror_fold_info

        if missing_cache_tags:
            remote_skip_tags, remote_skip_prefixes = avm_skip_mirror_fold_info(
                missing_cache_tags
            )
            mirror_skip_tags |= remote_skip_tags
            mirror_skip_prefixes |= remote_skip_prefixes
        if mirror_skip_tags:
            log.info(
                f"  CarInfo（{config.CAR_STATE_TOPIC}）在至少一个超声事件时刻为后视镜折叠，"
                f"对应 tag 将跳过 Step4–6: {sorted(mirror_skip_tags)}"
            )
        if mirror_skip_tags and (5 not in skip or 6 not in skip):
            with open(id_mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            filtered = {
                k: v for k, v in mapping.items() if int(k) not in mirror_skip_tags
            }
            fd, mirror_filtered_mapping_path = tempfile.mkstemp(
                suffix=".json", prefix="id_mapping_mirror_fold_skip_"
            )
            os.close(fd)
            with open(mirror_filtered_mapping_path, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
    vlm_id_mapping = mirror_filtered_mapping_path or id_mapping_path

    # Step 4
    if 4 not in skip:
        step4_generate_avm(
            args.samples_dir,
            args.generate_dir,
            skip_bag_prefixes=mirror_skip_prefixes,
            only_bag_prefixes=tag_bag_prefixes,
        )
    else:
        log.info(f"[跳过 Step 4]")

    # Step 5
    if 5 not in skip:
        step5_draw_images(
            vlm_id_mapping,
            args.read_data_dir,
            ignore_fs_types=args.ignore_fs_types,
            yuyan=args.yuyan,
            chaosheng_pixel_radius=args.chaosheng_pixel_radius,
        )
    else:
        log.info(f"[跳过 Step 5]")

    # Step 6
    if 6 not in skip:
        result_dir = config.RESULT_DIR
        if os.path.isdir(result_dir):
            log.info(f"[Step 6 前] 删除旧诊断结果目录: {result_dir}")
            shutil.rmtree(result_dir)
        elif os.path.lexists(result_dir):
            log.info(f"[Step 6 前] 删除路径: {result_dir}")
            os.remove(result_dir)
        step6_run_vlm(vlm_id_mapping, args.read_data_dir, model=args.model,
                      ignore_fs_types=args.ignore_fs_types,
                      debug_thinking=args.debug_thinking,
                      yuyan=args.yuyan)
    else:
        log.info(f"[跳过 Step 6]")

    if mirror_filtered_mapping_path and os.path.isfile(mirror_filtered_mapping_path):
        try:
            os.remove(mirror_filtered_mapping_path)
        except OSError:
            pass

    log.info("")
    log.info("=" * 60)
    log.info(f"  🎉 Pipeline 全部完成!")
    log.info(f"  日志已保存: {log_file}")
    log.info("=" * 60)


if __name__ == "__main__":
    # import sys
    # sys.argv = [
    #     "avp_vlm_pipeline_avm.py",
    #     "--id-mapping", "/home/jiangzirou/avp_promptkit/get_data/test.json",
    #     # "--mode", "draw",  # 调试鱼眼/绘图时先只跑 draw，避免走 VLM
    #     # "--model", "gemini-3-pro-preview",
    #     "--skip-steps", "1","2","6"
    # ]
    main()
