#!/usr/bin/env python3
"""
AVP 全流程 Pipeline

步骤:
  1. get_id_mapping.py         → get_data/id_mapping.json  ({tag_id: feishu_id})
  2. 已移除（原「获取 bag 列表」，冗余：Step 3 内部自行通过 meta_data 获取 bag）
  3. unpack_bag_for_avm + save_bag_data   解包 samples 并准备 read_data（每 tag 独立 BagReader；默认多 tag 并行，--unpack-workers 可调）
  4. run_standalone.sh          拼接鱼眼图（若 read_data 内后视镜折叠缓存判折叠则跳过对应 bag；无缓存 tag 不再远端补读）
  5. avp_vlm_pipeline_avm.py   绘制 AVM 标注图像（折叠 tag 从映射中剔除）
  6. EAS 微调模型三分类 → 映射「是否误检」（默认）；--openai-diagnose 时改走 VLM 大模型
  7. upload_to_feishu            诊断结果 CSV + 图片上传飞书电子表格（默认开启，追加去重）

日志自动保存到 diagnosis_logs/MMDD/pipeline_<时间戳>.log，同时在终端实时输出。

用法:
  python pipeline.py -p iffcom -v U9zPLpFvR                    # 默认：EAS 微调诊断 + 上传飞书
  python pipeline.py -p iffcom -v U9zPLpFvR --openai-diagnose  # 可选：VLM 大模型 API 诊断
  python pipeline.py -p iffcom -v U9zPLpFvR --skip-steps 1
  python pipeline.py ... --no-yuyan   # 关闭鱼眼抽帧与双图 VLM
  python pipeline.py ... --chaosheng-pixel-radius 40   # Step5 BEV 超声-相机关联半径（默认 30）
  python pipeline.py ... --unpack-workers 1   # Step3 强制串行（默认按 CPU 并行，上限 4）
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
TOTAL_STEPS = 7
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


def run(cmd, cwd=None, check=True, quiet=False, **kwargs):
    """运行子进程。quiet=True 时不捕获输出（不写日志、不打终端），用于 Step4 等刷屏二进制。"""
    cmd_str = " ".join(cmd)
    cwd = cwd or PROJECT_ROOT
    stdin = kwargs.pop("stdin", None)

    if quiet:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        proc.wait()
        if check and proc.returncode != 0:
            log.error(f"  命令失败 (exit code: {proc.returncode}): {cmd_str}")
            sys.exit(proc.returncode)
        return proc

    log.info(f"  $ {cmd_str}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        stdin=stdin,
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


def _collect_event_bag_prefixes(tag_ids, read_data_dir):
    """从 read_data 中收集有超声事件的 tag 对应的精确 Heavy bag_prefix → YYYYMM。

    数据来源：Step 3 的 save_data 写入的 event_heavy_bags.json（仅含触发超声的 bag）。
    无超声事件的 tag 自动跳过，无 event_heavy_bags.json 的 tag 也跳过（返回值中不含）。
    """
    from get_data.unpack_bag_for_avm import extract_bag_prefix, extract_yyyymm

    result = {}
    hit = 0
    miss_no_event = 0
    miss_no_cache = 0

    for tag_id in tag_ids:
        tag_dir = os.path.join(read_data_dir, str(tag_id))
        if not os.path.isdir(tag_dir):
            miss_no_event += 1
            continue

        cache_path = os.path.join(tag_dir, "event_heavy_bags.json")
        if not os.path.isfile(cache_path):
            has_ts = any(
                e.isdigit() and os.path.isdir(os.path.join(tag_dir, e))
                for e in os.listdir(tag_dir)
            )
            if has_ts:
                miss_no_cache += 1
            else:
                miss_no_event += 1
            continue

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                event_bags = json.load(f)
            for bag_name in event_bags:
                prefix = extract_bag_prefix(bag_name)
                yyyymm = extract_yyyymm(prefix)
                if prefix and yyyymm:
                    result[prefix] = yyyymm
            hit += 1
        except Exception:
            miss_no_cache += 1

    log.info(
        f"  bag 精确收集: {hit} 个 tag 命中缓存 → {len(result)} 个 bag，"
        f"无超声事件 {miss_no_event}，缺缓存 {miss_no_cache}"
    )
    return result, miss_no_cache


def step3_unpack_and_save_bag_data(
    tag_ids,
    samples_dir,
    read_data_dir,
    extract_fisheye=True,
    unpack_workers=DEFAULT_UNPACK_WORKERS,
):
    """解包 bag 并准备 read_data，返回有超声事件的精确 bag_prefix → yyyymm 映射。

    数据来源：save_data 在 scan_ultrasonic_events 后写入的 event_heavy_bags.json，
    只包含真正触发超声事件的 Heavy bag，而非该 tag 的全部 Heavy bag。
    """
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

    if pending:
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

    bag_prefixes, miss = _collect_event_bag_prefixes(tag_ids, read_data_dir)
    if miss:
        log.warning(
            f"  ⚠️  {miss} 个有超声事件的 tag 缺少 event_heavy_bags.json 缓存，"
            f"需重跑 Step 3（删除对应 read_data 后重新解包）以生成精确缓存"
        )
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

    for bag_name, _yyyymm in sorted(targets.items()):
        out_path = os.path.join(generate_dir, bag_name)
        if os.path.isdir(out_path) and os.listdir(out_path):
            log.info(f"  case_id={bag_name} 已拼接，跳过")
            continue
        proc = run(
            ["bash", run_sh,
             "--interval", "1",
             "-i", samples_dir,
             "-o", generate_dir,
             "-b", bag_name],
            cwd=avm_dir,
            check=False,
            quiet=True,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            log.info(f"  拼接完成 case_id={bag_name}")
        else:
            log.warning(
                f"  拼接失败 case_id={bag_name} exit_code={proc.returncode}"
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


# ── Step 6 EAS 分支 ──────────────────────────────────────────
def _collect_draw_image_to_flat(
    draw_image_dir,
    read_data_dir,
    dst_root,
    ignore_fs_types=None,
    only_tag_ids=None,
):
    """将 pipeline Step5 产出的 draw_image/<tag>/<ts>/ 转换为 EAS 需要的 images/crop/yuyan 平铺结构。

    only_tag_ids: 仅收集这些 tag 的数据；None 则收集全部。
    复用 tool/collect_raw_data.py 已有逻辑。

    Returns:
        本次收集到的 case stem 集合（如 {"123456_17700001", ...}）。
    """
    from tool.collect_raw_data import _copy_drawn_images, _crop_drawn_images

    tag_strs = [str(t) for t in only_tag_ids] if only_tag_ids else None
    scope = f"tag_ids={tag_strs}" if tag_strs else "全部"
    log.info(f"  draw_image → 平铺结构: {dst_root}（{scope}）")

    # 收集前记录 images/ 已有文件，用于计算本次增量
    images_dir = os.path.join(dst_root, "images")
    before = set(os.listdir(images_dir)) if os.path.isdir(images_dir) else set()

    _copy_drawn_images(draw_image_dir, dst_root, only_tag_ids=tag_strs)
    _crop_drawn_images(
        draw_image_dir,
        read_data_dir,
        dst_root,
        size=150,
        ignore_fs_types=ignore_fs_types or [],
        only_tag_ids=tag_strs,
    )

    after = set(os.listdir(images_dir)) if os.path.isdir(images_dir) else set()
    new_files = after - before
    collected_stems = {os.path.splitext(f)[0] for f in new_files}

    crop_dir = os.path.join(dst_root, "crop")
    yuyan_dir = os.path.join(dst_root, "yuyan")
    n_crop = len(os.listdir(crop_dir)) if os.path.isdir(crop_dir) else 0
    n_yuyan = len(os.listdir(yuyan_dir)) if os.path.isdir(yuyan_dir) else 0
    log.info(
        f"  本次收集: images={len(new_files)}, "
        f"目录总量: images={len(after)}, crop={n_crop}, yuyan={n_yuyan}"
    )
    return collected_stems


def step6_eas_diagnose(
    tag_ids,
    data_dir,
    output_dir,
    *,
    eas_base="",
    eas_token="",
    max_tokens=32,
    timeout=600,
    log_every=20,
    resume=False,
    draw_image_dir="",
    read_data_dir="",
    ignore_fs_types=None,
    id_mapping=None,
    project_key="",
):
    """用 EAS 微调模型对 images/crop/yuyan 三图做三分类→映射误检。

    流程：
      1. 从 draw_image/ 收集本次 tag_ids 的三图到 data_dir（pipeline_data/）
      2. 按 tag_ids 过滤 data_dir/images/ 中属于本次流程的 case
      3. 对每个 case 调 EAS 三分类 → 聚合映射「是否误检」
      4. 输出 eas_eval_predictions.jsonl + eas_labels.csv 到 output_dir（log/）
      5. 按 tag 聚合诊断结果，发飞书评论

    id_mapping: {tag_id_str: feishu_id} 映射，用于发飞书评论。
    """
    banner(6, "EAS 微调模型三分类 → 误检诊断")

    from tool.eas_eval import (
        _b64_image,
        _eas_auth_headers,
        _normalize_pred,
    )
    from tool.build_labels import build_row
    from prompts_engine.context.object_type_catalog import object_type_task_prompt
    import csv
    import time
    import requests
    from pathlib import Path
    from collections import defaultdict

    # Step 6a: 从 draw_image 收集本次 tag 的三图到 data_dir
    if draw_image_dir:
        _collect_draw_image_to_flat(
            draw_image_dir, read_data_dir, data_dir,
            ignore_fs_types=ignore_fs_types,
            only_tag_ids=tag_ids,
        )

    eas_base = (eas_base or config.EAS_BASE_URL).rstrip("/")
    token = eas_token or config.EAS_TOKEN
    url = f"{eas_base}/v1/chat/completions"
    headers = _eas_auth_headers(token)

    now = datetime.now()
    date_sub = now.strftime("%m%d")
    ts_tag = now.strftime("%Y%m%d_%H%M%S")
    eas_out_dir = os.path.join(output_dir, date_sub)
    os.makedirs(eas_out_dir, exist_ok=True)
    jsonl_path = os.path.join(eas_out_dir, f"eas_eval_predictions_{ts_tag}.jsonl")
    csv_path = os.path.join(eas_out_dir, f"eas_labels_{ts_tag}.csv")

    data_root = Path(data_dir)
    images_dir = data_root / "images"
    crop_dir = data_root / "crop"
    yuyan_dir = data_root / "yuyan"

    if not images_dir.is_dir():
        log.error(f"  images/ 目录不存在: {images_dir}")
        return

    # Step 6b: 按 tag_ids 过滤，只诊断本次流程的 case
    tag_prefixes = {str(t) for t in tag_ids}
    all_stems = sorted(
        p.stem for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    stems = sorted(s for s in all_stems if s.split("_")[0] in tag_prefixes)

    log.info(f"  数据目录: {data_dir}")
    log.info(f"  目录总量: {len(all_stems)} 个 case，本次 tag 匹配: {len(stems)} 个")
    log.info(f"  EAS: {eas_base}")

    SYSTEM_PROMPT = (
        "你是泊车环视场景的超声/视觉联合诊断模型，须结合多图作答。图例：红标=超声地面障碍物"
        "（点、短线或闭合多边形），为分析对象，表示超声在地面上的感知结果。绿线=邻车检测框"
        "投影到地面的多边形，表示邻车可能占用的地面区域。黄线=相机障碍在AVM上的投影轮廓，"
        "用于在鸟瞰中对齐真实可见障碍；判断时对齐黄线与真实障碍本体，比较红标与真实障碍的"
        "关系，勿将红标与黄线本身当作一对匹配目标。中心黑矩形=自车（上为车头、下为车尾），"
        "正在倒车入库。车位中心白箭头=预计倒车方向；若无箭头，默认沿车位中轴线直线倒车。"
        "白矩形框=仅遮挡车牌，与障碍物/标线无关，分析时完全忽略。AVM由鱼眼展开拼接：红标"
        "仅有地面投影、无高度语义；离地越高常渐淡/半透明或与背景融合，属成像与拼接特性，"
        "不等于该处无实物。须结合鱼眼透视理解障碍远近、立面与地面接触，区分竖直方向透视"
        "表现与地面接触位置，避免仅凭AVM上半部发虚误判空间关系。输入按顺序三张：①AVM"
        "鸟瞰：以红标为准；高处虚化不得单独作为无实体依据。②以超声障碍质心为中心的局部"
        "crop，用于聚焦红标。③与AVM主方位一致的单路鱼眼：绿/黄与AVM语义一致，图中不画"
        "红标，红标仍以AVM为准；作透视与尺度参考，减轻仅凭鸟瞰在远近、实体尺度与类型上的"
        "不确定；禁止在鱼眼与AVM之间做像素级距离换算或强行点配对。回答必须严格遵守用户"
        "给出的任务与可选项；只输出要求的标签或词，不要解释。"
    )

    TASKS = [
        (
            "entity_existence",
            "<image>任务：实体存在性判定。请判断红色超声高亮附近是否存在真实障碍。可选项：yes, no。",
        ),
        (
            "geometry_relation",
            "<image>任务：几何一致性判定。请判断红色超声高亮与附近真实障碍之间的几何关系。可选项：aligned, misaligned。",
        ),
        (
            "object_type",
            object_type_task_prompt(),
        ),
    ]

    def _resolve_images(stem):
        """返回 [avm, crop, yuyan] 三图路径列表，缺图返回 None。"""
        paths = []
        for d in (images_dir, crop_dir, yuyan_dir):
            found = None
            for ext in (".jpg", ".jpeg", ".png"):
                p = d / f"{stem}{ext}"
                if p.is_file():
                    found = p
                    break
            if found is None:
                return None
            paths.append(found)
        return paths

    def _call_eas(image_paths, task_prompt):
        user_text = task_prompt.replace("<image>", "", 1).strip()
        user_content = [
            {"type": "image_url", "image_url": {"url": _b64_image(p)}}
            for p in image_paths
        ]
        user_content.append({"type": "text", "text": user_text})
        payload = {
            "model": config.EAS_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            resp = requests.post(
                url, json=payload, headers=headers,
                timeout=(min(60, timeout), timeout),
            )
        except requests.exceptions.RequestException as e:
            return "", str(e)
        if resp.status_code != 200:
            return "", f"HTTP {resp.status_code}: {(resp.text or '')[:300]}"
        try:
            body = resp.json()
            pred = body["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return "", f"parse: {e}"
        return pred.strip(), ""

    done_ids: set = set()
    if resume and os.path.isfile(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("case_id") and not r.get("error"):
                        done_ids.add(r["case_id"])
                except json.JSONDecodeError:
                    pass
        log.info(f"  resume: 已有 {len(done_ids)} 条结果，跳过")

    all_preds: dict = defaultdict(dict)
    total = len(stems)
    stats = {"ok": 0, "error": 0, "skip": 0}
    mode = "a" if resume and os.path.isfile(jsonl_path) else "w"

    with open(jsonl_path, mode, encoding="utf-8") as out_f:
        for idx, stem in enumerate(stems, 1):
            if stem in done_ids:
                stats["skip"] += 1
                continue

            img_paths = _resolve_images(stem)
            if img_paths is None:
                log.warning(f"  [{idx}/{total}] {stem} 缺三图之一，跳过")
                stats["error"] += 1
                continue

            case_preds = {}
            has_error = False
            t0 = time.time()

            for task_name, task_prompt in TASKS:
                if task_name != "entity_existence" and case_preds.get("entity_existence") == "no":
                    break
                pred_raw, err = _call_eas(img_paths, task_prompt)
                pred_norm = _normalize_pred(pred_raw) if pred_raw else ""
                row = {
                    "case_id": stem,
                    "task": task_name,
                    "prediction": pred_raw,
                    "prediction_norm": pred_norm,
                    "error": err,
                    "latency_s": round(time.time() - t0, 2),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                if err:
                    has_error = True
                    break
                case_preds[task_name] = pred_norm

            elapsed = round(time.time() - t0, 2)
            if has_error:
                stats["error"] += 1
                log.warning(f"  [{idx}/{total}] {stem} ERR {elapsed}s")
            else:
                stats["ok"] += 1
                all_preds[stem] = case_preds
                if idx % log_every == 0:
                    log.info(
                        f"  [{idx}/{total}] {stem} OK "
                        f"entity={case_preds.get('entity_existence','')} "
                        f"geom={case_preds.get('geometry_relation','')} "
                        f"obj={case_preds.get('object_type','')} "
                        f"{elapsed}s"
                    )

    # 若 resume 模式下有旧结果，也需要把旧结果加载到 all_preds
    if resume and done_ids:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    cid = r.get("case_id", "")
                    task = r.get("task", "")
                    if cid and task and not r.get("error"):
                        all_preds[cid][task] = r.get("prediction_norm", "")
                except json.JSONDecodeError:
                    pass

    rows = []
    for stem in sorted(all_preds.keys()):
        preds = all_preds[stem]
        fields = {
            "entity": preds.get("entity_existence", ""),
            "geometry": preds.get("geometry_relation", ""),
            "object_type": preds.get("object_type", ""),
        }
        rows.append(build_row(stem, fields))

    fieldnames = ["case_id", "实体是否存在", "超声标记命中或偏移", "障碍物类型", "是否误检"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    misdetect_dist = Counter(r["是否误检"] for r in rows)
    log.info(f"  完成: 成功 {stats['ok']}, 失败 {stats['error']}, 跳过(resume) {stats['skip']}")
    log.info(f"  预测 jsonl: {jsonl_path}")
    log.info(f"  标签 CSV: {csv_path} ({len(rows)} 条)")
    log.info(f"  是否误检分布: {dict(misdetect_dist)}")

    # 按 tag 聚合诊断结果，发飞书评论
    if id_mapping and project_key:
        from collections import defaultdict as _dd
        tag_results = _dd(list)
        for r in rows:
            tag = r["case_id"].split("_")[0]
            ts = "_".join(r["case_id"].split("_")[1:])
            tag_results[tag].append((ts, r["是否误检"]))

        try:
            from comment.add_comment import FeishuCommentTester
            tester = FeishuCommentTester()
        except Exception as e:
            log.warning(f"  飞书评论初始化失败，跳过: {e}")
            tester = None

        if tester:
            for tag_str, items in sorted(tag_results.items()):
                feishu_id = id_mapping.get(tag_str)
                if not feishu_id:
                    continue
                lines = []
                for ts, verdict in items:
                    lines.append(f"时间戳{ts}：{verdict}")
                comment_record = "EAS诊断：\n" + "\n".join(lines)
                log.info(f"  飞书评论 tag={tag_str}: {comment_record}")
                try:
                    test_url = (
                        f"https://project.feishu.cn/{project_key}"
                        f"/case/detail/{feishu_id}"
                    )
                    tester.test_comment(test_url, comment_record)
                except Exception as e:
                    log.warning(f"  飞书评论发送失败 tag={tag_str}: {e}")
            log.info(f"  飞书评论发送完成")
    else:
        log.info(f"  未传入 id_mapping 或 project_key，跳过飞书评论")

    log.info(f"  ✅ EAS 微调模型诊断完成")
    return csv_path


# ── Step 7 ──────────────────────────────────────────────────
_FEISHU_OPEN_API = "https://open.feishu.cn/open-apis"
_DEFAULT_FEISHU_APP_ID = "cli_a6e0444aedfbd00b"
_DEFAULT_FEISHU_APP_SECRET = "8W1Art9TRWrV50C7QgITwbYbMMqLKI5x"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff"}
_MAX_RANGES_PER_BATCH = 200
_IMAGE_UPLOAD_TIMEOUT = 120
_REQ_TIMEOUT = 15.0


def _feishu_tenant_token():
    app_id = (
        os.environ.get("FEISHU_APP_ID", "").strip()
        or os.environ.get("LARK_APP_ID", "").strip()
        or _DEFAULT_FEISHU_APP_ID
    )
    app_secret = (
        os.environ.get("FEISHU_APP_SECRET", "").strip()
        or os.environ.get("LARK_APP_SECRET", "").strip()
        or _DEFAULT_FEISHU_APP_SECRET
    )
    import requests as _rq
    r = _rq.post(
        f"{_FEISHU_OPEN_API}/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=_REQ_TIMEOUT,
    )
    if r.status_code != 200:
        raise RuntimeError(f"获取 tenant_access_token HTTP {r.status_code}: {r.text[:2000]}")
    data = r.json()
    token = data.get("tenant_access_token")
    if not token:
        raise RuntimeError(f"响应中无 tenant_access_token: {data}")
    return str(token)


def _feishu_resolve_spreadsheet_token(url_or_token: str) -> str:
    """从 wiki/sheets URL 或直接 token 解析出 spreadsheet_token。"""
    import re
    import requests as _rq

    url_or_token = url_or_token.strip()
    # wiki 链接
    m_wiki = re.search(r"/wiki/([A-Za-z0-9]+)", url_or_token)
    if m_wiki:
        wiki_token = m_wiki.group(1)
        tenant = _feishu_tenant_token()
        resp = _rq.get(
            f"{_FEISHU_OPEN_API}/wiki/v2/spaces/get_node",
            headers={"Authorization": f"Bearer {tenant}"},
            params={"token": wiki_token},
            timeout=_REQ_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_wiki_node HTTP {resp.status_code}: {resp.text[:2000]}")
        j = resp.json()
        if j.get("code") not in (None, 0):
            raise RuntimeError(f"get_wiki_node 业务错误: {j}")
        node = (j.get("data") or {}).get("node") or {}
        t = node.get("obj_token") or node.get("node_token") or node.get("token")
        if not t:
            raise RuntimeError(f"Wiki node 无 obj_token: {j}")
        return str(t)

    # sheets 链接
    m_sheet = re.search(r"/sheets/([A-Za-z0-9]+)", url_or_token)
    if m_sheet:
        return m_sheet.group(1)

    # 直接 token
    return url_or_token


def _feishu_sheet_id(spreadsheet_token: str, headers: dict, sheet_index=0, sheet_title=None):
    """获取第一个可见 sheet 的 sheet_id。"""
    import requests as _rq
    url = f"{_FEISHU_OPEN_API}/sheets/v3/spreadsheets/{spreadsheet_token}/sheets/query"
    r = _rq.get(url, headers=headers, timeout=_REQ_TIMEOUT)
    sheets = []
    if r.status_code == 200:
        j = r.json()
        if j.get("code") in (None, 0):
            sheets = (j.get("data") or {}).get("sheets") or []
    if not sheets:
        url2 = f"{_FEISHU_OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/metainfo"
        r2 = _rq.get(url2, headers=headers, timeout=_REQ_TIMEOUT)
        if r2.status_code != 200:
            raise RuntimeError(f"sheets v3/v2 均失败: {r2.status_code}")
        j2 = r2.json()
        sheets = (j2.get("data") or {}).get("sheets") or []
    visible = [
        s for s in sheets
        if not s.get("hidden") and s.get("resource_type", "sheet") == "sheet"
    ]
    if not visible:
        raise RuntimeError("未找到可见 sheet")
    if sheet_title:
        for s in visible:
            if s.get("title") == sheet_title:
                return str(s.get("sheet_id")), sheets
        raise RuntimeError(f"未找到标题为 {sheet_title!r} 的工作表")
    sid = visible[sheet_index].get("sheet_id")
    return str(sid), sheets


def _feishu_grid_row_count(sheets, sheet_id):
    for s in sheets:
        if str(s.get("sheet_id")) == str(sheet_id):
            gp = s.get("grid_properties") or {}
            rc = gp.get("row_count")
            if rc is not None:
                return max(int(rc), 2)
    return 20000


def _feishu_read_col_a(spreadsheet_token, sheet_id, headers, header_rows, max_row):
    """读 A 列已有 case_id → {case_id: 1-based行号}，以及最后非空行号。"""
    import requests as _rq
    start = header_rows + 1
    if max_row < start:
        return {}, header_rows
    rng = f"{sheet_id}!A{start}:A{max_row}"
    r = _rq.get(
        f"{_FEISHU_OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_get",
        headers=headers, params={"ranges": rng}, timeout=_REQ_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"values_batch_get A 失败: {data}")
    vrs = (data.get("data") or {}).get("valueRanges") or []
    if not vrs:
        return {}, header_rows
    values = vrs[0].get("values") or []
    row_by_case = {}
    last = header_rows
    for i, row in enumerate(values):
        row_1 = start + i
        if not row:
            continue
        cell = row[0]
        cid = _feishu_cell_text(cell).strip()
        if not cid:
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            if cid.lower().endswith(ext):
                cid = cid[:-len(ext)].strip()
                break
        last = max(last, row_1)
        if cid not in row_by_case:
            row_by_case[cid] = row_1
    return row_by_case, last


def _feishu_cell_text(cell):
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, dict):
        for k in ("text", "cellText", "stringValue"):
            v = cell.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return str(cell).strip()


def _feishu_cell_has_content(cell):
    if cell is None:
        return False
    if isinstance(cell, str):
        return bool(cell.strip())
    if isinstance(cell, dict):
        if cell.get("type") == "embed-image" and cell.get("fileToken"):
            return True
        return bool(_feishu_cell_text(cell))
    return True


def _feishu_batch_update(spreadsheet_token, headers, value_ranges, timeout=_IMAGE_UPLOAD_TIMEOUT,
                         max_retries=5, retry_base=2):
    import requests as _rq
    for i in range(0, len(value_ranges), _MAX_RANGES_PER_BATCH):
        chunk = value_ranges[i:i + _MAX_RANGES_PER_BATCH]
        for attempt in range(max_retries):
            r = _rq.post(
                f"{_FEISHU_OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_batch_update",
                headers=headers, json={"valueRanges": chunk}, timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            code = data.get("code", 0)
            if code == 0:
                break
            if code == 90217 and attempt < max_retries - 1:
                wait = retry_base * (attempt + 1)
                log.warning(f"  飞书限流(90217)，{wait}s 后重试 ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"values_batch_update 失败: {data}")


def _feishu_write_image(spreadsheet_token, headers, sheet_id, cell, image_path,
                        timeout=_IMAGE_UPLOAD_TIMEOUT, max_retries=5, retry_base=2):
    import requests as _rq
    with open(image_path, "rb") as f:
        raw = f.read()
    if not raw:
        return
    name = os.path.basename(image_path)
    if "." not in name:
        name += ".png"
    rng = f"{sheet_id}!{cell}:{cell}"
    body = {"range": rng, "image": list(raw), "name": name}
    for attempt in range(max_retries):
        r = _rq.post(
            f"{_FEISHU_OPEN_API}/sheets/v2/spreadsheets/{spreadsheet_token}/values_image",
            headers=headers, json=body, timeout=timeout,
        )
        data = r.json() if r.status_code == 200 else {}
        code = data.get("code", -1)
        if r.status_code == 200 and code == 0:
            return
        if code == 90217 and attempt < max_retries - 1:
            wait = retry_base * (attempt + 1)
            log.warning(f"  飞书限流(90217) {cell}，{wait}s 后重试 ({attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        raise RuntimeError(f"values_image 失败 {cell} {image_path}: HTTP {r.status_code} {data}")


def _resolve_img(directory, case_id):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = os.path.join(directory, case_id + ext)
        if os.path.isfile(p):
            return p
    return None


def step7_upload_to_feishu(
    eas_csv_path,
    data_dir,
    feishu_url,
    image_delay=0.05,
):
    """将 EAS 诊断结果 CSV + 图片上传至飞书电子表格。

    表格列: A=case_id  B=avm  C=crop  D=yuyan
            E=实体是否存在  F=超声标记命中或偏移  G=障碍物类型  H=微调模型预测（是否误检）

    追加模式：已存在的 case_id 跳过新增行。
    """
    banner(7, "上传诊断结果到飞书电子表格")

    import csv
    import time

    if not eas_csv_path or not os.path.isfile(eas_csv_path):
        log.error(f"  ❌ CSV 文件不存在: {eas_csv_path}")
        return
    if not feishu_url:
        log.warning("  ⚠️  未指定飞书表格 URL (--feishu-sheet-url)，跳过 Step 7")
        return

    # 读 CSV
    rows_by_cid = {}
    with open(eas_csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("case_id") or "").strip()
            if not cid:
                continue
            rows_by_cid[cid] = {
                "entity": (row.get("实体是否存在") or "").strip(),
                "geometry": (row.get("超声标记命中或偏移") or "").strip(),
                "object_type": (row.get("障碍物类型") or "").strip(),
                "misdetect": (row.get("是否误检") or "").strip(),
            }
    if not rows_by_cid:
        log.warning("  CSV 中无有效数据，跳过")
        return
    log.info(f"  CSV: {eas_csv_path}，共 {len(rows_by_cid)} 条")

    images_dir = os.path.join(data_dir, "images")
    crop_dir = os.path.join(data_dir, "crop")
    yuyan_dir = os.path.join(data_dir, "yuyan")

    # 飞书认证 & 解析 spreadsheet
    spreadsheet_token = _feishu_resolve_spreadsheet_token(feishu_url)
    tenant = _feishu_tenant_token()
    headers = {
        "Authorization": f"Bearer {tenant}",
        "Content-Type": "application/json; charset=utf-8",
    }
    sheet_id, sheets = _feishu_sheet_id(spreadsheet_token, headers)
    log.info(f"  spreadsheet_token={spreadsheet_token}, sheet_id={sheet_id}")

    max_scan = _feishu_grid_row_count(sheets, sheet_id)
    header_rows = 1
    start_row = 2

    existing, last_content = _feishu_read_col_a(
        spreadsheet_token, sheet_id, headers, header_rows, max_scan,
    )
    log.info(f"  飞书已有 {len(existing)} 行 case_id，最后非空行={last_content}")

    # 分配行：已有 → 跳过；新增 → 追加到行尾
    case_ids = list(rows_by_cid.keys())
    next_new = max(last_content + 1, start_row)
    assignments = []
    skipped_existing = 0
    for cid in case_ids:
        if cid in existing:
            skipped_existing += 1
            continue
        assignments.append((cid, next_new))
        existing[cid] = next_new
        next_new += 1

    log.info(f"  新增: {len(assignments)}，跳过(已存在): {skipped_existing}")
    if not assignments:
        log.info("  无新增行，Step 7 结束")
        return

    # 写 A 列 case_id
    a_vrs = [
        {"range": f"{sheet_id}!A{row}:A{row}", "values": [[cid]]}
        for cid, row in assignments
    ]
    _feishu_batch_update(spreadsheet_token, headers, a_vrs)
    log.info(f"  写入 A 列: {len(a_vrs)} 个新行")

    # 写 E/F/G/H 标签列
    label_col_map = [("E", "entity"), ("F", "geometry"), ("G", "object_type"), ("H", "misdetect")]
    label_vrs = []
    for cid, row in assignments:
        info = rows_by_cid[cid]
        for col, key in label_col_map:
            label_vrs.append({
                "range": f"{sheet_id}!{col}{row}:{col}{row}",
                "values": [[info.get(key, "")]],
            })
    if label_vrs:
        _feishu_batch_update(spreadsheet_token, headers, label_vrs)
        log.info(f"  写入 E/F/G/H 标签: {len(label_vrs)} 格")

    # 写 B/C/D 图片列
    img_cols = [
        ("B", images_dir, "avm"),
        ("C", crop_dir, "crop"),
        ("D", yuyan_dir, "yuyan"),
    ]
    wrote = missed = 0
    for idx, (cid, row) in enumerate(assignments):
        for col_letter, directory, label in img_cols:
            img_path = _resolve_img(directory, cid)
            if not img_path:
                missed += 1
                continue
            cell = f"{col_letter}{row}"
            try:
                _feishu_write_image(spreadsheet_token, headers, sheet_id, cell, img_path)
                wrote += 1
            except RuntimeError as e:
                log.warning(f"  写图失败 {cell} {img_path}: {e}")
                missed += 1
            if image_delay > 0:
                time.sleep(image_delay)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(assignments):
            log.info(f"  图片进度: {idx + 1}/{len(assignments)}")

    log.info(f"  图片写入完成: 新写={wrote}, 缺失/失败={missed}")
    log.info(f"  ✅ Step 7 飞书上传完成")


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
                        help="飞书项目 Key（默认 iffcom）")
    parser.add_argument("-v", "--view-id", default="U9zPLpFvR",
                        help="飞书视图 ID（默认 当天缺陷数据 `U9zPLpFvR`）")
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
                        help="跳过指定步骤编号 (1/3/4/5/6/7)，如 --skip-steps 1 3")
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
                        help="Step6 记录 VLM 原始回复到 diagnosis_logs/MMDD/debug_thinking_*.txt")
    parser.add_argument("--log-dir", default=config.LOG_DIR,
                        help=f"日志输出目录 (默认: {config.LOG_DIR})")
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
    # ── Step6 诊断分支：默认 EAS 微调模型；--openai-diagnose 时走 VLM 大模型 ──
    parser.add_argument(
        "--openai-diagnose",
        action="store_true",
        help=(
            "Step6 改用 VLM 大模型 API 诊断（替代默认的 EAS 微调模型）。"
            "此模式下不执行 Step7 飞书表格上传。"
        ),
    )
    parser.add_argument(
        "--no-comment",
        action="store_true",
        help="Step6 EAS 诊断后不发送飞书工单评论",
    )
    parser.add_argument(
        "--eas-data-dir",
        default=config.PIPELINE_DATA_DIR,
        help=f"EAS 分支的图片数据目录（images/crop/yuyan），默认 {config.PIPELINE_DATA_DIR}",
    )
    parser.add_argument(
        "--eas-base",
        default="",
        help="EAS 服务根 URL（默认取 config.EAS_BASE_URL）",
    )
    parser.add_argument(
        "--eas-token",
        default="",
        help="EAS Authorization Token（默认取 config.EAS_TOKEN）",
    )
    parser.add_argument(
        "--eas-timeout",
        type=int,
        default=config.EAS_TIMEOUT,
        help=f"EAS 单次请求超时秒数（默认 {config.EAS_TIMEOUT}）",
    )
    parser.add_argument(
        "--eas-output-dir",
        default=config.LOG_DIR,
        help="EAS 诊断输出目录（jsonl + labels.csv，默认 diagnosis_logs/MMDD/）",
    )
    parser.add_argument(
        "--feishu-sheet-url",
        default="https://rqk9rsooi4.feishu.cn/wiki/PnvnwxXdrie48Mkmxalcm6uLnCf",
        help=(
            "Step7: 诊断结果上传到飞书电子表格（支持 /wiki/ 或 /sheets/ 链接）。"
            "默认上传；--openai-diagnose 或 --skip-steps 7 时不执行"
        ),
    )
    parser.add_argument(
        "--image-delay",
        type=float,
        default=0.05,
        help="Step7 两次图片写入间隔秒数（默认 0.05）",
    )
    args = parser.parse_args()

    if args.list_models:
        from vlm.VLM_API import _use_vertex_api
        if _use_vertex_api():
            print("Vertex 模式（generateContent），配置模型：")
            print(f"  1. {getattr(config, 'VLM_MODEL', 'gemini-3.1-pro-preview')}")
            print(f"  Base: {config.VLM_BASE_URL}")
            alt = getattr(config, "VLM_BASE_URL_ALT", "")
            if alt:
                print(f"  Alt:  {alt}")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=config.VLM_API_KEY, base_url=config.VLM_BASE_URL)
            models = client.models.list()
            print("可用模型列表：")
            for i, m in enumerate(models.data, 1):
                print(f"  {i}. {m.id}")
        sys.exit(0)

    _, log_file = setup_logging(args.log_dir)
    # 供 Step3 子进程内 BagReader 追加写入同一 pipeline 日志（print 不会进 FileHandler）
    os.environ["PIPELINE_LOG_FILE"] = os.path.abspath(log_file)

    id_mapping_path = args.id_mapping

    skip = set(args.skip_steps)

    # 全流程运行时清理中间产物，确保全新运行；跳步时保留已有数据
    if not skip:
        for d in [args.samples_dir, args.read_data_dir, args.generate_dir,
                  config.DRAW_IMAGE_DIR, config.RESULT_DIR]:
            if os.path.isdir(d):
                shutil.rmtree(d)
                log.info(f"  已清理: {d}")

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

    # Step 3（原 Step 2「获取 bag 列表」已移除，Step 3 内部自行通过 meta_data 获取 bag）
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
        if 4 not in skip:
            tag_bag_prefixes, _ = _collect_event_bag_prefixes(tag_ids, args.read_data_dir)
            if not tag_bag_prefixes:
                log.info("  未从 read_data 找到 event_heavy_bags 缓存，Step 4 将全量扫描 samples/")

    # 与 bag_reader 一致：Light bag CarInfo（CAR_STATE_TOPIC）在超声事件时刻若判后视镜折叠 → 不跑 Step4–6。
    # 仅使用 read_data 内已写入的后视镜折叠缓存；无缓存的 tag 不再远端补读 Light bag。
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
        if missing_cache_tags:
            log.info(
                f"  无后视镜折叠缓存文件的 tag 共 {len(missing_cache_tags)} 个，"
                f"已跳过远端补读（不再为这些 tag 读 Light bag）；"
                f"Step4–6 仅使用 read_data 内已有缓存做折叠剔除，无缓存 tag 不据此剔除"
            )
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
    eas_csv_path = None
    if 6 not in skip:
        if args.openai_diagnose:
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
            with open(id_mapping_path, "r", encoding="utf-8") as f:
                _id_map = json.load(f)
            eas_csv_path = step6_eas_diagnose(
                tag_ids,
                args.eas_data_dir,
                args.eas_output_dir,
                eas_base=args.eas_base,
                eas_token=args.eas_token,
                timeout=args.eas_timeout,
                resume=True,
                draw_image_dir=config.DRAW_IMAGE_DIR,
                read_data_dir=args.read_data_dir,
                ignore_fs_types=args.ignore_fs_types,
                id_mapping=None if args.no_comment else _id_map,
                project_key="" if args.no_comment else args.project_key,
            )
    else:
        log.info(f"[跳过 Step 6]")

    # Step 7: 默认 EAS 分支诊断后上传飞书；--openai-diagnose 不执行
    if 7 not in skip and not args.openai_diagnose and eas_csv_path:
        step7_upload_to_feishu(
            eas_csv_path,
            args.eas_data_dir,
            args.feishu_sheet_url,
            image_delay=args.image_delay,
        )
    elif 7 in skip:
        log.info(f"[跳过 Step 7]")
    elif args.openai_diagnose:
        log.info(f"[跳过 Step 7]（--openai-diagnose 模式不上传飞书表格）")

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
    #     "--skip-steps", "1","6"
    # ]
    main()
