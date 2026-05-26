#!/usr/bin/env python3
"""本地验证 proj/run.py 的 run 列表。

用法::
    python proj/test_user_command.py
    python proj/test_user_command.py --data-root /mnt/public-data/user/ziroujiang/train_data_v2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest.mock as mock

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "proj_run", os.path.join(_script_dir, "run.py")
)
_run = importlib.util.module_from_spec(_spec)
with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="")):
    _spec.loader.exec_module(_run)  # type: ignore[union-attr]

run = _run.run
job_name = _run.job_name
_remote_repo = _run._remote_repo
DATA_ROOT_DEFAULT = _run.DATA_ROOT
remote_code_tar_file = _run.remote_code_tar_file
output_dir = _run.output_dir
code_tar_file = _run.code_tar_file


def _join_run(cmds: list) -> str:
    return " && ".join(str(c) for c in cmds)


def _rewrite_for_sandbox(cmd: str, sandbox: str, data_root: str) -> str:
    ws = os.path.join(sandbox, "workspace")
    csi = os.path.join(sandbox, "csi-data-aly", "user", "ziroujiang")
    out = cmd.replace("/workspace", ws).replace(
        "/mnt/csi-data-aly/user/ziroujiang", csi
    )
    repo = _remote_repo
    datasets = os.path.join(csi, "datasets", "train_data_v2")
    model_out = os.path.join(csi, "model", job_name)
    start_sh = f"/bin/bash {ws}/{repo}/start_pai.sh"
    out = out.replace(
        start_sh,
        f"SKIP_TRAIN=1 DATA_ROOT={datasets} OUTPUT_DIR={model_out} {start_sh}",
    )
    return out


def _setup_sandbox(sandbox: str, data_root: str) -> None:
    ws = os.path.join(sandbox, "workspace")
    csi_user = os.path.join(sandbox, "csi-data-aly", "user", "ziroujiang")
    code_tar_dst = os.path.join(
        csi_user, "code_tar", os.path.basename(remote_code_tar_file)
    )
    model_dir = os.path.join(csi_user, "model", job_name)
    os.makedirs(os.path.dirname(code_tar_dst), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    _run.write_start_pai_sh()
    with tarfile.open(code_tar_file, "w:gz") as tar:
        tar.add(_script_dir, arcname=_remote_repo, filter=_run.exclude_function)
    shutil.copy2(code_tar_file, code_tar_dst)

    datasets_dst = os.path.join(csi_user, "datasets", "train_data_v2")
    os.makedirs(os.path.dirname(datasets_dst), exist_ok=True)
    if not os.path.exists(datasets_dst):
        os.symlink(os.path.abspath(data_root), datasets_dst)
    os.makedirs(ws, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="本地测试 run 列表")
    parser.add_argument("--data-root", default="")
    args = parser.parse_args()

    data_root = (args.data_root or "").strip() or DATA_ROOT_DEFAULT
    if not os.path.isfile(os.path.join(data_root, "dataset.jsonl")):
        alt = "/mnt/public-data/user/ziroujiang/train_data_v2"
        if os.path.isfile(os.path.join(alt, "dataset.jsonl")):
            print(f"[INFO] CPFS 不可用，改用: {alt}")
            data_root = alt
        else:
            print(f"[ERROR] 找不到 dataset.jsonl", file=sys.stderr)
            return 1

    pai_cmd = _join_run(run)
    print("=" * 60)
    print(f"run 列表共 {len(run)} 条（dr tjob 拼接后预览）")
    print("=" * 60)
    for i, c in enumerate(run):
        print(f"  [{i}] {c}")
    print()
    print("拼接后:")
    print(pai_cmd)
    print()

    if pai_cmd.lstrip().startswith("&&"):
        print("FAIL: 拼接后以 && 开头")
        return 1

    print("[1/3] bash -n 拼接命令 ...")
    r = subprocess.run(["bash", "-n", "-c", pai_cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL:", r.stderr)
        return 1
    print("OK")

    print("[2/3] 沙箱模拟（SKIP_TRAIN=1）...")
    with tempfile.TemporaryDirectory(prefix="pai_cmd_test_") as sandbox:
        _setup_sandbox(sandbox, data_root)
        test_cmd = _rewrite_for_sandbox(pai_cmd, sandbox, data_root)
        r = subprocess.run(["bash", "-euxo", "pipefail", "-c", test_cmd], text=True)
        if r.returncode != 0:
            return r.returncode

    print("[3/3] start_pai.sh / train.sh 语法检查 ...")
    for sh in ("start_pai.sh", "train.sh"):
        p = os.path.join(_script_dir, sh)
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {sh}:", r.stderr)
            return 1
    print("OK")
    print("\n全部通过。提交: dr tjob submit -p pai proj/run.py -t AVP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
