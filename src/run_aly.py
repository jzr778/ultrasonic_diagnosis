
import os
import tarfile
import subprocess

# 与本地 run.py 中 freespace3d 实验对齐，按需改 job_name / work_dir
project_name = "freespace3d"
job_name = "avp-freespace3d-c01-temporal-bev-fusion-0518-baseline"

cfg_file = "configs/configs_3dfreespace/task.py"
work_dir = f"./outputs/{job_name}"
deepkeer_project = project_name
deepkeer_name = job_name

# 复现实验排障开关：True 时注入更详细的分布式/算子调试环境变量
enable_debug_env = True

code_tar_file = "/mnt/yrfs/yujianguo/code_tar/" + job_name + ".tar.gz"
print('code_tar_file : ', code_tar_file)

port = "22"
remote_ip = "10.250.128.109"
output_dir = "/mnt/csi-data-aly/user/yujianguo/model/"
remote_code_tar_dir = "/mnt/csi-data-aly/user/yujianguo/code_tar/"
remote_code_tar_file = remote_code_tar_dir + job_name + ".tar.gz"

# 打包时排除大目录与无关文件（与 rsync 排除 outputs 一致）
_EXCLUDE_PATH_SEGMENTS = frozenset(
    {
        "outputs",
        "temp",
        "vision.egg-info",
        "unit_tests",
        "build",
        ".bin",
        "__pycache__",
        ".git",
        ".eggs",
        "trace_model",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        "htmlcov",
        ".ipynb_checkpoints",
        "wandb",
        ".cursor",
        "node_modules",
        ".tox",
        "venv",
        ".venv",
        "env",
        ".conda",
        "dist",
    }
)


def _should_exclude_tar_member(arcname):
    """arcname 形如 vision_train_framework/vision/..."""
    name = arcname.replace("\\", "/")
    parts = name.split("/")
    for seg in parts:
        if seg in _EXCLUDE_PATH_SEGMENTS:
            return True
        # 常见日志/缓存文件
        if seg.endswith(".log") and seg != "CHANGELOG":
            return True
    if name.endswith(".pyc"):
        return True
    if name.endswith(".pyo"):
        return True
    return False


def exclude_function(tarinfo):
    if _should_exclude_tar_member(tarinfo.name):
        return None
    return tarinfo


# 训练代码经常迭代，强制重新打包避免旧包污染
if os.path.exists(code_tar_file):
    os.remove(code_tar_file)
    print("Removed old tar, will re-pack with latest code")

print("Start to tar code !!!, please wait")
source_code_dir = os.getcwd()
with tarfile.open(code_tar_file, "w:gz", compresslevel=6) as tar:
    tar.add(source_code_dir, arcname=os.path.basename(source_code_dir), filter=exclude_function)

upload_commands = f"scp -r -P {port} {code_tar_file} root@{remote_ip}:{remote_code_tar_dir}"
result = subprocess.run(upload_commands, shell=True, text=True, capture_output=True)
print(upload_commands)
print(result)

# 若训练依赖公共数据软链，在此追加命令（路径请按集群实际修改）
ln_commands = [
    # f"mkdir -p /mnt/public-data/shared/public/trajcaching_v3/",
    # f"sudo ln -s /mnt/csi-data-aly/shared/public/trajcaching_v3/debs /mnt/public-data/shared/public/trajcaching_v3/debs",
]

# 解压后顶层目录名 = 本地打包时 cwd 的 basename（一般为 vision_train_framework）
_remote_repo = "vision_train_framework"

ln_commands = [
    f"mkdir -p /mnt/pubic-data/shared/public/trajcaching_v3/",
    f"sudo ln -s /mnt/csi-data-aly/shared/public/trajcaching_v3/debs /mnt/pubic-data/shared/public/trajcaching_v3/debs",
]

run = [
    *ln_commands,
    f"nvidia-smi",
    # f"export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
    f"cp {remote_code_tar_file} /workspace",
    f"cd /workspace",
    f"rm -rf {_remote_repo}",
    f"ls -l",
    f"tar zxvf {job_name}.tar.gz",
    f"if [ ! -d /workspace/{_remote_repo}/outputs ]; then ln -s {output_dir} /workspace/{_remote_repo}/outputs; fi",
    f"set +u",
    f"cd /workspace/{_remote_repo}/",
    f"python setup.py develop",
    f"apt install -y libturbojpeg || true",
    f"pwd",
    (
        "export TORCH_DISTRIBUTED_DEBUG=DETAIL; "
        "export CUDA_LAUNCH_BLOCKING=0; "
        "export NCCL_ASYNC_ERROR_HANDLING=1; "
        "export TORCH_SHOW_CPP_STACKTRACES=1; "
        "export CUDNN_DETERMINISTIC=0; "
        "export CUDNN_BENCHMARK=0; "
        "echo '[debug-env] enabled: TORCH_DISTRIBUTED_DEBUG=DETAIL, CUDA_LAUNCH_BLOCKING=0, "
        "NCCL_ASYNC_ERROR_HANDLING=1, TORCH_SHOW_CPP_STACKTRACES=1, CUDNN_DETERMINISTIC=0, "
        "CUDNN_BENCHMARK=0'"
    ) if enable_debug_env else "echo '[debug-env] disabled'",
    f"./tools/deeproute_dist_train.sh --cfg_file {cfg_file} "
    f"--work_dir={work_dir} "
    f"--deekeeper_project_name {deepkeer_project} "
    f"--deekeeper_experiment_name {deepkeer_name} ",
]


priority = 6
num_machine = 8
num_gpu = 16
name = job_name
gpu_type = "PPU"
platform = "pai"

image = "acr-yr-prod-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/ppu/vtf:vtf-v161-py38-cuda116-torch20-np123-u20-yjg-0106"