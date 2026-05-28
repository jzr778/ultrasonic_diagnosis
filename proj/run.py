
import os
import tarfile
import subprocess

# 按需改 job_name
project_name = "ultrasonic-diagnosis"
job_name = "pai-diagnosis-qwen35-27b-v3"

cfg_file = "train.sh"
DATA_ROOT = "/mnt/csi-data-aly/user/ziroujiang/datasets/all_data_v3"
work_dir = f"/mnt/csi-data-aly/user/ziroujiang/model/{job_name}"
deepkeer_project = project_name
deepkeer_name = job_name

# 复现实验排障开关：True 时注入更详细的分布式/算子调试环境变量
enable_debug_env = False

_script_dir = os.path.dirname(os.path.abspath(__file__))
# 本地打包目录（本机通常无 CPFS 挂载）；上传后 PAI 从 remote_code_tar_file 读取
_code_tar_dir = os.path.join(_script_dir, "code_tar")
os.makedirs(_code_tar_dir, exist_ok=True)
code_tar_file = os.path.join(_code_tar_dir, job_name + ".tar.gz")
print('code_tar_file : ', code_tar_file)

port = "22"
remote_ip = "10.250.130.131"
output_dir = f"/mnt/csi-data-aly/user/ziroujiang/model/{job_name}"
remote_code_tar_dir = "/mnt/csi-data-aly/user/ziroujiang/code_tar/"
remote_code_tar_file = remote_code_tar_dir + job_name + ".tar.gz"

# 打包时排除大目录与无关文件（与 rsync 排除 outputs 一致）
_EXCLUDE_PATH_SEGMENTS = frozenset(
    {
        "outputs",
        "output",
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
        "code_tar",
        "infer_results",
    }
)


def _should_exclude_tar_member(arcname):
    """arcname 形如 proj/..."""
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

mkdir_remote = (
    f"ssh -p {port} root@{remote_ip} "
    f"'mkdir -p {remote_code_tar_dir} {output_dir}'"
)
print("Prepare remote dirs:", mkdir_remote)
subprocess.run(mkdir_remote, shell=True, check=True)

upload_commands = f"scp -P {port} {code_tar_file} root@{remote_ip}:{remote_code_tar_file}"
print("Upload:", upload_commands)
result = subprocess.run(upload_commands, shell=True, text=True, capture_output=True)
print(result.stdout)
if result.returncode != 0:
    print("stderr:", result.stderr)
    raise SystemExit(f"scp failed (exit {result.returncode})")
print("Upload OK:", remote_code_tar_file)

# 若训练依赖公共数据软链，在此追加命令（路径请按集群实际修改）
ln_commands = [
    # f"mkdir -p /mnt/public-data/shared/public/trajcaching_v3/",
    # f"sudo ln -s /mnt/csi-data-aly/shared/public/trajcaching_v3/debs /mnt/public-data/shared/public/trajcaching_v3/debs",
]

# 解压后顶层目录名 = 本地打包时 cwd 的 basename（在 proj/ 目录下执行本脚本则为 proj）
_remote_repo = "proj"

ln_commands = [
    # f"mkdir -p /mnt/pubic-data/shared/public/trajcaching_v3/",
    # f"sudo ln -s /mnt/csi-data-aly/shared/public/trajcaching_v3/debs /mnt/pubic-data/shared/public/trajcaching_v3/debs",
]

run = [
    # "sleep infinity"
    # *ln_commands,
    f"pip install ms-swift==4.2.0 -i https://mirrors.aliyun.com/pypi/simple/",
    f"pip install qwen-vl-utils==0.0.14",
    f"cp {remote_code_tar_file} /workspace",
    f"cd /workspace",
    f"rm -rf {_remote_repo}",
    f"tar zxvf {job_name}.tar.gz",
    f"cd /workspace/{_remote_repo}/",
    f"export TRAIN_OUTPUT_DIR={output_dir}",
    f"chmod +x ./train.sh",
    f"bash -c ./train.sh",
]


priority = 6
num_machine = 2
num_gpu = 16
name = job_name
gpu_type = "PPU"
platform = "pai"

image = "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/training-service/vllm:qwen3.5-xpu2.0.0-accl-fixreasoning"
