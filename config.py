"""
项目统一配置。

敏感信息（密码、API Key 等）从 .env 文件或环境变量读取；
非敏感配置（路径、URL、topic 名等）直接定义为常量。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ── 简易 .env 加载器（不依赖 python-dotenv） ──

def _load_env_file(env_path):
    """读取 .env 文件，已存在的环境变量不会被覆盖。"""
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_env_file(PROJECT_ROOT / ".env")


# ============ DR 平台 / DPBAG 凭证 ============

DR_USERNAME = os.environ.get("DR_USERNAME", "")
DR_PASSWORD = os.environ.get("DR_PASSWORD", "")
DR_ENDPOINT = os.environ.get("DR_ENDPOINT", "https://drplatform-backend.deeproute.cn")
DR_TAG_QUERY_URL = f"{DR_ENDPOINT}/scene/tag/instance/query"

os.environ.setdefault("DPBAG_DP_USERNAME", DR_USERNAME)
os.environ.setdefault("DPBAG_DP_PASSWORD", DR_PASSWORD)

# ============ VLM API ============

VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
# openai：OpenAI 兼容 chat/completions；vertex：七牛 bypass Vertex generateContent
VLM_API_STYLE = os.environ.get("VLM_API_STYLE", "openai").strip().lower()
VLM_BASE_URL = os.environ.get(
    "VLM_BASE_URL",
    "https://api.qnaigc.com/bypass/vertex/v1",
)
# 备用 Base（主地址失败时可重试）；如 https://openai.qiniu.com/bypass/vertex/v1
VLM_BASE_URL_ALT = os.environ.get("VLM_BASE_URL_ALT", "").strip()
# vertex 模式下默认模型；pipeline --model auto 时使用
VLM_MODEL = os.environ.get("VLM_MODEL", "gemini-3.1-pro-preview")

# ============ 飞书 ============

FEISHU_PLUGIN_ID = os.environ.get("FEISHU_PLUGIN_ID", "")
FEISHU_PLUGIN_SECRET = os.environ.get("FEISHU_PLUGIN_SECRET", "")
FEISHU_USER_KEY = os.environ.get("FEISHU_USER_KEY", "")
FEISHU_ENDPOINT = os.environ.get("FEISHU_ENDPOINT", "https://project.feishu.cn/open_api")
FEISHU_PROJECT_KEY = os.environ.get("FEISHU_PROJECT_KEY", "iffcom")

# ============ 数据路径 ============

DATA_BASE = os.environ.get("AVP_DATA_BASE", "/mnt/public-data/user/ziroujiang/avp")

SAMPLES_DIR = os.path.join(DATA_BASE, "samples") # 鱼眼
GENERATE_DIR = os.path.join(DATA_BASE, "generate") # AVM
READ_DATA_DIR = os.path.join(DATA_BASE, "read_data") # 定位信息
DRAW_IMAGE_DIR = os.path.join(DATA_BASE, "draw_image") # 根据AVM和定位绘制图片
RESULT_DIR = os.path.join(DATA_BASE, "result_avm") # 大模型诊断结果

PROTO_DEBS_DIR = os.environ.get(
    "PROTO_DEBS_DIR",
    "/mnt/public-data/shared/public/trajcaching_v3/debs",
)
PROTO_LOCAL_DIR = str(PROJECT_ROOT / "get_data" / "proto")

# ============ Bag Topic ============

CHAOSHENG_TOPIC = "/planner/stop_objects"
CAMERA_TOPICS = [
    "/sensors/camera/panoramic_1_raw_data/compressed_proto",
    "/sensors/camera/panoramic_2_raw_data/compressed_proto",
    "/sensors/camera/panoramic_3_raw_data/compressed_proto",
    "/sensors/camera/panoramic_4_raw_data/compressed_proto",
]
CAMERA_NAMES = ["panoramic_1", "panoramic_2", "panoramic_3", "panoramic_4"]
OBSTACLE_TOPIC = "/perception/objects"
POSE_TOPIC = "/localization/pose"
PLANNING_TOPIC = "/planner/trajectory"
# Light bag 内 CAN 车身 CarInfo（proto 仍为 CarInfo）；misc.rear_view_mirror 为 bool：False=折叠，True=展开
CAR_STATE_TOPIC = "/canbus/car_info"
