import sys
import os
import re
from typing import Dict, List, Union
import json
sys.path.append("/mnt/pubic-data/shared/public/trajcaching_v3/debs/proto")
sys.path.append("/mnt/pubic-data/shared/public/trajcaching_v3/debs/scenariohouse")
# Add proto directory to Python path
# sys.path.append('/opt/deeproute/common/common-protocol/include/proto/')
sys.path.insert(0, "/mnt/pubic-data/shared/public/trajcaching_v3/debs/proto")
sys.path.insert(0, "/mnt/pubic-data/shared/public/trajcaching_v3/debs/scenariohouse")

# DrFile相关导入
try:
    from drfile.drfile_client import DrFileClient, ClientConfiguration
    from drfile.modules.fileGalaxy.model.request.drfile_request import (
        CopyFileRequest,
        DoesFileExistRequest,
    )
    from drfile.modules.sdk.model.request.file_transfer_request import GetFileRequest

    from dplib.env import EnvConfig
    from dplib import DpEngine

    DRFILE_AVAILABLE = True
except ImportError:
    print("警告: drfile模块未安装，meta.json加载功能将不可用")
    DRFILE_AVAILABLE = False

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ============ DrFile配置 ============
# 用于访问DrFile系统获取bag meta信息
PERCEPTIONTEAM_USERNAME = "perceptionteam"
PERCEPTIONTEAM_PASSWORD = "r6zR86V4*+=*"
DR_ENDPOINT = os.getenv("DR_ENDPOINT", "https://drplatform-backend.deeproute.cn")

# 全局DrFileClient实例（单例模式）
_dr_client_instance = None
_dr_engine_instance = None

env_config = EnvConfig()
env_config.retry_times = 5
env_config.dplib_retry_sleep_time = 20

def get_dr_client():
    """获取DrFileClient单例实例"""
    global _dr_client_instance
    if not DRFILE_AVAILABLE:
        return None

    if _dr_client_instance is None:
        if not PERCEPTIONTEAM_USERNAME or not PERCEPTIONTEAM_PASSWORD:
            print("警告: DrFile配置不完整，无法创建客户端")
            return None

        try:
            _dr_client_instance = DrFileClient(
                ClientConfiguration(
                    username=PERCEPTIONTEAM_USERNAME,
                    password=PERCEPTIONTEAM_PASSWORD,
                    endpoint=DR_ENDPOINT,
                    token_expired_time=-1,
                    show_config=False,
                    show_progress=False,
                    show_summary=False,
                )
            )
            print("DrFileClient初始化成功")
        except Exception as e:
            print(f"DrFileClient初始化失败: {e}")
            return None

    return _dr_client_instance

def get_dr_engine_instance():
    global _dr_engine_instance
    if _dr_engine_instance is None:
        _dr_engine_instance = DpEngine(
            username=PERCEPTIONTEAM_USERNAME,
            password=PERCEPTIONTEAM_PASSWORD,
            env_config=env_config,
        )
    return _dr_engine_instance


def parse_protobuf_data(data: bytes) -> Dict[str, float]:
    """
    解析protobuf格式的数据，提取ground_in_sensing的参数

    Args:
        data: 字节格式的protobuf数据

    Returns:
        ground_in_sensing中的参数字典
    """
    # 解码字节数据
    text = data.decode('utf-8')

    # 查找 ground_in_sensing 块
    pattern = r'ground_in_sensing\s*{([^}]+)}'
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return {}

    content = match.group(1)

    # 提取参数
    result = {}
    param_pattern = r'(\w+)\s*:\s*([\d.-]+)'
    param_matches = re.findall(param_pattern, content)

    for param_name, param_value in param_matches:
        result[param_name] = float(param_value)

    return result

def get_ground(trip_id):
    dr_engine = get_dr_engine_instance()
    trip_meta = dr_engine.get_trip_meta(trip_id=int(trip_id))
    trip_name = trip_meta.trip_name
    dr_client: DrFileClient = get_dr_client()
    ground = dr_client.download_bytes(
        GetFileRequest(
            namespace='trip',
            path=f"/{trip_name}/configs/ground.cfg",
        )
    )
    ground = parse_protobuf_data(ground)
    return ground

if __name__ == '__main__':
    trip_id = 80617951
    ground = get_ground(trip_id=trip_id)
    print(ground)
    # 保存
    output_path = "ground.json"
    output_path = os.path.join(os.path.dirname(__file__), output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground, f, indent=2)



