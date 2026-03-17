"""
DrFile / DpEngine 单例客户端。

所有需要访问远端 DrFile 的模块统一通过此模块获取客户端实例，
避免重复初始化和凭证硬编码。
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

try:
    from drfile.drfile_client import DrFileClient, ClientConfiguration
    from drfile.modules.sdk.model.request.file_transfer_request import GetFileRequest
    from dplib.env import EnvConfig
    from dplib import DpEngine
    DRFILE_AVAILABLE = True
except ImportError:
    print("警告: drfile 模块未安装，远端文件下载功能将不可用")
    DRFILE_AVAILABLE = False

_dr_client = None
_dr_engine = None


def _get_env_config():
    cfg = EnvConfig()
    cfg.retry_times = 2
    cfg.dplib_retry_sleep_time = 5
    return cfg


def get_dr_client():
    """获取 DrFileClient 单例。"""
    global _dr_client
    if not DRFILE_AVAILABLE:
        return None
    if _dr_client is None:
        _dr_client = DrFileClient(
            ClientConfiguration(
                username=config.DR_USERNAME,
                password=config.DR_PASSWORD,
                endpoint=config.DR_ENDPOINT,
                token_expired_time=-1,
                show_config=False,
                show_progress=False,
                show_summary=False,
            )
        )
    return _dr_client


def get_dr_engine():
    """获取 DpEngine 单例。"""
    global _dr_engine
    if not DRFILE_AVAILABLE:
        return None
    if _dr_engine is None:
        _dr_engine = DpEngine(
            username=config.DR_USERNAME,
            password=config.DR_PASSWORD,
            env_config=_get_env_config(),
        )
    return _dr_engine


def download_trip_config(trip_id, config_filename):
    """从 DrFile 下载指定 trip 的配置文件，返回原始字节。"""
    engine = get_dr_engine()
    trip_meta = engine.get_trip_meta(trip_id=int(trip_id))
    client = get_dr_client()
    return client.download_bytes(
        GetFileRequest(
            namespace='trip',
            path=f"/{trip_meta.trip_name}/configs/{config_filename}",
        )
    )
