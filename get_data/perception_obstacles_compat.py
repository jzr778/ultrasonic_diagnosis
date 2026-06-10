"""
补全 ``deeproute_perception_obstacle_pb2`` 中缺失的 ``PerceptionObstacles``。

部分仓库内的生成代码被截断，未注册容器消息，但 bag 中 /perception/objects 等 topic 仍使用该类型。
在默认 DescriptorPool 中增加仅含 time_measurement + perception_obstacle 的最小 FileDescriptor，
并写回 ``perception.deeproute_perception_obstacle_pb2.PerceptionObstacles``，供 bag_reader 等使用。
"""

import importlib
import os
import sys

_PATCH_FILE = "perception/avp_promptkit_perception_obstacles.proto"
_FULL_NAME = "deeproute.perception.PerceptionObstacles"


def _proto_dir_on_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import config
    proto_dir = config.PROTO_LOCAL_DIR
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)


def ensure_perception_obstacles_class():
    """返回 ``PerceptionObstacles`` 消息类；首次调用时向全局描述符池注册。"""
    _proto_dir_on_path()
    mod = importlib.import_module("perception.deeproute_perception_obstacle_pb2")
    if getattr(mod, "PerceptionObstacles", None) is not None:
        return mod.PerceptionObstacles

    from google.protobuf import descriptor_pb2
    from google.protobuf import message_factory
    from google.protobuf.descriptor_pool import Default

    pool = Default()

    try:
        desc = pool.FindMessageTypeByName(_FULL_NAME)
    except KeyError:
        desc = None

    if desc is None:
        fp = descriptor_pb2.FileDescriptorProto()
        fp.name = _PATCH_FILE
        fp.package = "deeproute.perception"
        fp.dependency.append("perception/deeproute_perception_obstacle.proto")

        mt = fp.message_type.add()
        mt.name = "PerceptionObstacles"
        f1 = mt.field.add()
        f1.name = "time_measurement"
        f1.number = 1
        f1.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f1.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
        f2 = mt.field.add()
        f2.name = "perception_obstacle"
        f2.number = 2
        f2.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        f2.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        f2.type_name = ".deeproute.perception.PerceptionObstacle"
        pool.Add(fp)
        desc = pool.FindMessageTypeByName(_FULL_NAME)

    cls = message_factory.GetMessageClass(desc)
    setattr(mod, "PerceptionObstacles", cls)
    return cls


ensure_perception_obstacles_class()
