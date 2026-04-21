"""Step3 子进程入口：与 pipeline 解耦，便于 multiprocessing spawn 安全 import。"""


def unpack_one_tag(payload):
    """单个 tag 解包 + save_data。返回 (tag_id, ok, err_msg)。"""
    tag_id, samples_dir, read_data_dir, extract_fisheye = payload
    from get_data.save_bag_data import save_data
    from get_data.unpack_bag_for_avm import unpack_tag

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
        return (tag_id, True, None)
    except Exception as e:
        return (tag_id, False, str(e))
