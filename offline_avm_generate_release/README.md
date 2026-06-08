离线 AVM 生成工具 - 使用说明
==============================

一、运行方式（任选其一）

  1) 使用脚本（推荐，会自动处理 OpenCV 库）
     ./run_standalone.sh -i <输入根目录> -b <bag名> -o <输出根目录>

  2) 直接运行二进制
     ./offline_avm_generate -i <输入根目录> -b <bag名> -o <输出根目录>

二、常用参数

  -i, --input    输入根目录（必须）
  -b, --bag-name  bag 名称，如 YR-C01-35_20260120_062850.Heavy_Topic_Group.bag（必须）
  -o, --output   输出根目录，结果在 输出目录/bag名/ 下（必须）
  -w, --width    BEV 图像宽度，默认 8
  --height       BEV 图像高度，默认 12
  --interval     采样间隔，默认 10
  -v, --verbose  详细输出

三、输入目录结构要求

  输入根目录/
    config/YYYYMM/bag名/data_index.csv
    config/YYYYMM/bag名/ground.cfg
    config/YYYYMM/bag名/cameras.cfg
    panoramic_1/YYYYMM/bag名/...
    panoramic_2/...
    panoramic_3/...
    panoramic_4/...

  YYYYMM 从 bag 名中的日期解析（如 20260120 -> 202601）。

四、环境说明

  - 本包已包含全部运行依赖（runfiles + lib/），对方机器无需安装任何 C++/OpenCV/CUDA 库。
  - 仅需 x86_64 Linux 与 glibc（系统自带），解压后直接运行 run_standalone.sh 即可。

五、示例

  ./run_standalone.sh -i /data/input -b YR-C01-35_20260120_062850.Heavy_Topic_Group.bag -o /data/output
