# AVP PromptKit — 泊车超声误检自动诊断系统

基于视觉语言模型（可选供应商api或微调模型），对泊车环视场景中的超声障碍物检测结果进行自动化误检诊断。系统从飞书项目获取待诊断 case，自动解包 bag 数据、生成 AVM 全景图、绘制标注图像、调用模型诊断，最终将结果回写飞书。

## 快速开始

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DR 平台、VLM API、飞书等凭证

# 2. 运行主流程（默认：微调模型诊断 + 上传飞书）
python pipeline.py

# 2.1 可选：大模型 API 诊断（不上传飞书表格）
python pipeline.py --openai-diagnose
```

## Pipeline 主流程

`pipeline.py` 是项目核心，串联 7 个步骤完成端到端诊断：

```
飞书项目视图
    │
    ▼
┌─ Step 1 ─┐  获取 tag_id ↔ feishu_id 映射
│           │  → get_data/id_mapping.json
└───────────┘
    │
    ▼
┌─ Step 3 ─┐  解包 bag → samples（鱼眼帧）+ read_data（超声/定位/规划）
│           │  每 tag 独立 BagReader，多 tag 可并行
└───────────┘
    │
    ▼
┌─ Step 4 ─┐  鱼眼拼接 AVM 全景图
│           │  调用 offline_avm_generate_release 的 C++ 工具
└───────────┘
    │
    ▼
┌─ Step 5 ─┐  绘制 AVM 标注图像
│           │  在 BEV 上叠加超声高亮、相机障碍投影、车位等标注
└───────────┘
    │
    ▼
┌─ Step 6 ─┐  模型诊断（二选一）
│  默认     │  EAS 微调模型三分类 → 映射「是否误检」
│  --openai-│  VLM 大模型（Gemini / OpenAI 兼容）多任务诊断
│  diagnose │  输出 jsonl + CSV 到 diagnosis_logs/MMDD/
└───────────┘
    │
    ▼
┌─ Step 7 ─┐  诊断结果 + 图片上传飞书电子表格（默认开启，追加去重）
│           │  --openai-diagnose 或 --skip-steps 7 时跳过
└───────────┘
```

> Step 2 已移除（原「获取 bag 列表」功能由 Step 3 内部完成）。

### 常用参数

| 参数 | 说明 |
|------|------|
| `-p / --project-key` | 飞书项目 Key（默认 `iffcom`） |
| `-v / --view-id` | 飞书视图 ID（默认 当天缺陷数据 `U9zPLpFvR`） |
| `--skip-steps 1 3` | 跳过指定步骤 |
| `--openai-diagnose` | 改用 VLM 大模型 API 诊断（跳过 Step 7） |
| `--model gemini-3-pro-preview` | `--openai-diagnose` 时指定 VLM 模型 |
| `--feishu-sheet-url` | Step 7 上传目标飞书表格（默认已配置，支持 wiki / sheets 链接） |
| `--skip-steps 7` | 跳过飞书表格上传 |
| `--no-yuyan` | 关闭鱼眼解包与鱼眼辅助 |
| `--chaosheng-pixel-radius 40` | Step 5 超声-相机关联半径（默认 30） |
| `--unpack-workers 1` | Step 3 并行解包数（默认 min(CPU, 4)） |

## 数据目录

所有中间产物默认位于 `AVP_DATA_BASE`（`/mnt/public-data/user/ziroujiang/avp`）：

```
avp/
├── samples/            # Step3: 解包的鱼眼原始帧
├── read_data/          # Step3: 超声/定位/规划结构化数据
├── generate/           # Step4: AVM 全景图
├── draw_image/         # Step5: 标注后的 AVM 图像
├── result_avm/         # Step6: VLM 诊断结果
└── diagnosis_logs/     # 运行日志 + EAS 的 jsonl/csv
    └── MMDD/           #   按日期分目录
```

EAS 分支另有平铺图片目录（`pipeline_data/`），从 `draw_image/` 按 tag 收集 `images/crop/yuyan` 三图。

## 项目结构

```
avp_promptkit/
├── pipeline.py                  # 主流程入口
├── config.py                    # 统一配置（凭证、路径、Topic）
├── .env.example                 # 环境变量模板
│
├── get_data/                    # 数据获取层
│   ├── get_id_mapping.py        #   飞书视图 → tag_id/feishu_id 映射
│   ├── bag_reader.py            #   远端 bag 读取与解包
│   ├── unpack_bag_for_avm.py    #   解包 bag 到 samples/
│   ├── save_bag_data.py         #   提取超声/定位/规划 → read_data/
│   ├── get_meta_data.py         #   DR 平台 tag 元数据查询
│   └── ...                      #   车辆配置、相机参数、地面参数等
│
├── vlm/                         # VLM 诊断引擎
│   ├── avp_vlm_pipeline_avm.py  #   Step5 绘图 + Step6 VLM 诊断主逻辑
│   ├── VLM_API.py               #   VLM API 调用封装（OpenAI / Vertex）
│   ├── panoramic_projector.py   #   鱼眼→BEV 投影
│   └── point2box_mindistance_avm.py  # 超声-相机障碍最近距离匹配
│
├── prompts_engine/              # Prompt 构建引擎（LangChain + Jinja2）
│   ├── configs/                 #   Agent 配置
│   ├── templates/               #   Prompt 模板
│   ├── context/                 #   Context 构建器
│   └── engine/                  #   Agent 基类、工厂、任务管理
│
├── comment/                     # 飞书项目评论
│   ├── add_comment.py           #   发送诊断结果评论
│   ├── get_comment_id.py        #   查询评论 ID
│   └── remove_comment.py        #   删除评论
│
├── offline_avm_generate_release/  # AVM 拼接工具（C++ 二进制 + 启动脚本）
│
├── tool/                        # 辅助工具脚本
│   ├── eas_eval.py              #   EAS 微调模型评测
│   ├── build_labels.py          #   三分类 → 是否误检 映射规则
│   ├── collect_raw_data.py      #   draw_image → 平铺结构转换
│   ├── upload_predictions_to_feishu.py  # 预测结果上传飞书表格
│   ├── sync_raw_images_to_feishu_sheet.py  # 原始图片同步到飞书表格
│   ├── export_feishu_labels_csv.py  # 从飞书表格导出标签 CSV
│   ├── diagnose_val_dataset.py  #   验证集离线诊断
│   ├── diagnose_liuyi_benchmark.py  # 六一 benchmark 批量诊断
│   ├── crop_read_data_chaosheng.py  # 超声质心局部裁剪
│   └── ...                      #   其他数据处理与可视化工具
│
└── proj/                        # 模型训练相关
    ├── run.py                   #   训练启动脚本
    └── train.sh                 #   训练 shell 脚本
```

## 环境变量

参照 `.env.example` 配置，主要分为四组：

| 分组 | 变量 | 说明 |
|------|------|------|
| DR 平台 | `DR_USERNAME` / `DR_PASSWORD` | 远端 bag 数据访问凭证 |
| VLM API | `VLM_API_KEY` / `VLM_BASE_URL` / `VLM_MODEL` | 大模型 API 配置 |
| 飞书 | `FEISHU_PLUGIN_ID` / `FEISHU_PLUGIN_SECRET` / `FEISHU_USER_KEY` | 飞书项目 API 凭证 |
| 数据路径 | `AVP_DATA_BASE` | 中间产物根目录（默认 `/mnt/public-data/user/ziroujiang/avp`） |
