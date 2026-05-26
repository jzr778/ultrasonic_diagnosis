# 4 * 20GiB
# 数据目录：jsonl 里的图片是相对路径（如 images/xxx.jpg），需告诉 Swift 从哪张目录开始找
DATA_ROOT="${DATA_ROOT:-/mnt/csi-data-aly/user/ziroujiang/datasets/train_data_v2}"
# 由 run.py 注入，与 output_dir 一致
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/mnt/csi-data-aly/user/ziroujiang/model/pai-diagnosis-qwen35-27b-5193}"
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
ROOT_IMAGE_DIR="${DATA_ROOT}" \
NPROC_PER_NODE=16 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
swift sft \
    --model /mnt/csi-data-aly/user/ziroujiang/model/Qwen3.5-27B \
    --tuner_type lora \
    --dataset "${DATA_ROOT}/dataset.jsonl" \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --group_by_length true \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 16 \
    --dataloader_num_workers 16