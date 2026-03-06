#!/bin/bash

# VideoMambaPro Finetuning Script for Micro-action (Experiment 6 - Weighted Loss + Fix Val + Import Fix)

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 项目根目录
PROJECT_ROOT="$SCRIPT_DIR/../../"

# 切换到 videamambapro 代码目录
cd "$PROJECT_ROOT/videomambapro"

# 预训练权重路径 - 使用 Tiny Checkpoint
PRETRAINED_WEIGHTS_PATH="./pretrained_models/Tiny_checkpoint.pth"

# 检查权重文件是否存在
if [ ! -f "$PRETRAINED_WEIGHTS_PATH" ]; then
    echo "Error: Pretrained weights not found at $PRETRAINED_WEIGHTS_PATH"
    echo "This is expected if you haven't downloaded them yet."
    echo "Checking alternate location (just in case)..."
    PRETRAINED_WEIGHTS_PATH="../videomambapro/pretrained_models/Tiny_checkpoint.pth"
fi

echo "Starting training (Exp 6 - Weighted Loss)..."
echo "Project Root: $PROJECT_ROOT"
echo "Weights: $PRETRAINED_WEIGHTS_PATH"
echo "Output Dir: $SCRIPT_DIR"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行微调脚本
# --nproc_per_node=4: 使用 4 张 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 关键改动 (Experiment 5):
# 1. mixup 0, cutmix 0: 禁用 Mixup/Cutmix，让模型直接学习原始图像/动作
# 2. lr 5e-4: 降低学习率，防止破坏 Backbone 特征 (比 1e-3 安全，比 1e-5 快)
# 3. layer_decay 0.8: 恢复分层衰减，保护底层特征
# 4. update_freq 4: 保持有效 Batch Size 128 (16*2*4)
# 5. epochs 30: 快速验证

# 注意: `run_class_finetuning.py` 默认 mixup=0.8. 设置 --mixup 0 禁用。

torchrun --nproc_per_node=4 run_class_finetuning.py \
    --model videomambapro_t16_in1k \
    --data_path "../datasets" \
    --data_set MyCustomDataset \
    --nb_classes 52 \
    --finetune "$PRETRAINED_WEIGHTS_PATH" \
    --log_dir "$SCRIPT_DIR" \
    --output_dir "$SCRIPT_DIR" \
    --batch_size 16 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 2 \
    --num_workers 4 \
    --opt adamw \
    --lr 1e-4 \
    --num_sample 1 \
    --drop_path 0.1 \
    --fc_drop_rate 0.0 \
    --layer_decay 0.8 \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 2 \
    --epochs 30 \
    --smoothing 0.0 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --color_jitter 0.0 \
    --update_freq 4 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval
