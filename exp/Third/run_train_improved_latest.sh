#!/bin/bash

# VideoMambaPro Finetuning Script for Micro-action (Improved)

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 项目根目录
PROJECT_ROOT="$SCRIPT_DIR/../../"

# 切换到 videamambapro 代码目录
cd "$PROJECT_ROOT/videomambapro"

# 预训练权重路径
PRETRAINED_WEIGHTS_PATH="./pretrained_models/Tiny_checkpoint.pth"

# 检查权重文件是否存在
if [ ! -f "$PRETRAINED_WEIGHTS_PATH" ]; then
    echo "Error: Pretrained weights not found at $PRETRAINED_WEIGHTS_PATH"
    echo "Please download the weights and place them in the correct location, or update the path in this script."
    # 尝试查找其他可能的路径
    ALT_PATH="../pretrained_models/Tiny_checkpoint.pth"
    if [ -f "$ALT_PATH" ]; then
        PRETRAINED_WEIGHTS_PATH="$ALT_PATH"
        echo "Found weights at $ALT_PATH"
    else
         echo "Creating a placeholder check, proceeding if you are sure..."
    fi
fi

echo "Starting training..."
echo "Project Root: $PROJECT_ROOT"
echo "Weights: $PRETRAINED_WEIGHTS_PATH"
echo "Output Dir: $SCRIPT_DIR"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行微调脚本 (使用 torchrun 启动多卡训练)
# --nproc_per_node=3: 使用 3 张 GPU
# 如果单卡，请改为 --nproc_per_node=1

# 使用 GPU 1, 2, 3
export CUDA_VISIBLE_DEVICES=1,2,3

torchrun --nproc_per_node=3 run_class_finetuning.py \
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
    --epochs 70 \
    --opt adamw \
    --lr 2e-4 \
    --layer_decay 0.9 \
    --drop 0.1 \
    --drop_path 0.1 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --test_best \
    --dist_eval
