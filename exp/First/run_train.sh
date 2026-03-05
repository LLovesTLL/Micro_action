#!/bin/bash

# VideoMambaPro Finetuning Script for Linux (Multi-GPU)

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 项目根目录 (假设当前脚本在 exp/action/ 下)
PROJECT_ROOT="$SCRIPT_DIR/../../"

# 切换到 videamambapro 代码目录
cd "$PROJECT_ROOT/videomambapro"

# 预训练权重路径 (请根据实际情况修改文件名或相对路径)
# 假设权重文件放在 videamambapro/pretrained_models 文件夹中
PRETRAINED_WEIGHTS_PATH="./pretrained_models/Tiny_checkpoint.pth"

# 检查权重文件是否存在
if [ ! -f "$PRETRAINED_WEIGHTS_PATH" ]; then
    echo "Error: Pretrained weights not found at $PRETRAINED_WEIGHTS_PATH"
    echo "Please download the weights and place them in the correct location, or update the path in this script."
    exit 1
fi

echo "Starting training..."
echo "Project Root: $PROJECT_ROOT"
echo "Weights: $PRETRAINED_WEIGHTS_PATH"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行微调脚本 (使用 torchrun 启动多卡训练)
# --nproc_per_node=4: 使用 4 张 GPU
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
    --sampling_rate 4 \
    --num_workers 8 \
    --epochs 50 \
    --opt adamw \
    --lr 5e-4 \
    --layer_decay 0.75 \
    --dist_eval
