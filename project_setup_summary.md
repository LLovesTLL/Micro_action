# VideoMambaPro 微动作识别项目配置日志

**日期**: 2026年2月27日
**任务**: 配置 VideoMambaPro 项目以使用自定义数据集（52类微动作）进行微调训练。

## 1. 项目与数据分析
*   **项目**: VideoMambaPro (基于 Mamba 架构的视频理解模型)。
*   **目标**: 使用自定义数据集进行微动作识别（Micro-action Recognition）。
*   **数据结构**:
    *   **标签文件**: 
        *   `datasets/annotations/train_list_videos.txt` (格式: `视频文件名.mp4 标签ID`)
        *   `datasets/annotations/val_list_videos.txt`
        *   `datasets/annotations/label_name.txt` (52个类别)
    *   **视频文件**:
        *   训练集: `datasets/train/*.mp4` (直接存放，无子文件夹)
        *   验证集: `datasets/val/*.mp4` (直接存放，无子文件夹)

## 2. 代码修改记录

### A. 数据集适配 (`datasets/build.py`)
为了识别自定义数据集格式，添加了 `MyCustomDataset` 逻辑：
*   **文件**: `d:\Administrator\My_File\毕业论文(微动作)\Baseline\VideoMambaPro-main\datasets\build.py`
*   **改动**:
    *   在 `build_dataset` 函数中增加 `if args.data_set == 'MyCustomDataset':` 分支。
    *   **路径映射**:
        *   训练模式: 读取 `annotations/train_list_videos.txt`，视频根目录设为 `train`。
        *   验证模式: 读取 `annotations/val_list_videos.txt`，视频根目录设为 `val`。
    *   **分隔符**: 强制使用空格 (`split=' '`) 解析列表文件。

### B. 训练脚本适配 (`videomambapro/run_class_finetuning.py`)
*   **文件**: `d:\Administrator\My_File\毕业论文(微动作)\Baseline\VideoMambaPro-main\videomambapro\run_class_finetuning.py`
*   **改动**:
    *   在参数解析器 (`parser`) 的 `--data_set` 选项中注册了 `'MyCustomDataset'`，使其成为可选参数。

## 3. 运行脚本创建

### A. Windows PowerShell 脚本 (本地调试用)
*   **文件**: `exp/action/run_train.ps1`
*   **配置**: 单卡训练，使用绝对路径。

### B. Linux Shell 脚本 (服务器训练用)
*   **文件**: `exp/action/run_train.sh`
*   **配置**: 
    *   **4 GPU 并行训练**: 使用 `torchrun --nproc_per_node=4` 启动。
    *   **相对路径**: 自动获取脚本位置，适应服务器不同的目录结构。
    *   **权重路径**: 默认寻找 `videomambapro/pretrained_models/videomambapro_t16_in1k_res224.pth`。
    *   **输出目录**: 日志和模型保存在脚本同级目录 (`exp/action/`)。

## 4. 训练指南 & Q&A

### 预训练权重
*   **模型**: `videomambapro_t16_in1k` (Tiny, Patch16, ImageNet-1K Pretrained)。
*   **原理**: 虽然是图像预训练权重，但模型会将其扩展为时空特征提取器。对于视频任务，使用 ImageNet 权重初始化是标准且有效的做法。
*   **下载**: 需下载 `videomambapro_t16_in1k_res224.pth` 并放置在 `videomambapro/pretrained_models/` 目录下。

### 服务器部署步骤 (Linux)
1.  **上传代码**: 保持目录结构完整 (`datasets/`, `videomambapro/`, `exp/`)。
2.  **上传权重**: 将 `.pth` 文件放入 `videomambapro/pretrained_models/`。
3.  **赋予权限**: `chmod +x exp/action/run_train.sh`
4.  **运行**: `./exp/action/run_train.sh` (建议使用 `nohup` 后台运行)。

## 5. 文件结构概览 (服务器端预期)

```text
ProjectRoot/
├── datasets/
│   ├── annotations/ (txt列表文件)
│   ├── train/       (训练视频)
│   └── val/         (验证视频)
├── videomambapro/   (代码核心)
│   ├── pretrained_models/
│   │   └── videomambapro_t16_in1k_res224.pth
│   └── run_class_finetuning.py
├── exp/
│   └── action/
│       ├── run_train.sh  (启动脚本)
│       └── ... (训练日志和输出模型将生成在这里)
└── ...
```
