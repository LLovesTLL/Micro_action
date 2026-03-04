# VideoMambaPro 微动作识别项目日志

## 1. 项目与数据分析
*   **项目**: VideoMambaPro (基于 Mamba 架构的视频理解模型)。
*   **仓库地址**: `https://github.com/LLovesTLL/Micro_action.git`
*   **目标**: 使用自定义数据集进行微动作识别（Micro-action Recognition）。
*   **数据结构**:
    *   **标签文件**: 
        *   `datasets/annotations/train_list_videos.txt` (格式: `视频文件名.mp4 标签ID`)
        *   `datasets/annotations/val_list_videos.txt`
        *   `datasets/annotations/label_name.txt` (52个类别)
    *   **视频文件**:
        *   训练集: `datasets/train/*.mp4` (直接存放，无子文件夹)
        *   验证集: `datasets/val/*.mp4` (直接存放，无子文件夹)

## 2. 工作日志 (Chronological Log)

### 📅 2026年2月27日
**任务**: 配置 VideoMambaPro 项目以使用自定义数据集（52类微动作）进行微调训练。

#### A. 代码修改记录

**1. 数据集适配 (`datasets/build.py`)**
为了识别自定义数据集格式，添加了 `MyCustomDataset` 逻辑：
*   **文件**: `d:\Administrator\My_File\毕业论文(微动作)\Baseline\VideoMambaPro-main\datasets\build.py`
*   **改动**:
    *   在 `build_dataset` 函数中增加 `if args.data_set == 'MyCustomDataset':` 分支。
    *   **路径映射**:
        *   训练模式: 读取 `annotations/train_list_videos.txt`，视频根目录设为 `train`。
        *   验证模式: 读取 `annotations/val_list_videos.txt`，视频根目录设为 `val`。
    *   **分隔符**: 强制使用空格 (`split=' '`) 解析列表文件。

**2. 训练脚本适配 (`videomambapro/run_class_finetuning.py`)**
*   **文件**: `d:\Administrator\My_File\毕业论文(微动作)\Baseline\VideoMambaPro-main\videomambapro\run_class_finetuning.py`
*   **改动**:
    *   在参数解析器 (`parser`) 的 `--data_set` 选项中注册了 `'MyCustomDataset'`，使其成为可选参数。

#### B. 运行脚本创建

**1. Windows PowerShell 脚本 (本地调试用)**
*   **文件**: `exp/action/run_train.ps1`
*   **配置**: 单卡训练，使用绝对路径。

**2. Linux Shell 脚本 (服务器训练用)**
*   **文件**: `exp/action/run_train.sh`
*   **配置**: 
    *   **4 GPU 并行训练**: 使用 `torchrun --nproc_per_node=4` 启动。
    *   **相对路径**: 自动获取脚本位置，适应服务器不同的目录结构。
    *   **权重路径**: 默认寻找 `videomambapro/pretrained_models/videomambapro_t16_in1k_res224.pth`。
    *   **输出目录**: 日志和模型保存在脚本同级目录 (`exp/action/`)。

#### C. 训练指南 & Q&A

**预训练权重**
*   **模型**: `videomambapro_t16_in1k` (Tiny, Patch16, ImageNet-1K Pretrained)。
*   **原理**: 虽然是图像预训练权重，但模型会将其扩展为时空特征提取器。对于视频任务，使用 ImageNet 权重初始化是标准且有效的做法。
*   **下载**: 需下载 `videomambapro_t16_in1k_res224.pth` 并放置在 `videomambapro/pretrained_models/` 目录下。

**服务器部署步骤 (Linux)**
1.  **上传代码**: 保持目录结构完整 (`datasets/`, `videomambapro/`, `exp/`)。
2.  **上传权重**: 将 `.pth` 文件放入 `videomambapro/pretrained_models/`。
3.  **赋予权限**: `chmod +x exp/action/run_train.sh`
4.  **运行**: `./exp/action/run_train.sh` (建议使用 `nohup` 后台运行)。

**文件结构概览 (服务器端预期)**
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

#### D. 版本控制与 GitHub 推送指南

**1. 配置 .gitignore**
为避免上传大文件 (视频、权重、数据集)，已配置 `.gitignore`。
*   **忽略文件**: `*.mp4`, `datasets/`, `pretrained_models/`, `output/`, `*.pth` 等。
*   **保留文件**: 核心代码、脚本 (`exp/action/*.sh`, `*.ps1`)、文档。

**2. 推送流程 (处理大文件被误添加的情况)**
如果在 `git add` 之前已有大文件被误添加进历史记录，导致 push 失败 (100MB 限制)，请按以下步骤彻底重置：

1.  **删除 .git 文件夹** (清除所有历史包袱):
    ```powershell
    Remove-Item -Recurse -Force .git
    ```
2.  **重新初始化 Git**:
    ```powershell
    git init
    ```
3.  **确保 .gitignore 生效**:
    (确认目录下有 .gitignore 文件)
4.  **重新添加与提交**:
    ```powershell
    git add .
    git commit -m "Initial commit for VideoMambaPro"
    ```
5.  **重命名主分支**:
    ```powershell
    git branch -M main
    ```
6.  **关联远程仓库**:
    ```powershell
    git remote add origin https://github.com/LLovesTLL/Micro_action.git
    ```
7.  **强制推送**:
    ```powershell
    git push -u origin main --force
    ```

### 📅 2026年3月3日
**任务**: 调试记录

#### A. 遇到的问题与解决方案
1.  **模块导入错误 (`ModuleNotFoundError`)**
    *   **现象**: `No module named 'models.videomamba'`
    *   **原因**: 代码重构后文件夹名为 `videomambapro`，但 `__init__.py` 中仍引用 `videomamba`。
    *   **解决**: 修正 `videomambapro/models/__init__.py` 中的导入路径，并设置 `PYTHONPATH` 包含项目根目录。

2.  **模型注册缺失 (`Unknown model`)**
    *   **现象**: `RuntimeError: Unknown model: videomambapro_t16_in1k`
    *   **原因**: 脚本指定的模型名称 `videomambapro_t16_in1k` 未在 `videomambapro.py` 中显示注册。
    *   **解决**: 在 `videomambapro/models/videomambapro.py` 中使用 `@register_model` 添加模型别名映射，关联到基础模型实现。

3.  **权重加载不匹配 (`KeyError` / Shape Mismatch)**
    *   **现象**: `KeyError: 'temporal_pos_embedding'` 或 `patch_embed` 权重维度不匹配 (2D vs 3D)。
    *   **原因**: 使用 ImageNet (2D图像) 预训练权重初始化 Video (3D视频) 模型。ImageNet 权重没有时间维度的位置编码，且 Patch Embedding 卷积核是 2D 的。
    *   **解决**: 修改 `run_class_finetuning.py` 的权重加载逻辑：
        *   忽略缺失的 `temporal_pos_embedding`（允许随机初始化）。
        *   (代码中已准备) 2D 权重膨胀 (Inflation) 逻辑以适配 3D 卷积核。

4.  **内存锁页错误 (`RuntimeError: pin_memory`)**
    *   **现象**: `RuntimeError: Caught RuntimeError in pin memory thread... CUDA error: invalid argument`
    *   **原因**: 训练过程中出现 OOM 或参数无效。经排查 `ulimit -a`，系统 `locked memory` 限制仅为 64MB。当 `batch_size=32` (`8` x 4卡) 且 `num_workers=8` 时，PyTorch DataLoader 试图锁定的内存超过系统限制。
    *   **解决**: 在 `run_train.sh` 中降低资源占用：
        *   **Batch Size**: `32` -> `16` (单卡 4 -> 4卡总 16)。
        *   **Num Workers**: `8` -> `4` (减少多进程内存开销)。

#### B. 实验运行状态
*   **当前配置**: 4卡并行训练，总 Batch Size 16。
*   **初始指标**: Epoch 0 初始 Loss 约为 4.15 (52类随机初始化的交叉熵损失约为 ln(52)≈3.95，数值在合理范围内)。
*   **下一步计划**: 观察 Loss 收敛情况；如验证集准确率过低，需检查 2D 权重加载是否正确生效。

#### C. 图像预训练模型训练视频模型的解释 (I3D / ImageNet-to-Video)

您提出的“使用图像预训练模型训练视频模型”是一种非常主流乃至标准的做法，通常被称为 I3D (Inflated 3D ConvNet) 或 ImageNet-to-Video transfer。

虽然这种方法被广泛接受并且效果显著，但相比于直接使用“视频预训练模型”，确实存在一些潜在的劣势或需要注意的地方：

1.  **时间建模能力的缺失**
    *   **问题**: ImageNet 图像预训练模型是在静态图片上训练的，它只学习到了空间特征（如形状、纹理、颜色），完全没有见过视频中的时间变化和运动信息。
    *   **后果**: 模型初始化时，对“时间”的概念是一张白纸（Temporal Pos Embedding 和时间维度的卷积权重都是初始化或简单复制的）。
    *   **补救**: 这通常不是大问题，因为微调（Finetuning）阶段模型会迅速学会时间特征。但这意味着微调的前期，模型需要花一些时间来“适应”视频数据。

2.  **收敛速度可能较慢**
    *   **问题**: 相比于已经在类似 YouTube-8M 或 Kinetics-400 这样的大规模视频数据集上预训练过的模型，从图像模型开始微调，通常需要更多的 Epoch 才能收敛到最优效果。
    *   **后果**: 您可能需要设置稍长一点的训练轮数（Epochs）。

3.  **对微调数据量的依赖**
    *   **问题**: 如果您的目标视频数据集（Micro_action）非常小，从未见过的“时间参数”可能因为数据量不足而无法充分学习，导致过拟合或无法捕捉复杂的时序动作。
    *   **对比**: 如果有一个已经在大型视频数据集上预训练好的模型，它已经学会了通用的“动作特征”，迁移到您的微小数据集上效果通常会更好。

**总结**
尽管存在上述“劣势”，但从图像预训练开始通常是最好的选择，除非您有非常昂贵的算力去自己跑一个大规模视频预训练，或者恰好找到了结构完全一致且在相关领域预训练过的视频模型。

**对于您的情况**:
*   这完全是正统且正确的操作路径。
*   您的模型会自动“膨胀”并将静态特征迁移到视频任务中，通常效果依然非常强劲。
*   如果遇到效果不理想，可以尝试增加微调的 **Epochs 数**，或者使用更强的**数据增强手段**。

#### D. 实验过程性的解读 (基于训练日志)

**1. 警告与环境信息**
*   **Warning: temporal_pos_embedding not found**: 
    *   **现象**: 权重缺少时间绝对位置编码。
    *   **原因**: 使用的 `Tiny_checkpoint.pth` 是 ImageNet-1K 图像预训练权重，不包含 VideoMamba 引入的时序组件。
    *   **结论**: 符合预期，模型将从零学习时序位置关系。
*   **DDP Warning (find_unused_parameters=True)**: 
    *   **现象**: PyTorch 提示正在使用全图扫描来查找未使用的参数。
    *   **原因**: 这是 VideoMambaPro 在 DDP 环境下的默认安全配置，防止复杂分支结构导致的反向传播错误。
    *   **结论**: 不影响精度，仅轻微增加开销，可忽略。

**2. 权重加载与匹配情况 (Weight Diagnosis)**
*   **未加载的权重 (预期内)**:
    *   `temporal_pos_embedding`: 跳过加载。
    *   `layers.*.mixer.*_b.*`: 这些是 **Backward Scan (反向扫描)** 的参数。由于 ImageNet 预训练的 Vim (Vision Mamba) 主要是单向扫描的，而 VideoMambaPro 增强为 Bidirectional Mamba，因此反向参数是全新初始化的。
*   **形状不匹配 (Size Mismatch) - 关键观测点**:
    *   `patch_embed.proj.weight`: 
        *   **Checkpoint**: `[192, 3, 16, 16]` (2D, ImageNet)
        *   **Current**: `[192, 3, 2, 16, 16]` (3D, Video)
        *   **解读**: 2D 卷积核未能自动“膨胀”成 3D 形式。这意味着第一层视觉特征提取能力完全重置为随机，需要从头训练。
    *   `head.weight/bias`: 分类头从 1000 类 (ImageNet) 变为 52 类 (Micro-action)，必须完全重置。

**3. 模型结构与优化策略** (详细复盘)
*   **模型架构 (VideoMambaPro Tiny)**:
    *   **输入层 (`patch_embed`)**: `Conv3d(3, 192, kernel_size=(2, 16, 16), stride=(2, 16, 16))`。将视频切分为时空 Patch。
    *   **核心层 (`layers`)**: 共 24 个 Block。
    *   **双向扫描机制 (Bi-Mamba)**: 每个 Block 包含成对的参数（如 `conv1d` 和 `conv1d_b`，`dt_proj` 和 `dt_proj_b`）。`_b` 后缀代表 Backward（反向）扫描路径。相比原始 Vision Mamba 的单向扫描，这种双向机制能同时捕捉视频的前向和后向时序依赖，显著提升动作理解能力。
*   **优化策略 (SOTA 配置)**:
    *   **逐层学习率衰减 (Layer-wise Learning Rate Decay)**:
        *   **机制**: `Assigned values: [0.00075..., ..., 1.0]`。
        *   **原理**: 网络深层（高语义、任务相关）使用全额学习率 (Scale=1.0)，网络浅层（低级特征、通用）使用极低学习率 (Scale=0.00075)。
        *   **目的**: 最大限度保留预训练模型在底层学到的通用视觉特征（边缘、纹理），同时让高层快速适应新的微动作分类任务。
    *   **参数分组 (Param Groups)**:
        *   `decay`: 卷积核、线性层权重 -> 施加 **Weight Decay (0.05)** 以防过拟合。
        *   `no_decay`: Bias、LayerNorm、位置编码 -> **Weight Decay (0.0)**，保持数值稳定性，防止模型坍塌。

**4. 训练进度与状态 (Training Dynamics)**
*   **Loss 趋势**: 
    *   `Epoch 0`: Loss 4.1567 -> 3.8567。
    *   **解读**: 初始 Loss 符合随机猜测 (ln52 ≈ 3.95)，且呈稳步下降趋势，证明网络梯度正常，正在有效汲取知识。
*   **计算效率**:
    *   **Throughput**: 预热结束后，单步耗时稳定在 4.8s 左右。
    *   **Data Loading**: `data_time` 极低 (0.0005s)，说明 4 个 Workers 足够喂饱 GPU，不存在 I/O 瓶颈。
*   **异常处理机制 (AMP Robustness)**: 
    *   早期出现的 `NaN/Inf` 是混合精度训练 (AMP) 初始化时的正常现象。
    *   Loss Scale 从 65536 自动降级至 32768 并稳定，表明系统已自动修正了数值溢出问题。
*   **资源占用**: 显存占用约 4.5GB (batch_size=4/gpu)，利用率较为保守，后续如有需求可增大 Batch Size。

**总结**: 
训练已成功跑通。各项指标（Loss 下降、计算速度、内存使用）均在健康范围内。模型结构确认使用了高级的双向扫描和逐层学习率衰减策略，为后续的高性能表现奠定了基础。

#### E. 实验结果分析与改进方向 (Results Analysis)

**1. 实验结果数据**
*   **测试集规模**: 83,790 样本 (Views/Clips)
*   **Top-1 准确率**: **11.48%**
*   **Top-5 准确率**: **39.24%**
*   **最终 Loss**: **3.416** (从初始 ~4.15 下降)
*   **训练耗时**: 18小时 24分

**2. 结果深度分析**
*   **有效学习验证**:
    *   52分类的随机猜测准确率仅为 $1/52 \approx 1.92\%$。
    *   当前 **11.48%** 的准确率约是随机猜测的 **6倍**，证明模型**已经开始学习**到有效的微动作特征，管道(Pipeline)是打通的。
*   **Top-1 与 Top-5 的差距**:
    *   Top-5 高达 39.24%，说明模型在面对输入视频时，往往能将正确类别排在前列，但缺乏足够的信心将其置于首位。这对于微动作（Fine-grained action）识别很常见，因为微动作之间差异极小（如“点头” vs “低头”），模型容易混淆。
*   **性能瓶颈推测**:
    *   **初始化劣势**: 正如日志分析提到的，`patch_embed` 层因维度不匹配而**随机初始化**。这意味着模型的第一层视觉提取器是从零开始学的，严重拖慢了收敛速度。
    *   **训练时长不足**: 18小时对于从零适应微动作领域（且首层随机初始化）可能还不够。Loss (3.416) 仍然处于较高水平，通常优秀的分类模型 Loss 应降至 1.0 以下。

**3. 下一步改进建议 (Action Plan)**
*   **策略一：修复权重初始化 (高优先级)**
    *   修改 `run_class_finetuning.py`，实现 `patch_embed` 的 **Central Inflation**（中心膨胀）或 **Mean Inflation**（均值膨胀）。将 ImageNet 的 2D 卷积核复制扩展到 3D，而不是随机初始化。这将显著提升前几轮的效果。
*   **策略二：延长训练轮数 (More Epochs)**
    *   目前的准确率仍在上升通道中，并未饱和。建议继续训练（Resume Training），多跑 20-50 个 Epochs。
*   **策略三：分析混淆矩阵**
    *   微动作中可能存在“长尾分布”或极其相似的类别。查看哪些类别准确率为 0，针对性调整数据采样策略。
*   **验证测试设置**:
    *   确认测试时是否使用了 Multi-view testing (如 4 clips x 3 crops)。虽然能提升精度，但推理极慢。如果是为了快速验证迭代，可暂时减少 view 数量。

### 📅 2026年3月4日
**任务**: 代码深度解读与微动作适配方案优化

#### A. 核心代码机制解读 (Detailed Analysis)

**1. 模型架构: 从 2D 到 3D 的进化 (`videomambapro/models/videomambapro.py`)**
*   **权重膨胀 (Weight Inflation)**:
    *   **机制**: 核心函数 `inflate_weight` 解决了 ImageNet 预训练权重 (2D) 与视频模型 (3D) 的维度不匹配问题。
    *   **实现**: 对于卷积层 (如 PatchEmbed)，它将 `(C_out, C_in, H, W)` 的 2D 卷积核扩展为 `(C_out, C_in, T, H, W)`。
    *   **策略**: 采用 **"Center Frame Initialization"** (中心帧初始化)。除中间时间步保留原 2D 权重外，其余时间步初始化为 0。
    *   **意义**: 这使得模型在微调初期表现得像一个 2D 特征提取器，随着训练进行，逐渐学会利用时间维度信息，极大加速了收敛。
*   **Mamba 核心 (SSM)**:
    *   **线性复杂度**: 相比 Transformer 的 $O(N^2)$ 计算量，Mamba 随序列长度线性增长，适合处理长视频序列。
    *   **双向扫描 (Bi-directional Scan)**: 代码中的 Block 包含了前向 (`conv1d`) 和后向 (`conv1d_b`) 两套参数。这克服了传统 SSM 单向因果的局限，让模型能同时利用“过去”和“未来”的帧信息来判定当前动作，这对判定微动作的起始和结束至关重要。

**2. 数据处理: 稀疏采样与增强 (`datasets/kinetics.py`)**
*   **稀疏采样 (Sparse Sampling)**:
    *   **逻辑**: 并不连续读取每一帧，而是通过 `frame_sample_rate` (步长) 进行跳帧采样。
    *   **公式**: 覆盖时间窗口 = `clip_len` (16帧) $\times$ `sampling_rate` (如 4)。
    *   **微动作适配**: 对于瞬时微动作，较大的步长会导致漏掉关键帧；因此我们在改进方案中将步长从 4 降至 2，以提高时间分辨率。
*   **Decord 解码**: 使用 `decord.VideoReader` 进行高效解码，支持直接定位到特定帧索引，避免了全视频解码的开销。

**3. 训练策略: 迁移学习的最佳实践 (`videomambapro/run_class_finetuning.py`)**
*   **层级学习率衰减 (Layer-wise LR Decay)**:
    *   **原理**: 网络不同层级的特征抽象程度不同。底层提取通用特征 (边缘/纹理)，高层提取语义特征 (动作类别)。
    *   **操作**: `optim_factory.py` 中实现了从底层到顶层逐渐增加学习率 (如底层 0.001x -> 顶层 1.0x)。这有效防止了微调过程中破坏预训练的底层通用视觉特征。
*   **随机深度 (Stochastic Depth / DropPath)**:
    *   **机制**: 在训练过程中随机“丢弃”整个残差块 (Block)。
    *   **作用**: 相当于训练了无数个子网络的集成，显著增强了模型在小数据集 (52类) 上的泛化能力，防止过拟合。

#### B. 实验方案优化 (针对 Micro-action)
针对微动作幅值小、速度快、易受强增强破坏的特点，制定了新版训练方案。

**1. 新建脚本**: `exp/experiments/run_train_improved.sh`

**2. 核心代码修复 (Critical Fix) - 手动膨胀 (`run_class_finetuning.py`)**:
*   **问题**: VideoMambaPro 第一层 (`patch_embed`) 权重 (3D) 与 ImageNet 预训练权重 (2D) 维度不匹配 `[192, 3, 2, 16, 16] vs [192, 3, 16, 16]`，导致该层**被随机初始化**。这使得模型在训练初期丧失了所有视觉特征提取能力，严重拖慢收敛。
*   **修复**: 在权重加载逻辑中增加了**手动膨胀 (Manual Inflation)** 代码。
    *   **实现**: 将 2D 卷积核复制到 3D 卷积核的中间帧 (Center Frame Initialization)，其余时间步置零。这确保模型在 Epoch 0 即拥有 ImageNet 级别的空间特征提取能力。

**3. 关键参数调整详情**:
*   **时序分辨率提升**: `sampling_rate` **4 → 2**。
    *   *理由*: 采样间隔从 4 帧降为 2 帧，使模型聚焦更短但更密集的动作窗口，防止漏掉转瞬即逝的微动作。
*   **数据增强减弱**:
    *   **AutoAugment**: `rand-m7-n4` → **`rand-m5-n2`** (降低几何变换强度，保留动作方向性)。
    *   **Mixup/Cutmix**: **禁用 (0.0)**。
    *   *理由*: 微动作依赖细微的纹理和结构变化，强力混合 (Mixup) 或遮挡 (Cutmix) 会破坏关键特征，导致模型困惑。
*   **正则化增强**:
    *   新增 `--drop 0.1` (Head Dropout) 和 `--drop_path 0.1`。
    *   *理由*: 在去除强力数据增强后，主动增加 Dropout 防止在小数据集 (52类) 上过拟合。

#### C. 模型对比决策: 为何不使用 VideoMamba (Pure Mamba) 预训练权重？
虽然 VideoMamba 提供了 K400/SSV2 的视频预训练权重，但经分析决定**不予使用**，原因如下：

*   **架构不兼容**: 
    *   **Block 结构**: VideoMambaPro 的 Block 采用了不同的残差连接方式 (Parallel Residual: `Add -> LN -> Mixer` with explicit residual management)，与原始 VideoMamba 并不完全一致。
    *   **辅助模块**: VideoMamba 包含复杂的 CLIP 对齐头 (`clip_decoder`) 和特定的位置编码逻辑，这些在 VideoMambaPro 的微调架构中是被移除的。直接加载会导致大量权重丢失 (`unexpected_keys`)，使得预训练的价值大打折扣。
*   **性价比考量**:
    *   通过修复 **Patch Embed 膨胀** 问题，我们已经让 VideoMambaPro 成功继承了 ImageNet 强大的空间特征。
    *   相比于强行适配不兼容的 K400 权重，使用“**ImageNet 空间底座 + 快速时序适应**”的策略更稳健，且避免了不可控的架构冲突bug。

#### D. 下一步计划
*   运行 `exp/experiments/run_train_improved.sh` (已包含权重膨胀修复)。
*   监控验证集准确率，预期前几个 Epoch 的性能将显著优于第一次实验 (不再是随机猜测)。
*   对比 "High Augmentation (前次实验)" 与 "Low Augmentation + Dense Sampling (本次改进)" 的性能差异。

