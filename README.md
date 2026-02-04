# A Two-Stage Deep Decoupling and Alignment Framework for Industrial Defect Detection  
# 面向工业缺陷检测的两阶段深度解耦与对齐框架

> This repository contains the official implementation of **GLA-CLIP**, a two-stage deep decoupled alignment framework for industrial defect detection.  
> 本仓库提供 **GLA-CLIP** 的官方实现：一个面向工业缺陷检测的两阶段深度解耦与对齐框架。

---

## 1. Overview / 总览

GLACLIP follows a **two-stage training strategy** to decouple and align global–local text anchors with visual features:

- **Stage 1 – Global–Local Text Anchor Disentanglement**  
  Only the text side (prompt learner) is trained to construct clear *normal* vs *anomaly* textual anchors.

- **Stage 2 – Visual–Text Alignment**  
  Visual encoder and local prompts are optimized to align image / pixel features with the frozen text anchors.

`asset/figure1.jpg` illustrates the overall architecture and training pipeline.
![alt text](asset/figure1.jpg)
![第一篇论文网络框架_2](https://github.com/user-attachments/assets/0dfe5a68-aaf6-4ff1-a4a2-fe4756ef79a5)

- 图 1（`asset/figure1.jpg`）展示了 GLACLIP 的整体结构：
  - 上半部分：Stage 1，全局与局部文本锚点的解耦与对比学习。
  - 下半部分：Stage 2，将视觉特征与已学得的文本锚点进行对齐，实现图像级与像素级异常检测。

```text
Stage 1: Global–Local Text Anchor Disentanglement
Stage 2: Visual–Text Alignment
```

Qualitative comparison with prior CLIP-based anomaly detectors (WinCLIP, AnomalyCLIP, AdaCLIP, AACLP, GlocalCLIP) is shown in `asset/figure2.jpg`.
- 图 2（`asset/figure2.jpg`）给出了与 WinCLIP、AnomalyCLIP、AdaCLIP、AACLP、GlocalCLIP 等方法在多类工业数据集上的可视化对比，GLA-CLIP 在异常区域定位上更加准确、背景响应更低。


---

## 2. Highlights / 方法亮点

The main contributions of this work are summarized as follows:  
本工作的主要贡献概括如下：

- **Architectural Decoupling / 架构级解耦**  
  We explicitly construct dual branches for **global Image–Text** and **local Patch–Text** alignment. The global branch focuses on establishing robust semantic anchors, while the local branch generates high-precision anomaly maps, structurally reducing the mutual interference between multi-granularity features.  
  我们显式构建全局图像–文本分支和局部补丁–文本分支：前者用于学习稳健的语义锚点，后者用于生成高精度异常热力图，从架构层面消除多粒度特征之间的相互干扰。

- **Lightweight Progressive Adaptation / 轻量级渐进适配**  
  With the visual backbone frozen, we introduce **Deep Hierarchical Text Prompts** and **text-side Residual Adapters**, injecting learnable compound tokens into the deep Transformer layers to achieve efficient knowledge transfer from natural-image pretraining to industrial defect semantics.  
  在冻结视觉主干的前提下，引入深层次层级文本提示与文本侧残差适配器，将可学习复合 token 注入 Transformer 深层，实现从自然图像域到工业缺陷域的高效、平滑知识迁移。

- **Two-Stage Global–Local Training Strategy / 全局–局部两阶段训练策略**  
  We adopt a decoupled two-stage paradigm: **Stage 1** freezes the visual encoder and uses global contrastive learning to construct robust normal / anomalous text anchors; **Stage 2** refines fine-grained defect semantics via pixel-level segmentation loss and local prompt fine-tuning under frozen global anchors. Extensive experiments on MVTec AD, VisA, BTAD and MPDD show that GLA-CLIP achieves state-of-the-art performance and strong cross-domain generalization without target-domain fine-tuning.  
  提出解耦的全局–局部两阶段训练范式：在 Stage 1 冻结视觉编码器，仅通过全局对比学习构建稳健的正常 / 异常文本锚点；Stage 2 在冻结全局锚点的约束下，通过像素级分割损失与局部提示微调细化缺陷语义。大量在 MVTec AD、VisA、BTAD、MPDD 上的跨域实验表明，GLA-CLIP 无需在目标域微调即可取得 SOTA 性能并展现出优异的泛化能力。


---

## 3. Performance & Ablation / 性能与消融实验

### 3.1 Main Results on Industrial Datasets / 工业数据集主结果

`asset/table1.png` (`Table 1`) reports the performance of GLACLIP and previous state-of-the-art zero-shot anomaly detection (ZSAD) methods on **MVTec AD**, **VisA**, **MPDD**, **BTAD** across both image-level and pixel-level metrics.

- 表 1（`asset/table1.png`）展示了 GLACLIP 在 MVTec AD、VisA、MPDD、BTAD 四个工业异常检测数据集上的图像级和像素级指标，与 WinCLIP、AnomalyCLIP、AdaCLIP、GlocalCLIP、AACLP 等方法对比：
  - 采用 AUROC、AP（图像级）和 AUROC、AUPRO（像素级）作为评价指标。
  - 红色粗体表示最佳结果，蓝色粗体表示次优结果，平均行汇总了各数据集上的整体表现。

### 3.2 Key Component Ablation / 关键模块消融

`asset/table2.png` (`Table 2`) presents the ablation of key components on MVTec AD and VisA datasets.

- 表 2（`asset/table2.png`）对模型的关键组件进行了消融实验：
  - 在 Base 模型基础上，逐步加入不同模块（例如改进的文本提示、残差适配器等），观察像素级与图像级指标的提升。

### 3.3 Two-Stage Training Strategy / 两阶段训练策略消融

`asset/table3.png` (`Table 3`) studies different training strategies on MVTec AD and VisA.

- 表 3（`asset/table3.png`）比较了仅训练 Stage 1、仅训练 Stage 2 以及完整两阶段训练三种策略：
  - 结果表明，两阶段联合训练在图像级与像素级上都取得了最优性能，验证了先学习文本锚点、再对齐视觉特征的有效性。


---

## 4. Project Structure / 项目目录结构

The core structure of this repository is as follows:

```text
GLACLIP0126/
├─ GLACLIP_lib/          # CLIP backbone, tokenizer and helper functions
├─ asset/                # Figures and tables used in the manuscript
├─ generate_dataset_json/ # Scripts to build meta.json for each dataset
├─ checkpoints/          # Training checkpoints (output)
├─ results/              # Testing and visualization results (output)
├─ train.py              # Two-stage training script
├─ test.py               # Evaluation script
├─ ablation_study.py     # Ablation experiments driver
├─ glocal_prompt_generator.py # Prompt learner & adapter modules
├─ dataset.py            # Unified industrial dataset loader
├─ metrics.py            # Image- and pixel-level metrics
├─ visualization.py      # Heatmap & t-SNE visualization
├─ loss.py, logger.py, utils.py
├─ requirements.txt      # Python dependencies
└─ Manuscript-0125.pdf   # Paper draft for this repository
```

> For a detailed algorithmic description and experimental settings, please refer to `Manuscript-0125.pdf`.


---

## 5. Environment & Installation / 环境与安装

We recommend using the provided `requirements.txt` to reproduce the environment.

### 4.1 Using Conda (recommended) / 使用 Conda（推荐）

```bash
# Create a new environment named GLACLIP
conda create -n GLACLIP python=3.10
conda activate GLACLIP

# Install Python dependencies
pip install -r requirements.txt
```

- 上述命令会创建名为 **`GLACLIP`** 的 Conda 虚拟环境，并根据 `requirements.txt` 安装本项目所需的全部依赖。


---

## 6. Datasets / 数据集

This repo currently supports several industrial anomaly detection benchmarks (configured in `dataset.py`):

- **MVTec AD**  
- **VisA**  
- **MPDD**  
- **BTAD**

The expected data format is described by a `meta.json` under each dataset root, where each entry records image paths, mask paths, class names and anomaly labels.  
`Dataset` and `generate_class_info` in `dataset.py` handle dataset-specific details.

当前代码主要支持以下工业异常检测数据集（见 `dataset.py`）：

- **MVTec AD**：经典工业缺陷检测基准。
- **VisA**：多类视觉异常检测数据集。
- **MPDD**：多产品缺陷检测数据集。
- **BTAD**：BTAD 工业表面缺陷数据集。

> 请按照原论文或下面的脚本说明组织数据目录结构，并确保每个数据集根目录下存在 `meta.json` 文件，用于描述图像、掩码路径及类别信息。

### 6.1 Dataset download / 数据集下载

Please download the original datasets from their official websites and place them under `./data` (or any path you prefer):

- `./data/mvtec` – MVTec AD
- `./data/visa` – VisA
- `./data/mpdd` – MPDD
- `./data/btad` – BTAD

你可以从各数据集的官网获取原始数据，并按照如下目录组织（可根据需要调整根目录，注意与下面生成 `meta.json` 时的 `root` 保持一致）：

```text
data/
  mvtec/
    bottle/ ...
    cable/ ...
    ...
  visa/
    split_csv/1cls.csv
    images/ ...
  mpdd/
    bracket_black/ ...
    ...
  btad/
    01/ ...
    02/ ...
    03/ ...
```

### 6.2 Generate meta.json / 生成 meta.json 文件

For each dataset, a `meta.json` file is required under the dataset root. It is produced by the scripts in `generate_dataset_json/`:

- **MVTec AD**: `generate_dataset_json/mvtec.py` (class `MVTecSolver`)
- **VisA**: `generate_dataset_json/visa.py` (class `VisASolver`)
- **MPDD**: `generate_dataset_json/mpdd.py` (class `MpddSolver`)
- **BTAD**: `generate_dataset_json/btad.py` (class `BtadSolver`)

Usage pattern（以 MVTec 为例）：

1. Edit the `root` argument in the `__main__` block to your local path, e.g.

   ```python
   if __name__ == '__main__':
       runner = MVTecSolver(root='E:/data/mvtec')
       runner.run()
   ```

2. Run the script:

   ```bash
   python generate_dataset_json/mvtec.py
   ```

3. The script will write `meta.json` to `root` and print the number of normal/anomaly samples.

VisA、MPDD、BTAD 的脚本用法类似：

```bash
# 根据你的数据路径修改脚本里的 root，然后分别运行：
python generate_dataset_json/visa.py
python generate_dataset_json/mpdd.py
python generate_dataset_json/btad.py
```

> 生成 `meta.json` 后，`dataset.py` 中的 `Dataset` 类会自动读取该文件，并根据 `mode` 字段（train/test）构建训练或测试数据集。


---

## 7. Usage: Training, Evaluation & Visualization / 使用：训练、测试与可视化

- **Training / 训练**  
  The model is trained with a **two-stage strategy** implemented in `train.py` (`TwoStageTrainer`). A typical command is:

  ```bash
  python train.py \
    --train_data_path <TRAIN_DATA_ROOT> \
    --dataset <DATASET_NAME> \
    --save_path <CHECKPOINT_DIR>
  ```

  Checkpoints and a `train_info.json` file will be saved under `save_path`. For full hyper-parameter settings, please refer to the argparse block at the end of `train.py` and the experimental setup in the manuscript.  
  训练脚本会在 `save_path` 下保存各阶段的 checkpoint 以及 `train_info.json`；完整超参数配置请参考 `train.py` 末尾的 argparse 定义与论文中的实验设置。

- **Evaluation / 测试与评估**  
  The evaluation script `test.py` loads a trained prompt learner and reports image-level and pixel-level metrics:

  ```bash
  python test.py \
    --data_path <TEST_DATA_ROOT> \
    --dataset <DATASET_NAME> \
    --checkpoint_path <CHECKPOINT_PATH> \
    --save_path <RESULT_DIR>
  ```

  The script logs per-class metrics and saves `test_results.json` and `test_info.json` under `save_path` for further analysis.  
  测试脚本会输出每类的图像级 / 像素级指标，并在 `save_path` 下保存 `test_results.json` 与 `test_info.json` 以便复现论文表格中的结果。
---

## 8. Core Files / 核心文件说明

- `train.py` – Two-stage training script implementing `TwoStageTrainer` and CLI arguments.  
  两阶段训练脚本，实现 `TwoStageTrainer` 以及完整的命令行接口。

- `test.py` – Evaluation script for image-level & pixel-level anomaly detection with logging and JSON export.  
  测试与评估脚本，计算图像级 / 像素级指标并导出 JSON 结果。

- `glocal_prompt_generator.py` – Defines `GLACLIP_PromptLearner`, `ResidualAdapter` and `TripletLoss`, i.e., the global–local prompt generator and adapter modules.  
  定义 `GLACLIP_PromptLearner`、`ResidualAdapter` 与 `TripletLoss` 的文件，是全局–局部提示生成与对比损失的核心实现。

- `dataset.py` – Dataset wrapper and class-mapping utilities for industrial datasets.  
  数据读取与类别映射工具，支持 MVTec AD、VisA、MPDD、BTAD 等数据集。

- `metrics.py` – Implements image-level and pixel-level metrics such as AUROC, AP, AUPRO, F1.  
  指标计算模块，实现图像级和像素级的各类指标。

- `visualization.py` – Visualization tools for anomaly maps and t-SNE feature plots.  
  可视化模块，用于生成异常热力图与 t-SNE 特征可视化。

- `GLACLIP_lib/` – Wrapper of CLIP backbone, tokenizer and utility functions (`load`, `compute_similarity`, etc.).  
  CLIP 主干、分词器以及若干工具函数的封装。


---

## 9. Citation / 引用

If you find this repository helpful for your research, please consider citing the corresponding paper (bibtex entry to be added when the paper is officially available).

如果本代码对你的研究工作有帮助，欢迎在论文中引用我们的工作（正式发表后会在此补充 BibTeX）。

---

## 10. Contact / 联系方式

For questions or issues, please open an issue in this repository.  
如有问题或建议，请在本仓库中提交 issue 进行反馈。

