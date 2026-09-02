# MMS 数据目录说明

本目录保存工作台使用的 M&Ms 数据索引、预处理数组和质量检查结果。实验协议固定为：Vendor A（Siemens）是源域，Vendor B/C/D 是目标域。Vendor C 的 25 名 `Training/Unlabeled` 患者不进入正式目标流，因此不会参与推理适配或指标计算。

## 目录结构

```text
data/
├── manifests/
│   ├── patients.csv       # 患者级信息
│   ├── volumes.csv        # ED/ES 体积、原始尺寸与 spacing
│   ├── slices.csv         # 每张切片的图像、mask 路径和 z_index
│   └── qc_report.json     # 预处理和数据量汇总
├── processed/
│   └── 2d_1p5mm_256/     # 本地 256×256 NPY 数组，不上传 Git
├── qc/                    # 质量检查图片和失败记录
├── splits/
│   └── mms_vendor_a_to_bcd_tta.json
└── visual/                # 数据可视化图片
```

`processed/` 已被 `.gitignore` 排除。代码和 manifest 中可以保存相对路径，但不要把医学图像、mask 或生成的模型文件提交到 GitHub。

## 预处理数组

每张图像和标签都是一个 `256×256` 的 `.npy` 文件：

- 图像：`float32`，形状为 `[256, 256]`，强度已归一化到 `[0, 1]`；
- 标签：`uint8`，形状为 `[256, 256]`；
- 标签定义：`0=background`、`1=RV`、`2=MYO`、`3=LV`；
- 网络输入由 `data.py` 增加通道维，成为 `[1, 256, 256]`。

默认 `patient_volume` 评估在过滤后的 `256×256×Z` 前景切片栈上报告 3D Dice 和 `HD95_px`。`slice_random` 模式先过滤、再在 Vendor 内打乱目标切片，并逐张报告 2D Dice 和 `HD95_2d_px`；两种模式都不进行 native-grid 逆变换。

## Manifest 用途

- `patients.csv`：vendor、中心、病理类型和 ED/ES frame 等患者级信息；原始 Windows 路径只作为历史元数据，不用于运行。
- `volumes.csv`：原始/重采样尺寸、spacing、切片数和前景统计。
- `slices.csv`：运行时的主要索引。Source train/validation 和 B/C/D 测试流只保留 `has_fg=1` 且 `fg_pixels>0` 的切片；过滤在顺序打乱和组 batch 前完成。
- `qc_report.json`：345 名患者、690 个 ED/ES 体积及各 vendor 数量的预处理汇总。

完整数据清单仍保留所有 345 名患者；正式 C 域测评只使用 Validation 10 名和 Testing 40 名患者，共 50 名患者、100 个 ED/ES 体积。

## 标签隔离

测试时适配方法只能接收图像张量。数据流使用 manifest 中预计算的标签派生字段 `has_fg` 做协议级切片筛选，但不在适配前加载 mask 数组；只有预测返回后，外层评估器才调用 `load_mask`。该口径必须标记为 foreground-only，不得与原始全切片协议混为一组结果。

数据加载入口位于项目根目录的 `data.py`：

```python
from data import build_source_loaders, build_target_slice_loader, build_target_stream

train_loader, val_loader = build_source_loaders(cfg, seed=2022)
vendor_b_stream = build_target_stream("B", cfg, order_seed=2022)
vendor_b_random_slices = build_target_slice_loader("B", cfg, order_seed=2022)
```

随机切片 loader 的 dataset item 只包含图像、mask 路径和到达元数据；运行器必须先完成整批适配与预测，之后才能通过 dataset 加载 mask 计算逐切片指标。
