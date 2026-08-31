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

当前评估按项目约定直接在 `256×256×Z` 网格上进行，只报告 3D Dice 和以 pixel 为单位的 `HD95_px`，不进行 native-grid 逆变换。

## Manifest 用途

- `patients.csv`：vendor、中心、病理类型和 ED/ES frame 等患者级信息；原始 Windows 路径只作为历史元数据，不用于运行。
- `volumes.csv`：原始/重采样尺寸、spacing、切片数和前景统计。
- `slices.csv`：运行时的主要索引。每个 checkpoint seed 在各 Vendor 内独立打乱患者，随后按 `patient → ED/ES → z_index` 读取。
- `qc_report.json`：345 名患者、690 个 ED/ES 体积及各 vendor 数量的预处理汇总。

完整数据清单仍保留所有 345 名患者；正式 C 域测评只使用 Validation 10 名和 Testing 40 名患者，共 50 名患者、100 个 ED/ES 体积。

## 标签隔离

测试时适配方法只能接收图像张量。`MMSTargetVolumeDataset` 首先加载图像并保留 mask 路径；只有 `run_volume` 完整返回预测后，外层评估器才调用 `load_mask`。任何 TTA 方法都不得导入数据模块或利用 B/C/D 标签选择阈值、优化器或方法变体。

数据加载入口位于项目根目录的 `data.py`：

```python
from data import build_source_loaders, build_target_stream

train_loader, val_loader = build_source_loaders(cfg, seed=2022)
vendor_b_stream = build_target_stream("B", cfg, order_seed=2022)
```
