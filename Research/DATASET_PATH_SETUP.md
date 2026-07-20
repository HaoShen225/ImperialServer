# Dataset Path Setup for Backbone Training

本文档说明合作伙伴在服务器上运行 `backbone_training_MeanTeacher.py` 时，如何布置 ACDC 源域数据和 MMS 目标域数据路径。

## 1. 代码不会自带数据集

GitHub 仓库只包含训练和评估代码，不包含 ACDC/MMS 数据本体。运行前需要先把已经预处理好的数据集复制到服务器本地磁盘。

训练脚本读取的是 normalized + processed `.npy` 数据：

- ACDC source domain: 用于 Mean Teacher backbone 训练。
- MMS target domain: 训练结束后用于 4 个 vendor 上的源模型性能评估。

## 2. 推荐的服务器目录结构

建议在服务器上建立类似下面的目录：

```text
/data/Cardiac/
  ACDC_normalized/
    manifests/
      patients.csv
    processed/
      2d_1p5mm_256/
        metadata.jsonl
        train/
          ...
        val/
          ...
        test/
          ...

  MMS_normalized/
    manifests/
      patients.csv
    processed/
      2d_1p5mm_256/
        vendor_A_Siemens/
          metadata.jsonl
          ...
        vendor_B_Philips/
          metadata.jsonl
          ...
        vendor_C_GE/
          metadata.jsonl
          ...
        vendor_D_Canon/
          metadata.jsonl
          ...
```

具体的 `.npy` 文件可以位于各自 processed 子目录下，只要 `metadata.jsonl` 中的 `image` 和 `mask` 字段能够正确指向这些文件即可。

## 3. ACDC 目录要求

ACDC 根目录通过训练脚本参数 `--dataset-root` 指定，例如：

```bash
--dataset-root /data/Cardiac/ACDC_normalized
```

该目录下必须包含：

```text
ACDC_normalized/
  manifests/
    patients.csv
  processed/
    2d_1p5mm_256/
      metadata.jsonl
```

`manifests/patients.csv` 至少需要包含以下列：

- `patient_id`
- `group`
- `output_split`

训练时只会使用 `output_split == train` 的病例。当前协议默认 ACDC train split 为 80 个病例。`TrainLoader` 会根据 `seed` 在每个 `group` 内随机选择指定数量的有标注病例，其余 train 病例作为无标注病例。

`processed/2d_1p5mm_256/metadata.jsonl` 中每一行至少需要包含：

- `slice_id`
- `patient_id`
- `image`
- `mask`
- `phase`
- `z_index`
- `output_split`

其中 `image` 和 `mask` 可以是相对于 `ACDC_normalized` 的路径，也可以是绝对路径。推荐使用相对路径，便于不同服务器迁移。

## 4. MMS 目录要求

MMS 根目录通过训练脚本参数 `--target-dataset-root` 指定，例如：

```bash
--target-dataset-root /data/Cardiac/MMS_normalized
```

该目录下必须包含：

```text
MMS_normalized/
  manifests/
    patients.csv
  processed/
    2d_1p5mm_256/
      vendor_A_Siemens/
        metadata.jsonl
      vendor_B_Philips/
        metadata.jsonl
      vendor_C_GE/
        metadata.jsonl
      vendor_D_Canon/
        metadata.jsonl
```

`manifests/patients.csv` 至少需要包含以下列：

- `patient_id`
- `vendor`
- `vendor_name`
- `domain`

当前代码支持的 vendor/domain 对应关系为：

| Vendor 参数 | Vendor name | Domain directory |
| --- | --- | --- |
| `A` | `Siemens` | `vendor_A_Siemens` |
| `B` | `Philips` | `vendor_B_Philips` |
| `C` | `GE` | `vendor_C_GE` |
| `D` | `Canon` | `vendor_D_Canon` |

每个 vendor 目录下的 `metadata.jsonl` 每一行至少需要包含：

- `slice_id`
- `patient_id`
- `image`
- `mask`
- `phase`
- `z_index`

推荐额外保留以下字段，便于检查和记录：

- `vendor`
- `vendor_name`
- `domain`
- `pathology`
- `has_fg`
- `rv_pixels`
- `myo_pixels`
- `lv_pixels`

`TestLoader` 会校验 `MMS_normalized/manifests/patients.csv` 中的病人列表与对应 vendor 的 `metadata.jsonl` 是否一致。

## 5. 在服务器上运行训练

假设仓库克隆到：

```bash
/home/user/TTA_Project
```

并且数据放在：

```bash
/data/Cardiac/ACDC_normalized
/data/Cardiac/MMS_normalized
```

推荐从仓库根目录运行：

```bash
cd /home/user/TTA_Project

python Research/backbone_training_MeanTeacher.py \
  --dataset-root /data/Cardiac/ACDC_normalized \
  --target-dataset-root /data/Cardiac/MMS_normalized \
  --output-root /home/user/TTA_Project/Research/backbone_params
```

默认行为：

- 训练 `patients = 1,2,3`
- 训练 `seeds = 0,1,2,3,4`
- 每个设置训练 200 epochs
- 每个 backbone 训练结束后，在 MMS `A,B,C,D` 四个 vendor 上做 patient-wise 源模型评估

输出会保存到：

```text
Research/backbone_params/
  Patient1/
    Seed0/
      baseline_model_with_metadata.pt
      checkpoint_final.pt
      training_log.csv
      run_config.json
      eval_source_model/
        eval_summary.csv
        patient/
          vendor_A_Siemens/
          vendor_B_Philips/
          vendor_C_GE/
          vendor_D_Canon/
```

## 6. Smoke test

第一次在服务器上部署时，建议先跑一个极小测试，确认路径和依赖都正常：

```bash
cd /home/user/TTA_Project

python Research/backbone_training_MeanTeacher.py \
  --dataset-root /data/Cardiac/ACDC_normalized \
  --target-dataset-root /data/Cardiac/MMS_normalized \
  --output-root /home/user/TTA_Project/Research/backbone_params \
  --patients 1 \
  --seeds 0 \
  --epochs 1 \
  --max-steps-per-epoch 1 \
  --eval-vendors A \
  --max-eval-patients 1 \
  --overwrite
```

如果只想检查训练，不想做 MMS 评估，可以加：

```bash
--skip-eval
```

## 7. 常见路径问题

### 问题 1: 服务器不是 Windows，默认路径不能用

代码里保留了本机默认路径：

```text
D:\Running_Place\PostGraduateProjects\dataset\Cardiac\ACDC_normalized
D:\Running_Place\PostGraduateProjects\dataset\Cardiac\MMS_normalized
```

服务器上不要依赖这些默认值，直接使用：

```bash
--dataset-root /your/server/path/ACDC_normalized
--target-dataset-root /your/server/path/MMS_normalized
```

### 问题 2: 找不到 `.npy` 文件

检查 `metadata.jsonl` 中的 `image` 和 `mask` 字段：

- 如果是相对路径，它们必须相对于数据集根目录成立。
- 如果是绝对路径，它们必须是服务器上的真实路径。

推荐使用相对路径，例如：

```json
{"image": "processed/2d_1p5mm_256/train/patient001_ED_z00_image.npy", "mask": "processed/2d_1p5mm_256/train/patient001_ED_z00_mask.npy"}
```

### 问题 3: MMS vendor 校验失败

如果报错类似 metadata/manifest mismatch，检查：

- `MMS_normalized/manifests/patients.csv` 中该 vendor 的 `patient_id`
- 对应 `processed/2d_1p5mm_256/vendor_* /metadata.jsonl` 中的 `patient_id`
- 两边病人集合必须一致

### 问题 4: 输出目录没有写权限

训练会保存模型参数和日志。请确认 `--output-root` 指向的目录可写，并且磁盘空间足够。完整训练会生成多个 `.pt` checkpoint，不建议把这些权重提交到 GitHub。

## 8. 最小数据路径检查脚本

如果需要在正式训练前单独检查 loader，可以在仓库根目录运行：

```bash
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path("Research").resolve()))

from helper.dataloader import TrainLoader, TestLoader

acdc = Path("/data/Cardiac/ACDC_normalized")
mms = Path("/data/Cardiac/MMS_normalized")

train_loader = TrainLoader(
    labeled_cases_per_class=1,
    seed=0,
    batch_size=16,
    labeled_batch_size=8,
    dataset_root=acdc,
)

print("ACDC labeled patients:", len(train_loader.labeled_patients))
print("ACDC unlabeled patients:", len(train_loader.unlabeled_patients))
print("ACDC labeled slices:", train_loader.labeled_slice_count)
print("ACDC unlabeled slices:", train_loader.unlabeled_slice_count)

for vendor in ["A", "B", "C", "D"]:
    test_loader = TestLoader(vendor, batch_size=4, dataset_root=mms)
    print(vendor, test_loader.domain, "patients:", len(test_loader.patient_ids), "slices:", test_loader.slice_count)
PY
```

请把脚本中的 `/data/Cardiac/ACDC_normalized` 和 `/data/Cardiac/MMS_normalized` 替换为服务器实际路径。
