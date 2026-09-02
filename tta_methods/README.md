# 测试时适配方法说明

本目录实现统一协议下的八种测试时适配方法。所有方法直接继承 `BaseTTA`，共享相同源 checkpoint、目标流、arrival batch、timing 和 reset policy。

## 公共接口

`tta_methods/__init__.py` 暴露 `METHODS` 注册表和 `build_method`。每个方法必须实现：

```python
setup()                 # 配置模型模式、可训练参数、优化器和方法状态
adapt(images)           # 只使用无标签图像执行一次逻辑更新
predict(images)         # 返回 logits，不修改适配状态
```

`BaseTTA.process_batch` 统一执行 `adapt_then_predict` 或 `predict_then_adapt`。不要在子类覆盖该函数。

`AdaptationResult` 记录 loss、已见样本/像素数、选择数、是否更新以及方法诊断。它还可以携带不序列化的临时 `probe_payload`；该 payload 只能在适配和预测完成后由评估层与标签对照，不能影响方法状态。随机方法只能使用自己的 `torch.Generator`；reset 必须恢复模型、teacher、memory、RBN buffer、计数器和随机数状态，并重建 optimizer。

## 方法概览

| 方法 | 主要更新与状态 |
|---|---|
| Source | 冻结模型，使用源域 BN running statistics，不适配 |
| TENT | 更新全部 BN affine，最小化像素熵 |
| EATA | TENT 参数范围，加可靠性/冗余过滤和 Vendor A Fisher 正则 |
| SAR | 排除 encoder layer4 的 BN affine，同一 batch 上执行两次 SAM 和恢复机制 |
| CoTTA | 更新全模型，维护 EMA teacher、source anchor、多尺度预测和随机恢复 |
| RoTTA | 用 RBN 替换 BN，维护 EMA teacher 与 CSTU memory |
| RoID | BN affine、soft-likelihood-ratio、certainty/diversity weighting 和源权重融合 |
| DeYO | BN affine、patch shuffle、前景像素 entropy/PLPD 过滤 |

正式 TENT profile 固定为：全部 BN affine、普通像素熵、每个 arrival batch 更新一步、`SGD(lr=6.25e-5, momentum=0.9, weight_decay=0)`；病人流使用 BS=4，随机切片流使用 BS=8。

正式 SAR profile 保留官方两阶段 SAM 机制：第一次反传只构造 sharpness perturbation，第二次筛选是第一次筛选的子集，随后由底层 `SGD(lr=6.25e-5, momentum=0.9, weight_decay=0)` 执行实际更新。运行记录额外报告两轮筛选伪标签的全像素准确率、真值前景像素准确率和筛选覆盖率；真实 mask 始终位于适配边界之外。

## SEG-MOD

EATA、RoTTA、RoID 和 DeYO 原本面向图像分类，本项目对密集分割采用锁定的 SEG-MOD：

- EATA：使用每张切片的空间均值类别概率作为冗余描述符；
- RoTTA：memory category 使用“空背景或主导前景类”，强增强只改变强度；
- RoID：使用切片级概率/熵聚合，并关闭容易受背景主导的 prior correction；
- DeYO：在原预测前景像素上计算 PLPD，使用 4×4 patch shuffle。

结果中同时保存 `profile_verified` 和 `profile_kind`，避免把 SEG-MOD 表述为分类论文的逐字复现。所有阈值固定在 `config.yaml`，不得使用目标域标签调参。

## 模块边界

- `common.py` 只放完全同语义的熵、BN surgery、optimizer、EMA 和状态辅助函数。
- SAM、Fisher、augmentation、RBN、memory、prior/weighting 和 PLPD 保留在所属方法目录。
- `tta_methods/` 禁止导入 `data.py`、`metrics.py` 或 `run_tta.py`。
- 普通 batch-stat 方法必须移除 BN running buffers；RoTTA 的 RBN buffer 必须可序列化并可 reset。

正式运行示例：

```bash
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python run_tta.py \
  --config config.yaml \
  --method tent \
  --source-seed 2022 \
  --vendors B C D \
  --device cuda
```

EATA 正式运行前需使用 `train_source.py --fisher` 生成与源 checkpoint 对应的 Fisher artifact。
