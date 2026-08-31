# 测试说明

本目录验证工作台的工程正确性和科学实验不变量。除数据协议测试外，大多数测试使用 `conftest.py` 中的小型四分类 BN 分割网络，因此不需要训练完整 ResUNet-34。

## 测试文件

| 文件 | 主要覆盖内容 |
|---|---|
| `test_data.py` | 患者划分无重叠、B/C/D 目标流、C 无标注排除、切片顺序、尾 batch、mask 延迟加载 |
| `test_metrics.py` | 3D Dice、`HD95_px`、缺失类别有限对角线惩罚、患者级 bootstrap |
| `test_model.py` | ResUNet-34 输出形状、随机初始化可复现性、decoder BN、CE+Dice 和 AdamW 参数组 |
| `test_protocol.py` | 无标签 `run_volume` 边界、空标签防护、mask permutation immunity、两种 timing |
| `test_reaggregate.py` | source-only 离线过滤、派生哈希记录、拒绝自适应结果 |
| `test_methods.py` | 八方法构造/更新、BN policy、trainable scope、随机 reset replay、RoTTA RBN |
| `test_state.py` | 方法导入隔离和未知配置键立即报错 |

## 运行方式

使用项目现有的 `TTA` conda 环境：

```bash
cd /rds/general/user/hs225/home/mms_TTA
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python -m pytest -q
```

只运行某一类门禁：

```bash
# 数据与 split
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python -m pytest -q tests/test_data.py

# TTA 方法生命周期
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python -m pytest -q tests/test_methods.py

# 指标约定
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python -m pytest -q tests/test_metrics.py
```

涉及 `data/processed/` 的测试需要本地 MMS `.npy` 数据存在。合成方法测试只依赖 PyTorch，不读取目标 mask 来执行适配。

## 源训练 smoke test

完整单元测试通过后，可用真实 Vendor A 数据执行短 CPU smoke：

```bash
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python train_source.py \
  --config config.yaml \
  --seed 2022 \
  --device cpu \
  --smoke-test
```

该命令只训练两个 batch、验证一个体积，并检查 loss 有限、参数确实更新、checkpoint 可重载且重载 logits 一致。生成物写入 `checkpoints/Stochastic_Ini/smoke/`，不会提交到 Git。

提交代码前至少运行完整 `pytest -q` 和 `git diff --check`。任何涉及协议、状态或方法更新范围的修改，都应同时添加对应的回归测试。
