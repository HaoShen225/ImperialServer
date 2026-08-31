# 数据划分与目标流说明

本目录保存实验运行时使用的固定划分入口。所有数据集划分都在患者级完成；目标域只对既定患者集合的到达顺序做可复现打乱，不会重新划分患者。

## 文件说明

### `vendor_a_split.json`

定义 Vendor A 源域训练/验证划分的来源：

- 训练集：60 名患者；
- 验证集：15 名患者；
- 两者都来自 Vendor A 的 `Training/Labeled`；
- 五个源模型 seed 使用完全相同的患者划分。

文件通过 selector 引用 `data/splits/mms_vendor_a_to_bcd_tta.json` 中的患者列表，避免维护两份可能不一致的 ID。

### `target_streams.json`

定义 Vendor B/C/D 的权威测试流规则：

1. vendor 顺序由实验命令或 `config.yaml` 指定；
2. 对每个 Vendor 独立使用 checkpoint seed 打乱合格患者列表；
3. 同一 seed 的 Source、TENT 和 SAR 使用完全相同的患者顺序；
4. 每名患者固定先 ED、后 ES，患者作为不可拆分的顺序单元；
5. 每个体积内部按 `z_index` 升序，最后一个不足四张的 arrival batch 仍保留；
6. `Training/Unlabeled` 在打乱前被排除，不参与适配或评分。

当前协议版本为 v3。Vendor C 只包含 Validation 10 名和 Testing 40 名患者，共 50 名患者；原始数据中的 25 名 `Training/Unlabeled` 患者仍保留在 manifest，但不属于测评序列。

实际完整患者列表保存在 `data/splits/mms_vendor_a_to_bcd_tta.json`。运行器会把完整协议文件、`target_streams.json` 的 SHA-256、顺序 seed、完整患者顺序及其 SHA-256 写入实验结果；`vendor_a_split.json` 作为受 Git 版本控制的源域 selector，指向同一个协议文件。

## 修改约束

- 不允许使用 Vendor B/C/D 标签调整患者顺序、方法超参数或 SEG-MOD 设计；顺序 seed 必须等于对应的 checkpoint seed。
- 修改患者列表、排除规则或顺序会改变实验协议，必须提升 JSON 的 `version` 并重新生成所有可比较结果。只有完全无更新的 source-only 记录可以使用 `reaggregate_source_results.py` 离线过滤，并且必须记录旧协议哈希。
- `patient`、`vendor`、`never` 是模型状态的 reset policy，不会改变这里定义的数据顺序。
- 正式独立域比较使用 A→B、A→C、A→D；continual 设置使用 A→B→C→D 且跨 vendor 不重置。

划分和批处理约束由以下测试保护：

```bash
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python -m pytest -q tests/test_data.py
```
