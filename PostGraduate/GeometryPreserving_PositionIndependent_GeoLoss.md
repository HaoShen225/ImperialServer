# Geometry-Preserving Loss 的 Position-Independent 改进方案

## 背景动机

当前 Geometry-Preserving RadialGate 使用 source GT mask 构造 signed-distance 权重：

$$
w(v)=\frac{\min(dist(v,\partial Y),d_{clip})}{d_{clip}}
$$

然后约束风格校准后的 student foreground probability 不要偏离 frozen SADG teacher：

$$
L_{geo}
=
\frac{
\sum_v w(v)\left|P_\theta^{fg}(v)-P_0^{fg}(v)\right|
}{
\sum_v w(v)+\epsilon
}
$$

这个设计的优点是直观：靠近 GT 边界的位置允许变化，远离边界的稳定背景和前景内部不应被风格校准改变。

但它仍然是一个 pixel-wise spatial weighting。虽然它不是固定图像坐标 mask，而是 anatomy-relative mask，但如果 source foreground 位置分布较窄，loss 仍可能受到 source foreground spatial distribution 的影响。

因此可以考虑把几何约束从：

$$
\text{pixel-wise source-GT spatial mask}
$$

改成更弱位置依赖的形式：

$$
\text{distance-to-boundary / teacher-confidence / boundary-distribution constraint}
$$

核心目标不变：

> Style canonicalization should preserve anatomy geometry, but the constraint should not overfit source foreground positions.

---

## 方案一：Distance-Bin Balanced Geometry Loss

### 直觉

当前 loss 对所有像素直接加权求和，容易被大面积背景区域主导，也可能隐式依赖 source 前景在图像中的位置。

Distance-bin balanced loss 保留 signed-distance 的几何思想，但不直接按像素位置累计，而是先按“到结构边界的距离”分桶。

也就是说，我们不关心某个点在图像左上角还是右下角，只关心它属于哪类几何区域：

- 边界附近；
- 中等距离区域；
- 远离边界的稳定背景或前景内部。

### 定义

给定 GT foreground union：

$$
Y_{fg}(v)=\mathbb{1}[y(v)>0]
$$

计算 foreground boundary：

$$
\partial Y
$$

以及像素到 boundary 的距离：

$$
d(v)=dist(v,\partial Y)
$$

将距离划分为若干 bins：

$$
B_k=\{v\mid d(v)\in [a_k,a_{k+1})\}
$$

在每个 bin 内计算 student 与 teacher 的 foreground probability 差异：

$$
L_k
=
\frac{1}{|B_k|+\epsilon}
\sum_{v\in B_k}
\left|P_\theta^{fg}(v)-P_0^{fg}(v)\right|
$$

最后对所有 bins 均匀或加权聚合：

$$
L_{geo}^{bin}
=
\sum_k \alpha_k L_k
$$

其中最简单设置是：

$$
\alpha_k=\frac{1}{K}
$$

### 优点

- 保留“远离边界更稳定”的几何直觉。
- 减少大背景面积对 loss 的主导。
- 比 pixel-wise SDT weighting 更少依赖 source foreground 的具体图像位置。
- 解释清楚，适合作为当前方法的自然升级。

### 缺点

- 仍然使用 source GT boundary。
- 需要选择 distance bins。
- 如果某些 bin 像素很少，需要处理空 bin。

### 推荐超参数

可以先用固定 bins：

$$
[0,2),[2,4),[4,8),[8,16),[16,32),[32,\infty)
$$

或者在代码中设置：

```text
geo_bin_edges_px = 0,2,4,8,16,32
```

对于最后一档：

$$
d(v)\ge 32
$$

单独作为 far-region bin。

---

## 方案二：Teacher-Confidence Geometry Loss

### 直觉

当前 \(w(v)\) 来自 GT distance map。另一种更 position-independent 的做法是让 frozen SADG teacher 自己决定哪些位置可靠。

如果 teacher 对一个位置非常确定，那么 student 在风格校准后不应该大幅改变该位置的 foreground probability。

如果 teacher 本身不确定，例如边界附近或模糊区域，就降低约束强度。

### 定义

teacher foreground probability：

$$
P_0^{fg}(v)
=
\sum_{c>0}softmax(f_0(x))_c(v)
$$

student foreground probability：

$$
P_\theta^{fg}(v)
=
\sum_{c>0}softmax(f_\theta(C_\phi(x)))_c(v)
$$

一种简单置信度权重是：

$$
w_{conf}(v)
=
\left|P_0^{fg}(v)-0.5\right|
$$

也可以用 entropy：

$$
w_{conf}(v)
=
1-H(P_0(v))
$$

然后：

$$
L_{geo}^{conf}
=
\frac{
\sum_v w_{conf}(v)
\left|P_\theta^{fg}(v)-P_0^{fg}(v)\right|
}{
\sum_v w_{conf}(v)+\epsilon
}
$$

更稳的版本是只在 teacher 高置信区域计算：

$$
\Omega_{conf}
=
\{v\mid P_0^{fg}(v)>\tau_{fg}\}
\cup
\{v\mid P_0^{fg}(v)<\tau_{bg}\}
$$

例如：

$$
\tau_{fg}=0.8,\quad \tau_{bg}=0.2
$$

于是：

$$
L_{geo}^{conf}
=
\frac{
\sum_{v\in\Omega_{conf}}
w_{conf}(v)
\left|P_\theta^{fg}(v)-P_0^{fg}(v)\right|
}{
\sum_{v\in\Omega_{conf}}w_{conf}(v)+\epsilon
}
$$

### 优点

- 不直接使用 GT distance map 作为 spatial mask。
- 更像 teacher-student geometry consistency。
- 对 source foreground position distribution 的理论负担更小。
- 实现简单。

### 缺点

- teacher 如果有远处 false positive，student 可能被迫模仿。
- 如果 teacher 对 target-like hard cases 本身不稳，约束可能限制 student 改善。
- 需要设置 confidence threshold。

### 推荐超参数

```text
geo_conf_fg_thresh = 0.8
geo_conf_bg_thresh = 0.2
geo_conf_weight = abs_fg_prob
```

可以先比较：

1. no threshold；
2. high-confidence threshold；
3. entropy-based confidence。

---

## 方案三：Boundary-Distribution Consistency

### 直觉

前两个方案仍然是逐像素约束。第三个方案更进一步：不要求 student 和 teacher 在每个像素都一致，而是要求它们的结构边界分布一致。

这更像在约束 shape/boundary geometry，而不是约束 foreground probability map。

### Soft Boundary Map

对 foreground probability 取梯度幅值：

$$
B(P)=|\nabla P|
$$

例如：

$$
B(P)
=
|\nabla_x P|+|\nabla_y P|
$$

student boundary：

$$
B_\theta(v)=B(P_\theta^{fg})(v)
$$

teacher boundary：

$$
B_0(v)=B(P_0^{fg})(v)
$$

最直接的 boundary consistency 是：

$$
L_{bd}
=
\|B_\theta-B_0\|_1
$$

### Distance-Weighted Boundary Mass

也可以不逐像素对齐 boundary，而是约束 boundary mass 相对 GT boundary 的距离分布。

给定：

$$
d_y(v)=dist(v,\partial Y)
$$

计算 student 的 distance-weighted boundary mass：

$$
M_\theta
=
\sum_v d_y(v)B_\theta(v)
$$

teacher 的 distance-weighted boundary mass：

$$
M_0
=
\sum_v d_y(v)B_0(v)
$$

约束：

$$
L_{bd-dist}
=
\left|M_\theta-M_0\right|
$$

更稳的版本是归一化：

$$
\bar{M}_\theta
=
\frac{
\sum_v d_y(v)B_\theta(v)
}{
\sum_v B_\theta(v)+\epsilon
}
$$

$$
\bar{M}_0
=
\frac{
\sum_v d_y(v)B_0(v)
}{
\sum_v B_0(v)+\epsilon
}
$$

然后：

$$
L_{bd-dist}
=
\left|\bar{M}_\theta-\bar{M}_0\right|
$$

### 优点

- 更接近“边界几何一致性”。
- 不强制 student 逐像素模仿 teacher。
- 对小幅位置变化更宽容。
- 适合解释 HD95，因为 HD95 本质上依赖边界距离。

### 缺点

- 实现和调参更复杂。
- 梯度边界图可能受 softmax 平滑程度影响。
- 如果只约束 boundary distribution，可能无法充分抑制内部孔洞或远处小 FP。

### 推荐超参数

```text
geo_boundary_type = sobel_l1
lambda_boundary = 0.05 or 0.1
geo_boundary_distance_clip_px = 32
```

第一版建议先实现最简单的：

$$
L_{bd}=\|B(P_\theta^{fg})-B(P_0^{fg})\|_1
$$

如果效果不稳，再做 distance-weighted boundary mass。

---

## 三个方案对比

| 方案 | 是否用 source GT | 是否逐像素 | Position dependence | 实现难度 | 推荐程度 |
|---|---:|---:|---:|---:|---:|
| Distance-bin balanced SDT | 是 | bin 内平均 | 低 | 中 | 高 |
| Teacher-confidence | 否，或弱依赖 | 是 | 中低 | 低 | 高 |
| Boundary-distribution | 可选 | 可逐像素或分布式 | 低 | 中高 | 中 |

---

## 推荐实验矩阵

为了保持 ICASSP 短文故事清晰，建议不要一次性把三个都混起来。先做如下矩阵：

```text
SADG
LF-Barycenter-RadialGate
GeoPres-SDT-PixelWeighted        # 当前版本
GeoPres-SDT-BinBalanced          # 方案一
GeoPres-TeacherConfidence        # 方案二
GeoPres-BoundaryConsistency      # 方案三
```

如果时间有限，优先做：

```text
GeoPres-SDT-PixelWeighted
GeoPres-SDT-BinBalanced
GeoPres-TeacherConfidence
```

因为这两种替代方案最容易实现，也最容易解释。

---

## 推荐论文叙事

当前版本可以描述为：

> We use source GT signed-distance maps to weight geometry consistency, encouraging the canonicalized model to preserve stable foreground/background regions while allowing uncertainty near anatomical boundaries.

升级后的 position-independent 版本可以描述为：

> To avoid encoding source-specific foreground locations, we aggregate geometry consistency over distance-to-boundary bins or teacher-confidence regions, making the constraint depend on structural reliability rather than absolute spatial coordinates.

最推荐的主线是：

$$
\text{Pixel-weighted SDT}
\rightarrow
\text{Distance-bin balanced SDT}
$$

因为它是当前方法的最自然升级：保留 geometry-preserving 直觉，同时减少 source foreground spatial bias。

