问题设定：
在目标域上的测试阶段，仅使用无标注的目标域数据对模型进行更新以达到更好的目标域性能。
各个协议下的严格要求：
| 类型                                                    |           测试时可见数据 | 是否跨 batch 累积 |       是否重置模型 | 典型目标                      | 特点                                                           | 代表方法/关键词                                 |
| ----------------------------------------------------- | ----------------: | -----------: | -----------: | ------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| **Single-sample TTA / Single-image TTA**              |         单张图像或单个样本 |            否 |          通常是 | 熵最小化、一致性、增强平均             | 最严格；不能依赖 batch statistics；适合医学图像单病例推理                        | TTT, MEMO, single-image TTA              |
| **Test-Time Batch Adaptation, TTBA**                  |     一个 mini-batch |            否 | 通常每 batch 重置 | BN 统计、熵最小化、一致性            | 介于 single-image 和 online 之间；不同 batch 预测相互独立                  | BN adaptation, TENT-episodic, MEMO       |
| **Test-Time Domain Adaptation, TTDA / SFDA-like TTA** |          整个目标域测试集 |  是，通常多 epoch |            否 | 伪标签、自训练、聚类、特征对齐           | 比 offline TTA 更接近 source-free domain adaptation；能利用目标域整体分布   | SFDA, SHOT-like, TeST                    |
| **Test-Time Prior Adaptation, TTPA**                  | 目标域输出分布或 batch 预测 |           可选 |           可选 | 调整类别先验 (p_t(y))           | 主要处理 label shift / class prior shift，不一定改 backbone           | Prior adaptation, label-shift correction |
| **Continual TTA / CTTA**                              |            非平稳连续流 |            是 |            否 | 熵、自训练、teacher-student、记忆库 | 目标分布随时间变化；核心问题是 error accumulation 和 catastrophic forgetting | CoTTA, EATA, SAR                         |
| **Open-world / Open-set TTA**                         |         目标域可能有未知类 |           可选 |           可选 | 拒识、异常检测、开放集校准             | 不假设目标类别完全等于源类别；更难                                            | Open-set TTA, universal TTA              |
| **Source-available TTA**                              |      测试时仍可访问部分源数据 |           可选 |           可选 | 源-目标对齐、正则保持               | 严格来说通常不算 fully TTA，更接近 UDA/online DA                         | Source-available adaptation              |
| **Source-free / Fully TTA**                           |      只有源模型和目标测试数据 |           可选 |           可选 | 熵、BN、伪标签、一致性              | 最标准、最严格的 TTA；不访问源数据和目标标签                                     | TENT, CoTTA, MEMO                        |
以下是Literature Review


<table>
  <thead>
    <tr>
      <th>编号</th>
      <th>Title</th>
      <th>Backbone</th>
      <th>TTA类型</th>
      <th>创新点</th>
      <th>对比的SoTA</th>
      <th>它的问题</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>TENT: Fully Test-Time Adaptation by Entropy Minimization (TENT)</td>
      <td>HRNet-W18 fully convolutional segmentation network，源域任务：全量监督训练。</td>
      <td>Fully TTA / Source-free TTA / Online batch TTA，参数更新范围：BN Affine，每张图单独估计statistic。</td>
      <td>Entropy minimization; BN affine update; target BN statistics</td>
      <td>Source-only, BN Adaptation, Pseudo-labeling, TTT, AugMix, ANT, ANT+SIN, SHOT</td>
      <td>I. 对伪标签不加筛选地反传，易放大高置信错误；<br>II. 只更新 BN affine，适应能力有限；<br>III. 小 batch 下 BN 统计不稳。</td>
    </tr>
    <tr>
      <td>2</td>
      <td>When Confidence Fails: Revisiting Pseudo-Label Selection in Semi-supervised Semantic Segmentation (CSL)</td>
      <td>ResNet backbone + DeepLabv3+ decoder, Xception-65</td>
      <td>不是TTA，是伪标签训练方法</td>
      <td>I. 指出了无标签训练中筛选可靠伪标签的重要性；<br>II. 从“只看最大置信度”改成“置信度分布可分性”；<br>III. 用谱松弛 / 凸优化思想做 sample-adaptive pseudo-label selection，避免调参；<br>IV. Gaussian smooth loss weight；<br>V. 不再硬丢弃低可靠像素，而是赋予边界像素较低的权重</td>
      <td>非TTA方法，无参考价值</td>
      <td>I. 正确伪标签的筛选仅看模型分割头自信程度，而忽略了更早出现distortion导致分割错误这种情况，可能出现‘自信的错’；<br>II. 其中的Disperssion Residual的准确估计过分依赖分类类别数量足够大，在寡类别（3-4类）的医疗影像分割任务中筛选出的伪标签有些噪（已经经过实验确认）；<br>III. 谱松弛算法计算开销过于巨大。<br>IV. 无显式伪标签纠错能力。</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation (InTEnt)</td>
      <td>U-Net + attention layers + middle block between encoder and decoder (修改了经典的U-Net架构)</td>
      <td>Single-sample TTA / Single-image TTA，参数更新范围是BN Statistics </td>
      <td>I. 提出‘single image TTA’ 协议；<br>II. 发现‘single image TTA’ 协议中BN statistic比 BN affine更重要；<br>III. 在计算目标域BN statistics时用多个moments更新，得到多个adapted model,用每个adapted model在扰动前后的posterior entropy之差（sharpness）或entropy给每个adapted model的posterior打分加权，集成多个moment更新下的预测结果。<br>IV. 不更新模型，天然不会出现模型负迁移。</td>
      <td>MEMO, TENT, SAR, FSeg, SITA</td>
      <td>I. 冻结模型全部参数且不更新模型，无显式接收新结构能力，是否有能力对抗强结构偏移存疑。<br>II. 选模信号仅取决于模型自信程度，而非结构/几何证据的合理性，可能出现‘自信的错’；<br>III. 只能单图推理，对于更常见的Volume Data处理效率低且天然不适配。</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Gradient Alignment Improves Test-Time Adaptation for Medical Image Segmentation (GraTA)</td>
      <td>ResUNet-34，源域任务是标准监督训练，全量源域数据用于训练。</td>
      <td>Source-free / Fully TTA（主实验为Single-image TTA），参数更新范围是BN affine，每张单图单独更新BN Statistic。</td>
      <td>I. 将‘梯度是否可靠’作为TTA方向准确判据；<br>II. 用强弱增强一致性损失与entropy minimization损失做梯度对齐，以保留正确的梯度方向；<br>III. 用强弱增强一致性损失与entropy minimization损失两个梯度之间的cosine similarity做step size规划，减小了冲突梯度对模型的影响</td>
      <td>DUA, DIGA, MedBN, TENT, DLTTA, DomainAdapter, SAR (并非原文的LN/GN版backbone), DeTTA</td>
      <td>I. 两项梯度分别对应强弱增强一致性损失与entropy minimization损失，不加筛选地计算两项损失则两个梯度都会带噪，不能默认带噪梯度的公共方向就是有益的；<br>II. 采用entropy minimization做auxiliary gradient，可能出现‘自信的错’的auxiliary，会带偏正确的梯度方向；<br>III. 强弱增强一致性损失依赖对侧视图的重增强，推理开销极大。</td>
    </tr>
    <tr>
      <td>5</td>
      <td>TOWARDS STABLE TEST-TIME ADAPTATION IN DYNAMIC WILD WORLD (SAR)</td>
      <td>ResNet50-LN/GN (源域模型用的就是GN/LN版)</td>
      <td>Source-free TTA</td>
      <td>I. 发现在wild-TTA场景（指测试流的domain随时间变化）下BN statistic是最大的不稳定因素；<br>II. 引入了可靠伪标签筛选机制，利用posterior vector entropy来筛选可靠伪标签，只用reliable mask内部的prediction做优化；<br>III. 可靠组内部由改为最小化局部邻域内最坏情况下的 entrop（两步Forward一步backward，第一次算梯度方向，在模型参数上加梯度方向*step算出最坏扰动；第二步用带最坏扰动的模型forward input，算出的梯度用来更新未经扰动的模型）；<br>IV. 防止posterior collapse（给所有posterior相同类别的高概率）：当输出熵低于阈值则回滚模型参数。</td>
      <td>TENT, EATA, TTT, MEMO, DDA</td>
      <td>I. 可靠性判定仅依靠posterior vector，‘自信的错’；<br>II. 仅利用可靠组内部梯度更新模型权重，造成大量梯度浪费。</td>
    </tr>
    <tr>
      <td>6</td>
      <td>ProDA / Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation (ProDA)</td>
      <td>DeepLabv2 + ResNet-101</td>
      <td>不是TTA，是伪标签训练方法</td>
      <td>I. 提出了伪标签训练中固定initial prediction作为锚点可以有效避免过漂移带来的负迁移；<br>II. 提出基于prototype的标签修正机制（取输入seg head 的原型计算每个posterior vector对应的prototype posterior，然后用这个prototype posterior调制initial prediction作为修正后的伪标签监督）；<br>III. prototype 在 mini-batch 中用 EMA 在线更新以适配目标域；<br>IV. 设计了Symmetric Cross Entropy用于吸收伪标签错误；<br>V. 原型一致性监督：约束同一张图扰动前后posterior vector和prototype指向性同时不变。</td>
      <td>非TTA方法，无参考价值</td>
      <td>I. 将全图的伪标签都纳入可监督范围，做法太硬，可能累计误差；<br>II. 原文默认一类一原型，天然不适配结构高度异质的医疗影像数据集。</td>
    </tr>
    <tr>
      <td>7</td>
      <td>PASS: Test-Time Prompting to Adapt Styles and Semantic Shapes in Medical Image Segmentation (PASS)</td>
      <td>2D U-Net with ResNet-34 backbone</td>
      <td>Source-free TTA (存疑)，Continual TTA</td>
      <td>I. 提出Input Decorator结构，用于将目标域的图像风格处理成源域风格；<br>II. 显式量化每类class的形状，用class shape prior作为目标约束；<br>III. 在Latent Space中维护一个shape prompt bank，其中存放源域中的shape prompt，测试时通过通过点积注意力查询相似度最高的shape prompt。</td>
      <td>PTBN, DUA, TENT, RN-CR, SAR, OCL, CoTTA, TIPI, ProSFDA, VPTTA, AdaMI, DAE, DPG</td>
      <td>I. 风格prompt在测试集上训练，当Batch size小时会过拟合（两个卷积层构成的style prompt自由度过高）；<br>II. 点积注意力的shape prompt bank默认源域/目标域病人体位/姿态差距不能过大，否则会失效（卷积层的平移等变性）；<br>III. shape prompt bank体量过于巨大，无异于直接存源域数据，可能不算严格的source-free TTA。</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Each Test Image Deserves A Specific Prompt: Continual Test-Time Adaptation for 2D Medical Image Segmentation (VPTTA)</td>
      <td>ResUNet-34 (OD/OC) / PraNet + Res2Net backbone (Polyp)</td>
      <td>Continual TTA</td>
      <td>I. 冻结模型，只训练Prompt模块，避免错误累积；<br>II. 提出Low-frequency Prompt，一个点乘Prompt调整输入图像的频域表示中的低频部分，模拟了传统图像处理中的风格偏移处理；<br>III. 建立Prompt Bank，其中存放历史的prompt，推理每张图片时检索历史prompt作为当前prompt的初始化；<br>IV. 提出风格偏移造成BN statistic偏移，用对齐BN statistics作为目标来优化prompt；<br>V. 提出 source-target statistics warm-up，让 prompt 从 easy-to-hard 逐渐对齐。</td>
      <td>TENT, CoTTA, DLTTA, DUA, SAR, DomainAdaptor</td>
      <td>I. 假设域偏移主要为风格偏移，对于非风格偏移无法处理（主要是结构/体位/形变这类）；II. 在复现实验中发现‘频域进行风格-结构解耦’策略过于粗糙，虽然可以大幅提升Dice，却造成了HD95暴涨，经过消融实验，发现造成HD95恶化的主要原因是 VPTTA 的动态 warm-up BN statistics机制。</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Mixture of Prototypes for Test-time Adaptive Segmentation (MoE)</td>
      <td>SegFormer-B5, DeepLabV2 + ResNet-101</td>
      <td>Continual TTA</td>
      <td>I. 将传统的prototypical TTA改写成了多原型范式，每个class对应多个原型（expert），测试时用weighted source prototypes估计目标域prototypes，适配class内结构更复杂的场景；<br>II. 测试域推理时冻结整个模型，仅优化一个小型的prototype gating network为每个类别加权得到目标域class prototype，用来算prototypical posterior；<br>III. 优化目标为最小化Prototypical Posterior Entropy。</td>
      <td>TENT, CoTTA, BECoTTA, SVDP, C-MAE</td>
      <td>I. Gating Network为全连接架构，要求默认每个class原型数一致，在医疗影像中不同class体积差异过大不满足这个假设；<br>II. Prototypical posterior与model posterior相加得到calibrated predictions，需要对bottleneck resolution的prototypical posterior进行重采样对齐model posterior的分辨率；<br>III. 无显式处理‘prototypical/model posterior冲突’的能力，在二者冲突时（即多数分割error发生的情况），优化方向随机，不一定能纠正错误；<br>V. 这篇论文中的experts是对全图生效的，即全图的每个class还是只能用一个prototypes，只不过这个prototyp是全部class prototype加权得到的。</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Continual Test-Time Domain Adaptation (CoTTA)</td>
      <td>WideResNet-28, ResNeXt-29, ResNeXt-50, SegFormer-B5, 源域任务：标准监督训练，全量源域数据用于训练backbone。</td>
      <td>Continual TTA，更新范围：整个模型。</td>
      <td>I. 首次明确Continual TTA的两大瓶颈：伪标签高噪和源域知识遗忘；<br>II. 提出了EMA teacher方法；<br>III. 提出了强弱增强一致性损失与weak-teacher-strong-student范式；<br>IV. 每次更新student model后随机将student model的部分参数替换为source model的，显式避免源域遗忘</td>
      <td>BN Stats Adapt, TENT, Test Aug (很少，因为这基本上是第一篇continual TTA 协议论文)</td>
      <td>I. 双模型交替优化，推理时资源消耗极大；<br>II. teacher-student之间的差不加过滤就加入损失噪声太大。</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Efficient Deformable Convolutional Prompt for Continual Test-Time Adaptation in Medical Image Segmentation (EDCP)</td>
      <td>Res34UNet, 3D U-Net</td>
      <td>Continual TTA</td>
      <td>I. 提出了deformable convolutional prompt：用一个offset可变的卷积层（kernel每个元素受到一对2个方向的offsets）作为prompt处理输入图片，将输入图像处理成backbone能看懂的形式；<br>II. 设计了offset predictor用于估计亚像素级像素偏移。</td>
      <td>TENT, CoTTA, SAR, VPTTA</td>
      <td>I. prompt会对输入图像进行空间上的偏移/形变，分割是在deform之后的图像上做的，这会导致输出的掩码也是deform后的版本，但是由于inverse deformation无法估计，输出的掩码可能较输入图像有形态失真；<br>II. 需要同时训练可变形卷积层和offset predictor两个模块，在小batch size下容易过拟合。</td>
    </tr>
    <tr>
      <td>12</td>
      <td>The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization (DUA)</td>
      <td>ResNet-26, WideResNet-40-2,</td>
      <td>Source-free TTA</td>
      <td>I. 采用动量更新的方式更新目标域BN统计量，动量随着迭代自适应地变化（指数衰减）；<br>II. 为适配small-batch TTA情形，用多版增强的目标域影像构建mini batch稳定训练。</td>
      <td>TTT, TENT, NORM</td>
      <td>I. 伪标签不加筛选；<br>II. ‘自信的错’。</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Source-Free Domain Adaptation for Medical Image Segmentation via Prototype-Anchored Feature Alignment and Contrastive Learning (SFDA)</td>
      <td>classic U-Net</td>
      <td>Test-Time Domain Adaptation</td>
      <td>I. 用 source classifier weights 作为 source prototypes；<br>II. TTA过程中的优化目标：每个 target pixel feature 靠近相似的 source prototype且每个 source prototype 都能在 target batch 中找到对应 target pixels，防止所有像素坍缩到背景/主导类别；<br>III. 将目标域像素嵌入向量PDF建模为GMM，类别先验为GMM weight，EM优化估计目标域中每个class的出现概率作为class prior probability；<br>IV. 利用高熵标签作为最小概率类别的负标签，与可靠标签同时放入损失做对比学习；<br>V. 目标域训练时将可靠预测对应的embedding拉向自身对应的原型。</td>
      <td>DPL, AdaMI, FSM</td>
      <td>I. 推理时需配置的超参数过多；<br>II. 每类单原型，不适合前景结构复杂的场景；<br>III. 并非Online TTA。</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Prototype bank-driven test-time adaptation for medical ultrasound image segmentation (PBTTA)</td>
      <td>U-Net</td>
      <td>continual TTA</td>
      <td>I. 动量更新BN统计量；<br>II. 用高置信像素构造当前图像的类别原型，作为目标域可靠组；<br>II. 将原型后验与模型后验加权相加融合，以修正预测。</td>
      <td>TENT, CoTTA, SAR, DomainAdaptor, MedBN, DIGA, VPTTA</td>
      <td>I. 置信组划分仅看p_max，‘自信的错’；<br>II. 推理时超参数过多，配置麻烦。</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Dynamically Instance-Guided Adaptation: A Backward-free Approach for Test-Time Domain Adaptive Semantic Segmentation (DIGA)</td>
      <td>DeepLabV2 + ResNet101</td>
      <td>Continual TTA</td>
      <td>I. 冻结全部模型参数，仅在推理时用动量更新BN statistics，避免负迁移；<br>II. 利用p_max高的前景预测掩码提取目标域class prototypes，并用动量更新源域prototype bank，用于估计目标域原型；<br>III. 用目标域原型计算prototypical posterior，与model posterior融合以修正模型预测。</td>
      <td>TENT, EATA, IN, DUA, SITA</td>
      <td>I. 默认每个class只有单个原型，不适配复杂前景结构；<br>II. 选择可靠标签只靠p_max，‘自信的错’；<br>III. 全部融合和thresholding采取硬阈值操作1，超参数过多；<br>IV. 仅调BN statistics，等于默认源域与目标域之间仅存在风格偏移，对结构漂移无效。</td>
    </tr>
    <tr>
      <td>16</td>
      <td>DLTTA: Dynamic Learning Rate for Test-Time Adaptation on Cross-Domain Medical Images (DLTTA)</td>
      <td>U-Net (2/3-D), DenseNet-121 pretrained with ImageNet101</td>
      <td>Continual TTA</td>
      <td>I. 将历史样本的bottleneck feature存进memory bank，在推理时查询当前图像与memory bank中feature map之间的L2距离，将L2距离最小的K个feature map当作suppory；<br>II. 平均support对应的model posterior得到reference prior（依旧是逐位置平均）；<br>II. 设计了对称KL散度函数并用它来衡量model prediction与reference prior之间的差异；<br>III. 根据差异计算TTA时的步长，差异大步长大，反之步长小。</td>
      <td>PTBN, TTT, TENT, ATTA, UDA</td>
      <td>I. 逐位置计算语义相似度，默认语义对齐是空间层面的，当support和target体位/姿态不同则查询失效，且会引入空间位置相关的错误归纳偏置；<br>II. KL散度的计算也是空间对齐，同理。</td>
    </tr>
    <tr>
      <td>17</td>
      <td>DomainAdaptor: A Novel Approach to Test-time Adaptation (DomainAdapter)</td>
      <td>ResNet18, ResNet50</td>
      <td>Source-free TTA</td>
      <td>I. 动量更新BN statistic，动量由source-batch、image-batch 统计距离自动生成；<br>II. 把 source statistics 隐式吸收到 BN affine 中，避免直接 finetune 混合统计量时出现 weight-statistics mismatch；<br>III. 通过 temperature scaling 软化高置信预测，让高置信样本也产生足够梯度，提高reliable mask内部信息利用率。</td>
      <td>DeepAll, Ada BN, ARM, SLR, TENT, LAME</td>
      <td>I. 损失函数还是熵最小化，无显式纠错能力； <br>II. 梯度仅来自固定可靠组，结构覆盖受限，不一定能够扩大覆盖结构。</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Efficient Test-Time Model Adaptation without Forgetting (EATA)</td>
      <td>ResNet26, ResNet50, ResNet101, ResNet152</td>
      <td>Continual TTA (非分割而是分类)</td>
      <td>I. 只用可靠且非冗余样本更新，利用硬阈值筛选低熵的prediction作为可靠标签；<br>II. 冗余筛选：保存历史目标域posterior vector，用余弦相似度表征当前样本的posterior和历史样本的最大相似度，并设置硬阈值筛选低相似度样本（非冗余样本）；<br>III. 设计了Fisher anti-forgetting，用于防止源域遗忘及负迁移。</td>
      <td>非TTA分割，无参考价值。</td>
      <td>I. 正确预测基本上余弦相似度都很高，用余弦相似度会滤掉太多监督信号，梯度信号太弱； <br>II. 可靠性判定只看model posterior，‘自信的错’。</td>
    </tr>
    <tr>
      <td>19</td>
      <td>MedBN: Robust Test-Time Adaptation against Malicious Test Samples (MedBN)</td>
      <td>ResNet26, ResNet50</td>
      <td>Source-free TTA</td>
      <td>I. 为防止BN Statistic的均值被极端值带偏，将BN Normalization的均值替换成了中位数。</td>
      <td>TeBN, TENT, ETA / EATA, SAR, SoTTA, sEMA, mDIA</td>
      <td>I. 医疗影像中embedding中位数太容易偏（由于不同影像中背景组分可能不同，它们的embedding如何分布不好预测），可能还不如直接用mean。</td>
    </tr>
    <tr>
      <td>20</td>
      <td>MEMO: Test Time Robustness via Adaptation and Augmentation (MEMO)</td>
      <td>ResNet26, ResNet50, ResNet101, WSL ResNeXt-101</td>
      <td>Source-free TTA (主实验为Single-sample TTA)</td>
      <td>I. 提出了单图多增强一致性约束，即不止约束单次增强前后的model posterior一致，也要求多个增强版本预测结果一致；<br>II. 设计了Marginal Entropy Loss，最小化该损失可同时使预测自信并且多增强预测一致。</td>
      <td>TENT, TTT, Robust Training系列</td>
      <td>I. 本质上还是只看分割头的posterior，‘自信的错’。</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Uncertainty Reduction for Model Adaptation in Semantic Segmentation (UncertaintyReduction)</td>
      <td>DeepLabV2 + ResNet-101 + ASPP</td>
      <td>Test-Time Domain Adaptation</td>
      <td>I. 多个 auxiliary decoders 做 dropout consistency：推理时通过多版本drop-out增强Feature map，输入多个辅助分割头并要求辅助分割头输出与主分割头（冻结）一致，迫使embeddings远离不确定边界；<br>II. pseudo labeling：用p_max做可靠性判据，筛选出可靠组，在可靠组内部计算CE损失作为语义锚点。</td>
      <td>AdaptSegNet, AdvEnt, FCAN, CBST, MRKLD, LRENT, CBST, MRKLD, LRENT</td>
      <td>I. 本质上还是只看分割头的posterior，‘自信的错’。<br>II. 仅靠p_max筛选可靠组，可靠组内过噪；<br>III. 不同class用不同的threshold，超参数太多，不够实用；<br>IV. 多个分割头提供dropout consistancy，资源浪费严重。</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Towards Better Stability and Adaptability: Improve Online Self-Training for Model Adaptation in Semantic Segmentation (DTST)</td>
      <td>DeepLabV2 + ResNet-101, VGG-16 + ASPP</td>
      <td>Test-Time Domain Adaptation</td>
      <td>I. 指出domain adaption失稳是由‘不合时宜的teacher model更新’导致的；<br>II. 用SND (soft neighbourhood density) 估计整张图的posterior是否成簇，整张图的预测是否可靠，当SND上升说明student model向正向进化了，此时可以吸收student model；<br>III. 用类别平均后验概率监测是否存在劣势class，为防止劣势class被忽略，提出了Training-Consistency based Resampling策略，即将带有可靠伪标签的目标域中的对应的类别前景贴到带分割图像中，并同时将对应的伪标签作为目标的一部分加强训练模型区分劣势类别的能力。</td>
      <td>HCL, UDA, SDF</td>
      <td>I. 本质上还是只看分割头的posterior，‘自信的错’。<br>II. 仅靠p_max筛选可靠组，可靠组内过噪；<br>III. TCR策略直接将其他图贴到当前的图上，有可能构造出不合理的结构。</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Tree Energy Loss: Towards Sparsely Annotated Semantic Segmentation (Energy Loss Tree)</td>
      <td>DeepLabV3 + ResNet-101, HRNet-W48</td>
      <td>稀疏标签训练（不是TTA）</td>
      <td>I. 提出了深层/浅层特征两颗树的图建模方式，浅层树以全图token/像素为节点，以像素/token与四邻像素的灰度距离/特征相似度为边权重，构造Minimum Span Tree；<br>II. 用级联高层/底层两个树滤波实现稀疏标注的全图传递，构建高精度伪标签，实现了‘利用灰度特征补齐深层特征的失真’这一目的，对于边界结构发现有奇效（不再是只相信输入分割头的特征且不受原型量化噪声的影响）；III. 四临边建树，O(HW)且同样能够实现全图的信息传递。 </td>
      <td>不是TTA，无参考价值。</td>
      <td>I. ‘跨边界即不同类’的假设太强。<br>II. ‘稀疏标签可用’的假设下，可能部分的‘树枝岛’无可靠标签监督，这提醒我们在做方法迁移时要采取更软的策略处理可靠区域外部标签或者干脆全局软加权；<br>III. 所有伪标签权重一致，并没有更加相信可靠部分，忽略不可靠部分；<br>IV. 直接级联高低两层树滤波，做法太硬，相当于仅保留两棵树的图上交集，且引入了更大的路径衰减。</td>
    </tr>
  </tbody>
</table>



