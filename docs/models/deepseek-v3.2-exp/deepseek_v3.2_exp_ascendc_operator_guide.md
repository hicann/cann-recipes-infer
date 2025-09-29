# NPU DeepSeek-V3.2-Exp Ascend C 融合算子优化


面向 DeepSeek-V3.2-Exp 架构，本次发布两个全新的融合算子：`LightningIndexer` (LI) 以及`SparseFlashAttention` (SFA)。其中`LightningIndexer`算子是本次 DeepSeek 新架构中最特别的部分之一，用于对重要的上下文进行稀疏选择，当前 Top-$k$ 计算已经融入`LightningIndexer`算子，相较于 Top-$k$ 外置收益大约20%；`SparseFlashAttention`则是针对稀疏 Attention 场景进行了特殊优化的 Attention 算子。

A3中的 Cube 核与 Vector 核数比为 $1:2$，每1个 Cube 核与2个 Vector 核构成一个组核 (一个 AICore)，其中一个组核的结构如下图所示，其中 L1、L0A、L0B、L0C 为 Cube 核的片上 Buffer，UB 为 Vector 核的片上 Buffer。

<div align="center">
    <img src="./figures/framework.png" width="950" />
</div>



在本篇技术报告中，我们将对两个融合算子在 A3 上的具体优化进行细致地阐述，包括：具体的算子 Tiling 设计、Cube 核与 Vector 核间的流水优化、计算流程的实现优化、以及下一步即将开展的优化等内容；同时将介绍当前融合算子在提升易用性上的进展。

## Highlights

- 细致的算子 Tiling 及 Pipeline 设计，发挥融合算子优势
- 充分利用硬件特性，基于 Ascend C API 高效实现
- 聚合访存，减少指令下发，提高离散访存性能
- 算子开发及调测的易用性提升

## Outline

- [LightningIndexer](#lightningindexer)
    - [LI 算子 Tiling 设计](#li-算子-tiling-设计)
    - [LI Pipeline 设计](#li-pipeline-设计)
    - [ReLU 随路实现](#relu-随路实现)
    - [Top-$k$ 指令实现](#top-kkk-指令实现)
    - [LI Next Feature](#li-next-feature)
- [SparseFlashAttention](#sparseflashattention)
    - [SFA 算子 Tiling 设计](#sfa-算子-tiling-设计)
    - [SFA Pipeline 设计](#sfa-pipeline-设计)
    - [访存优化](#访存优化)
    - [SFA Next Feature](#sfa-next-feature)
- [易用性提升](#易用性提升)
    - [TilingKey 模板化](#tilingkey-模板化)
    - [分核计算下沉](#分核计算下沉)
    

## LightningIndexer

`LightningIndexer`基于一系列操作得到每一个 token 对应的 Top-$k$ 个位置。对于某个 token 对应的 Index Query $Q_{index}\in\R^{g\times d}$，给定上下文 Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中 $g$ 为 GQA 对应的 group size，$d$ 为每一个头的维度，$S_{k}$ 是上下文的长度，`LightningIndexer`的具体计算公式如下：
$$
\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
$$

可拆分为如下计算流程：
1. 计算矩阵乘法：$S = Q_{index}@K_{index}^T$
2. 计算激活函数：$S'=\text{ReLU}(S)$
3. 计算广播乘法：$S_W=(W@[1]_{1\times S_{k}})\odot S'$
4. 沿G轴进行Reduce操作：$Score=[1]_{1\times g}@ S_W$
5. 对 $Score$ 进行 $\text{Top-}k$ 计算，即获取数值排序前 $k$ 个的结果，并返回其对应的 Index

在 Prefill 或开启 MTP 的 Decode 场景，多个 token 对应的 $Q_{index}$ 会被合并统一计算，本文后续部分不对单/多个 token 对应的 $Q_{index}$ 加以区分。

实际开发中需要充分利用昇腾硬件的特征以实现更好的性能，以下将详细介绍当前的解决方案及具体实现方式。

### LI 算子 Tiling 设计

算子的 Tiling 设计决定了储存空间是否被充分利用，进而影响整体的算力利用率以及算子性能。针对`LightningIndexer`，本章分别阐述核间和核内的 Tiling 设计。

由于片上 Buffer 空间有限，任何计算本质上都需要进行切块，因此需要确定一次迭代计算的基本块大小，即核间的 Tiling 设计。当前`LightningIndexer`算子将基本块大小设置为$(512,512)$：即一次迭代计算完成512行以减少右矩阵的重复搬运 (以$g=64$为例，一次会合并8个token统一计算)；同时会完成 $S_{k}$ 方向上粒度为512的计算，这个值的选取可以保证流水线头尾开销可控以及 Cube 核与 Vector 核之间同步开销可被计算掩盖。

核内 Tiling 设计采用了如下方案：
- L1 空间划分为如下部分：
  - $Q_{index}$ 矩阵常驻：$512\times 128\times2\text{ Bytes}=128\text{ KB}$，L1层级基本块为 $(512,128)$，在实际计算时可做更细致的搬运，即一次只搬运$(256,128)$大小的数据到L1以降低搬运头开销
  - $K_{index}$ 矩阵 3-Buffer 循环复用：$3\times256\times 128\times2\text{ Bytes}=192\text{ KB}$，L1层级基本块为 $(256,128)$
- L0A，L0B，L0C 使能 Double Buffer，分别划分为$2\times32\text{ KB},2\times32\text{ KB},2\times64\text{ KB}$，一次搬入 L0A 和 L0B 的矩阵块大小都为$(128,128)$，这是一种对昇腾硬件较为亲和的设置，可以提高算力利用率。

这样的方式可以保证在计算过程中 $Q_{index}$ 只需要搬运一次，并且降低 $K_{index}$ 矩阵的重复搬运次数：假设 $Q_{index}$ 矩阵 L1 层级基本块大小为 $(128,128)$，而行方向总的 token 数为 $S_q$，那么 $K_{index}$ 矩阵需要重复搬运 $\lceil S_q/128\rceil$ 次；选取基本块大小为 $(512,128)$ 可将重复搬运次数降低到 $\lceil S_q/512\rceil$。由于第一个计算过程矩阵乘的启动需要右矩阵的搬运，为右矩阵分配多块 Buffer 可消除计算流程因为 Buffer 占用造成的阻塞，所以为 $K_{index}$ 分配 3-Buffer 可以提升带宽爬坡期的搬运效率。


### LI Pipeline 设计

`LightningIndexer`的计算流程只有两个阶段：即矩阵乘和 ReLU 操作 ($C_1$) 以及后续的多个向量计算 ($V_1$)。为了防止多个阶段的串行执行，需要提前下发 (Preload) 一次 $C_1$ 以实现流水的互相掩盖。

<div align="center">
    <img src="./figures/LIpipeline.png" width="600" />
</div>

### ReLU 随路实现

`LightningIndexer`首先需要进行矩阵乘法，并对结果进行ReLU计算。由于矩阵乘法和 ReLU 计算分别发生在不同的计算单元，朴素的实现需要将矩阵乘法的结果从 Cube 核搬运至 Vector 核进行。而昇腾NPU上 Cube 核的 FixP 组件提供随路 ReLU 能力，即可以在矩阵乘法完成之后直接在数据搬运的过程中随路完成 ReLU 操作，不需要显式进行 ReLU 计算。除 ReLU 之外，FixP 组件还有更多强大功能，具体可参考[基础API: FixP相关](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0251.html)。

### Top-$k$ 指令实现

`LightningIndexer`融合算子的核心是在长达数十万的序列中，为每个 token 高效地筛选出分数最高的 $k$（例如2048）个索引。同时，对于算子而言，Top-$k$ 的计算必须是准确无误的，不能采用近似算法求解。

当前的实现方案基于昇腾支持的排序指令进行全量排序，top-$k$ 计算方案过程分为三步：  
- **分组排序**：通过`VBS32`指令将每32个 token 按照其 $Score$ 进行稳定降序排列，输出其排序向量以及对应索引向量，直到将整个序列的 token 分组排序完毕。  

<div align="center">
    <img src="./figures/topkstep1.png" width="600" />
</div>

- **归并**：通过`VMS4v2`指令将至多4组（可为2/3组）长度为32、128、512的已排序向量进行归并，直到合并后的向量长度达到2048。合并排列仍然按照其对应 $Score$ 进行稳定降序排列，最终输出的向量长度为4组输入向量长度之和，同时输出其对应索引向量。  
<div align="center">
    <img src="./figures/topkstep2.png" width="600" />
</div>

- **规约**：采用`VMS4v2`指令的耗尽模式 (当某个输入向量的可用数据已全部读取完毕，排序指令立即暂停执行)，将至多4组长度为2048已排序的向量进行归并。取出前 $k=2048$ 个参数组成的向量与另外未参与合并排序的有序向量重复进行 Top-$k$ 计算，直到将所有长度为2048的有序向量比较完毕，得到最终所有向量中得分最高的2048个对应的 Index。  

<div align="center">
    <img src="./figures/topkstep3.png" width="600" />
</div>

假设上下文长度为 $S_k$ ，**分组排序**与**归并**步骤的计算量为 $4S_k$。**规约**步骤的计算量无法精确评估，采用正态分布结合蒙特卡洛法模拟，结果显示第一次的计算量接近 $4k$ ，从第二次开始，由于已经有了 $k$ 个较大数的向量参与运算，计算量开始大幅减少直到逼近耗尽极限 $k$ 。该部分计算量经过模拟分析后可表示为 $0.36S_k+3.32k$ 。因此Top-$k$ 流程总计算量 $totalCount = 4.36S_k + 3.32k$

### LI Next Feature

针对`LightningIndexer`算子，后续将在算法和工程上持续优化，主要的方向是：
- `LightningIndexer`算子涉及到复杂的 Vector 计算，集中在赋权加和以及 Top-$k$ 计算，通过探索减少计算量的算法以降低耗时。
- `LightningIndexer`算子支持量化，减少访存量并充分利用 Cube 核的强大能力进行性能优化。

## SparseFlashAttention

随着大模型上下文长度的增加，Sparse Attention 的重要性与日俱增，这一技术通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。Sparse Attention 计算可表示为

$$
\text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
$$
其中 $\tilde{K},\tilde{V}$ 为基于某种选择算法 (如`LightningIndexer`) 得到的重要性较高的 Key 和 Value，一般具有稀疏或分块稀疏的特征，$d_k$ 为 $Q,\tilde{K}$ 每一个头的维度。


本次公布的`SparseFlashAttention`是面向 Sparse Attention 的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。`SparseFlashAttention`算子沿用 [FlashAttention](https://arxiv.org/pdf/2307.08691) 的计算流程，分为四个阶段：
- $C_1$：$Q@K^T$
- $V_1$：online softmax
- $C_2$：$P@V$
- $V_2$：rescaling $O$

### SFA 算子 Tiling 设计

`SparseFlashAttention`一次迭代计算的基本块大小为$(256,512)$。

算子核内 Tiling 设计如下：
- L1 空间划分为如下部分：
  - $Q/P$ 矩阵分时复用 ($P$矩阵是softmax计算的结果)：$4\times128\times 288\times2\text{ Bytes}=288\text{ KB}$，即在一个 $C_1$ 或 $C_2$ 阶段中，左矩阵分四次搬运到 L1 中，最终在一次迭代内常驻于 L1，其中 $Q$ 矩阵每次的搬运量大小为 $(128,288)$，$P$ 矩阵每次的搬运量大小为 $(128,256)$
  - $K/V$ 矩阵 3-Buffer 循环复用：$3\times128\times 288\times2\text{ Bytes}=216\text{ KB}$，其中$K$矩阵每次的搬运大小为$(128,288)$，$V$矩阵每次的搬运大小为$(256,128)$
- L0A，L0B，L0C使能 Double Buffer，分别划分为$2\times32\text{ KB},2\times32\text{ KB},2\times64\text{ KB}$，在 $C_1$ 阶段一次搬入 L0A 和 L0B的矩阵块大小分别为 $(128,96),(96,128)$；在 $C_2$ 阶段一次搬入 L0A 和 L0B 的矩阵块大小都为 $(128,128)$。


### SFA Pipeline 设计

对于复杂的融合算子而言，Cube 核和 Vector 核之间的高效协同是发挥硬件算力的核心挑战，流水掩盖作为连接二者的核心调度机制，其设计优劣直接决定了算子性能。


由于沿用 [AMLA](https://gitcode.com/ascend-tribe/ascend-inference-cluster/blob/main/OptimizedOps/AMLA-%E4%BB%A5%E5%8A%A0%E4%BB%A3%E4%B9%98%E7%9A%84%E9%AB%98%E6%80%A7%E8%83%BD%E6%98%87%E8%85%BEMLA%E7%AE%97%E5%AD%90.pdf) 算法，`SparseFlashAttention`除最终的归一化操作外只有三个计算阶段$C_1,V_1,C_2$。相较于`LightningIndexer`算子而言，`SparseFlashAttention`算子的计算流程更为复杂，依赖关系更多，朴素的流水排布会导致计算过程中出现大量的空泡。如下图的上半部分所示，由于计算流程中的不同阶段之间存在数据依赖，不论是 Cube 流或 Vector 流上都出现了很多的空闲部分，导致算力利用率较低。本算子采用 Preload 流水排布以消除依赖，实现了除头尾以外的平台期部分 Cube 流完美掩盖 Vector 流，效果如下图下半部分所示，可以看到这种方式相较于朴素的流水排布有较大的性能收益。

<div align="center">
    <img src="./figures/pipeline.png" width="1000" />
</div>

### 访存优化

实际的访存行为可以分为两个步骤：指令下发及数据搬运。对于非 Sparse 场景的 Attention 算子，数据访存一般是连续的，因此单次数据搬运耗时相较于指令下发耗时更长，可实现连续的多次访存，达到较高的访存带宽。然而对于 Sparse Attention 的离散访存而言，单次访存的耗时大幅下降，甚至可能小于指令下发的耗时，这就会导致严重的指令下发阻塞，如下图所示。

<div align="center">
    <img src="./figures/instruction_launch.png" width="800" />
</div>

Attention 计算从 Cube 核发起，访存指令也由 Cube 核发射。基于 A3 芯片 Cube 核和 Vector 核数比为 $1:2$ 的特性，利用 Vector 核发射访存指令，将指令的下发效率提高一倍。另一方面，单次访存的数据量本身也会影响实际达到的带宽，而离散访存单次的访存量较小，这会导致数据搬运本身的耗时高于预期。针对这一问题，可以将离散的数据点两两聚合，利用 [srcGap](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/ascendcopapi/atlasascendc_api_07_0103.html#ZH-CN_TOPIC_0000002449889097__table9182515919) 参数实现成对的数据访存，提升访存带宽，如图所示。

<div align="center">
    <img src="./figures/stride.png" width="800" />
</div>

### SFA Next Feature

后续将探索`SparseFlashAttention`算子的量化支持以应对长上下文场景下显存占用过大的挑战。


## 易用性提升
### TilingKey 模板化

TilingKey 是算子区分不同的实现而将代码进行区分的方法，目标是以不同的分支处理不同的特性。然而 TilingKey 本身不易于记忆和理解，因为它们通常是较长又没有明确含义的数字，严重影响代码的可读性和可编程性。

为了提高算子代码的可读性，降低开发难度，本次开源的两个全新算子均采用 Ascend C 的 Tiling 模板编程方法替代传统的 TilingKey 编程，只需要定义模板参数及其对应的组合即可实现分支管理，减少对 TilingKey 具体数值的依赖，提高代码的管理效率。详细内容可参考[Tiling模板编程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)。

### 分核计算下沉

Tiling 是算子执行前的预处理阶段，在此阶段 Host 侧会计算 workspace 占用以及分核策略等算子执行需要的内容，在单算子模式下，可能会出现 Host Bound 的情况。通过[静态图下沉技术](https://mp.weixin.qq.com/s?__biz=MzA4MzM0ODMyMA==&mid=2447512608&idx=1&sn=c093b0529cf83f57a1609f27a4fdb7ce&scene=21&poc_token=HKIs0WijUEkVAZiRySCCHG8qH1y2Hk9fbFfAAxs4)，我们规避了这种情况。

然而对于 Attention 类算子而言，实际输入的不同 Batch 对应的上下文长度可能不同，因此 Tiling 中分核策略的计算对具体上下文长度 (actualSeqLengths) 存在值依赖，这使得 Attention 类算子无法完全下沉到静态图执行，导致整网产生过多小的静态子图，引入大量静态子图调度开销；同时，进行分核计算时需要将对应的数据从 Device 侧向 Host 侧拷贝，造成整网性能较差。

<div align="center">
    <img src="./figures/tiling1.png" width="800" />
</div>

针对这一问题，Attention 类算子采用了 Tiling 下沉的解决方案，将分核计算下沉到 Device 侧的 AICPU 进行。整网运行时，分核计算依赖的数据就绪后即可启动，且可以与算子执行流并行进行。然而，对于分核计算依赖前序算子的情况，这种方法依然无法避免执行断流。从具体使用的角度看，这种处理方式引入了额外的 AICPU Tiling 逻辑，在自定义算子流程中需要另外开发对应的 Tiling func，并配合下沉图执行调度实现，功能复杂、链条较长，导致自定义开发和调测难度较高。

<div align="center">
    <img src="./figures/tiling2.png" width="800" />
</div>


本次发布的`LightningIndexer`及`SparseFlashAttention`算子进行了进一步的 Tiling 下沉优化，将分核计算流程从 AICPU 下沉到 AICore 内部执行，简化调用链条，消除 Tiling 依赖前序算子引入的调度开销，开发和调试流程得到了简化。

<div align="center">
    <img src="./figures/tiling3.png" width="800" />
</div>
