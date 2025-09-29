# 基于 PyPTO 的 Lightning Indexer 和 Deepseek Indexer Attention 算子开发实践

## 简介

近年 AI 大模型规模出现爆炸式增长，其对算子性能的要求也越来越高。受限于硬件内存大小和带宽约束，大模型时代的算子需充分考虑计算过程优化、内存读写优化、不同层级的内存复用等多方面因素，而提升算子性能的重要手段之一便是算子融合。所谓算子融合，是指通过将多个计算合并到一个算子 Kernel 中，以减少内核启动开销和内存读写次数，提高内存复用率，从而提高算子性能。虽然算子融合性能收益显著，但开发融合算子也常常伴随巨大的挑战：

1. 融合意味着单算子中有多个计算逻辑（如 MatMul、Softmax、ReLU），代码量大，复杂度高；
2. 融合算子通常有多个输入输出，对内存访问的要求高，一旦核内不同层级的 SRAM 利用不善，性能会大打折扣；
3. 为了达到高性能，需要使用大量当代硬件定制特性，导致算子难以跨平台复用；
4. 通常存在大量可配置参数，手动调优困难且不可持续。

随着融合算子的规模逐渐增大，由于开发难度的原因，手写融合算子的方式越来越难以为继。基于此，算子编程领域兴起了一种针对现代硬件的高效易用计算编程范式——基于 Tile 编程。它将大规模计算任务和数据分解为更小、更易管理的“块”（Tile），从而更好地利用硬件的内存层次结构和并行计算能力。

PyPTO 是 CANN 推出的一款高效的自研编程框架，旨在简化算子开发流程，大幅提升算子编程易用性，同时保持高性能计算能力，给开发者提供面向 MPMD （类似 Mega Kernel ）开发的、基于 Tensor/Tile 的编程范式。相比于现有算子开发工具，其主要特点包括：

1. 自动生成读写指令，开发者无需手动管理核内 SRAM。同时，开发者可以通过参数配置控制 Tiling 切分和 SRAM 利用策略，从而达到对核内 SRAM 的高效利用。
2. 定义了一组抽象的 Parallel Tile Operation (PTO) Instruction （PTO 指令），开发者编写的程序都会转换为一组 PTO 指令。通过在不同硬件平台实现对应 PTO 指令，可自动完成算子的跨平台迁移，不再需要手动适配。
3. 提供了一套 MPMD (Multiple Program Multiple Data) 运行时。相比于 SPMD (Single Program Multiple Data)，MPMD 无需进行全局同步，而是利用数据的生产消费关系构建任务间依赖，使不同核在同一时间执行不同程序，充分利用多核并行能力。
4. 提供了一套集成开发环境，用于可视化算子编译及运行过程，帮助开发者定位和调试算子精度/性能问题，实现算子快速迭代。

下图展示了 PyPTO 的主要架构：

<p align="center">
  <img src="./figures/arch.png" width="50%" alt="示例图片">
  <center>PyPTO 架构</center>
</p>

## 特性介绍

### 基于 Tensor/Tile 的编程范式

PyPTO 是一个基于 Tensor/Tile 的编程范式，其核心思想是使用 Tensor/Tile 作为数据的基本表达方式，通过一系列对 Tensor 的基本运算来描述并组装完整的计算流程。

- Tensor 作为基础数据结构，计算过程是对 Tensor 类型的数据进行数学运算。
- 将原始 Tensor 进行切分后得到的小块数据，称为 Tile，而这个切分过程称为 Tiling。Tiling 使得处理器核可以根据核内 SRAM 容量大小，容纳计算过程用到的 Tile 数据，并复用这些 Tile 以达到减少对 Global Memory 访问的目的。
- 基于 PyPTO 开发的代码整体风格简洁明晰，开发者可以聚焦于算法本身的计算逻辑和各个 OP 的 Tile Shape 设置，无需感知复杂的内存、调度、硬件等底层实现细节，对于开发者更加友好，开发难度大幅下降。

### Parallel Tile Operation

在 PyPTO 中，对数据的操作被称为 Operation。Tensor Operation 在经过 Tiling 之后转换为 Tile Operation。框架提供了一组基本的 Tile Operation 用于底层硬件抽象，可针对不同芯片适配，实现算子的跨平台支持。此外，开发者可以自定义扩充 Tile Operation，以支持新的硬件平台，或者得到特定平台的高效性能。

下表列举了一些 PTO 指令实例：

| 分类                         | 指令          | 功能描述                             |
| -------------------------- | ----------- | -------------------------------- |
| **Element Op（逐元素操作）**      | TADD        | 执行两个 Tile 数据块内逐元素的加法运算           |
|                            | ...       | ... |
| **Tile-Scalar Op（块与标量操作）** | TADDS       | 将标量与 Tile 数据块内逐元素相加              |
|                            | ...       | ... |
| **Axis Op（按轴操作）**          | TROWSUM     | 对 Tile 数据块的每一行执行求和               |
|                            | ...       | ... |
| **MATMUL Op（矩阵乘操作）**       | MAMULB      | 执行两个 Tile 矩阵数据块的乘法               |
|                            | ...       | ... |
| **Fix Pipe Op（固定管线操作）**    | TMOV        | 将 Tile 元素从源寄存器搬移到目的寄存器（可带随路运算）   |
|                            | ...       | ... |
| **MEM Op（访存操作）**           | TLOAD       | 将数据从内存加载到 Tile 数据块               |
|                            | TSTORE      | 将 Tile 数据块写回到内存                  |
|                            | ...       | ... |
| **Complex Op（复杂操作）**       | TCVT        | 执行 Tile 数据块的类型转换                 |
|                            | ...       | ... |

### 核内 SRAM 的管理

PyPTO 为开发者提供两种 Tiling 调优的方式：
1. 设定 Tile Shape 参数，框架按内置 Tiling 方法将 Tensor Operation 计算图展开成 Tile Operation 计算图；
2. 开发者可以自定义 Tiling 方法，可以实现更特化，更高效的展开逻辑。

通过 Tiling 得到 Tile Operation 粒度的计算图后，框架会对计算图进行切分，并为切分后的子图插入数据搬运指令，以使用各层级的核内 SRAM。每个子图的输入输出都放在 Global Memory 上，而子图内只会使用核内 SRAM 用于存放中间数据。

PyPTO 的子图切分过程会考虑以下因素：

1. Operation 亲和性，在同一类型的核上执行的 Operation 会被尽可能分配到同一子图。
2. 子图同构性，框架会尽可能保证切分出的子图是可复用的，以减少编译耗时，降低 ICache Miss 率。
3. 子图并行度和重复数据搬运的权衡，过大的子图会导致核空闲，过小的子图会导致同一份数据向不同核重复搬运。开发者可以通过配置参数调整切分策略，在并行度和 SRAM 利用率间取得平衡。

下图是一个算子的 Tiling 和切分过程示例：

图中展示了A乘以常数后与B相加，然后与W进行矩阵乘，最后对尾轴进行累加的一个计算示例。其中A，B的维度为（64, 64），W的维度为（64, 32）。

<p align="center">
  <img src="./figures/TensorGraph.jpg" width="60%" alt="示例图片">
  <center>Tensor 图示例</center>
</p>

MulS，Add和RowSum均为Vector操作，设置其Tile Shape为（32, 32），Matmul为Cube操作，不进行切分。经过Tiling之后，得到如下Tile计算图。

<p align="center">
  <img src="./figures/TileGraph.jpg" width="60%" alt="示例图片">
  <center>Tile 图示例</center>
</p>

针对A3的CV分离架构，框架会将Matmul单独切成一个子图，Matmul前的MulS + Add切成一个子图，Matmul之后的RowSum切成一个子图。其中MulS_Add子图会被调用四次，RowSum子图会被调用两次，每次处理不同的Tile块。

<p align="center">
  <img src="./figures/ExecGraph.jpg" width="60%" alt="示例图片">
  <center>Execution Graph 图示例</center>
</p>

### MPMD 运行时

经过上一阶段的切图，会得到许多小的子图，这是 PyPTO 的最小执行单元，每个子图会编译成可执行文件，生成对应任务，被分发到不同的核上执行。PyPTO 的运行时是 MPMD 的，整个运行过程可分为如下几步：

1. 根据子图间的数据依赖，创建任务间依赖；
2. 调度 AICPU 根据任务的拓扑顺序下发无前置依赖的任务到空闲 AI 核上，AI 核收到任务后立即异步执行；
3. 任务执行完成后调度 AICPU 收到回调信息，解除后续任务的依赖并继续尝试下发。

### Human-In-The-Loop 的调优思路

从算子描述到上板执行，整个过程需要经过 Tile 展开、切图、运行时调度等众多阶段，其中许多阶段面对的都是 NP-Hard 问题，因此期望由框架自动生成高效性能的融合算子是不现实的。为此，框架提供了一套可视化的集成开发环境，使开发者可以查看每个阶段的输出信息，控制各阶段的运行过程，并将修改后的算子重新提交到 NPU 上执行，得到新的结果和性能数据。通过这一机制，开发者可以快速迭代，在较短的周期内开发出性能优异的算子。

下图展示了 PyPTO 的算子开发和调优过程：

<p align="center">
  <img src="./figures/pypto_workflow.png" width="40%" alt="示例图片">
  <center>PyPTO 工作流</center>
</p>

## 融合算子示例

PyPTO 是 MPMD 的调度思想，和 Mega Kernel 思想类似，并支持开发者自行选定融合范围。本文以 DeepSeek V3.2 中 Lightning Indexer 和 Deepseek Indexer Attention 的开发为实例，从融合算子开发者的视角，阐述在 PyPTO 的编程框架下，如何进行不同融合范围的算子的开发、以及通过 Human-In-The-Loop 的方式调优。

### Lightning Indexer

Lightning Indexer 是 DeepSeek V3.2 的 Deepseek Indexer Attention 计算的一部分。其计算流程图如下：

<p align="center">
  <img src="./figures/DeepseekLightningIndexer.png" width="40%" alt="示例图片">
  <center>LightningIndexer</center>
</p>

1. 开发者根据计算流程图，进行融合算子编码开发，其对应的 PyPTO 源代码可以参考：[PyPTO Deepseek Lightning Indexer](../../../ops/pypto/src/lightning_indexer_pto/op_kernel/lightning_indexer_topk.cpp)。
2. 基于 PyPTO 完成开发后， PyPTO 提供了完善的可视化集成开发环境，为开发者提供了不同阶段的产物。其中最为重要的是在 NPU 上运行得到的性能 Profiling 数据，即下图所示的泳道图，开发者通过泳道图可以分析融合算子中各部分的耗时，进而开展后续的性能调优。

<p align="center">
  <img src="./figures/DeepseekLightningIndexer_0.png" width="60%" alt="示例图片">
  <center>LightningIndexerSwimlane</center>
</p>

- 该泳道图的横轴表示时间，纵轴由一条条的泳道构成，一条泳道表示一个 AI 核。每条泳道上，随时间排布了对应 AI 核上执行的任务。
- 每个任务上标记了一个标签，该标签是开发者写在源码中的语义标签 (Semantic Label)，用来辅助开发者理解当前硬件在执行的任务对应到哪一段前端代码。
- 选中一个任务时，可以显示其前后依赖，用来观察任务的执行顺序、依赖间隔等是否合理。
- 在泳道图上，可以看到不同颜色（即前端开发者写的不同代码）在同一时间不同计算核并行执行。传统的 SPMD 的调度要求同一时刻执行相同程序，而 MPMD 的运行时会根据实际运行时解依赖情况来调度任务执行，当一个任务解依赖完成后，就可以和在不同 AI 核上的任务异步并行执行。

#### 泳道图分析

当开发者写完 PyPTO 的代码，获取第一版的开箱性能后，需要根据泳道图分析可能的优化手段。上述泳道图展示的是在典型 Shape（4 Batch， KV-Cache 64K 长度）下的执行情况，直觉上能看到几个可以优化的“异常点”：
1. MatMul 的 Cube 任务前序无依赖，为什么没有在第一时间执行？
2. 主体的 Cube 和 Vector 任务都有三列，但是第三列只有部分核在工作，没有很好的对齐尾部。
3. 最后的 TopK 似乎有严重的拖尾。

接下来，PyPTO 的开发者可以针对上述问题进行分析。
- 通过对异常点 1 的分析，发现框架在处理数据搬运类 Operation 的时候，对不需要数据搬运的场景优化能力缺失。
- 通过对异常点 2 的分析，发现由于是 4 Batch 64K 的场景，Page Attention 中给定的 Block Size 为 128，那么 MatMul 切分出了 `4*64*1024/128=2048` 个任务，程序中通过配置调整任务 L1 复用为 32，即最后有 `2048/32=64` 个任务，无法平均分配到各个核上（当前硬件共有 24 个 Cube 核）。
- 同样由于是 4 Batch 的场景，且最终的 TopK 任务是单核的归并，所以在 Vector 核上有拖尾。我们可以尝试优化最后的排序、归并任务来减小拖尾。

#### 改进措施及结果

- 针对异常点 1 ，通过修改相应的 Pass ，补齐泛化能力，从而提升性能 (10%) 。与传统黑盒的调优框架不同， PyPTO 通过可视化的集成开发环境辅助开发者发现算子甚至框架的问题，开发者可以选择本地修复或者向开源社区反馈。
- 针对异常点 2 ，开发者可以选择通过调整 L1 复用来达到特定 Shape 性能和动态 Shape 泛化性能的平衡。只是在本例中，L1 复用设置为 32 可以达到较好的泛化性。
  ```cpp
  Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 32);
  ```
- 在对异常点 2 分析的过程中，发现 MatMul 的输出粒度为 128，进而发现 Tile Shape 同样设置为 128，这对硬件计算并不是最高效的，因此选择调整 Tile Shape，让 MatMul 把数据输出到连续内存后再进行后续大颗粒度的 Vector 运算，提升性能 (17%) 。
  ```cpp
  TileShape::Current().SetVecTile({64, 256});
  ```

在经过上述优化后，得到新版的泳道图如下，可以看到其性能有明显的优化 (27%) 。

<p align="center">
  <img src="./figures/DeepseekLightningIndexer_1.png" width="60%" alt="示例图片">
  <center>LightningIndexerSwimlaneNew</center>
</p>

#### 下一步计划

经过上述优化，泳道图已经比较密集。但是图中仍然有一些可优化的部分：

* MatMul 的子图太大会导致后续 Vector 任务延迟开始，第一列 Cube 任务下方仍然有大量的空白时间。通过对原始计算逻辑的分析，得知这一部分可以并行，因此后续可以考虑通过控制复用程度对此进行优化。
* 最后的 TopK 是通过排序归并得出的结果，可以考虑将排序部分放入前面 Vector 运算，最后只做归并，从而减小拖尾时间。

### Deepseek Indexer Attention

Lightning Indexer 算子展示了 PyPTO 允许开发者可以控制算子融合的范围。接下来，本文通过 Deepseek Indexer Attention 的融合示例，展示 PyPTO 的整网融合能力。

<p align="center">
  <img src="./figures/DeepseekIndexerAttention.png" width="50%" alt="示例图片">
  <center>DeepseekIndexerAttention</center>
</p>

Deepseek Indexer Attention 的计算逻辑图如图所示，主要由 MLA Prolog、Indexer Prolog、Lightning Indexer、Sparse Flash Attention 4 个相对独立的部分构成，可以看出在更大的范围内融合有更大的并行度可以挖掘。PyPTO 允许开发者先手动写这独立的四个部分，并通过框架将四个部分融合成一个完整的 Deepseek Indexer Attention 大算子。因此，大算子的开发策略如下：
* 对于复杂融合算子，可以将复杂融合算子拆分成若干个相对独立的子算子；根据泳道图对各个子算子进行独立性能调优。
* 当各子算子接近最优性能后，再把子算子们进行更大程度的融合，以期获得完整融合算子的最优性能。

其对应的 PyPTO 源代码可以参考：[PyPTO Deepseek Indexer Attention](../../../ops/pypto/src/deepseek_indexer_attention/op_kernel/deepseek_indexer_attention.cpp)。与 Lightning Indexer 的融合算子相同，也可以上板得到 Deepseek Indexer Attention 的泳道图：

<p align="center">
  <img src="./figures/DeepseekIndexerAttention_0.png" width="60%" alt="示例图片">
  <center>DeepseekIndexerAttentionSwimlane</center>
</p>

#### 泳道图分析

大融合算子思想的优势主要在于，将所有相对独立的子图融合在一起，提高内存利用率，并充分发挥NPU的性能。因此，可以从图中看到，DIA 整图会比单独的Lighting Indexer图更加复杂，所有的计算步骤融合在一起，乱序调度。从图中，可以很清晰地快速定位到性能瓶颈点：

* 离散 Gather 部分的耗时占比高达 70% 以上。该部分的主要操作是 SparseFlashAttention 中，根据 Lightning Indexer 计算得到的 TopK index，从 kv_cache/kr_cache 中离散地读取对应 index 的 kv 进行聚合的过程。该聚合过程当前通过在 Vector 核搬运完成离散到连续的转换，再写到 GM Workspace中，由 Cube 核搬运到 L1 使用，既浪费了内存，又因重复搬运带来了极大的性能损耗。
* Indexer Prolog 子图 matmul 的任务过多。matmul M轴当前为1，未进行合轴处理，利用率过低。
* 存在较大的流水线 bubble。两个子图之间存在未消除的barrier，导致所有后续无依赖的任务也同样停滞，出现了较大的 Pipeline bubble。
* reshape 空子图导致任务启动延时。reshape未充分优化，导致存在的空子图，后续依赖该 reshape 结果的任务停滞。

#### 改进措施及结果

针对上述识别的关键性能瓶颈点，逐一分析、解决：

* PyPTO 允许开发者自定义 Tile Operation，因此，针对该场景，需要开发特定的 TileOp，其功能是根据 index 直接从 GM 离散搬运到 L1 的连续空间上，使其能够直接进行 Matmul 的计算。这样，可以省去离散转连续的额外workspace的申请，并去除冗余的搬运，性能提升非常明显。可以参考如下示意图：

<p align="center">
  <img src="./figures/DeepseekIndexerAttentionGather.png" width="60%" alt="示例图片">
  <center>自定义 Gather Tile Op</center>
</p>

* 将 b * s1 合轴，调整 Tiling，撑大 M 轴，用满 L0A，充分提高 Matmul 的计算效率，性能倍增。
* 消除两个子图之间的 barrier，将两个子图充分融合，不依赖的 task 可以提前启动，可以消除流水线的气泡。
* reshape操作，不改变数据排布，消除后，原本依赖 reshape 的 task 的 delay 消除，并减少解依赖的时间。

在经过上述分析和优化后，得到新版的泳道图如下，可以看到，使能以上优化措施后，该融合算子的运行时长缩短了 75% 。

<p align="center">
  <img src="./figures/DeepseekIndexerAttention_1.png" width="60%" alt="示例图片">
  <center>DeepseekIndexerAttentionSwimlaneNew</center>
</p>

#### 下一步计划

经过上述优化，泳道图中的几个关键性能瓶颈点已经被消除，性能也有了比较大的提升。基于新的泳道图，我们可以继续挖掘性能优化点，持续提升性能。

* Task 排布相对离散，任务调度不够紧密，需要分析任务之间的依赖，消除 Pipeline Bubble 。
* SparseAttention 中，离散的数据搬运效率较低，需要提高搬运的并行度。
* 多个子图之间，仍然存在较大的空隙，需要再进一步的融合，将有数据依赖的计算整合到同一个 LOOP 中。

期望经过以上优化后，泳道图中的任务会排布更加紧密。总的来说，性能优化是需要分多轮迭代的，每一轮解决关键的性能瓶颈点后，再基于新的泳道图进行分析、持续优化。