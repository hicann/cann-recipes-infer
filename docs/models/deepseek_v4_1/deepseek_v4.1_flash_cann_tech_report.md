# DeepSeek\-V4\.1\-Flash CANN优化实践

## 引言

DeepSeek\-V4\.1\-Flash 是 DeepSeek\-V4 系列的混合大模型，原生支持多模态，引入Engram架构实现模型能力进一步飞跃。并通过模型结构一系列创新：KVShare、Indexer Share、FP4低精量化等技术使长序列下 Attention 的计算量与访存量保持低位，降低历史kvcache存储压力，并实现推理成本大幅降低。CANN 已支持V4\.1 Flash模型，并开源相关算子和模型的参考实现。

---

## 核心亮点（Key Highlights）

1. **昇腾950原生支持 FP8/FP4 混合精度**：主干 FP8 权重 \+ MoE 专家 FP4 的原生混合精度，KV Cache 分精度存储——SWA 缓存 FP8、压缩 KV 与 Index 缓存 FP4，配合长序列 LightningIndexer 高效执行。

2. **昇腾950原生支持 Engram FP8 \+ 灵衢组网 UB Host Offload**：Engram 表 FP8 量化存储，经灵衢组网 UB 实现 CPU Host offload，多流并行掩盖 Engram 查找与 H2D 传输开销。

3. **昇腾原生支持 Sparse Attention FP8/FP4 混合输入**：单次 sparse\_attn 融合滑窗 FP8 与压缩稀疏 FP4 两路 KV，支持混合精度输入。

4. **昇腾亲和 LightningIndexer 二级召回**：候选块预筛（一级）\+ topk 打分（二级）两级召回，长序列下打分候选规模与上下文长度解耦。

5. **昇腾原生支持 多模态（ViT \+ LLM）理解大模型接入**：ViT 编码 \+ LLM 解码的原生多模态架构，图像不经过外部 caption 模型、直接由视觉编码器映射进 LLM 隐层空间，与文本 token 在同一残差流中参与推理。

6. **昇腾950 单机 16 卡部署、支持长序列**。


整体部署策略沿用DeepSeekV4的EP并行方案，针对模型的新结构特征，设计实现NPU亲和的并行策略，当前支持`950PR/DT`代际昇腾芯片部署，提供长达1M序列的高性能推理能力。[模型推理代码](../../../models/deepseek_v4_1/README.md)与低精度 [稀疏Attention](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/sparse_flash_attention_antiquant) / [LightningIndexer](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/quant_lightning_indexer) / [Engram](https://gitcode.com/cann/ops-transformer/blob/master/mc2/common/docs/torchapi_ElasticBuffer.md) 等算子实现均已在开源。


---

## 模型结构

### 2\.1 总体架构

![model_arch.png](figures/model_arch.png)

主干为 40 层 MoE Transformer（hidden 5120 / 64 查询头 / 1 KV 头），按「压缩者 / 消费者」分工切成前 20 层 Encoder（L0–L19）与后 20 层 Decoder（L20–L39）。每层 Block 是 **mHC \+ Attention \+ MoE** 三元组，残差流以 `hc_mult=4` 路并行副本承载。注意力侧沿用 SWA（`sliding_window=128`）\+ 压缩稀疏注意力（2:1 压缩）路线，压缩 KV 由 4 个源层集中生产、其余层只读共享 Cache，长序列下注意力读取量与计算量保持近常数。以下分「整体结构 / mHC / 压缩与存储」三块展开，详细算子级影响见第 3 章。

#### 分层与mHC 结构

![model_block.png](figures/model_block.png)

- **分层分工**：L0–L1 为 SWA（不压缩）、L2–L19 为 2倍压缩CSA、L20–L39 为 1倍压缩CSA。其中仅 `[2, 8, 14, 20]` 四个源层真正执行压缩并产出共享压缩 KV，组内其余层直接消费源层 Cache；在后20层中，`[20, 24, 28, 32, 36]` 层复用L20阐述的InderexCache并阐述IndexerTopk并提供给其余层消费。

- **流水接力**：每个子层产出的 mix 系数供下一个子层使用而非自用，系数由残差流自身推导；计算由 AscendC 算子「mHC 折叠\-扩展」承载。

#### 压缩与存储结构

![model_compress.png](figures/model_compress.png)

- **SWA 滑窗与压缩**：所有层维护 `sliding_window=128` 的 KV 环形缓存（FP8量化）。Prefill 整段计算填 ring，Decode 每步写入位置模 128 槽位并回读整个窗口。

- **压缩稀疏注意力**：Compressor 在2倍压缩时对输入分别投影出 kv 与 score，对每组 2 个 token 做 `softmax(score)` 加权求和（全程 FP32）得 1 个 KV latent；1倍压缩时退化为一次纯投影 \+ Norm。压缩 KV 缓存仅 4 个源层维护（`[B, L/ratio, 512]`，FP4），其余层只读。

- **shared\_kv\_indexer 两级稀疏索引**：indexer层获取量化后的inderexCache后（FP4）。两级召回：一级通过候选与分组topk机制，最多获取 16384 个候选位置；二级在候选块内打分取 `index_topk=512` 个压缩位置，与滑窗 128 位置拼接后经**一次** `sparse_attn` 完成两路混合注意力。`[2, 8, 14, 20, 24, 28, 32, 36]` 共 8 个 index 源层（Encoder 3 \+ Decoder 5），非源层复用最近源层的 topk 结果。

- **KV Cache 布局**：持久化三种缓存——压缩 KV（FP4，4 源层）、index KV（FP4，同源）、滑窗 KV（FP8，每层固定 64 KiB，不持久化，命中/恢复时向前补算 128 token 重建）；压缩组内共享，Decode 每 query 实际读取量被 `topk=512 + 窗口 128` 限制为 640 个位置（详见 3\.3）。

- **MQA**：沿用 Deepseek\-V4 提出的 MQA 结构作为 Attention 核心逻辑。

### 2\.2 量化方案

#### 2\.2\.1 精度布局

混合精度贯穿整网：

- **权重与激活**：FP8 稠密权重，MoE 专家 FP4；激活量化采用 MXFP8量化，scale 可直接以 E8M0 存储

- **KV Cache 分精度存储**：滑窗 KV FP8、压缩 KV FP4、IndexCache FP4（详见 2\.1）。

- **Engram 表**：大表保持 FP8 存储，查询时按行反量化到 BF16。

- **精度敏感路径**：门控（Gate）输出 FP32，避免精度对专家选择影响。

#### 2\.2\.2 融合算子

本章节介绍KV量化的稀疏Attention在昇腾950上的优化实践。

##### 2\.2\.2\.1 双 AI Core 共享 KV Cache

###### 动机与问题

昇腾 950 支持 Cube L0C/L1 Buffer 和 Vector Unified Buffer 的直接 CV 数据传递，提高核内数据利用率，减少 L2 和 GM 层的数据交换。但是这也会给算子的 tiling 设计带来一些挑战：Cube 核或者 Vector 核产生的数据需要即时地被另一个核消费，L1 cache 及 UB 的空间会更加紧张，同时流水的排布更加复杂。

对于 DeepSeek 的 Attention 机制，K 和 V 为同一份 tensor，单 head，`head_dim = 512`。考虑到 K 和 V 实际上是同一份，因此我们需要在 L1 Cache 中为 KV 预留三份 buffer，这样可以保证访存没有 bubble。如下图所示，仅考虑 FlashAttention 主要的计算流程，其中 Sx 表示第 x 次 softmax 计算，Oy 表示第 y 次 rescale 计算。

![ops_attention.png](figures/ops_attention.png)

L1 中 3buffer KV 占用空间大小为：`128 × 512 × 2 × 3 = 384 KB`，L1 还剩余空间大小为 `128 KB`，由于还需要存储每次 softmax 的结果 P，因此行方向基本块大小只能切到 64，为 Q 预留空间大小为 `64 × 512 × 2 = 64 KB`。此处如果通过减少 KV 序列方向的基本块大小调大行方向基本块大小，会导致 rescale 计算次数过多，引入大量向量化计算，同时在 UB 上需要常驻更大尺寸的输出 tensor \(rescale 计算需要保持 FP32，如果行方向基本块过大会导致占用 UB 过多\)，依然不可行。

对于稀疏 Attention 而言，性能瓶颈在于对 KV 的离散搬运，也就是 V0 阶段的 mergeKV 操作，这一部分我们会利用 Vector 核进行搬运，通过聚合两个 token 的访存减少指令数，对于量化场景还会涉及到对 KV 的反量化。

当 Group Size `G = 128` 时，单个 token 对应的 attention 需要处理 128 行 Q，而基本块仅能容纳 64 行，因此 **需要两次迭代** 才能完成一个 token 的计算，这意味着 V0（mergeKV）阶段对同一批 KV 需要搬运两次，带来大量冗余的数据搬运开销，性能较差。

**核心矛盾**：L1/UB 容量不足 → 基本块行数受限 → 迭代次数翻倍 → KV 搬运冗余。

###### Chromosomal Crossover

为解决上述矛盾，我们采用一种染色体交错\(Chromosomal Crossover\)实现方式：**每两个 AICore 配对，协同处理同一个 token 对应的 G=128 的计算**。DeepSeek 的 FlashMLA 仓也采用了这种方式：[A Deep Dive Into The Flash MLA FP8 Decoding Kernel on Hopper](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Fdeepseek-ai%2FFlashMLA%2Fblob%2Fmain%2Fdocs%2F20250929-hopper-fp8-sparse-deep-dive.md%23crossover&from=https%3A%2F%2Fgitcode.com%2Fcann%2Fops-transformer%2Fwiki%2F%25E7%25A8%2580%25E7%2596%258FAttention%25E6%2580%25A7%25E8%2583%25BD%25E4%25BC%2598%25E5%258C%2596%25E5%25AE%259E%25E8%25B7%25B5.md&lang=zh&theme=white)。

具体流程如下：

1. **V0 阶段（mergeKV）**：两个 AICore 处理同一个 token`G=128`的计算，各自负责离散搬运并聚合当前 token 所需 KV 的一半。各自对离散的 KV Cache 做反量化后，将聚合结果写出到 GM 上的共享区域进行交叉共享。

2. **C1 阶段（Q@Kᵀ 计算）**：两个 AICore 在 C1 阶段**读取同一块共享的 KV**，各自用自己那部分`G=64`的 Q 与共享 KV 做 MatMul，进而完成后续的 attention 计算。

![ops_attn_share.png](figures/ops_attn_share.png)

这样，每个 AICore 的 L1 只需容纳 `S_inner = 128` 的 KV（三份 buffer 共 384 KB）\+ `S_q = 64` 的 Q（64 KB）\+ `(64,128)`的 P（16 KB\*2， 开启 double buffer），满足容量约束；而 G=128 的计算由两个 AICore 并行完成，无需迭代两次。

##### 2\.2\.2\.2 稀疏地址计算：VF 向量化实现

###### 动机与功能

PagedAttention 下，KV cache 按 block 物理分页存放，逻辑上连续的 KV 序列在物理上被 `block_table` 重映射为 PA 物理块。稀疏注意力又只选中部分 block\(`sparse_indices`\)。因此每个被选中的逻辑索引 `s2Idx` 需要换算成真实物理地址:

![ops_addr.png](figures/ops_addr.png)

```Plain Text
blkIdx  = s2Idx / blockSize                       // 落在第几个逻辑块
blkOff  = s2Idx % blockSize                        // 块内偏移
phyBlk  = block_table[blkIdx]                       // 逻辑块 → 物理块号(查表)
phys_addr(元素) = phyBlk * kvStride + blkOff * kvDim
```

- `kvStride`: KV cache物理块之间的元素跨度。

- `kvDim`:单 token 的特征维\(每行元素数\)。

- **结果是 int64**:`phyBlk * kvStride` 会超过 2^32，必须用 64 位地址。

`GetKVPhyAddr` 要做的事情是把一整行 query 选中的 `total` 个逻辑索引批量换算成 `total` 个 int64 物理地址，写回 `kvPhyAddrGm`，供后续 `GetRealS2Addr` 取址搬 KV。 `GetKVPhyAddr` 用 VF 实现，把以上换算物理地址换算公式**向量化、并把查表从 GM 标量访问改成 UB 向量 gather**。

###### 标量实现 vs 向量化实现

向量寄存器以 **256B 为单位**:`RegTensor<uint32_t>` 64 个 int32 = 64×4B = **256B**。VF 每个 loop 用 **2 个寄存器**\(`_1`/`_2`\)→ 一次处理 **128 个逻辑索引**; 输出是 int64 \(拆成 L/H 两个 int32\)。

**向量化收益:**

1. **循环次数 ÷128**:`total` 个索引从 `total` 次标量迭代降到 `ceil(total/128)` 次 VF 迭代。

2. **消除 GM 标量查表延迟**:`block_table` 一次性 `CopyPaTableToUb` 到 UB，之后用 `DataCopyGather` 替代 `GetValue` 逐个访 GM。

3. **除/模变移位**:`blockSize` 取 2 的幂, 除法和取模降为 `ShiftRights`\+`Muls`\+`Sub`。

4. **int64 地址 SIMD 化**:用 `Mull`/`Add`/`AddC` \+ carry mask 在寄存器内完成 64 位运算, 无需标量 64 位 ALU 逐个算。

##### 2\.2\.2\.3 核间流水排布

每个 S2 基本块均需用到一份 bf16 的 KV 矩阵（`s2BaseSize=128 × D=512`，合计 **128KB**）。N=128 的场景下，该 KV 矩阵经 GM 在两个 AIC 间共享（G=128 跨 2 个 AIC、各算 64 头）：由 AIV 的 `ProcessVec0` 在 UB 算出后写到 GM 的 `v0ResGm`，AIC 使用时再经 `IterateLoadQK` 从 GM `DataCopy` 到 L1 的 KV 矩阵缓冲 `l1RightBuffers`。该 KV 矩阵自产出起持续驻留至被 bmm2 消费——既作为 bmm1 的 KV 矩阵（当 K，算 Q·Kᵀ），又作为 bmm2 的 KV 矩阵（当 V，算 P·V）。

正因为一份 KV 矩阵要在 L1 里被 `load→bmm1→bmm2` 连续消费，因此需要在 L1 预留**三份 buffer**（`l1RightBuffers`，按 `taskIdMod3` 轮转），使相邻基本块的取数与计算彼此错开、消除访存 bubble。

这三份 buffer 能否被持续填满、连续轮转，正是本节流水排布要解决的问题。

一份 KV 矩阵每经过一轮基本块循环便前进一步，依次走 **vec0 产出（AIV）→ load 取数 GM→L1（AIC）→ bmm1（AIC）→ bmm2（AIC）** 四步。收益的关键，在于 `load` 与 `vec0` 落在哪一轮基本块循环：

**初版实现（预取深度 2）——取数与产出同一轮**：`PRELOAD_NUM = 2`、`runInfo[3]`，参与轮转的 KV 矩阵 3 份。`load` 与 `vec0`落在同一轮、且取的是同一块；AIC 必须等 AIV 的 `ProcessVec0` 产出并跨核同步后才能 `IterateLoadQK`。

**修改后（预取深度 3）——load 比 vec0 晚一轮**：把预取 `PRELOAD_NUM` 从 2 提到 3（`runInfo[4]`），让 `load` 与 `vec0` 解耦到相邻两轮基本块循环——`vec0` 在本轮产出并写 GM，`load` 推迟到下一轮才从 GM 取到 L1。

![ops_flow.png](figures/ops_flow.png)

上图按依赖关系排布三条流水线：**aiv**（向量核，依次做 `vec0` 产出 KV、以及 softmax 等后续向量处理）、**cube MTE2**（`load`，把 KV 从 GM 搬入 L1）、**MMAD**（cube 的 `bmm1`/`bmm2` 矩阵计算）；同色块表示复用同一块 L1 KV buffer（`l1RightBuffers` 按 `taskIdMod3` 轮转，故第 1、4 块同色、第 2、5 块同色，依此类推）。修改前，MMAD 需要等待其前置的 load 任务完成，而 load 又需要等待对应的 v0，从而产生气泡（图中红色 idle 段）；修改后 `load` 取上一轮已在 GM 就绪的 KV，无需等待，从而消除气泡。

需特别说明：图中 **load 第四块 qk 时，其起点被推迟到第一轮 qk 的 ****`bmm2`**** 算完之后才能启动**（红色箭头）。原因在于 L1 只有三块 KV buffer 循环复用，第四块与第一块落在同一块 L1 空间上，必须等第一块 KV 的最后一个消费者 `bmm2` 读完、腾出该 buffer，第四块的 KV 才能覆盖写入这块 L1。不过此刻 MMAD 正忙于前序基本块的 bmm 计算，这段 buffer 等待被压在 cube 计算之后、完全掩盖，并不会让 MMAD 等待；三块 buffer 因而得以连续轮转，以三份 buffer 消除 bubble”的设计落到实处。

### 2\.3 Engram n\-gram 记忆

#### 2\.3\.1 Engram 结构

![ops_engram_ub.png](figures/ops_engram_ub.png)

- **注入位置**：L1、L14 两层，在 Block 内 Attention/MoE 之前向残差流写入 n\-gram 查找结果；

- **作用**：以可训练的 n\-gram 哈希记忆补充固定上下文之外的语言先验，查表结果经门控注入残差流，与 mHC 多路残差兼容；

- **配置**：两张表分别 384,008,192 / 384,018,432 行 × 256 维（FP8 存储，按行反量化）；8 头；max\_ngram\_size=4（覆盖 2\-gram 至 4\-gram）；压缩词表 99,092。

- **计算流程**

    - **压缩词表映射**：全部 token 先映射到 99,092 个规范化压缩 id——NFKC → NFD → 去重音 → 小写 → 空白折叠，使大小写/重音/空白差异归一，哈希等价；

    - **滚动哈希**：每个 \(layer, lookback\) 用独立奇数乘子，滚动 XOR 累积出 2\-gram 至 4\-gram 哈希；每个 \(ngram 长度, head\) 组合落入**互不重复的素数桶区间**（从 `engram_vocab_size=16,000,000` 起依次取素数），8 头 × 3 个 ngram 长度 = 24 个独立哈希列；

    - **查表**：每 token 读取 24 个哈希列 × 256 维行，拼接后经一次 Linear（6144 → 25600）拆出 4 路 key 与共享 value；

    - **门控写入**：残差流与 key 的归一化点积经「带符号 sqrt \+ sigmoid」得到门控值，门控后的 value 加回残差流；位置直通掩码可让指定位置跳过注入。

- **并行与存储实现**

    - **存储**：两表本体 2×384M×256×1B ≈ 197 GB，采用多级多卡切分，同时将其 offload 到 host 内存上存储开销。量化时，采用 mxfp8 量化，其 scale factor 数据相对较小，全量存储到 device 内存中，减少小数量通信的开销。

    - **并行**：Engram Fetch过程采用多流并发和后台 URMA 通信的方式和前序层并行 overlap。

#### 2\.3\.2 通信实现\-灵衢 UB

本次对 Engram 的通信部分做了相应的加速算子实现

Engram Table offload 到 host 侧，通过 UB 能力从分布式的 host memory 中 fetch engram 表。

![ops_engram_module.png](figures/ops_engram_module.png)

通过 URMA 转 UBMem 的方式进行互通，获取 Engram 表数据

1）将 host memory 注册到 device 侧，获取到 device 侧可以访问的地址

2）使用 hcomm 的能力建立 urma 通信连接

3）使用 AIV 组网通信任务放入到 URMA 通信队列中，后续 urma 后台执行，不占用计算核与计算进行 overlap

**支持量化能力**

量化的 scaling factor 参数数据量要小很多，当前没有offload 到 host 侧，全量 engram table 的 scaling factor 参数都是放在 device 的 local memory 中，由 AIV 进行 gather 到输出上。（后续当 engram 表非常大的时候，也考虑将 scaling factor offload 到 host memory 上）

![ops_engram_quant.png](figures/ops_engram_quant.png)





### 2\.4 ViT 视觉编码器

视觉侧（ViT 编码器 \+ Aligner 架构与配置）**与已开源的 DeepSeek\-V4\-Flash\-Vision\-Exp 主体保持一致**。

图像以 token span 形式并入序列，逐行插入换行分隔符保持空间布局语义；MoE 门控为图像 token 维护视觉 token 专属的 bias；图像 token 同时被 Engram 屏蔽（置 DEAD），n\-gram 统计不跨越图文边界。

视觉侧引入后，LLM 主干注意力**全程保持因果**，这是 V4\.1\-Flash 相对开源主体的关键差异：视觉内容以 token span 形式并入序列后与文本 token 同等对待，全程无需为图像 span 引入双向 mask；**双向注意力仅存在于 ViT 编码器内部。**

---

## 影响与变化

整体上，V4 的压缩是**逐层私有**的——每层压缩自己的 KV，因此每层都是 KV 生产者；V4\.1 的压缩是**分层共享**的——压缩集中在 4 个源层，其余层只读共享 Cache。这一差异同时决定了算力分布（Compressor  \& Indexer 的执行密度）与 KVCache存储架构的部署特性。

### 3\.1 共享 KV 的影响

Compressor压缩侧：V4 的压缩逐层私有，Compressor 压缩动作从几乎全层削减到 4 层（减少近10倍），FP4 激活量化写回同样集中到 4 层。代价是源层产出更多KVCache，对源层 Cache 的读写热性提出更高要求。

Indexer索引侧：V4 的 Indexer 在 CSA 层逐层单级执行 topk，打分覆盖本层压缩 Cache 全量，长序列下 topk 是主要耗时；V4\.1 的 Indexer 仅在源层执行（减少近2\.5倍），面向后4个Indexer层通过二级召回的方式大幅减少了topk目标个数（仅16K个）大幅减少了耗时占用。

### 3\.2 SMLA/sparse\_flash\_mla：topk 固定

实际参与注意力计算的压缩位置固化为 topk=512，因此**实际计算与搬运的 KV 固化为 topk 大小**，不随压缩 Cache 总量增长。面向V4\.1 的更长KV存储场景，固定topk使得KV候选长度、较低的压缩倍率并不会制约sparse\_attn计算效率。

### 3\.3 Prefill 前 21 层 / 后 19 层

Prefill 时，前 21 层（L0–L20，含 4 个 KV 源层）计算全量 qkv；后 19 层（L21–L39）只消费共享压缩 KV，因此只需对每个请求计算 window\-size（128）大小的 q（用于生成滑窗 Cache），前序位置的计算可整体省略。这与 3\.4 的「Prefill 尾部截断」对应：后 19 层计算量从 O\(L\) 降至 O\(W=128\)。

### 3\.4 新增算子带来的带宽增量

- **Engram**（新增）：每 token 的哈希查表与查表投影权重流式读是 Decode 带宽的新增项，权重读取量与 MoE 激活专家权重读取同量级；384M 行的查表规模对 HBM 容量也提出要求。

- **ViT \+ Aligner**（新增，仅视觉阶段）：视觉编码为算力型负载，与文本 Prefill 可解耦执行（见 4\.1）。



下面以 L=1,048,576（1M）序列为例细化说明整体差异：

||DeepseekV4|DeepseekV4\.1|说明|
|---|---|---|---|
|share\-cache \+ FP4|21 层 CSA 4× 压缩、20 层 HCA 128× 压缩、FP8 存储**≈ 3\.0 GB/ 1M 序列**|**640 MB 压缩 KV \+ 160 MB IndexCache ≈ 0\.8 GB/ 1M序列**|降低约 **3\.9× KV存储**，计算量随topk变化|
|重计算 SWA|每存储Block需保存64KB 窗口 Cache|**无需存储，仅用于PD传输**|长期存储历史Cache用于PrefixCache命中时池化代价极高，**重算代价可控**|
|Engram|\-|基于hostDDR，分布式offloadEngram记忆|与序列长度无关|

---

## 并行与部署

### 4\.1 EPD 三段分离：VisionEncoder / Prefill / Decode

模型推理天然分为三个工作负载特征迥异的阶段，工程实现中也显式分离：

- **VisionEncoder（视觉编码）**：算力密集场景。ViT 编码器对单图执行全双向注意力，经 Aligner 对齐到主干隐层，输出仅在 prefill 首 chunk 的 span 覆盖时被消费（详见 2\.4）。

- **Prefill**：算力密集场景。压缩集中于 L0–L20 的 4 个源层，后 19 层可尾部截断只算最后 128 token（详见 3\.4）。可使用 **CP \+ EP **的并行方式。

- **Decode**：带宽密集型。每 token 主要开销为搬运一轮权重，受带宽限制；滑窗 ring buffer 与压缩中间状态槽位按位置模 ratio 递增更新，增量计算开销恒定。可使用 **DP \+ EP **的并行方式。其中 **Engram **使用**多机多卡切分**，以 **host DDR 内存作为扩展**（详见 2\.3\.3）。

三阶段资源特征（算力 / 算力 / 带宽）不同，可部署在不同硬件配比上，昇腾950代际芯片提供PR/DT两种形态，支撑灵活部署。

### 4\.2 Prefix Cache 命中

池化 KV Cache 的持久化边界（只存压缩 KV \+ index KV \+ 压缩中间状态，不存滑窗 Cache）决定了命中场景的处理方式：

- **命中判定**：新请求 prompt 与既有序列共享前缀时，命中区间直接复用已持久化的压缩 KV 与 index KV，跳过对应区间的压缩计算；命中hash判定条件仅需覆盖历史KV的hash，无关SWA型类LinearCache，易于命中；

- **SWA 感受野补偿**：滑窗注意力仅覆盖最近 128 个 token，命中点之前的滑窗 Cache 未被持久化，因此命中场景需要在命中点**向前补算 128 个 token（window\-size）**，重建各层滑窗 Cache，弥补 SWA 的感受野；其他历史压缩KV由PA存储直接命中获取，无需回退计算。
