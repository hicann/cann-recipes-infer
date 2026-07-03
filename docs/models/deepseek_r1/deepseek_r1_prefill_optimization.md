# 基于Atlas A3集群的DeepSeek-R1模型prefill阶段推理性能优化实践
## 概述
本文主要介绍基于Atlas A3 系列产品的DeepSeek-R1模型prefill阶段并行部署和性能优化策略。主要采用了混合切分策略、Double-Routing、micro-batch等关键优化技术，最终在Atlas A3 系列产品实现perfect eplb下6.5k tokens/卡/s的吞吐性能。

## MLA (Multi-Head Latent Attention)部署策略优化
### DP (Data Parallelism)切分策略
prefill在MLA部分可以采用纯DP计算策略，每卡部署完整模型权重，不对模型做任何切分，该策略有以下优点：
1. 纯DP切分可以避免多卡冗余KV-Cache (Key-Value Cache)，每卡只需要存储本卡所接收请求的KV-Cache数据量，各卡之间的KV-Cache互不相同，无冗余数据。
2. MLA部分无通信算子，可以避免通信开销，prefill阶段更容易用满算力，在请求数据量较大时更易获得高吞吐。

<div align="center">
<img src="./figures/image_attention_dp.png" alt="attention dp" width="30%">
</div>

但是该切分策略会导致以下问题：

1. 计算负载不均：实际场景中，输入请求的序列长度往往参差不齐，处理较短序列的卡会提前完成计算，随后必须等待批次中最长序列的处理完成，导致计算周期被浪费。
2. TTFT (Time to First Token)被延长：若实时请求数量低于卡数，部分卡将处于空闲状态；若为凑齐完整批次而延迟处理，会延长首token生成时间；若直接处理不完整批次，则会导致计算资源利用不充分。

针对上述问题，通过对TP (Tensor Parallelism)切分策略改进，本样例提出SP-TP-SP (Sequence Parallelism, SP)混合切分策略，对以上问题进行优化。

### SP-TP-SP混合切分策略

针对输入序列，采用batch和seq_len合轴的方式，转换为token数量，对token进行SP切分，均匀的分发到各个卡上，计算过程如下图所示：

<div align="center">
<img src="./figures/image_attention_sp.png" alt="attention sp" width="30%">
</div>

1. 将多个请求序列拼接后并padding到卡数的最小整数倍，进行SP切分，此时每张卡获取的token数相同。
2. FlashAttention（FA）计算部分采用TP策略，通过将Attention head均匀分布到每张计算卡上来实现，确保计算负载在多卡间均衡分布。
3. FlashAttention计算完成之后，通过AlltoAll通信，将TP转换为SP切分，通信算子的通信量更小，性能更优；此时，o_proj由于需要全载权重，占用内存会变高。

上述切分策略，每张卡的MLA阶段计算量相等，计算负载均匀。假设有3个输入input0，input1和input2，把batch和seq_len合轴后，将token均匀分布到Rank0~Rank3上，down_proj包含q_a_proj和kv_a_proj计算，q_up_proj和kv_up_proj分别表示q_b_proj和kv_b_proj计算，此时每张卡计算数据量一致，计算负载均匀。具体计算流程如下图所示：

<div align="center">
<img src="./figures/image_data_split_sp.png" alt="attention data split sp" width="80%">
</div>

上述策略有以下优势：

1. **TTFT更短**：尤其是长序列场景，比如1个128K的请求，此时可以将计算均匀分布到各个卡上完成计算，获得更短时延。
2. **MLA部分的计算负载均匀**：输入序列只需要padding到卡数的倍数即可，各个卡的计算量相等，几乎无冗余计算。

但是该策略会引入以下问题，需要通过相应的优化手段解决：

1. 每卡需要存储完整的KV-Cache，各卡之间的KV-Cache存在冗余，占用内存高，可通过将KV-Cache数据分卡存储进行规避。
2. 相对于纯DP策略，MLA计算引入AllGather和AlltoAll通信，该通信时间可以进一步通过micro-batch进行优化。

**总结**：当前在prefill场景中，当请求量较大或KV-Cache内存占用较高时，可以考虑采用DP切分策略，以提升吞吐量；当请求数量较少、分布不均匀或单次请求序列长度较长时，可以考虑采用SP-TP-SP混合切分策略，以降低时延。

## MoE (Mixture of Experts)部署策略优化
MoE计算阶段采用EP (Expert Parallelism)切分策略，即将路由专家均匀分布到每张卡上，同时每张卡上同步存放一个共享专家。路由专家采用**Double-Routing**的计算策略完成计算，其核心在于每张卡需要完成2次init-routing和finalize-routing操作。该方案有以下优点：

1. 当负载相对较均匀时，相对于AllGather的方案，峰值内存较低，每张卡只需要获取对应专家计算的token即可，不存在数据冗余。

2. 相比于每张卡上获取完整token数据量，通信量更低，通信耗时更优。

该方案的主要计算流程如下图所示：
<div align="center">
<img src="./figures/image_double_routing.png" alt="Double-Routing" width="60%">
</div>

主要计算步骤如下：

1. 调用npu_moe_init_routing_v2接口，对本卡上的token进行expand并排序得到expand_x，同时获取每个专家需要计算的token个数即tokens_per_expert，以及用于恢复token原始排序的参数expanded_row_idx（**第一次routing操作**）。
    ```python
    expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(...)
    ```
2. 通过对tokens_per_expert进行AlltoAll操作和ReduceSum操作，获得每张卡需要向其他卡发送的token数量(**input_splits**)，以及每张卡需要从其他卡上拿到的tokens数量（**output_split**），该参数主要用于AlltoAll通信，各卡分别获取对应的token。
3. 调用AlltoAll通信接口，对每张卡上的token进行交互，每张卡上的专家获取本专家需要计算的token。
    ```python
    dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)
    ```
4. 此时每张卡上已经获取到本卡专家需要计算的token，但是处于乱序。因此，本卡需要再完成一次re_routing操作，得到排序后的token分布即hidden_states_ordered_by_experts，以及本卡上每个专家需要计算的token数量tokens_per_local_expert，同时得到用于恢复本卡token顺序的参数gathered_ids_unsort（**第二次routing操作**）。
    ```python
    hidden_states_ordered_by_experts, _, gathered_ids_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(...)
    ```

5. 完成路由专家和共享专家计算。
    ```python
    hidden_states_ordered_by_experts = experts(**args)
    ```
6. 对完成计算的token进行排序恢复（**第一次finalize-routing操作，与第二次routing操作相对应**），此处采用torch.index_select()接口性能更优。
    ```python
    new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
    ```
7. 调用AlltoAll通信接口，进行token多卡交互，把本卡上的token获取回来。
    ```python
    dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)
    ```
8. 调用finalize_routing接口，对本张卡上的token进行顺序恢复，得到原始排序的token（**第二次finalize-routing操作，与第一次routing操作相对应**），同时可直接通过该接口完成路由专家和共享专家的加权求和。
    ```python
    hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=hidden_states_share, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )
    ```

## TND格式支持
常规FlashAttention采用BNSD的格式（[batch, num_of_head, seq_len, head_dim]）计算，而对于每个请求长度均不确定的场景，可能需要padding操作，导致计算资源的浪费。而TND格式可以对batch和seq_len进行合轴处理即[batch*seq_len, num_of_head, head_dim]，转换为token数，无需padding数据即可完成计算，此功能依赖于FlashAttention的功能支持。

针对TND（[token, num_of_head, head_dim]）格式，需要把所有输入的请求合成一个list进行处理。假设有4个请求，seq_len长度分别为128，256，512和1024，在FlashAttention计算时，需要把所有的请求合成一个list，即actual_seq_lengths_kv=[128, 256, 512, 1024]即可完成对应计算，无需把请求padding到统一的长度。

尤其对于不定长输入的场景，TND格式解决了易用性以及计算资源浪费的问题。但是，实际计算时，TND格式会引入以下问题：

1. **数据排布不亲和**：对于FlashAttention计算不亲和，数据访问不连续，FlashAttention带宽效率较低。
2. **引入额外计算开销**：原始BNSD的格式需要使用transpose变成TND格式，引入一些非必要的计算开销。

为避免上述问题，实际在FlashAttention计算中采用**NTD**的格式，提升带宽效率的同时，也减少了transpose等计算开销。在prefill阶段MLA部分实际的计算流程如下图所示：

<div align="center">
<img src="./figures/image_attention_ntd_format.png" alt="ntd format" width="80%">
</div>

采用该计算方式的优点如下：
1. 相对于TND格式，FlashAttention的计算效率提升15%。
2. 配合matmul权重拆分以及FlashAttention支持独立的NoPE(No Positional Encoding)和RoPE(Rotary Positional Encoding)输入，可以完全消除split/concat/transpose等额外计算开销，提升MLA阶段的计算效率。

## micro-batch优化
prefill阶段通常属于计算bound的场景，同时又有一定比例的通信耗时开销，通信耗时占比可达30%。而计算和通信由不同的硬件单元完成，如果此时计算和通信能够并行运行，将进一步提升prefill阶段的性能。理想情况下，通信耗时可以完全被计算掩盖，这样计算效率将极大提升。

因此，可以把输入拆分成两个micro-batch，采用双流的方式，构造两条流水线，通过定制化的流水排布，把计算和通信并行起来，最大化prefill阶段的计算效率。

将输入拆分成两个micro-batch，数据被一分为二，会改变prefill阶段每一层计算的shape，shape的变动会导致计算和通信的效率发生变化；同时在eager模式下，task下发的数量也会多一倍，对host要求更高。因此在prefill阶段做micro-batch优化时，需要确认以下问题的影响：

1. **计算和通信性能线性度较好**：要求输入数据切分前计算和通信的性能，近似等于切分后两份数据的计算和通信性能加和，这样可以最小化避免因shape变化导致的性能劣化。
2. **计算未出现host bound**：如果模型本身就是host bound场景，此时通过双流实现micro-batch会导致host性能劣化，整体耗时劣化。

### MLA-DP和MoE-EP场景micro-batch策略
MLA采用DP切分的情况下，整个prefill阶段的通信算子主要来源于MoE阶段的AlltoAll通信（dispatch阶段和combine阶段）。因此，该阶段micro-batch的核心目标就是通过计算掩盖dispatch和combine两个阶段的通信耗时，最大化prefill阶段的计算效率。
该场景的流水排布如下图所示：

<div align="center">
<img src="./figures/image_prefill_dp_mb.png" alt="micro-batch" width="80%">
</div>

其主要步骤如下：

1. **计算和通信划分**：将prefill的计算划分为6个阶段，分别为：atten、init_routing、dispatch、mlp、combine、finalize_routing，
图中： atten主要包含MLA和o_proj计算，dispatch主要包含将token分发到对应专家的AlltoAll通信，mlp主要包含路由专家和共享专家计算，combine主要包含把token恢复到各卡上的AlltoAll通信。

2. **micro-batch预处理**：将输入划分成2个micro-batch进行处理，假设每卡上面有2个4k请求，则每个micro-batch处理1个4k请求，当2个micro-batch全部计算完成时，则完成所有的计算处理。

3. **通信处理**：由于通信和计算并行，此时不能使能HCCL_OP_EXPANSION_MODE的AIV模式，采用默认SDMA模式。如果采用AIV模式，则会导致通信和计算硬件资源抢占，时延变长。

4. **创建流**：针对DP+EP的切分策略场景，将所有的计算放在默认计算流上。
   ```python
   curr_stream = torch.npu.current_stream()
   ```
    新建一条流，用于通信：
   ```python
   stream1 = torch.npu.Stream()
   ```

5. **流水排布**：将计算放到当前流上进行，如下所示：
   ```python
    with torch.npu.stream(curr_stream):
        # 在curr_stream进行atten计算
        atten()
   ```
   将通信放到stream1上进行计算：
   ```python
    with torch.npu.stream(stream1):
        # 在stream1进行atten计算
        dispatch()
   ```
   针对需要同步的位置，可以采用以下方式进行两条流的同步操作：
   ```python
   event0 = curr_stream.record_event()
   ...
   stream1.wait_event(event0)
   ```

通过使能micro-batch计算，dispatch和combine阶段的通信完全被计算掩盖，整个prefill阶段完全处于计算bound的状态，芯片侧无空闲时间，因此端到端的计算效率得到提升。

基于Atlas A3 系列产品，prefill采用2机共计16卡环境部署，输入序列为64*4K，假设MoE层专家负载均匀条件下，性能测试结果如下：

| 切分策略 | 时延(ms) | 总吞吐(tokens/s) | 平均吞吐(tokens/卡/s) |
|------|------|------|------|
| MLA-DP32 + MoE-EP32 | 3128   | 83805 | 5237
| MLA-DP32 + MoE-EP32 + micro-batch | 2512 | 104356 | 6522 |

该切分策略下，开启micro-batch特性可获得25%的性能提升。

## Matmul类算子使能weightNZ格式
针对DeepSeek模型中的Matmul类计算，可以将模型权重中的weight Tensor处理成昇腾AI处理器亲和的格式（weightNZ），这样可以进一步提升Matmul的计算效率。

weightNZ格式为512Bytes对齐的格式，可以有效提升硬件的带宽访问效率，以int8类型的Tensor，shape是[16384, 7168]为例，转换为NZ格式为[224, 1024, 16, 32]

## 附录
[环境部署以及样例执行](../../../models/deepseek_r1/README.md)
