# custom-npu_moe_init_routing_group_quant

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

- 算子功能：MoE的routing计算，根据MoE Gating TopK的计算结果做routing处理，支持不量化和动态量化模式。

- 计算公式：  

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdx：

    $$
    expandedRowIdx[sortedRowIdx[i]]=i
    $$

  3.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsum：

    $$
    expertTokensCountOrCumsum[i]=Histogram(sortedExpertIdx)
    $$

  4.计算quant结果：
    - 动态quant：
        - 若不输入scale：
            $$
            dynamicQuantScaleOut = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOut)
            $$
        - 若输入scale:
            $$
            dynamicQuantScaleOut = row\_max(abs(x * scale)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOut)
            $$
  
  5.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedX[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
    $$

  6.expandedRowIdx的有效元素数量availableIdxNum计算方式为，expertIdx中activeExpertRange范围内的元素的个数
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## 函数原型
```
custom.npu_moe_init_routing_group_quant(Tensor x, Tensor expert_idx, Tensor? scale=None, Tensor? offset=None, int active_num=-1, int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=-1, int expert_tokens_num_type=-1, bool expert_tokens_num_flag=False, int quant_mode=-1, SymInt[] active_expert_range, int row_idx_type=-1, int group_size=128) ->(Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>MOE的输入，即token特征输入，对应公式中x。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_idx</td>
      <td>输入</td>
      <td>每一行特征对应的K个处理专家，里面元素专家id不能超过专家数。对应公式中 expertIdx。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选输入</td>
      <td>表示用于计算quant结果的参数。如果不输入表示计算时不使用scale，对应公式中scale。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>可选输入</td>
      <td>表示用于计算quant结果的偏移值。在非量化场景下和动态quant场景下不输入，对应公式中offset。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>active_num</td>
      <td>属性</td>
      <td>表示总的最大处理row数，输出expanded_x只有这么多行是有效的。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expert_capacity</td>
      <td>属性</td>
      <td>表示每个专家能够处理的tokens数，取值范围大于等于0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expert_num</td>
      <td>属性</td>
      <td>表示专家数，expert_tokens_num_type为key\_value模式时，取值范围为[0, 5120], 其它模式取值范围[0, 10240]。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>drop_pad_mode</td>
      <td>属性</td>
      <td>表示是否为 DropPad 场景，取值为 0 和 1。
        <li>0：表示 Dropless 场景，该场景下不校验 expert_capacity。</li>
        <li>1：表示 DropPad 场景。</li></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expert_tokens_num_type</td>
      <td>属性</td>
      <td>取值为0、1和2 。
        <li>0：表示 comsum 模式。</li>
        <li>1：表示 count 模式，即输出的值为各个专家处理的 token 数量的累计值。</li>
        <li>2：表示 key\_value 模式，即输出的值为专家和对应专家处理 token 数量的累计值。</li></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expert_tokens_num_flag</td>
      <td>属性</td>
      <td>取值为false和true。
        <li>false：表示不输出 expert_tokens_count_or_cumsum。</li>
        <li>true：表示输出 expert_tokens_count_or_cumsum。</li></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>取值为-1、0、1、2、3、4、5。
        <li>-1：表示不量化场景。</li>
        <li>0：表示静态 quant 场景。</li>
        <li>1：表示动态 quant 场景。</li>
        <li>2：表示MXFP8量化场景，expanded_x量化到FLOAT8_E5M2。</li>
        <li>3：表示MXFP8量化场景，expanded_x量化到FLOAT8_E4M3FN。</li>
        <li>4：表示PerGroup量化，group_size固定为128，expanded_x量化到FLOAT8_E5M2，scale的dtype为float。</li>
        <li>5：表示PerGroup量化，group_size固定为128，expanded_x量化到FLOAT8_E4M3FN，scale的dtype为float。</li>
      </td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>active_expert_range</td>
      <td>可选属性</td>
      <td>长度为2，数组内的值为[expertStart, expertEnd], 表示活跃的expert范围在expertStart和expertEnd之间，左闭右开。要求值大于等于0，并且expertEnd不大于expertNum。不输入则不使能专家筛选功能。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>row_idx_type</td>
      <td>属性</td>
      <td>表示expanded_row_idx使用的索引类型，取值为0、1。（性能模板仅支持1）
        <li>0：表示gather类型的索引。</li>
        <li>1：表示scatter类型的索引。</li></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expanded_x</td>
      <td>输出</td>
      <td>根据expert_idx进行扩展过的特征。非量化场景下数据类型同x，量化场景quant_mode为0、1时数据类型支持INT8，quant_mode为2、3时数据类型分别支持FLOAT8_E5M2、FLOAT8_E4M3FN。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expanded_row_idx</td>
      <td>输出</td>
      <td>expanded_x和x的索引映射关系，前available_idx_num\*H个元素为有效数据，其余无效数据，当row_idx_type为0时，无效数据由-1填充；当row_idx_type为1时，无效数据未初始化。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_tokens_count_or_cumsum</td>
      <td>输出</td>
      <td>
        <li>在expert_tokens_num_type为1的场景下，表示active_expert_range范围内expert对应的处理token的总数。</li>
        <li>在expert_tokens_num_type为2的场景下，表示active_expert_range范围内token总数为非0的expert，以及对应expert处理token的总数。</li></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expanded_scale</td>
      <td>输出</td>
      <td>输出量化计算过程中scale的中间值。</li></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明
- 输入值域限制：
  - active_num 当前未使用，校验需等于NUM_ROWS*K。
  - expert_capacity 当前未使用，仅校验非空。
  - offset 当前未使用。
  - drop_pad_mode 当前只支持0，代表 Dropless 场景。
  - expert_tokens_num_type 当前只支持 1 和 2，分别代表 count 模式和 key\_value 模式。
  - expert_tokens_num_flag 只支持 true，代表输出 expert_tokens_count_or_cumsum。
  - quant_mode: 支持-1、0、1、2、3、4、5，其中-1表示不量化，0表示静态量化（输出INT8），1表示动态量化（输出INT8），2表示MXFP8量化到FLOAT8_E5M2，3表示MXFP8量化到FLOAT8_E4M3FN。4模式是groupSize固定为128的PerGroup量化，expand_x量化到float8_e5m2类型，scale为float类型。5模式是groupSize固定为128的PerGroup量化，expand_x量化到float8_e4m3类型，scale为float类型。

- 输入shape约束：
  - x ： shape为 (N, H)
  - expert_id : shape为(N, topK)
  - scale : shape为(N, )或者(expert, H)

- 输出shape约束：
  - expanded_x : shape为 (N*K, H)
  - expanded_row_idx : shape为 (N*K)
  - expert_tokens_count_cumsum : expert_tokens_num_type为1时，shape为 (expert, ) expert_tokens_num_type为2时，shape为 (expert, 2)
  - expanded_scale : quant_mode = 4或者quant_mode = 5时，shape为 (N*K, M)，其中M=Ceil(H, 128) 

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

    
## 调用示例

- 详见 [test_npu_moe_init_routing_group_quant.py](../examples/test_npu_moe_init_routing_group_quant.py)