# YAML Parameter Description

## Basic Config
- model_name: "deepseek_v3"                         # string type
- model_path: "/data/models/DeepSeek-V3.2-bf16/"    # string type
- exe_mode: "ge_graph"                              # string type. Only support ["ge_graph", "eager"]
- world_size: 128                                   # int type

## Model Config
- mm_quant_mode: A16W16           # Linear operation quant mode. currently only support A16W16
- gmm_quant_mode: A16W16          # GMM operation quant mode. currently only support A16W16
- pa_block_size: 128              # PA Block Size value. support [128, 256] 
- enable_weight_nz: True          # whether use nz-weight format for better performance. support [False, True]
- with_ckpt: True                 # whether load ckpt. support [False, True]
- enable_multi_streams: True      # whether enable multistream for better performance. support [False, True]
- enable_profiler: True           # whether enable profiling. support [False, True]
- enable_cache_compile: False     # whether enable cache compile for better performance. support [False, True]
- prefill_mini_batch_size: 0      # mini_batch_size for prefill stage. 
- perfect_eplb: False             # whether enable, test uniform scenario of MoE experts
- enable_auto_split_weight: True  # whether enable auto-split weight. support [False, True]

## Data Config
- dataset: "default"  # support ["default" "InfiniteBench" "LongBench"]
- input_max_len: 8192 # the input max length 
- max_new_tokens: 100 # max new tokens
- batch_size: 128     # Global batch size

## Parallel Config
- cp_size: 128         # Context Parallel Number. if cp_size > 0, cp_size should equal to world_size. Only active at prefill stage
- attn_tp_size: 1     # Attention TP Number
- oproj_tp_size: 8    # Oproj TP Number. Only support when attn_tp_size == 1
- dense_tp_size: 1    # Dense MLP TP Number
- moe_tp_size: 1      # MoE TP Number
- embed_tp_size: 16   # Embed TP Number
- lmhead_tp_size: 16  # LMHead TP Number
