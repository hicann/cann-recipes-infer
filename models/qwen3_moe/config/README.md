# YAML Parameter Description

## Basic Config
- model_name: "qwen3_moe"                           # string type
- model_path: "/data/models/origin/Qwen3-235B-A22B" # string type
- exe_mode: "ge_graph"                              # string type. Only support ["ge_graph", "eager"]
- world_size: 128                                   # int type

## Model Config
- tokenizer_mode: default         # support ["default", "chat"]
- with_ckpt: True                 # whether load ckpt. support [False, True]
- enable_profiler: True           # whether enable profiling. support [False, True]
- enable_cache_compile: False     # whether enable cache compile for better performance. support [False, True]
- perfect_eplb: False             # whether enable, test uniform scenario of MoE experts
- enable_auto_split_weight: True  # whether enable online-split weight. support [False, True]

## Data Config
- dataset: "default"  # support ["default" "InfiniteBench" "LongBench"]
- input_max_len: 8192 # the input max length 
- max_new_tokens: 100 # max new tokens
- batch_size: 128     # Global batch size

## Parallel Config
- attn_tp_size: 1     # Attention TP Number
- moe_tp_size: 1      # MoE TP Number
- embed_tp_size: 16   # Embed TP Number
- lmhead_tp_size: 16  # LMHead TP Number
