### YAML Parameter Description
The configuration instructions in the YAML file can be found below.
```yaml
Basic Config
  model_name: "GLM-5.2"                   # The model name. String type
  model_path: "/data/models/GLM-5.2-W8A8" # The model path. String type
  exe_mode: "ge_graph"                              # The execution mode. Only support ["ge_graph", "eager", "npugraph_ex"]
  world_size: 128                                   # The world size. Int type

Model Config
  enable_offload: False            # Whether enable KV offload. Support [False, True]
  shared_indexer_offload: False    # Whether use the GLM-5.2 shared-indexer offload path. Support [False, True]
  enlarge_pool_size: False         # Whether enlarge the resident token pool from 8K to 16K for shared-indexer offload. Support [False, True]
  pa_block_size: 128              # PA Block Size value. Support [128, 256]
  enable_weight_nz: True          # Whether use nz-weight format for better performance. Support [False, True]
  with_ckpt: True                 # Whether load ckpt. Support [False, True]
  enable_multi_streams: True      # Whether enable multistream for better performance. Support [False, True]
  enable_profiler: True           # Whether enable profiling. Support [False, True]
  enable_cache_compile: False     # Whether enable cache compile for better performance. Support [False, True]
  prefill_mini_batch_size: 0      # Mini_batch_size for prefill stage.
  perfect_eplb: False             # Whether enable, test uniform scenario of MoE experts. Support [False, True]
  enable_online_split_weight: True  # Whether enable online-split weight. Support [False, True]
  next_n: 1                       # Steps using multi-token prediction. Support [0, 1, 2, 3]

Data Config
  dataset: "default"  # Support ["default" "InfiniteBench" "LongBench"]
  input_max_len: 8192 # The input max length
  max_new_tokens: 100 # Max new tokens
  batch_size: 128     # Global batch size

Parallel Config
  cp_size: 128        # Context Parallel Number. When using CP, cp_size should be the same as world_size; otherwise, set cp_size to 1. Only active at prefill stage
  attn_tp_size: 1     # Attention TP Number
  oproj_tp_size: 8    # Oproj TP Number. Only support when attn_tp_size == 1
  dense_tp_size: 1    # Dense MLP TP Number
  moe_tp_size: 1      # MoE TP Number
  embed_tp_size: 16   # Embed TP Number
  lmhead_tp_size: 16  # LMHead TP Number
```
