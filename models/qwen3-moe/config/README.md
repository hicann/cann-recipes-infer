### YAML Parameter Description
The configuration instructions in the YAML file can be found below.
```yaml
Basic Config
  model_name: "qwen3_moe"                           # The model name. String type
  model_path: "/data/models/origin/Qwen3-235B-A22B" # The model path. String type
  exe_mode: "ge_graph"                              # The execution mode. Only support ["ge_graph", "eager", "acl_graph"]
  world_size: 128                                   # The world size. Int type

Model Config
  with_ckpt: True                 # Whether load ckpt. Support [False, True]
  enable_profiler: True           # Whether enable profiling. Support [False, True]
  enable_cache_compile: False     # Whether enable cache compile for better performance. Support [False, True]
  perfect_eplb: False             # Whether enable, test uniform scenario of MoE experts. Support [False, True]
  enable_auto_split_weight: True  # Whether enable onlinesplit weight. Support [False, True]

Data Config
  dataset: "default"  # Support ["default", "LongBench"]
  input_max_len: 4096 # The input max length
  max_new_tokens: 256 # The max new tokens
  batch_size: 128     # The global batch size

Parallel Config
  attn_tp_size: 1     # Attention TP Number
  moe_tp_size: 1      # MoE TP Number
  embed_tp_size: 16   # Embed TP Number
  lmhead_tp_size: 16  # LMHead TP Number
```