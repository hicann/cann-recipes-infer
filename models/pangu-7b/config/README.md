### YAML Parameter Description
The configuration instructions in the YAML file can be found below.
```yaml
Basic Config
  model_name: "pangu_7B"                              # The model name. String type
  model_path: "/dev/shm/ckpts/openPangu-Embedded-7B" # The model path. String type
  exe_mode: "acl_graph"                              # The execution mode. Support ["ge_graph", "eager", "acl_graph"]
  world_size: 1                                      # The world size. Int type

Model Config
  mm_quant_mode: A16W16       # Support ["A16W16", "A8W8"]
  gmm_quant_mode: A16W16      # Support ["A16W16", "A8W8"]
  with_ckpt: True             # Whether load ckpt. Support [False, True]
  enable_profiler: False      # Whether enable profiling. Support [False, True]
  enable_cache_compile: False # Whether enable cache compile. Support [False, True]
  enable_weight_nz: False     # Whether enable weight NZ format. Support [False, True]
  enable_online_split_weight: True # Whether enable online split weight. Support [False, True]
  tokenizer_mode: "default"   # Support ["default", "chat"]

Data Config
  dataset: "default"   # Support ["default", "LongBench"]
  input_max_len: 2048  # The input max length
  max_new_tokens: 256  # The max new tokens
  batch_size: 1        # The global batch size

Parallel Config
  attn_tp_size: 1   # Attention TP Number
  moe_tp_size: 1    # MoE TP Number
  embed_tp_size: 1  # Embed TP Number
  lmhead_tp_size: 1 # LMHead TP Number
```
