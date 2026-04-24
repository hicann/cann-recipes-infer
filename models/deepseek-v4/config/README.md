### YAML Parameter Description
The configuration instructions in the YAML file can be found below.
```yaml
Basic Config
  model_name: "deepseek_v4"                           # Model name. String type
  model_path: "/data/models/deepseek_v4_int8_w8a8"    # Weights path. String type
  exe_mode: "npugraph_ex"                               # Execution mode. Only support ["eager", "npugraph_ex"]
  world_size: 128                                       # Global rank num. Int type

Model Config
  pa_block_size: 128              # PA Block Size value. Support [128]
  with_ckpt: True                 # Whether load ckpt. Support [False, True]
  enable_multi_streams: True      # Whether enable multistream to improve performance. Support [False, True]
  enable_profiler: True           # Whether enable profiling. Support [False, True]
  enable_cache_compile: False     # Whether enable cache compile for better successive performance. Support [False, True]
  prefill_mini_batch_size: 0      # Mini_batch_size for prefill stage. Support [0, 1, 2, 3]
  perfect_eplb: False             # If enabled, will force uniform selection of MoE experts. Support [False, True]
  enable_online_split_weight: True  # Whether enable online-split weight. Support [False, True]
  next_n: 1                       # Steps using multi-token prediction. Support [0, 1, 2, 3]
  platform_version: "A3"          # inference platform. Support ["A3", "950"]
  enable_pypto: False             # Whether enable pypto operators. Support ["True", "False"]

Data Config
  dataset: "default"  # Support ["default" "InfiniteBench" "LongBench"]
  input_max_len: 8192 # Max input prompt length
  max_new_tokens: 256 # Max inferred new tokens
  batch_size: 128     # Global batch size
  temperature: 1.0    # Float that controls the randomness of the sampling. Lower values make the model more deterministic,
                      # while higher values make the model more random. Zero means greedy sampling.

Parallel Config
  cp_size: 1          # Prefill CP Number. Only support [1, world_size]
  attn_tp_size: 1     # Attention TP Number. Only support [1]
  oproj_tp_size: 1    # Oproj TP Number. Only support [1, 4, 8]
  moe_tp_size: 1      # MoE TP Number. Only support [1]
  embed_tp_size: 16   # Embed TP Number. Only support [1, 4, 8 16]
  lmhead_tp_size: 16  # LMHead TP Number. Only support [1, 4, 8 16]

```
