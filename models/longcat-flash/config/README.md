### YAML文件配置说明
The configuration instructions in the YAML file can be found below.
```yaml
Basic Config
model_name: "longcat_flash"       # The model name. String type
model_path: "your_model_path"     # The model path. String type
exe_mode: "ge_graph"              # The execution mode. Only support ["ge_graph", "eager"]
world_size: 32                    # The world size. Int type

Model Config
  enable_profiler: False     # Whether enable profiling. Support [False, True]
  with_ckpt: True            # Whether load ckpt. Support [False, True]
  enable_online_split_weight: True   # Whether enable auto-split weight. Support [False, True]
  pa_block_size: 128         # PA Block Size value. Support [128, 256]
  perfect_eplb: False        # Whether enable, test uniform scenario of MoE experts. Support [False, True]
  moe_chunk_max_len: 1024    # Moe layer chunk max length. Int type
  next_n: 0                  # Number of steps using multi-token prediction. Support [0, 1, 2]
  enable_multi_stream: 0 # Whether enable multistream for better performance, set for different core numbers. Support [0, 1, 2] with 0 for not using multistream. AFD scenario, the switch will not take effect.
  enable_mla_prolog: True    # Whether use mla_prolog fusion operator. Support [False, True]
  enable_cache_compile: False # Whether enable cache compile for better performance. Support [False, True]
  enable_superkernel: False   # Whether enable superkernel. Support [False, True]
  enable_prefetch: True       # Whether enable prefetch. support [False, True]
  decode_only: False          # Whether enable decode-only test for long-sequence scenarios. Support [False, True]
  enable_afd: False           # Whether enable attention-ffn disaggragation. support [False, True]. Note: if True, the first half of the ranks for world_size will serve as FFN, and the second half will serve as attn.

data_config:
  dataset: "LongBench"       # Support ["default", "LongBench"]
  input_max_len: 4608        # The input max length
  max_new_tokens: 32         # The max new tokens
  batch_size: 64             # The global batch size

parallel_config:
  attn_tp_size: 1     # Attention TP Number
  o_proj_tp_size: 8   # O_proj TP Number. When attn_tp_size > 1, attn_tp_size must be equal to o_proj_tp_size
  moe_tp_size: 1      # MoE TP Number
  dense_tp_size: 8    # Dense MLP TP Number
  embed_tp_size: 1    # Embed TP Number
  lmhead_tp_size: 1   # LMHead TP Number
  kvp_size: 1         # KVCache KVP Number. When kvp_size > 1, o_proj_tp_size must be equal to kvp_size. Support kvp_size subject to world_size % kvp_size = 0, defaults to 1(disabled).
```