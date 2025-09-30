# YAML Parameter Description

## Basic Config
- model_name: "gpt_oss_20b"         # string type
- model_path: "your_model_path"     # string type
- exe_mode: "eager"                 # string type. Only support "eager"
- world_size: 1                     # int type

## Model Config
- enable_profiler: False            # whether enable profiling. support [False, True]
- enable_auto_split_weight: True    # whether enable auto-split weight. support [False, True]

## Data Config
- dataset: "default"                # support ["default" "InfiniteBench" "LongBench"]
- input_max_len: 32                 # the input max length 
- max_new_tokens: 100               # max new tokens
- batch_size: 1                     # Global batch size

## Parallel Config
- tp_size: 16                       # LMHead/Attention/MoE TP Number