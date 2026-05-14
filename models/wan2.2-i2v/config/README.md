### YAML Parameter Description

Wan2.2-I2V inference parameters are maintained in `config/*.yaml`. `infer.sh` uses `14b_cfg2_ulysses4.yaml` by default.

Default configs:

- Single-card baseline / DiT cache: `14b_single.yaml`
- 8-card high-resolution inference: `14b_cfg2_ulysses4.yaml`

```yaml
model_args:
  ckpt_dir: "/path/to/Wan2.2-I2V-14B"
                                   # Weight directory.
  image: "examples/i2v_input.JPG"   # Input image path for I2V.
  prompt: "Summer beach ..."        # Text prompt.
  task: "i2v-A14B"                  # Task. Options: ["t2v-A14B", "i2v-A14B", "ti2v-5B", "animate-14B", "s2v-14B"].
  size: "640*360"                   # Output size. Options: ["720*1280", "1280*720", "480*832", "832*480", "640*360", "360*640", "704*1280", "1280*704", "1024*704", "704*1024"].
  frame_num: 61                     # Output frame count. Constraint: 4n+1.
  sample_steps: 40                  # Number of denoising steps.
  sample_solver: "unipc"            # Sampling solver. Options: ["unipc", "dpm++"].
  sample_shift: null                # FlowMatch shift. null means task default.
  sample_guide_scale: null          # CFG guidance scale. null means task default.
  base_seed: 100                    # Random seed.
  convert_model_dtype: true         # Whether to cast model parameters for inference. Options: [false, true].
  offload_model: null               # Whether to offload model to CPU. Options: [false, true, null].
  cfg_size: 1                       # CFG parallel degree. Current CFG parallel values: 1 or 2.
  ulysses_size: 1                   # Ulysses sequence parallel degree. Must divide the task head count.
  ring_size: 1                      # Ring attention parallel degree.
  tp_size: 1                        # Tensor parallel degree. Current code only accepts 1.
  dit_fsdp: true                    # Whether to enable DiT FSDP. Options: [false, true].
  t5_fsdp: true                     # Whether to enable T5 FSDP. Options: [false, true].
  t5_cpu: false                     # Whether to place T5 on CPU. Options: [false, true].
  vae_parallel: true                # Whether to enable VAE parallelism. Options: [false, true].
  use_prompt_extend: false          # Whether to enable prompt extension. Options: [false, true].
  prompt_extend_method: "local_qwen"
                                   # Prompt extension method. Options: ["dashscope", "local_qwen"].
  prompt_extend_target_lang: "zh"  # Prompt extension target language. Options: ["zh", "en"].
  quant_mode: 0                     # Quantization workflow. Options: [0, 1, 2, 3].

model_name: "wan2.2-i2v"            # Model name. Options: ["wan2.2-i2v"].
world_size: 1                       # Number of launched processes. Must equal cfg_size * ulysses_size * ring_size * tp_size.
master_port: 29600                  # torchrun master port.
entry_script: "generate.py"         # Entry script. Options: ["generate.py"].

env_vars:
  PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
                                   # NPU memory allocator config.
  TASK_QUEUE_ENABLE: "2"            # NPU task queue config.
  CPU_AFFINITY_CONF: "1"            # CPU affinity config.

dit_cache:
  method: "NoCache"                 # DiT cache method. Options: ["NoCache", "FBCache", "TeaCache"].
  enable_separate_cfg: true         # Whether to cache CFG branches separately. Options: [false, true].
  params:
    # FBCache / TeaCache
    rel_l1_thresh: 0.05             # Relative L1 threshold.

    # TeaCache
    coefficients: []                # TeaCache polynomial coefficients.
    warmup: 2                       # Number of initial full-compute steps.
```

Notes:

- `14b_single.yaml` can use DiT cache methods `NoCache`, `FBCache`, and `TeaCache`.
- Multi-card configs and DiT cache are mutually exclusive. Keep `dit_cache.method: "NoCache"` in multi-card configs.
