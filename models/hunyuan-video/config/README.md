### YAML Parameter Description

Hunyuan-Video inference parameters are maintained in `config/*.yaml`. Select a config by setting `YAML_FILE_NAME` in `infer.sh`.

Default configs:

- Single-card baseline: `single.yaml`
- 8-card sequence parallel: `sp8.yaml`
- Single-card FP8: `single_fp8.yaml`
- Single-card sparse attention: `single_sparse.yaml`
- 8-card sparse attention: `sp8_sparse.yaml`
- 8-card Ring sparse overlap: `sp8_sparse_overlap.yaml`
- One-stop platform config: `single_platform.yaml`

```yaml
model_args:
  model-base: "ckpts"              # Weight root directory. Relative paths are resolved from models/hunyuan-video/.
  prompt: "A cat walks ..."        # Text prompt for video generation.
  video-size: [720, 1280]          # Output size in [H, W].
  video-length: 129                # Output frame count. Constraint: 4n+1.
  infer-steps: 50                  # Number of denoising steps.
  seed: 42                         # Random seed.
  embedded-cfg-scale: 6.0          # Embedded CFG guidance scale.
  flow-shift: 7.0                  # FlowMatch timestep shift.
  flow-reverse: true               # Whether to use reverse flow scheduling. Options: [false, true].
  use-cpu-offload: true            # Whether to enable CPU offload. Options: [false, true].
  extract_q_k_data: false          # Whether to extract QK data for sparse attention offline profiling. Options: [false, true].
  extract_path: "path/to/qk_dir"   # Output directory for extracted QK data. Required when extract_q_k_data is true.
  ulysses-degree: 8                # Ulysses sequence parallel degree. Multi-card configs use this field.
  ring-degree: 1                   # Ring attention degree. Sparse configs currently require 1.
  use-vae-parallel: true           # Whether to enable VAE parallelism. Options: [false, true].
  fa-perblock-fp8: true            # Whether to enable FP8 FA activation quantization. Options: [false, true].
  mm-mxfp8: true                   # Whether to enable MXFP8 matmul quantization. Options: [false, true].
  dit-weight: "/abs/path/ckpt.pt"  # Optional DiT checkpoint path.
  model: "HYVideo-T/2-cfgdistill"  # DiT architecture. Options: ["HYVideo-T/2", "HYVideo-T/2-cfgdistill"].
  model-resolution: "720p"         # Model resolution preset. Options: ["540p", "720p"].
  precision: "bf16"                # DiT precision. Options: ["fp32", "fp16", "bf16"].
  seed-type: "auto"                # Seed source. Options: ["file", "random", "fixed", "auto"].

model_name: "hunyuan-video"        # Model name. Options: ["hunyuan-video"].
world_size: 1                      # Number of launched processes. Multi-card configs require world_size = ulysses-degree * ring-degree.
master_port: 29600                 # torchrun master port.
entry_script: "sample_video.py"    # Entry script. Options: ["sample_video.py"].

dit_cache:
  method: "NoCache"                # DiT cache method. Options: ["NoCache", "FBCache", "TeaCache", "TaylorSeer"].
  params:
    # FBCache / TeaCache
    rel_l1_thresh: 0.05            # Relative L1 threshold. Larger values are faster but may reduce quality.

    # TeaCache
    coefficients: []               # TeaCache polynomial coefficients.
    warmup: 2                      # Number of initial full-compute steps.

    # TaylorSeer
    n_derivatives: 3               # Taylor expansion order.
    skip_interval_steps: 4         # Full-compute interval.
    cutoff_steps: 1                # Number of final full-compute steps.
    offload: true                  # Whether to offload TaylorSeer history states to CPU. Options: [false, true].

sparse:
  method: "SVG"                    # Sparse attention method. Options: ["no_sparse", "TopK", "SVG"].
  block_size_Q: 128                # Q-axis block size.
  block_size_K: 512                # K-axis block size.
  model: "HunyuanVideo"            # Sparse module model type. Options: ["HunyuanVideo"].
  params:
    TopK:
      sparse_time_step: "10-49"    # Active denoising step range. Format: "start-end".
      sparsity_files_path: "./sparsity/720x1280x129/v3"
                                   # Offline profiling sparsity file directory.
      CAC_threshold: 0.66          # TopK threshold.
    SVG:
      sparse_time_step: "14-49"    # Active denoising step range. Format: "start-end".
      sparsity: 0.8                # SVG sparsity ratio.
      sample_mse_max_row: 5000     # Maximum sampled rows for MSE.
      context_length: 256          # SVG context length.
```

Notes:

- Sparse attention and DiT cache are mutually exclusive. Keep `dit_cache.method: "NoCache"` in sparse configs.
- `TopK` requires sparsity files that match `video-size` and `video-length`.
- `extract_q_k_data` is used to generate QK data for sparse attention offline profiling. Set `extract_path` to a writable directory when enabling it.
- TaylorSeer may require high host memory at large resolutions and long frame counts.
