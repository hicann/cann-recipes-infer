### YAML Parameter Description

Wan2.2-I2V inference parameters are maintained in `config/*.yaml`. `infer.sh` uses `14b_cfg2_ulysses4.yaml` by default.

Set `YAML_FILE_NAME` to select a file under `config/`, or set `WAN_CONFIG` to an absolute path or a path relative to the model directory. `WAN_CONFIG` takes precedence when both are set.

Default configs:

- Single-card baseline / DiT cache: `14b_single.yaml`
- CANNLab one-stop development platform template: `14b_single_platform.yaml`
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
  prompt_extend_target_lang: "zh"   # Prompt extension target language. Options: ["zh", "en"].
  quant_data_dir: null              # I2V W8A8 artifact root; config.json.quantization_config selects the runtime.
  quant_layout: "nd_view"           # Runtime device layout for the artifact weights.
  quant_ignored_layers: []          # Optional BF16 fallback overrides from the source checkpoint.

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
- `14b_single_platform.yaml` is read by `infer_platform.sh` and patched with the local ModelScope weight path at runtime.
- Multi-card configs and DiT cache are mutually exclusive. Keep `dit_cache.method: "NoCache"` in multi-card configs.

### TaylorSeer parameters

Use `14b_taylorseer.yaml` for the single-card Atlas A3, BF16 `i2v-A14B` adaptation. Set the local weight/image paths and run from `models/wan2.2-i2v`:

```bash
WAN_CONFIG=config/14b_taylorseer.yaml bash infer.sh
```

`WAN_CONFIG` selects a YAML path; when unset, the launcher uses `YAML_FILE_NAME` or its default. See the [model README](../README.md#四档复现) for the A3 reproduction scenario and [technical details](../../../docs/models/wan2.2-i2v/wan2.2-i2v_optimization.md#taylorseer) for the implementation and measured results.

```yaml
dit_cache:
  method: "TaylorSeer"
  enable_separate_cfg: true
  params:
    n_derivatives: 2
    skip_interval_steps: 3
    warmup: 2
    cutoff_steps: 2
    offload: false
    prefetch: false
    prefetch_distance: 1
    cache_group_size: 4
    max_inflight: 2
    host_cache_budget_bytes: null
    pinned_budget_bytes: null
    device_cache_budget_bytes: null
```

| Parameter | Default | Meaning and supported values |
| --- | --- | --- |
| `method` | `NoCache` if omitted | `NoCache`, `FBCache`, `TeaCache`, or `TaylorSeer`; unknown names fail instead of silently disabling the cache |
| `enable_separate_cfg` | `true` | Must be `true` for TaylorSeer; conditional and unconditional histories remain separate |
| `n_derivatives` | `2` | Integer 0–3; 0 uses only the last full residual; unavailable higher orders are not fabricated |
| `skip_interval_steps` | `3` | Integer ≥1; distance between scheduled full steps, so 3 permits at most two intervening prediction steps; 1 makes every step full |
| `warmup` | `2` | Integer ≥1; full-compute warmup for each expert epoch; the first periodic full at local step `warmup` is also executed |
| `cutoff_steps` | `2` | Integer ≥0; force full computation for the last `min(cutoff_steps, T)` steps of the whole request |
| `offload` | `false` | Boolean; store Taylor factors on Host instead of keeping all factors on NPU |
| `prefetch` | `false` | Boolean; requires `offload: true`; enables speculative H2D for the next group |
| `prefetch_distance` | `1` | Only integer 1 is supported; means one cache group, not one denoising step; never crosses CFG/expert boundaries |
| `cache_group_size` | `4` | Positive integer no larger than either expert's block count; the final group may be shorter |
| `max_inflight` | `2` | Only integer 2 is supported; two H2D/NPU slots, with a separate single D2H staging bundle and one update credit |
| `host_cache_budget_bytes` | `null` | Positive integer or `null`; cap for ordinary Host factors, including one publication window |
| `pinned_budget_bytes` | `null` | Positive integer or `null`; cap for two H2D plus one D2H pinned staging bundle |
| `device_cache_budget_bytes` | `null` | Positive integer or `null`; cap for cache factors/slots and reserved cache scratch, excluding model weights/activations |

`null` budgets resolve to finite shape-dependent requirements and are reported as `WAN_CACHE_BUDGET`; they do not mean unlimited memory. With FP32 feature bytes `b = 4 * B * S * C_hidden`, maximum order `q`, groups `G`, bundle `U = (q + 1) * b`, and two CFG branches, resident factors require `C_cache = 2 * G * U`. Offload uses at most `C_cache + U` ordinary Host bytes, `3U` pinned bytes, and a cache-related NPU budget of `3U + 4b`; the resident path uses `C_cache + U + 4b` on NPU. Model weights, activations, framework workspaces, and allocator reservation are additional. A smaller configured cap fails before cache allocation.

Boolean values are not valid integers. Unknown or duplicate keys, string values for boolean cache flags, unsupported derivative orders, invalid intervals/distances, `prefetch: true` without offload, or Taylor parameters under another method cause a nonzero startup failure. TaylorSeer requires batch 1, one process/device, all parallel sizes equal to 1, and no FSDP, VAE parallelism, quantization or concurrent requests.

`offload_model` controls model weights, independently of Taylor cache `offload`. The parser accepts both `--offload_model` and `--offload_model true/false`. YAML `offload_model: true` produces the flag. The generic launcher omits boolean `false`, so use the quoted string `offload_model: "false"` when an explicit false argument is needed. The four-case reproduction uses `offload_model: true`.

`model_args.reuse_offload_memory` defaults to `false`; the [A3 reproduction scenario](../README.md#四档复现) selects `true`. The generate CLI accepts `--reuse_offload_memory [true|false]`. Only `true` together with model offload skips the pre-denoising and retired-expert allocator flushes. Expert CPU/NPU moves, request-tail/VAE/exception cleanup and cache drain remain unchanged. With `offload_model: false`, even an `init_on_cpu` expert switch keeps the old flush. This option requires single-process I2V without FSDP, SP, parallel CFG/TP or VAE parallelism; allocator reservation must be measured separately from live cache bytes.

For rollback, select `method: "NoCache"` and remove Taylor-only parameters, or use `14b_single.yaml` with the same workload fields. Switching `offload` off also requires `prefetch: false`. Reconfigure between completed requests or restart inference; do not mutate flags during generation.

### Dual-expert W8A8 configuration

`14b_quantization.yaml` runs Wan2.2-I2V with W8A8 Attention/FFN Linear layers. Set `model_args.quant_data_dir` to the artifact root containing `config.json`; its `quantization_config` selects the W8A8 loading path.

This configuration uses `NoCache`. W8A8 quantization cannot be combined with TaylorSeer.

| Field | Description |
|---|---|
| `quant_mode` | I2V mode selected from YAML: 0 normal, 1 calibration export, 2 model export, 3 quantized inference. |
| `quant_data_dir` | Artifact root containing both noise experts and their quantization configuration. |
| `quant_layout` | Device weight layout: `nd_view`, `nd_contiguous`, or `nz`. |
| `quant_ignored_layers` | Optional exact/glob layer names restored from the source BF16 checkpoint. |

I2V reads all four quantization settings once from the launch YAML into the model config; `WanI2V` consumes that config. With `infer.sh`, use `model_args`; for direct `generate.py` invocation, select the YAML with `--cache_config`. The common CLI options remain accepted for launcher compatibility and other tasks.

The common entry point keeps quantization modes 0/1/2/3: normal inference, calibration export, quantized-model export, and quantized inference. I2V modes 1/2 retain the baseline branches and do not load W8A8 artifacts automatically. Conversion and launch commands are in the [model README](../README.md#w8a8-量化推理).
