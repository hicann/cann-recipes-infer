### YAML Parameter Description

Hunyuan-Image-3.0 inference parameters are maintained in `config/*.yaml`. `infer.sh` uses `ep8_cfg.yaml` by default. The default config is EP8 + CFG parallel + VAE parallel with 16 cards.

```yaml
model_args:
  model-id: "./ckpts/weight_ep8"    # Converted EP/TP weight directory.
  prompt: "A cinematic ..."         # Text prompt for image generation.
  image-size: "1024x1024"           # Output image size. Options include "auto", "HxW", "H:W", and "WxH".
  seed: 42                          # Random seed.
  diff-infer-steps: 50              # Number of denoising steps.
  reproduce: true                   # Whether to enable deterministic generation. Options: [false, true].
  attn-impl: "npu"                  # Attention implementation. Options: ["sdpa", "flash_attention_2", "npu"].
  moe-impl: "npu_grouped_matmul"    # MoE implementation. Options: ["eager", "flashinfer", "npu_grouped_matmul"].
  moe-ep: true                      # Whether to enable MoE expert parallelism. Options: [false, true].
  moe-tp: false                     # Whether to enable MoE tensor parallelism. Options: [false, true].
  use-system-prompt: "None"         # System prompt mode. Options: ["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"].
  bot-task: "image"                 # Generation task mode. Options: ["image", "auto", "think", "recaption"].
  rewrite: 0                        # Whether to rewrite the prompt. 0 disables rewriting.
  sys-deepseek-prompt: "universal"  # DeepSeek rewrite system prompt. Options: ["universal", "text_rendering"].
  save: "image.png"                 # Output image path.
  verbose: 0                        # Log verbosity.

model_name: "hunyuan-image-3.0"     # Model name. Options: ["hunyuan-image-3.0"].
world_size: 16                      # Number of launched processes. With CFG_PARALLEL="1", this is usually 2 * attention TP size.
master_port: 10086                  # torchrun master port.
entry_script: "run_image_gen.py"    # Entry script. Options: ["run_image_gen.py"].

env_vars:
  CFG_PARALLEL: "1"                 # Whether to enable CFG parallelism. Options: ["0", "1"].
  USE_VAE_PARALLEL: "1"             # Whether to enable VAE parallelism. Options: ["0", "1"].
  CPU_AFFINITY_CONF: "2"            # CPU affinity configuration.
```

Notes:

- `ep8_cfg.yaml` is the recommended default config for converted EP8 weights.
- Set `CFG_PARALLEL: "0"` and `world_size: 8` to disable CFG parallel.
- `model-id` must point to converted weights, not the original HuggingFace weight directory.
