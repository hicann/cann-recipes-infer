### YAML Parameter Description

SANA-Video inference parameters are maintained in `config/*.yaml`. `infer.sh` uses `2b_480p_single.yaml` by default.

Default configs:

- Local offline inference: `2b_480p_single.yaml`
- A8W8 quantized inference: `2b_480p_single_a8w8.yaml`
- A4W4 quantized inference: `2b_480p_single_a4w4.yaml`
- Platform template: `2b_480p_single_platform.yaml`

```yaml
model_args:
  config: "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml"
                                   # Upstream model structure config.
  model_path: "./SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"
                                   # DiT / transformer checkpoint path. Platform template may use hf:// URI.
  vae.vae_pretrained: "./SANA-Video_2B_480p/vae/Wan2.1_VAE.pth"
                                   # VAE checkpoint path.
  text_encoder.text_encoder_name: "./gemma-2-2b-it"
                                   # Gemma text encoder local HuggingFace directory.
  txt_file: "asset/samples/video_prompts_samples.txt"
                                   # Text prompt file. One prompt per line.
  work_dir: "output/sana_t2v_video_results"
                                   # Output directory.
  cfg_scale: 6                      # CFG guidance scale.
  motion_score: 30                  # Motion strength hint.
  flow_shift: 8                     # FlowMatch timestep shift.
  model.fp32_attention: "False"     # Whether to use FP32 attention. Options: ["False", "True"].
  quant_type: "bf16"                # Quantization mode. Options: ["bf16", "a8w8", "a4w4"].

model_name: "sana-video"            # Model name. Options: ["sana-video"].
world_size: 1                       # Number of launched processes. Can be increased for DP.
master_port: 29600                  # accelerate main process port.
entry_script: "inference_video_scripts/inference_sana_video.py"
                                   # Entry script. Options: ["inference_video_scripts/inference_sana_video.py"].
launcher: "accelerate"              # Launcher. Options: ["accelerate", "torchrun"].

launcher_args:
  mixed_precision: "bf16"           # accelerate mixed precision. Options: ["no", "fp16", "bf16", "fp8"].

env_vars:
  DISABLE_XFORMERS: "1"             # Whether to disable xFormers. Options: ["0", "1"].
  HF_HUB_OFFLINE: "1"               # HuggingFace Hub offline mode. Options: ["0", "1"].
  TRANSFORMERS_OFFLINE: "1"         # Transformers offline mode. Options: ["0", "1"].
  HF_DATASETS_OFFLINE: "1"          # Datasets offline mode. Options: ["0", "1"].
```

Notes:

- `2b_480p_single.yaml` uses local offline weights.
- `2b_480p_single_a8w8.yaml` and `2b_480p_single_a4w4.yaml` enable MXFP A8W8 / A4W4 quantization.
- `2b_480p_single_platform.yaml` is used by `infer_platform.sh` as a platform template.
