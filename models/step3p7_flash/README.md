# Step-3.7-Flash NPU 推理

Step-3.7-Flash 是多模态 MoE 大模型（视觉编码器 + MoE Decoder-only 文本主干 + 3 层 MTP），文本主干约 197B 参数 / BF16 ~394 GB。本目录提供其在昇腾 Atlas A3 上的 8 卡 / 16 rank 推理实现（基于 cann-recipes-infer 统一流程）：既能跑纯文本，也能「看图说话」。

**特性一览**

- **混合注意力**：12 个全局层 + 33 个滑窗层（window=512），各自独立的 RoPE 与 KVCache 池
- **大规模 MoE EP**：288 experts / top-8，EP16 专家并行（18 experts/rank）
- **Decode 图模式**：npugraph_ex（aclgraph 捕获重放），Decode 单步 ~85 ms → ~28 ms
- **图文理解**：视觉编码器 + 图像注入，复用已优化的文本主干，纯文本路径零影响

> 默认 8 卡 / 16 rank BF16；MTP 默认不启用。

## 模型架构（文本主干）

| 模块 | 配置 |
|------|------|
| 层数 | 45 decoder 层（dense 层 0-2，MoE 层 3-44） |
| Hidden / 词表 | 4096 / 128896 |
| 注意力 | GQA，混合 full（每 4 层 1 层，共 12 层，Q=64/KV=8，θ=5e6，partial_rotary=0.5，叠加 llama3 yarn）/ sliding（33 层，Q=96/KV=8，window=512，θ=1e4，partial_rotary=1.0） |
| 注意力特性 | per-head q/k RMSNorm（head_dim=128）；head-wise sigmoid 输出门控 |
| MoE | 288 experts，top-8，sigmoid + router_bias 路由（fp32 gate）+ 重归一化，scaling=3.0；每层 1 个 shared expert（dim=1280） |
| SwiGLU clamp | 层 43/44：MoE limit=7，shared limit=16 |
| 规模 | 文本主干 ~197B，BF16 ~394 GB（含视觉/MTP 全量 ~201B / 402.7 GB） |

## 并行策略（8 卡 / 16 rank，Atlas A3）

单 die 无法容纳 ~394 GB BF16 主干，需 8 卡 / 16 rank 分布。默认配置：

| 模块 | 切分 | 说明 |
|------|------|------|
| Attention | `attn_tp=1`（DP16） | 数据并行最大化吞吐 |
| MoE | `moe_tp=1`（EP16） | 288 / 16 = 18 experts/rank |
| Dense FFN（层 0-2） | `dense_tp=8` | |
| Embed / LMHead | `tp=16` | |

> EP 固定为 16（受 experts/rank ≤ 24 约束）；`attn_tp` 可取 {1, 2, 4, 8}。其他并行组合可在 `config/` 下自行调整。

## 环境与权重

- 环境：CANN 9.0.0 + PyTorch 2.8.0 + torch_npu 2.8.0 + transformers 5.0.0（详见 `requirements.txt`）。
- 权重：从 [stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) 下载，路径由 YAML 的 `model_config.model_path` 指定（默认 `/data/model/Step-3.7-Flash`）。

## 运行

复用统一流程入口 `executor/scripts/infer.sh`，在仓库根目录、source CANN 环境后执行：

```bash
bash executor/scripts/infer.sh --model step3p7_flash --yaml step3p7_flash_rank16_attndp16_ep16.yaml   # 8 卡 / 16 rank eager
```

### Decode 图模式

图模式（npugraph_ex / aclgraph）仅作用于 Decode（Prefill 保持 eager），显著降低单步时延：

```bash
bash executor/scripts/infer.sh --model step3p7_flash --yaml step3p7_flash_rank16_attndp16_ep16_decode_npugraphex.yaml  # npugraph_ex 图模式
```

| 执行模式 | Decode 单步 | 相对 eager |
|----------|------------|-----------|
| eager | ~85 ms | 1× |
| npugraph_ex | **~28 ms** | **~3×** |

> npugraph_ex 捕获 Decode 计算图并重放；部分 torch_npu 版本需额外适配（如 `static_kernel` import 修复，见配置文件注释）。

## 图像输入

图像经视觉编码器（perception_encoder，47 层 ViT + 下采样 + 投影）编码为 image embedding，在 `<im_patch>` 占位处注入文本序列，复用文本主干的 decoder 与 KVCache。图像推理由 `infer_vision.sh` 运行，与纯文本路径相互独立。

```bash
cd models/step3p7_flash
bash dataset/fetch_test_image.sh                                   # 取一张样例图（图不入仓，也可用自己的图）
bash infer_vision.sh                                               # 默认配置 + 样例图
bash infer_vision.sh step3p7_flash_rank16_attndp16_ep16.yaml "Describe this image." /path/to/img.jpg
```

**示例**（8 卡 / 16 rank BF16，贪心解码；样例图：沙发上的两只猫）

```
Prompt:  Describe this image in detail.
Output:  The main subjects are two cats sleeping on a pink surface — a couch or
         sofa covered with a pink blanket. The left cat is a tabby with grey /
         brown / black stripes and is wearing a green collar; the right cat is
         larger, a tabby with brown / orange tones and a fluffy belly. Near the
         top there are two remote controls resting on the couch.
```

> 作为对照：同一 prompt **不带图** 时，模型只能凭空臆造出与该图无关的内容 —— 说明图像确实参与了条件生成，而非被忽略。

视觉依赖见 `requirements.txt`（torchvision / Pillow），纯文本推理不需要。

模型注册于 `executor/core/support_models.py`（key = `step3p7_flash`）。
