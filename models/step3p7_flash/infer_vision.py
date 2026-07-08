# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end image+text inference for Step-3.7-Flash."""

import argparse
import json
import logging
import os
import struct
import sys

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup: this harness must import both the executor framework and the
# model-local modules (vision_step3p7 / processing_step3p7).
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "..", ".."))
for p in (REPO_ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.append(p)


# ---------------------------------------------------------------------------
# Vision-weight loading.
# ---------------------------------------------------------------------------
def _read_safetensors(path, wanted):
    out = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
        tmap = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                "F64": torch.float64, "I64": torch.int64, "I32": torch.int32}
        for name in wanted:
            if name not in meta:
                continue
            info = meta[name]
            beg, end = info["data_offsets"]
            fh.seek(base + beg)
            raw = fh.read(end - beg)
            dt = tmap.get(info["dtype"])
            if dt is None:
                raise ValueError(f"unsupported safetensors dtype {info['dtype']!r}")
            t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(info["shape"])
            out[name] = t.clone()
    return out


def load_vision_weights(weight_dir):
    """Read all 667 vision tensors (vision_model.* + vit_large_projector.weight)."""
    with open(os.path.join(weight_dir, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    vkeys = [k for k in wm
             if k.startswith("vision_model.") or k == "vit_large_projector.weight"]
    by_shard = {}
    for k in vkeys:
        by_shard.setdefault(wm[k], []).append(k)
    weights = {}
    for shard, keys in by_shard.items():
        weights.update(_read_safetensors(os.path.join(weight_dir, shard), keys))
    return weights


# ---------------------------------------------------------------------------
# Vision embeds: run tower on whole image + sub-patches, merge in placeholder
# order (faithful to HF Step3p7Model._process_image_input: per image, the
# sub-patch features come FIRST, then the whole-image features).
# ---------------------------------------------------------------------------
def build_image_embeds(tower, pixel_values, patch_pixel_values, num_patches):
    """Return flattened [num_placeholders_total, hidden] in placeholder order."""
    param = next(tower.parameters())
    device, dtype = param.device, param.dtype
    whole = tower(pixel_values.to(device).to(dtype))  # (N_img, 169, H)
    patch = (tower(patch_pixel_values.to(device).to(dtype))
             if patch_pixel_values is not None and patch_pixel_values.shape[0] > 0
             else None)  # (N_sub, 81, H)
    merged = []
    cur = 0
    for i, npatch in enumerate(num_patches):
        if npatch > 0:
            sub = patch[cur:cur + npatch]               # (npatch, 81, H)
            merged.append(sub.reshape(-1, sub.shape[-1]))
        merged.append(whole[i].reshape(-1, whole.shape[-1]))  # (169, H)
        cur += npatch
    return torch.cat(merged, dim=0)                     # (total_placeholders, H)


# ---------------------------------------------------------------------------
# Forward wrapper: inject image_embeds on the prefill call only.
# ---------------------------------------------------------------------------
def install_vision_injection(causal_lm, image_embeds, image_token_id):
    """Wrap Step3p5ForCausalLM.forward (instance) to add image_embeds on prefill.

    Decode calls (and any call without a prefill forward_metadata) pass through
    untouched, so KVCache / graph-mode behaviour is unchanged after Prefill.
    """
    orig_forward = causal_lm.forward

    def wrapped(*args, **kwargs):
        fm = kwargs.get("forward_metadata", None)
        if fm is not None and getattr(fm, "is_prefill", False):
            kwargs["image_embeds"] = image_embeds
            kwargs["image_token_id"] = image_token_id
        return orig_forward(*args, **kwargs)

    causal_lm.forward = wrapped
    return orig_forward


def remove_vision_injection(causal_lm, orig_forward):
    causal_lm.forward = orig_forward


# ---------------------------------------------------------------------------
# Drive one pre-tokenized request through the standard scheduling loop.
# ---------------------------------------------------------------------------
def run_pretokenized(llm, input_ids, batch_per_dp):
    """Add `batch_per_dp` copies of the same pre-tokenized request, run to done.

    With attn_dp the engine expects batch_size_per_dp_rank requests per rank; we
    replicate the single prompt so every DP rank prefills the same sequence (the
    same image_embeds is injected on every rank's prefill).
    """
    llm.scheduler.reset()
    rids = []
    for _ in range(max(1, batch_per_dp)):
        rid = llm.scheduler.add_request(prompt="<image+text>", input_ids=input_ids.clone())
        rids.append(rid)
    while llm.scheduler.has_work():
        if not llm.scheduler.run_step(llm.engine):
            break
    outs = []
    for rid in rids:
        req = llm.scheduler.pop_finished_request(rid)
        if req is None:
            outs.append(("", "error"))
            continue
        valid = (req.output_id_list[:req.valid_output_len]
                 if req.valid_output_len is not None else req.output_id_list)
        text = llm.engine.tokenizer.decode(torch.tensor(valid), skip_special_tokens=True)
        outs.append((text, req.finish_reason))
    return outs


def build_chat_prompt(tokenizer, user_text, with_image):
    """Render the chat template; image item emits one <im_patch> placeholder."""
    if with_image:
        content = [{"type": "image"}, {"type": "text", "text": user_text}]
    else:
        content = user_text
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def main():
    from executor.core.config import InferenceConfig
    from executor.offline.offline_inference import OfflineInference
    from executor.utils.logging_config import setup_logging
    from models.vision_step3p7 import VisionConfig, Step3p7VisionTower
    from models.processing_step3p7 import Step3p7Processor

    setup_logging()
    parser = argparse.ArgumentParser(description="Step-3.7-Flash image+text inference")
    parser.add_argument("--yaml_file_path", type=str, required=True)
    parser.add_argument("--image", type=str,
                        default=os.path.join(MODEL_DIR, "dataset", "test_image.jpg"))
    parser.add_argument("--prompt", type=str,
                        default="Describe this image in detail.")
    parser.add_argument("--weight_dir", type=str, default=None,
                        help="override checkpoint dir (default: config model_path)")
    parser.add_argument("--skip_text_control", action="store_true",
                        help="skip the pure-text zero-regression control run")
    args = parser.parse_args()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    global_rank = local_rank + rank_offset

    with open(args.yaml_file_path, "r") as f:
        import yaml
        yaml_dict = yaml.safe_load(f)
    config = InferenceConfig.from_dict(yaml_dict, global_rank=global_rank,
                                       local_rank=local_rank)
    weight_dir = args.weight_dir or config.model_config.model_path
    device = f"npu:{local_rank}"

    # ----- build engine (model + KV cache + warmup), executor untouched -----
    llm = OfflineInference(config)
    tokenizer = llm.engine.tokenizer
    causal_lm = llm.engine.model  # Step3p5ForCausalLM

    attn_dp_size = config.parallel_config.attn_dp_size
    attn_tp_size = config.parallel_config.attn_tp_size
    batch_per_dp = config.scheduler_config.batch_size // max(1, attn_dp_size)

    # ----- vision tower + weights on this rank -----
    with open(os.path.join(weight_dir, "config.json")) as f:
        full_cfg = json.load(f)
    vc = VisionConfig.from_hf_config(full_cfg)
    dtype = torch.bfloat16
    tower = Step3p7VisionTower(vc, use_fa=True).to(device).to(dtype).eval()
    vweights = load_vision_weights(weight_dir)
    n_loaded = tower.load_weights((k, v) for k, v in vweights.items())
    logger.info("[vision] vision tower loaded %d / 667 weights", len(n_loaded))

    # ----- processor: image + text -> expanded input_ids + pixel tensors -----
    from PIL import Image
    proc = Step3p7Processor(tokenizer)
    image = Image.open(args.image)
    chat_text = build_chat_prompt(tokenizer, args.prompt, with_image=True)
    mm = proc(chat_text, images=[image], device=device, dtype=dtype)
    input_ids = mm["input_ids"]
    image_token_id = mm["image_token_id"]
    n_placeholders = int((input_ids == image_token_id).sum().item())
    logger.info("[vision] expanded input_ids: %d tokens, %d <im_patch> placeholders, "
                "num_patches=%s", input_ids.numel(), n_placeholders,
                mm.get("num_patches"))

    # ----- vision embeds (whole + sub-patches), placeholder order -----
    image_embeds = build_image_embeds(
        tower, mm["pixel_values"], mm.get("patch_pixel_values"),
        mm["num_patches"])
    logger.info("[vision] image_embeds shape=%s (expect [%d, %d])",
                tuple(image_embeds.shape), n_placeholders, vc.text_hidden_size)
    if image_embeds.shape[0] != n_placeholders:
        raise ValueError(
            f"image_embeds rows {image_embeds.shape[0]} != "
            f"placeholder count {n_placeholders}")

    # ----- inject + run prefill/decode through the standard loop -----
    orig_forward = install_vision_injection(causal_lm, image_embeds, image_token_id)
    try:
        outs = run_pretokenized(llm, input_ids, batch_per_dp)
    finally:
        remove_vision_injection(causal_lm, orig_forward)
    for i, (text, reason) in enumerate(outs[:1]):
        logger.info("[vision][image] request %d (finish=%s): %s", i, reason, text)

    # ----- pure-text zero-regression control (no injection installed) -----
    if not args.skip_text_control:
        text_prompt = build_chat_prompt(tokenizer, args.prompt, with_image=False)
        text_ids = torch.tensor(
            tokenizer(text_prompt, add_special_tokens=False)["input_ids"],
            dtype=torch.long)
        ctrl = run_pretokenized(llm, text_ids, batch_per_dp)
        for i, (text, reason) in enumerate(ctrl[:1]):
            logger.info("[vision][text-control] request %d (finish=%s): %s",
                        i, reason, text)


if __name__ == "__main__":
    main()
