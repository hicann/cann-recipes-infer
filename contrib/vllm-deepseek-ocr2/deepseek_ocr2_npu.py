"""Ascend fused MOE implementation using vllm-ascend"""
import torch
try:
    import torch_npu
    print("[INFO] torch_npu loaded")
except ImportError:
    print("[WARN] torch_npu not available, running on CPU")
except Exception as e:
    print(f"[WARN] Failed to import torch_npu: {e}")

try:
    # Import Ascend MOE implementation from vllm-ascend
    from vllm_ascend.ops.fused_moe import (
        fused_experts,
        select_experts,
    )
    import vllm.model_executor.layers.fused_moe as moe_module

    # Save original fused_moe for fallback
    _original_fused_moe = moe_module.fused_moe

    def ascend_fused_moe(hidden_states, w1, w2, gating_output, topk,
                         renormalize=False, inplace=False, **kwargs):
        """Ascend NPU fused MOE implementation using vllm-ascend"""
        # Use vllm-ascend's select_experts instead of vllm's fused_topk
        # This avoids torch.ops._moe_C dependency
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=gating_output,
            top_k=topk,
            use_grouped_topk=False,  # 简单模式
            renormalize=renormalize,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
        )

        # Use Ascend's fused_experts for computation
        return fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk,
            expert_map=None
        )

    # Replace fused_moe in module with ascend_fused_moe
    moe_module.fused_moe = ascend_fused_moe
    print("[INFO] Successfully patched fused_moe with Ascend fused_experts")

except Exception as e:
    print(f"[WARNING] Failed to patch fused_moe with Ascend fused_experts: {e}")
    import traceback
    traceback.print_exc()
