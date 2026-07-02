# SPDX-License-Identifier: MIT
# Ascend NPU MOE Patch for DeepSeek-OCR-2
# Based on vllm-ascend (https://gitee.com/ascend/vllm-ascend)

"""Replace CUDA-specific MOE ops with vllm-ascend native implementation."""

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("[WARNING] torch_npu not found, NPU acceleration unavailable")


def _apply_moe_patch():
    """Apply MOE patch if vllm-ascend is available."""
    try:
        from vllm_ascend.ops.fused_moe import fused_experts, select_experts
    except ImportError:
        print("[WARNING] vllm-ascend not installed, skip MOE patch")
        return False

    try:
        import vllm.model_executor.layers.fused_moe as moe_module
    except ImportError:
        print("[WARNING] vllm not installed, skip MOE patch")
        return False

    def ascend_fused_moe(
        hidden_states,
        w1,
        w2,
        gating_output,
        topk,
        renormalize=False,
        inplace=False,
        **kwargs
    ):
        """Ascend NPU fused MOE implementation."""
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=gating_output,
            top_k=topk,
            use_grouped_topk=False,
            renormalize=renormalize,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
        )
        return fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk,
            expert_map=None,
        )

    moe_module.fused_moe = ascend_fused_moe
    print("[INFO] Ascend NPU MOE patch applied")
    return True


_apply_moe_patch()
