# coding=utf-8
# This code is copied from the LongCat-Flash-Chat implementations.
# (https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/blob/main/configuration_longcat_flash.py)
# Copyright (c) 2025 Meituan

"""LongcatFlash model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


LONGCAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LongcatFlashConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongcatFlashModel`]. It is used to instantiate an LongcatFlash
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LongcatFlash.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LongcatFlashModel`]
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        ffn_hidden_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP representations.
        expert_ffn_hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE representations.
        num_layers (`int`, *optional*, defaults to 61):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 128):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor or routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the query/key heads that don't use rotary position embeddings.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_method (`str`, *optional*, defaults to `"MLA"`):
            The attention method to use.
        initializer_range (`float`, *optional*, defaults to 0.006):
            The initializer range for the model.
        router_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the router.
        zero_expert_num (`int`, *optional*, defaults to `None`):
            The number of zero experts to use.
        zero_expert_type (`str`, *optional*, defaults to `None`):
            The type of zero expert to use.

    ```python
    >>> from transformers import LongcatFlashModel, LongcatFlashConfig

    >>> # Initializing a LongcatFlash style configuration
    >>> configuration = LongcatFlashConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "longcat_flash"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "local_colwise",
        "layers.*.mlp.experts.*.up_proj": "local_colwise",
        "layers.*.mlp.experts.*.down_proj": "local_rowwise",
        "layers.*.mlps.*.gate_proj": "local_colwise",
        "layers.*.mlps.*.up_proj": "local_colwise",
        "layers.*.mlps.*.down_proj": "local_rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=7168,
        ffn_hidden_size=18432,
        expert_ffn_hidden_size=2048,
        num_layers=61,
        num_nextn_predict_layers=1,
        num_attention_heads=128,
        num_key_value_heads=None,
        n_routed_experts=256,
        routed_scaling_factor=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        moe_topk=8,
        norm_topk_prob=False,
        hidden_act="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        attention_method='MLA',
        initializer_range=0.006,
        router_bias=False,
        zero_expert_num=None,
        zero_expert_type=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.expert_ffn_hidden_size = expert_ffn_hidden_size
        self.num_layers = num_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.moe_topk = moe_topk
        self.num_experts_per_tok = moe_topk
        self.norm_topk_prob = norm_topk_prob
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.attention_method = attention_method
        self.initializer_range = initializer_range
        self.router_bias = router_bias
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        if self.attention_method == "MLA":
            self.head_dim = qk_rope_head_dim
        else:
            ValueError('attention_method should be one of ["MLA"]')


        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.quant_config = kwargs.get('quant_config', None)

        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_hidden_layers(self):
        return self.num_layers


__all__ = ["LongcatFlashConfig"]
