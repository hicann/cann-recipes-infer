## built-in
from typing import List
import math
from dataclasses import dataclass

## third-party
from sympy import isprime
import torch
import torch.distributed as dist
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex
from executor.core.config import InferenceConfig, CommManager, PlatformVersion
from module.linear import VocabParallelEmbedding, ReplicatedLinear, set_weight_attrs


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max_ngram_size-gram), each split
    over `n_heads` heads. Every (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Ids go through the compressed table, then each position is hashed with the `max_ngram_size - 1`
    tokens before it. Look-back stops at the start of the sequence and at any dead token (an image
    span, cached as DEAD), so an n-gram never spans one. The cache carries all of this across the
    prefill/decode split.
    """

    DEAD = -1

    def __init__(
        self,
        config,
        infer_config,
        layout: EngramLayout,
        tokenizer
    ):
        super().__init__()
        self.layout = layout
        self.config = config
        self.infer_config = infer_config
        self.max_ngram_size = layout.max_ngram_size
        # every hash multiplier derives from the compressed vocab size, so a mismatch there would
        # silently rehash the whole table
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        # Add this assert check after adopting the new tokenizer.
        # assert vocab_size == config.engram_compressed_vocab_size, (vocab_size, config.engram_compressed_vocab_size)
        self.pad_id = token_map[config.engram_pad_id]
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)

        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)

    def _get_ngram_hashes(
        self,
        input_ids: torch.Tensor,
        prefix_input_ids: torch.Tensor = None,
        ngram_shift_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = input_ids
        B, T = x.shape
        prefix = prefix_input_ids
        if prefix_input_ids is not None:
            mask = prefix_input_ids == self.DEAD
            prefix_input_ids = torch.where(mask, self.pad_id, prefix_input_ids)

        def shift_k_pad(k: int) -> torch.Tensor:
            if k == 0:
                return x
            shifted = F.pad(x, (k, 0), mode="constant", value=self.pad_id)[:, :T]
            return shifted

        def shift_k_prefix(k: int) -> torch.Tensor:
            """Shift a decode query after prepending its real token history."""
            if k == 0:
                return x
            if prefix is None:
                raise ValueError(
                    "prefix_input_ids can not be None on the decoding phase."
                )
            history_input = torch.cat([prefix, x], dim=1)
            start = prefix.shape[1] - k
            if start < 0:
                raise ValueError(
                    f"prefix_input_ids length {prefix.shape[1]} is insufficient for shift {k}"
                )
            shifted = history_input[:, start : start + T]
            return shifted

        shift_fn = shift_k_prefix if prefix is not None else shift_k_pad
        base_shifts = [shift_fn(k) for k in range(self.max_ngram_size)]
        # [B, L, max_ngram_size] for decode, [1, T, max_ngram_size] for prefill
        shift_stack = torch.stack(base_shifts, dim=-1)
        if ngram_shift_mask is not None:
            mask = ngram_shift_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            pad_value = shift_stack.new_full((), self.pad_id)
            shift_stack = torch.where(mask, shift_stack, pad_value)
        tokens = shift_stack

        # XOR the multiplied ids together one lookback at a time, so the running value after step i
        # is the hash of the (i+1)-gram; each lands in its own prime-sized bucket range
        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_engram_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    def forward(
        self, input_ids, is_prefill, prefix_input_ids=None, ngram_shift_mask=None
    ):
        input_ids = self.token_map[input_ids]
        if prefix_input_ids is not None:
            prefix_input_ids = self.token_map[prefix_input_ids]
        if is_prefill:
            input_ids = input_ids.unsqueeze(0)
        else:
            input_ids = input_ids.view(-1, 1) # not support speculative decoding
        return self._get_ngram_hashes(
            input_ids,
            prefix_input_ids=prefix_input_ids,
            ngram_shift_mask=ngram_shift_mask,
        )


class MultiHeadEmbedding(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        infer_config,
        layer_id: int,
        engram_layout: EngramLayout,
        comm_manager=None,
    ):
        self.layout = engram_layout
        self.num_heads = self.layout.n_heads
        self.embedding_dim = dim
        self.layer_id = layer_id
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        custom_params = self.infer_config.model_config.custom_params
        self.engram_offload = bool(custom_params.get("enable_engram_offload", False))
        self.engram_tp_size = int(custom_params.get("engram_tp_size", 1))
        parallel_config = self.infer_config.parallel_config
        self.world_size = int(parallel_config.world_size)
        self.attn_dp_size = int(parallel_config.attn_dp_size)
        self.engram_dp_size = self.world_size // self.engram_tp_size
        self.engram_tp_group = None
        if self.engram_tp_size > 1 and self.comm_manager is not None:
            self.engram_tp_group = self.comm_manager.get_group(
                f"engram_tp_group_{self.layer_id}"
            )
        # Keep the rank-local vocabulary information for the non-offloaded
        # TP lookup path. VocabParallelEmbedding only performs the local
        # embedding lookup; this module must reduce the masked results.
        self.engram_tp_rank = (
            dist.get_rank(self.engram_tp_group)
            if self.engram_tp_group is not None
            else 0
        )
        self._engram_offloaded = False
        padded_total_N = (
            (num_embeddings + self.engram_tp_size - 1) // self.engram_tp_size
        ) * self.engram_tp_size

        super().__init__(
            vocab_size=padded_total_N,
            hidden_size=dim,
            padding_idx=None,
            params_dtype=(
                torch.bfloat16 if self.engram_offload else torch.get_default_dtype()
            ),
            tp_size=self.engram_tp_size,
            tp_rank=self.engram_tp_rank,
        )
        self.engram_vocab_size_per_rank = self.input_size_per_partition
        set_weight_attrs(
            self.weight, {"engram_fp8_weight_loader": self.fp8_weight_loader}
        )

    def weight_loader(self, param, loaded_weight):
        target = self.input_size_per_partition * self.engram_tp_size
        if loaded_weight.shape[0] < target:
            loaded_weight = torch.cat(
                (
                    loaded_weight,
                    loaded_weight.new_zeros(
                        target - loaded_weight.shape[0], loaded_weight.shape[1]
                    ),
                ),
                dim=0,
            )
        super().weight_loader(param, loaded_weight)

    def fp8_weight_loader(self, param, quant_weight, scale):
        """Dequantize and load only this rank's Engram embedding shard."""
        if quant_weight.ndim != 2 or scale.ndim != 2:
            raise ValueError(
                "Engram FP8 weight and scale must both be two-dimensional, "
                f"got weight={tuple(quant_weight.shape)}, scale={tuple(scale.shape)}"
            )

        rows, cols = quant_weight.shape
        scale_rows, scale_cols = scale.shape
        if scale_rows != rows or scale_cols <= 0 or cols % scale_cols != 0:
            raise ValueError(
                "Engram FP8 scale shape does not match the weight: "
                f"weight={tuple(quant_weight.shape)}, scale={tuple(scale.shape)}, "
                "expected one scale row per embedding row and an integral "
                "number of values per scale"
            )
        block_size = cols // scale_cols

        shard_rows = self.input_size_per_partition
        shard_start = self.engram_tp_rank * shard_rows
        shard_end = min(shard_start + shard_rows, rows)
        if tuple(param.shape) != (shard_rows, cols):
            raise RuntimeError(
                "Engram local embedding shape mismatch: "
                f"parameter={tuple(param.shape)}, expected={(shard_rows, cols)}"
            )

        param.data.zero_()
        if shard_start < shard_end:
            local_scale = scale[shard_start:shard_end].to(
                device=quant_weight.device, dtype=torch.float32
            )
            dequant_weight = (
                quant_weight[shard_start:shard_end].to(torch.float32)
                .unflatten(-1, (-1, block_size))
                * local_scale.unsqueeze(-1)
            ).flatten(-2).to(torch.bfloat16)
            param.data[:shard_end - shard_start].copy_(dequant_weight)

    def offload_weights(self):
        """Move the loaded local BF16 embedding shard into host storage."""
        from cann_ops_transformer.ops import ElasticBuffer

        if not self.engram_offload:
            raise RuntimeError("offload_weight_loader is only valid in offload mode")
        if self._engram_offloaded:
            return
        if self.weight.dtype != torch.bfloat16:
            raise TypeError(
                "ElasticBuffer only accepts BF16 Engram embeddings, "
                f"but the loaded shard has dtype={self.weight.dtype}"
            )
        num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
            self.input_size_per_partition, self.embedding_dim, torch.bfloat16
        )
        self.engram_buffer = ElasticBuffer(
            self.engram_tp_group,
            num_cpu_bytes=num_cpu_bytes,
            explicitly_destroy=True,
        )
        # Safetensors are yielded from CPU. ElasticBuffer's device-side write
        # API consumes an NPU tensor, so keep this transfer temporary and do
        # not retain a second copy as a module member.
        self.engram_buffer.engram_write(self.weight)
        del self.weight
        self._engram_offloaded = True

    def forward(
        self, input_ids: torch.Tensor, is_prefill: bool = False
    ) -> torch.Tensor:
        input_shape = input_ids.shape
        num_heads = input_shape[-1]
        flat_input_ids = input_ids.reshape(-1, num_heads)

        # In DSV4, attention workers may hold different request batches. When
        # the Engram TP group spans more DP replicas than the attention group,
        # gather their hash IDs first, just like calc_input_embeddings().
        allgather_ratio = 1
        local_tokens = flat_input_ids.shape[0]
        if (
            not self.engram_offload
            and self.engram_tp_group is not None
            and self.attn_dp_size > self.engram_dp_size
        ):
            allgather_ratio = self.engram_tp_size
            if is_prefill:
                max_tokens = torch.tensor(
                    [local_tokens], dtype=torch.long, device=flat_input_ids.device
                )
                dist.all_reduce(
                    max_tokens, op=dist.ReduceOp.MAX, group=self.engram_tp_group
                )
                max_tokens = int(max_tokens.item())
                gather_input = F.pad(
                    flat_input_ids, (0, 0, 0, max_tokens - local_tokens), value=0
                )
            else:
                max_tokens = local_tokens
                gather_input = flat_input_ids
            gathered = flat_input_ids.new_empty(max_tokens * allgather_ratio, num_heads)
            dist.all_gather_into_tensor(
                gathered, gather_input, group=self.engram_tp_group
            )
            flat_input_ids = gathered

        if self.engram_offload:
            shape = flat_input_ids.shape
            wait_callable = self.engram_buffer.engram_fetch(
                flat_input_ids.reshape(-1).to(torch.int32)
            )
            output = wait_callable()
            return output.view(*shape, self.embedding_dim)
        if self.engram_tp_group is None or self.engram_tp_size == 1:
            return (
                super()
                .forward(flat_input_ids)
                .view(*flat_input_ids.shape, self.embedding_dim)
            )

        # engram_tp_size > 1 and enable_engram_offload = False
        shard_start = self.engram_tp_rank * self.engram_vocab_size_per_rank
        local_input_ids = flat_input_ids - shard_start
        owned = (local_input_ids >= 0) & (
            local_input_ids < self.engram_vocab_size_per_rank
        )
        local_input_ids = local_input_ids.masked_fill(~owned, 0)
        output = super().forward(local_input_ids)
        output = output * owned.unsqueeze(-1).to(output.dtype)
        if self.attn_dp_size <= self.engram_dp_size:
            dist.all_reduce(output, group=self.engram_tp_group)
        else:
            reduced = output.new_empty(max_tokens, output.shape[-2], output.shape[-1])
            dist.reduce_scatter_tensor(reduced, output, group=self.engram_tp_group)
            output = reduced[:local_tokens] if is_prefill else reduced
        return output.view(*output.shape[:-1], self.embedding_dim)


class Engram(nn.Module):
    def __init__(self, config, infer_config, layer_id, engram_layout, prefix, comm_manager=None):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.layer_hash_index = engram_layout.layer_ids.index(layer_id)

        self.multi_head_embedding = MultiHeadEmbedding(
            engram_layout.num_embeddings[self.layer_hash_index],
            engram_layout.head_dim,
            infer_config=self.infer_config,
            comm_manager=self.comm_manager,
            engram_layout=engram_layout,
            layer_id=self.layer_id,
        )
        self.n_hash_cols = (engram_layout.max_ngram_size - 1) * engram_layout.n_heads
        self.wkv = ReplicatedLinear(self.n_hash_cols * engram_layout.head_dim,
                                    self.hidden_size * (config.hc_mult + 1),
                                    params_dtype=torch.float8_e4m3fn,
                                    quant_config=config.quant_config,
                                    prefix=f"{prefix}.wkv",
                                    )
        self.eps = config.rms_norm_eps
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, self.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, self.hidden_size))
        self.clamp_value = 1e-6

    def forward(self, hidden_states, engram_hash, is_prefill=False, engram_metadata=None, image_mask=None):
        """
        Forward pass for Engram layer.
        hidden_states: [T, HC_MULT, D]
        engram_metadata: preprocessed Engram runtime tensors from attention metadata.
        image_mask: True at image spans, where the gate shuts for pass-through.
        """
        engram_metadata = engram_metadata or {}

        embeddings = self.multi_head_embedding(
            engram_hash,
            is_prefill=is_prefill,
        ).reshape(
            -1,
            self.multi_head_embedding.embedding_dim
            * self.n_hash_cols
        )

        kv = self.wkv(embeddings)
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h, eps = hidden_states.float(), self.eps
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        gate_input = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(dot >= 0, gate_input, -gate_input))
        if image_mask is not None:
            gate = gate.masked_fill(image_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_states.dtype)
