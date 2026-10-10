# coding=utf-8
# Adapted from
# https://huggingface.co/meituan-longcat/LongCat-Next
#   modular_longcat_next.py - causal depth-transformer head
#   (FlashVarLenAttention / CasualDepthTransformerLayer / CasualDepthTransformerHead),
#   re-implemented here as HeadAttention / DepthLayer / DepthWeight / ParallelDepthHead
#   for multi-rank execution and graph capture;
#   modeling_longcat_ngram.py - N-gram embedding math reused by ShardedNgram.
# Copyright (c) 2026 Meituan
# The LongCat-Next portions above are released under the MIT License (see that repository's LICENSE file).
#
# Copyright 2024 The HuggingFace Inc. team. (generation-loop conventions taken from GenerationMixin)
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# The additions and modifications in this file are licensed under the Apache License, Version 2.0:
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

"""Single interface layer between LongCat-Next and the cann-recipes-infer executor.

Loads weights and configuration, connects the official generation state machine to the
framework's execution, paged KV cache and graph compilation, and drives the offline
multimodal requests. The weights directory only supplies weights and JSON configuration:
no Python file is read from it or written to it.
"""

import json
from contextlib import contextmanager
from pathlib import Path
import copy
import logging
from types import SimpleNamespace
from datetime import timedelta
import re
import os
import random
import sys
import time
from importlib.metadata import version
import torch
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import can_return_tuple
import numpy as np
from accelerate import init_empty_weights
from transformers import GenerationConfig
from models.longcat_next.models import modeling_longcat_next as longcat_model
from models.longcat_next.models.components import modular_longcat_next_audio, modular_longcat_next_visual
from models.longcat_next.models.configuration_longcat_next import LongcatNextConfig
from models.longcat_next.utils.tokenizer import LongcatNextTokenizer
from executor.core.kv_cache.cache_utils import prepare_block_tables
from executor.utils.forward_metadata import ForwardMetaData
from executor.utils.graph_utils import compile_model_forward
from executor.utils.profiler_context import ProfilerPhase
from module.linear import ColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from executor.core.engine.execution_engine import ExecutionEngine
from executor.core.support_models import load_model_classes
from models.longcat_next.models.components.processing_longcat_next import LongcatNextProcessor


# ---------------------------------------------------------------------------
# weight and configuration loading (the weights directory supplies data only)
# ---------------------------------------------------------------------------

@contextmanager
def component_dtype(dtype=torch.bfloat16):
    """Match HF construction: default BF16, explicit FP32 declarations survive."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(previous)


def load_official(model_path):
    """Load the configuration that ships next to the weights.

    The checkpoint directory supplies weights and JSON configuration only: no
    Python file is imported from model_path and nothing in it is written. The
    model definition itself is this repository's modeling_longcat_next.py.
    """
    root = Path(model_path)
    config = LongcatNextConfig.from_pretrained(str(root), local_files_only=True)
    config._attn_implementation = "sdpa"  # pylint: disable=protected-access
    return config


def load_components(model_path, components, device, parallel_heads=()):
    """components maps original checkpoint prefixes to instantiated modules.

    Heads have shard loaders; official modules are built under init_empty_weights.
    No complete HF backbone is constructed or loaded again.
    """
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        mapping = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        filename = root / "model.safetensors"
        with safe_open(str(filename), framework="pt", device="cpu") as reader:
            mapping = {name: filename.name for name in reader.keys()}
    tensors, owners = {}, {}
    for prefix, module in components.items():
        state = module.state_dict(keep_vars=True)
        for name, tensor in state.items():
            key = prefix + "." + name
            tensors[key] = tensor
            owners[key] = (module, name)
    missing = sorted(set(tensors) - set(mapping))
    if missing:
        raise RuntimeError(f"Required multimodal weights missing ({len(missing)}): {missing[:20]}")
    relevant = {name for name in mapping if any(name.startswith(p + ".") for p in components)}
    extra = sorted(relevant - set(tensors))
    if extra:
        raise RuntimeError(f"Unmapped multimodal checkpoint tensors ({len(extra)}): {extra[:20]}")
    by_file = {}
    for name in tensors:
        by_file.setdefault(mapping[name], []).append(name)
    for filename, names in sorted(by_file.items()):
        with safe_open(str(root / filename), framework="pt", device="cpu") as reader:
            for name in names:
                value = reader.get_tensor(name)
                module, local_name = owners[name]
                parameter = tensors[name]
                if module in parallel_heads and local_name.startswith("heads."):
                    # C+1 contains an actual end marker. Pad beyond that marker.
                    level = int(local_name.split(".")[1])
                    size = module.codebook_sizes[level] + 1
                    if value.shape[0] != size:
                        raise ValueError(f"{name}: expected {size} vocabulary rows")
                    padded = ((size + module.tp - 1) // module.tp) * module.tp
                    value = pad_rows(value, padded)
                loader = getattr(parameter, "weight_loader", None)
                if loader is not None:
                    loader(parameter, value)
                else:
                    if tuple(value.shape) != tuple(parameter.shape):
                        raise ValueError(f"{name}: checkpoint {value.shape} != module {parameter.shape}")
                    dtype = parameter.dtype
                    set_module_tensor_to_device(module, local_name, device, value=value, dtype=dtype)
                materialized = (
                    module.get_parameter_or_buffer(local_name) if hasattr(module, "get_parameter_or_buffer")
                    else (module.get_parameter(local_name) if isinstance(parameter, torch.nn.Parameter)
                          else module.get_buffer(local_name)))
                if materialized.dtype != parameter.dtype or materialized.is_meta:
                    raise RuntimeError(f"{name}: materialized dtype/device violates constructed parameter contract")
    for module in components.values():
        if any(p.is_meta for p in module.parameters()):
            raise RuntimeError("Unmaterialized parameter after multimodal weight loading")
        module.to(device).eval()
    for head in parallel_heads:
        head.finish_loading()


def pad_rows(value, padded):
    if value.shape[0] == padded:
        return value
    result = value.new_zeros((padded, *value.shape[1:]))
    result[:value.shape[0]].copy_(value)
    return result


# ---------------------------------------------------------------------------
# execution bridge: official token positions -> infer backbone and paged KV cache
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _clone_metadata(metadata):
    """Give replay metadata its own writable storage, outside the compiled call.

    Metadata holds length/index tensors, not the model's KV-cache storage.
    In particular, expanded CFG views must never become copy_ destinations.
    """
    result = copy.copy(metadata)
    for name, value in vars(metadata).items():
        if isinstance(value, torch.Tensor):
            value = value.clone(memory_format=torch.contiguous_format)
        elif isinstance(value, dict):
            value = {key: tensor.clone(memory_format=torch.contiguous_format)
                     for key, tensor in value.items()}
        elif isinstance(value, list):
            value = value.copy()
        setattr(result, name, value)
    return result


def _copy_decode_input(target, value, name):
    # copy_ permits broadcasting/casting; a captured input must keep its exact
    # signature. Check metadata only (no device synchronization or tensor dump).
    if target.shape != value.shape or target.dtype != value.dtype or target.device != value.device:
        raise ValueError(f"Decode input {name} changed shape, dtype or device")
    try:
        target.copy_(value)
    except RuntimeError as exc:
        # The command transport summarizes exceptions, so keep the field and
        # layout in the message even when the inner traceback is unavailable.
        raise RuntimeError(f"Decode input {name}: destination shape={tuple(target.shape)}, "
                           f"stride={target.stride()}; {exc}") from exc


class VocabLookup(nn.Module):
    """Global-ID lookup backed by the already-loaded vocabulary shards."""
    def __init__(self, table, group, tp, rank):
        super().__init__()
        self.table, self.group, self.tp, self.rank = table, group, tp, rank

    @property
    def weight(self):
        return self.table.weight

    def forward(self, ids, mask=None):
        width = self.table.weight.shape[0]
        local = ids - self.rank * width
        valid = (local >= 0) & (local < width)
        if mask is not None:
            valid = valid & mask
        x = self.table(torch.where(valid, local, torch.zeros_like(local))) * valid.unsqueeze(-1)
        if self.tp > 1:
            dist.all_reduce(x, group=self.group)
        return x


class ShardedNgram(nn.Module):
    """HF's stateless formula with the existing combined shard reduction.

    Context remains an explicit input: the HF state machine owns its update.
    No second rolling cache, boolean indexing or per-table collectives.
    """
    def __init__(self, source, embedding, prefill_shift):
        super().__init__()
        self.source, self.embedding = source, embedding
        self.prefill_shift = prefill_shift
        self.mods = source.precompute_vocab_mods()

    def forward(self, ids, context):
        source, s = self.source, ids.shape[1]
        tokens = torch.cat([context[:, -(source.n - 1):], ids], dim=-1)
        tokens = torch.where(source.is_oe_ignored(tokens), torch.zeros_like(tokens), tokens)
        shifts = {}
        for n in range(2, source.n + 1):
            shifts[n] = (source.shift_right_ignore_eos_unrolled(tokens, n - 1, source.config.eos_token_id)
                         if s == 1 else self.prefill_shift(tokens, n - 1, source.config.eos_token_id))
        x = self.embedding(ids)
        local = []
        for n in range(2, source.n + 1):
            for j in range(source.k):
                index = (n - 2) * source.k + j
                hashes = source.get_ngram_ids(tokens, shifts, self.mods[(n, j)], n)
                indices = (hashes % source.embedder_vocab_sizes[index])[:, -s:]
                local.append(source.lookup_embedding_local(source.embedders[index], indices))
        for projection, values in zip(source.post_projs, source.reduce_ngram_embeddings(local)):
            x = x + projection(values.contiguous())
        return torch.where(source.is_oe_ignored(ids).unsqueeze(-1), x,
                           x / (1 + source.k * (source.n - 1)))


def make_ngram(model, embedding):
    if model.ngram_embeddings is None:
        raise RuntimeError("Multimodal integration requires the original N-gram embedding")
    return ShardedNgram(model.ngram_embeddings, embedding, longcat_model.ngram_prefill_shift)


class FeatureCall(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

    def forward(self, ids, positions, metadata, embeddings):
        return self.backbone.forward_features(ids, positions, metadata, embeddings)


class TextCall(nn.Module):
    """One graph/command for an ordinary text step, with explicit HF history."""
    def __init__(self, model, ngram):
        super().__init__()
        self.backbone, self.ngram = model, ngram

    def forward(self, ids, positions, metadata, context):
        embeddings = self.ngram(ids.view(-1, 1), context).view(ids.shape[0], -1)
        hidden = self.backbone.forward_features(ids, positions, metadata, embeddings)
        return self.backbone.forward_lm_head(hidden)


class BackboneSession:
    def __init__(self, engine, hf_config):
        self.engine, self.device = engine, engine.device
        self.model = engine.main_worker.model
        source = self.model.model
        cm = engine.comm_manager
        self.embedding = VocabLookup(source.embed_tokens, cm.get_group("embed_tp_group"),
                                     source.embed_tp_size, source.embed_tp_rank)
        self.ngram = make_ngram(source, self.embedding)
        self.capacity = engine.max_total_len
        self.length, self.batch = 0, None
        request_slots = engine.infer_config.scheduler_config.batch_size_per_dp_rank
        self.requests = [SimpleNamespace(request_id=i) for i in range(request_slots)]
        self.graphs, self.static_inputs = {}, {}
        self.tables = {}
        self.compile_cache_root = None
        self.attention_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=self.device))

    def begin(self):
        manager = self.engine.kvcache_manager
        if manager is None:
            raise RuntimeError("Multimodal bridge requires framework paged KV cache")
        for request in self.requests:
            manager.free(request.request_id)
        for request in self.requests:
            if not manager.allocate_slots(request.request_id, 0, self.capacity):
                raise MemoryError(f"Could not reserve {len(self.requests)} independent KV sequences")
        self.length, self.batch = 0, None
        self.tables.clear()
        return None

    def ngram_embeddings(self, ids, context):
        return self.ngram(ids, context)

    def _metadata(self, batch, qlen, positions, rope_positions, prefill):
        device = self.device
        q = torch.full((batch,), qlen, dtype=torch.long, device=device)
        kv = torch.full((batch,), self.length + qlen, dtype=torch.long, device=device)
        cu_q, cu_kv = q.cumsum(0), kv.cumsum(0)
        manager = self.engine.kvcache_manager
        if batch not in self.tables:
            self.tables[batch] = prepare_block_tables(
                self.requests[:batch], manager, manager.get_block_table_max_lens(self.capacity),
                device, batch_size=batch, is_prefill=prefill)
        tables = self.tables[batch]
        # repeat materializes one element per request. For decode S=1,
        # expand(...).reshape(-1) instead retains stride=0 across CFG rows.
        physical_positions = positions.repeat(batch)
        # Requests here are equal-length, fully reserved sequences. Vectorize
        # the official helper's page*block_size+offset formula without .item().
        slots = {}
        for cache_manager in manager.single_type_managers:
            key, block_size = cache_manager.manager_key, cache_manager.block_size
            if cache_manager.attn_type != "FullAttention" or getattr(cache_manager, "compress_ratio", 1) != 1:
                raise ValueError("LongCat multimodal requires uncompressed FullAttention KV pages")
            indices = (positions // block_size).view(1, qlen).expand(batch, -1)
            pages = tables[key].gather(1, indices)
            slots[key] = (pages * block_size + positions.remainder(block_size)).reshape(-1).contiguous()
        metadata = ForwardMetaData(
            is_prefill=prefill, is_warm_up=False,
            attention_mask=self.attention_mask if prefill else None,
            kv_len=physical_positions if prefill else physical_positions.view(batch),
            actual_seq_lengths_q=q, actual_seq_lengths_kv=kv,
            actual_seq_lengths_cu_q=cu_q, actual_seq_lengths_cu_kv=cu_kv,
            prompt_tokens=batch * qlen if prefill else 0,
            block_table=tables, slot_mapping=slots,
        )
        if self.engine.exe_mode == "npugraph_ex" and not prefill:
            metadata.actual_seq_lengths_list_q = [qlen] * batch
            metadata.actual_seq_lengths_list_kv = [self.length + qlen] * batch
            metadata.actual_seq_lengths_cu_list_q = [qlen * (i + 1) for i in range(batch)]
            metadata.actual_seq_lengths_cu_list_kv = [(self.length + qlen) * (i + 1) for i in range(batch)]
        return metadata, rope_positions.reshape(-1).contiguous()

    def _decode(self, embeddings, rope_positions, metadata, token_ids=None):
        batch = embeddings.shape[0]
        graph_key = ("text" if token_ids is not None else "features", batch)
        if graph_key not in self.static_inputs:
            ids = torch.zeros(batch, dtype=torch.long, device=self.device)
            self.static_inputs[graph_key] = (
                ids, rope_positions.clone(memory_format=torch.contiguous_format),
                _clone_metadata(metadata),
                embeddings.clone(memory_format=torch.contiguous_format),
            )
            caller = TextCall(self.model, self.ngram) if token_ids is not None else FeatureCall(self.model)
            if self.engine.exe_mode == "eager":
                compiled = caller.forward
            else:
                config = copy.deepcopy(self.engine.infer_config)
                config.model_config.output_path = str(self.compile_cache_root / f"{graph_key[0]}_b{batch}")
                # Tensor shapes are fixed per interface. Only the Python FA
                # length values remain symbolic in npugraph_ex.
                for value in self.static_inputs[graph_key]:
                    fields = vars(value).values() if isinstance(value, ForwardMetaData) else (value,)
                    for field in fields:
                        tensors = field.values() if isinstance(field, dict) else (field,)
                        for tensor in tensors:
                            if isinstance(tensor, torch.Tensor):
                                torch._dynamo.mark_static(tensor)  # pylint: disable=protected-access
                if self.engine.exe_mode == "npugraph_ex":
                    import torchair
                    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
                    torchair.patch_for_hcom()
                    torch._dynamo.config.inline_inbuilt_nn_modules = False  # pylint: disable=protected-access
                    # Only baseline options available in torch_npu 2.8. Do not
                    # pass the optional super-kernel keys from newer releases.
                    options = {"frozen_parameter": True,
                               "static_kernel_compile": config.model_config.enable_static_kernel}
                    if config.model_config.enable_cache_compile:
                        compiled = torch.npu.npugraph_ex.inference.cache_compile(
                            caller.forward, cache_dir=os.path.join(config.model_config.output_path, "compile_cache"),
                            dynamic=True, options=options)
                    else:
                        compiled = torch.compile(caller.forward, backend="npugraph_ex", fullgraph=True,
                                                 dynamic=True, options=options)
                else:
                    compiled = compile_model_forward(caller.forward, config)
            self.graphs[graph_key] = compiled
            logger.info("Multimodal decode interface prepared: batch=%d, mode=%s; first call may compile",
                        batch, self.engine.exe_mode)
        ids, static_positions, static_metadata, static_embeddings = self.static_inputs[graph_key]
        if token_ids is not None:
            _copy_decode_input(ids, token_ids.reshape(-1), "input_ids")
        _copy_decode_input(static_positions, rope_positions, "position_ids")
        _copy_decode_input(static_embeddings, embeddings, "inputs_embeds")
        for name, value in vars(metadata).items():
            target = getattr(static_metadata, name)
            if isinstance(value, torch.Tensor):
                _copy_decode_input(target, value, name)
            elif isinstance(value, dict):
                for key, tensor in value.items():
                    _copy_decode_input(target[key], tensor, f"{name}.{key}")
            else:
                setattr(static_metadata, name, value)
        return self.graphs[graph_key](ids, static_positions, static_metadata, static_embeddings)

    def text_step(self, ids, context, cache_positions, rope_positions, positions=None):
        """Ordinary text has no media substitutions; retain HF sampling outside."""
        b, s = ids.shape
        if context.shape != (b, self.ngram.source.n - 1):
            raise ValueError("Text decode requires a fixed padded N-gram history")
        if self.length == 0 or s != 1:
            hidden = self.forward(self.ngram(ids, context), cache_positions, rope_positions, positions)
            return self.model.forward_lm_head(hidden)
        self._validate_append(b, s, cache_positions, rope_positions, positions)
        self.engine.profiler.set_status(ProfilerPhase.DECODE)
        metadata, flat = self._metadata(b, 1, cache_positions, rope_positions, False)
        logits = self._decode(context, flat, metadata, token_ids=ids)
        self.length += 1
        self.engine.profiler.step()
        return logits

    def _validate_append(self, b, s, cache_positions, rope_positions, positions):
        if positions is None:
            positions = cache_positions.cpu().tolist()
        validate_positions(positions, s, self.length, self.capacity)
        if b not in range(1, len(self.requests) + 1):
            raise ValueError("Batch exceeds reserved request slots")
        if self.batch is not None and self.batch != b:
            raise ValueError("CFG batch changed after KV was populated; start image CFG in the initial prefill")
        if rope_positions.shape != (b, s):
            raise ValueError("position_ids shape must match [B,S]")
        self.batch = b

    def forward(self, embeddings, cache_positions, rope_positions, positions=None):
        b, s, h = embeddings.shape
        self._validate_append(b, s, cache_positions, rope_positions, positions)
        if h != self.model.config.hidden_size:
            raise ValueError("Backbone expects [B=1 or 2, S, hidden_size]")
        if self.length == 0:
            self.engine.profiler.set_status(ProfilerPhase.PREFILL)
            metadata, flat_positions = self._metadata(b, s, cache_positions, rope_positions, True)
            hidden = self.model.forward_features(
                torch.zeros(b * s, dtype=torch.long, device=self.device), flat_positions, metadata,
                embeddings.reshape(b * s, h).contiguous())
            self.length += s
            self.engine.profiler.step()
            return hidden
        # The state machine may insert several prefix tokens. Existing MLA
        # prefill attends only to the current chunk, so append these in order
        # through decode (with the previously allocated KV), never fresh prefill.
        for i in range(s):
            self.engine.profiler.set_status(ProfilerPhase.DECODE)
            metadata, flat_positions = self._metadata(b, 1, cache_positions[i:i + 1], rope_positions[:, i:i + 1], False)
            hidden = self._decode(embeddings[:, i].contiguous(), flat_positions, metadata)
            self.length += 1
            self.engine.profiler.step()
        return hidden

    def warm_up(self, text=True, features=True, cfg=False):
        """Compile outside measured requests. Each variant owns static inputs."""
        variants = ([("text", 1)] if text else []) + (
            [("features", 1)] if features else []) + ([("features", 2)] if cfg else [])
        profiler_enabled = self.engine.profiler.enable_profiler
        self.engine.profiler.enable_profiler = False
        try:
            for kind, batch in variants:
                self.begin()
                ids = torch.ones(batch, 2, dtype=torch.long, device=self.device)
                context = torch.zeros(batch, self.ngram.source.n - 1, dtype=torch.long, device=self.device)
                pos = torch.arange(2, device=self.device)
                self.forward(self.ngram(ids, context), pos, pos.repeat(batch, 1), [0, 1])
                for step in (2, 3, 4):
                    pos = torch.tensor([step], device=self.device)
                    if kind == "text":
                        self.text_step(ids[:, :1].contiguous(), context, pos, pos.repeat(batch, 1), [step])
                    else:
                        self.forward(self.ngram(ids[:, :1], context), pos, pos.repeat(batch, 1), [step])
                torch.npu.synchronize()
            self.begin()
        finally:
            self.engine.profiler.enable_profiler = profiler_enabled


def validate_positions(positions, sequence_length, cached_length, capacity):
    """Validate physical cache positions independently of RoPE position IDs."""
    if len(positions) != sequence_length or not positions:
        raise ValueError("cache_position length must equal the current embedding sequence length")
    if positions != list(range(cached_length, cached_length + sequence_length)):
        raise ValueError(f"Non-contiguous cache append: cached={cached_length}, positions={positions[:8]}")
    if positions[-1] >= capacity:
        raise ValueError(f"Sequence exceeds KV capacity {capacity}; increase input/output capacity in YAML")



def warm_up_heads(head_call, batches, hf_config, device):
    """Warm the single head graph. Identical calls on every rank.

    The graph no longer depends on the codebook level, so one signature per
    (modality, batch) is enough; first and last level are both exercised.
    """
    for modality, batch_sizes in batches.items():
        cfg = hf_config.visual_config if modality == "visual" else hf_config.audio_config
        depth = len(cfg.vq_config.codebook_sizes)
        for batch in batch_sizes:
            hidden = torch.zeros(batch, 1, hf_config.hidden_size, dtype=torch.bfloat16, device=device)
            ids = torch.zeros(batch, depth, dtype=torch.long, device=device)
            for level in (0, depth - 1, 0):
                for _ in range(3):
                    head_call(hidden, ids, level, modality)


class HeadCall(nn.Module):
    """One level-independent graph body: depth transformer + all codebook heads.

    ``level`` is deliberately absent. It used to be a compile-time constant
    (module attribute, ``self.heads[level]``, ``x[:, level]``), which forced one
    npugraph_ex graph per codebook level. The depth transformer's eight
    positions do not depend on the level, so the graph returns every level's
    logits and the caller slices the requested level eagerly.
    """

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden, embeddings):
        return self.head(hidden, embeddings)


def compile_head(caller, config, cache_dir):
    """Use locked 2.8 APIs; keep head options independent of backbone options."""
    import torchair
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

    torchair.patch_for_hcom()
    torch._dynamo.config.inline_inbuilt_nn_modules = False  # pylint: disable=protected-access
    options = config.model_config.custom_params["multimodal"]
    if options["head_exe_mode"] == "ge_graph":
        from executor.utils.graph_utils import compile_model_forward as compile_head_forward

        head_config = copy.deepcopy(config)
        head_config.model_config.exe_mode = "ge_graph"
        head_config.model_config.enable_cache_compile = options.get("head_enable_cache_compile", False)
        head_config.model_config.enable_dynamic_graph = False
        head_config.model_config.enable_static_kernel = False
        head_config.model_config.output_path = str(cache_dir)
        return compile_head_forward(caller.forward, head_config)

    compile_options = {"frozen_parameter": True, "static_kernel_compile": False}
    if options.get("head_enable_cache_compile", False):
        return torch.npu.npugraph_ex.inference.cache_compile(
            caller.forward, cache_dir=str(cache_dir / "compile_cache"),
            dynamic=False, options=compile_options)
    return torch.compile(caller.forward, backend="npugraph_ex", dynamic=False,
                         fullgraph=True, options=compile_options)


class HeadGraphRunner:
    """One head graph per (modality, batch); the codebook level stays data.

    Compiling ``level`` into the graph produced eight graphs per signature and
    forced the dynamo recompile budget up. Nothing in the depth transformer
    depends on the level, so the graph is built once per signature and the
    level is applied afterwards by an eager slice.
    """

    def __init__(self, heads, config, cache_root, batches, compiler=compile_head):
        self.heads = heads
        self.config, self.cache_root, self.compiler = config, cache_root, compiler
        options = config.model_config.custom_params["multimodal"]
        self.mode = options.get("head_exe_mode", "eager")
        self.allowed = {(modality, batch)
                        for modality, head in heads.items() for batch in batches[modality]}
        self.graphs, self.inputs, self.callers = {}, {}, {}
        if self.mode != "eager":
            # Only (modality, batch) variants exist now; keep a small margin for
            # the warm-up pass without disabling recompilation diagnostics.
            count = max(1, len(self.allowed))
            torch._dynamo.config.recompile_limit = max(  # pylint: disable=protected-access
                torch._dynamo.config.recompile_limit, count)
            torch._dynamo.config.accumulated_recompile_limit = max(  # pylint: disable=protected-access
                torch._dynamo.config.accumulated_recompile_limit, count)

    def __call__(self, modality, hidden, embeddings, level):
        key = (modality, hidden.shape[0])
        if key not in self.allowed:
            raise ValueError(f"Head signature {key} was not reserved by the request manifest")
        head = self.heads[modality]
        depth = len(head.codebook_sizes)
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise ValueError("Head hidden must have shape [batch, 1, hidden_size]")
        expected = (hidden.shape[0], depth - 1, hidden.shape[-1])
        if tuple(embeddings.shape) != expected:
            raise ValueError(f"Head embedding input shape {tuple(embeddings.shape)} != {expected}")
        if embeddings.dtype != hidden.dtype or embeddings.device != hidden.device:
            raise ValueError("Head embedding input changed dtype/device")
        if not 0 <= level < depth:
            raise ValueError(f"Head level {level} outside [0, {depth})")

        if self.mode == "eager":
            return head.select(head(hidden, embeddings), level)

        if key not in self.graphs:
            # clone, not contiguous alone: even an already-contiguous caller
            # input must not become the replay buffer or alias another graph.
            buffers = tuple(value.clone(memory_format=torch.contiguous_format)
                            for value in (hidden, embeddings))
            for value in buffers:
                torch._dynamo.mark_static(value)  # pylint: disable=protected-access
                torch._dynamo.mark_static_address(value)  # pylint: disable=protected-access
            caller = HeadCall(head).eval()
            path = self.cache_root / f"{modality}_b{key[1]}"
            logger.info("Compile %s head B=%s mode=%s (one graph, all levels)",
                        modality, key[1], self.mode)
            self.inputs[key], self.callers[key] = buffers, caller
            self.graphs[key] = self.compiler(caller, self.config, path)
        for target, value in zip(self.inputs[key], (hidden, embeddings)):
            if target.shape != value.shape or target.dtype != value.dtype or target.device != value.device:
                raise ValueError(f"Head input for {key} changed shape/dtype/device")
            target.copy_(value)
        return head.select(self.graphs[key](*self.inputs[key]), level)


# ---------------------------------------------------------------------------
# TP implementation of the official causal depth head
# ---------------------------------------------------------------------------

def reduce_sum(x, tp, group):
    if tp > 1:
        # Depth FFN einsum may return a strided [B, L, D] view. HCCL
        # requires a contiguous buffer; use and return the reduced copy.
        # No dtype conversion or arithmetic is performed by contiguous().
        x = x.contiguous()
        dist.all_reduce(x, group=group)
    return x


class DepthWeight(nn.Module):
    """The official FFN reads reshaped weights directly, with codebook axis l.

    W1: [t,l,d], W2: [d,l,t]. Split t; NEVER split l or a flat W2 input.
    Biases exist in the checkpoint but are unused by the official einsums.
    """

    def __init__(self, dimension, depth, scale, tp, rank, down=False):  # pylint: disable=huawei-too-many-arguments
        super().__init__()
        self.d, self.l, self.t = dimension, depth, scale * dimension // depth
        self.tp, self.rank, self.down = tp, rank, down
        shape = (dimension, depth, self.t // tp) if down else (self.t // tp, depth, dimension)
        self.weight = nn.Parameter(torch.empty(shape, dtype=torch.bfloat16), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(dimension if down else scale * dimension,
                                              dtype=torch.bfloat16), requires_grad=False)
        self.weight.weight_loader = self.load_weight

    def load_weight(self, param, value):
        expected = (self.d, self.l * self.t) if self.down else (self.l * self.t, self.d)
        if tuple(value.shape) != expected:
            raise ValueError(f"Depth FFN weight shape {tuple(value.shape)} != {expected}")
        full = value.reshape(self.d, self.l, self.t) if self.down else value.reshape(self.t, self.l, self.d)
        shard = full.narrow(2 if self.down else 0, self.rank * (self.t // self.tp), self.t // self.tp)
        param.data.copy_(shard)


class HeadAttention(nn.Module):
    def __init__(self, dimension, depth, tp, rank, group):
        super().__init__()
        self.depth, self.heads, self.tp, self.group = depth, dimension // 128 // tp, tp, group
        args = dict(tp_size=tp, tp_rank=rank, params_dtype=torch.bfloat16, return_bias=False)
        self.q_proj = ColumnParallelLinear(dimension, dimension, bias=True, **args)
        self.k_proj = ColumnParallelLinear(dimension, dimension, bias=False, **args)
        self.v_proj = ColumnParallelLinear(dimension, dimension, bias=True, **args)
        # Add the replicated bias ONCE, after reduction.
        self.out_proj = RowParallelLinear(dimension, dimension, bias=True, skip_bias_add=True, **args)
        # FIA bool masks use True for blocked positions. This is an owned,
        # graph-external buffer, not a checkpoint tensor or a depth KV cache.
        self.register_buffer("causal_mask", torch.ones(depth, depth, dtype=torch.bool).triu(1),
                             persistent=False)

    def _attention(self, q, k, v):
        if q.device.type == "npu":
            # NPU SDPA may select npu_fusion_attention_v3, which the installed
            # GE backend cannot convert. Use the supported inference interface
            # in ALL NPU modes; preserve BNSD, scale and the eight-step causal
            # mask. No dynamic lengths, quantization, padding or dropout.
            # A-only variant: sparse_mode left at the original 0 so eager
            # numerics are byte-identical to the pre-change implementation.
            # Use this file for the E1/E2 experiments in README section 6.
            return torch.ops.npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=self.heads, num_key_value_heads=self.heads,
                input_layout="BNSD", atten_mask=self.causal_mask,
                scale=128 ** -0.5, sparse_mode=0,
                pre_tokens=2147483647, next_tokens=0, inner_precise=0)[0]
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

    def forward(self, x):
        b = x.shape[0]
        q, k, v = [proj(x).view(b, self.depth, self.heads, 128).transpose(1, 2).contiguous()
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        y = self._attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, self.depth, self.heads * 128).contiguous()
        return reduce_sum(self.out_proj(y), self.tp, self.group) + self.out_proj.bias


class DepthLayer(nn.Module):
    def __init__(self, dimension, depth, scale, tp, rank, group):  # pylint: disable=huawei-too-many-arguments
        super().__init__()
        head_partition(dimension, depth, scale, tp)
        self.tp, self.group = tp, group
        self.self_attention = HeadAttention(dimension, depth, tp, rank, group)
        self.layernorm1 = T5LayerNorm(dimension)
        self.layernorm2 = T5LayerNorm(dimension)
        self.linear1 = DepthWeight(dimension, depth, scale, tp, rank)
        self.linear2 = DepthWeight(dimension, depth, scale, tp, rank, down=True)

    def forward(self, x):
        residual = x + self.self_attention(self.layernorm1(x))
        y = torch.einsum("bld,tld->blt", self.layernorm2(residual), self.linear1.weight)
        y = F.gelu(y)
        y = torch.einsum("blt,dlt->bld", y, self.linear2.weight)
        return residual + reduce_sum(y, self.tp, self.group)


class ParallelDepthHead(nn.Module):
    def __init__(self, config, modality, tp, rank, group):
        super().__init__()
        cfg = config.visual_config if modality == "visual" else config.audio_config
        prefix = "image" if modality == "visual" else "audio"
        d = getattr(cfg, prefix + "_head_transformer_dims")
        layers = getattr(cfg, prefix + "_head_transformer_layers")
        scale = getattr(cfg, prefix + "_head_transformer_ffn_scale")
        self.codebook_sizes = list(cfg.vq_config.codebook_sizes)
        self.tp, self.group = tp, group
        self.hidden_norm = T5LayerNorm(config.hidden_size)
        self.hidden_proj = ReplicatedLinear(config.hidden_size, d, bias=False,
                                           params_dtype=torch.bfloat16, return_bias=False)
        self.transformer_layers = nn.ModuleList([
            DepthLayer(d, len(self.codebook_sizes), scale, tp, rank, group) for _ in range(layers)
        ])
        self.headnorm = T5LayerNorm(d)
        widths = [((size + 1 + tp - 1) // tp) * tp for size in self.codebook_sizes]
        self.head_widths = widths
        self.head_offsets = [sum(widths[:i]) for i in range(len(widths))]
        self.heads = nn.ModuleList([
            ColumnParallelLinear(d, width, bias=True,
                                 tp_size=tp, tp_rank=rank, params_dtype=torch.bfloat16, return_bias=False)
            for width in widths
        ])

    def forward(self, hidden, code_embeddings):
        # Same seven lookups/cumsum and eight-position causal computation as HF.
        # Level-independent: each codebook head runs on its own depth position and
        # callers slice the level they need in select(), so one compiled graph
        # stays valid for all eight levels.
        x = torch.cat([hidden.reshape(-1, 1, hidden.shape[-1]), code_embeddings.cumsum(1)], dim=1)
        x = self.hidden_proj(self.hidden_norm(x))
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.headnorm(x)
        return torch.cat([head(x[:, i].contiguous()) for i, head in enumerate(self.heads)], dim=-1)

    def select(self, logits, level):
        """Take this level's slice, then gather the TP vocab shards.

        ``logits`` is the concatenation of all codebook heads' *rank-local*
        slices, laid out level by level. The level slice therefore has to be
        taken BEFORE the gather: all_gather concatenates rank by rank, so a
        gather-first layout would be rank-major and every level would read
        another rank's fragment. Each width is a multiple of tp, so the local
        offset is exact.
        """
        start = self.head_offsets[level] // self.tp
        # Sliced views are non-contiguous; the collective needs a plain buffer.
        local = logits[..., start:start + self.head_widths[level] // self.tp].contiguous()
        if self.tp > 1:
            pieces = [torch.empty_like(local) for _ in range(self.tp)]
            dist.all_gather(pieces, local, group=self.group)
            logits = torch.cat(pieces, dim=-1)
        else:
            logits = local
        # Includes the official end-of-codebook row; excludes TP padding only.
        return logits[..., :self.codebook_sizes[level] + 1].contiguous()

    def finish_loading(self):
        for module in self.modules():
            if isinstance(module, (ColumnParallelLinear, RowParallelLinear, ReplicatedLinear)):
                # Keep standard format for the initial precision reference.
                module.quant_method.process_weights_after_loading(module, is_nz=False)


def head_partition(dimension, depth, ffn_scale, tp):
    intermediate = ffn_scale * dimension // depth
    if dimension % 128 or (dimension // 128) % tp or intermediate % tp:
        raise ValueError("Head dimensions do not admit the requested attention/depth-FFN TP split")
    return dimension // tp, intermediate // tp


# ---------------------------------------------------------------------------
# official generation state machine adapter
# ---------------------------------------------------------------------------

class RemoteModule(nn.Module):
    def __init__(self, executor, operation, device):
        super().__init__()
        self.executor, self.operation = executor, operation
        # PreTrainedModel.device needs at least one parameter even for a text
        # request without encoders. This zero-sized marker owns no model weight.
        self.marker = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)

    @property
    def weight(self):
        # HF uses this solely to determine the embedding's device.
        return self.marker

    def forward(self, *args, **kwargs):
        return self.executor.call(self.operation, *args, **kwargs)


class RemoteNgram(RemoteModule):
    def forward(self, input_ids, ngram_context=None):
        if ngram_context is None:
            ngram_context = input_ids.new_empty((input_ids.shape[0], 0))
        return self.executor.call("ngram", input_ids, ngram_context)


class RemoteHead(RemoteModule):
    def forward(self, hidden_states, code_ids, embedding, level):
        return self.executor.call(self.operation, hidden_states, code_ids, level=int(level))


def build_hf_adapter(config, executor, components, device, tokenizer):
    class FrameworkCache(longcat_model.NgramCache):
        def __init__(self, config):
            super().__init__(config=config)
            self.framework_length = 0

        def get_seq_length(self, layer_idx=0):
            return self.framework_length

    class FrameworkParent(nn.Module):
        def forward(  # pylint: disable=huawei-too-many-arguments
                self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, cache_position=None,
                use_cache=None, **kwargs):
            if not isinstance(past_key_values, FrameworkCache) or inputs_embeds is None:
                raise ValueError("Multimodal bridge requires composed embeddings and FrameworkCache")
            b, s, _ = inputs_embeds.shape
            if cache_position is None:
                cache_position = torch.arange(past_key_values.framework_length,
                                              past_key_values.framework_length + s, device=device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0).expand(b, -1)
            elif position_ids.shape[0] == 1 and b > 1:
                position_ids = position_ids.expand(b, -1)
            hidden = executor.call("backbone", inputs_embeds, cache_position, position_ids,
                                   positions=cache_position.cpu().tolist())
            past_key_values.framework_length += s
            return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)

    class InputAdapter(longcat_model.LongcatNextMultimodalModel, FrameworkParent):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = config
            self.embed_tokens = RemoteModule(executor, "embedding", device)
            self.ngram_embeddings = RemoteNgram(executor, "ngram", device)
            self.visual_tokenizer = components.get("model.visual_tokenizer")
            self.audio_tokenizer = components.get("model.audio_tokenizer")
            # Original placeholder and codebook constants, without constructing
            # the original backbone. Their values and input routing stay in HF.
            self._init_multimodal_constants(config)
            for name, value in self.named_buffers(recurse=False):
                setattr(self, name, value.to(device))

    mro = InputAdapter.mro()
    if mro[mro.index(longcat_model.LongcatNextMultimodalModel) + 1] is not FrameworkParent:
        raise RuntimeError("Unexpected HF inheritance: backbone interception is not valid")

    class GenerationAdapter(longcat_model.LongcatNextForGeneration):
        # The generation state machine, including the CFG batch expansion, is the
        # official implementation now carried by modeling_longcat_next.py: inherited
        # as-is, never re-wrapped or patched at runtime.

        def __init__(self):
            PreTrainedModel.__init__(self, config)
            self.model = InputAdapter()
            self.vocab_size = config.vocab_size
            self.lm_head = RemoteModule(executor, "lm_head", device)
            self.visual_head = RemoteHead(executor, "visual_head", device)
            self.audio_head = RemoteHead(executor, "audio_head", device)
            self.text_tokenizer = tokenizer

        @can_return_tuple
        def forward(  # pylint: disable=huawei-too-many-arguments
                self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                cache_position=None, logits_to_keep=0, visual_inputs=None, visual_ids=None,
                audio_inputs=None, audio_ids=None, audio_text_ids=None,
                multimodal_generation_status=None, visual_generation_config=None,
                audio_generation_config=None, **kwargs):
            # Fuse only ordinary text. Media/control substitutions remain in
            # the inherited HF forward, including its audio text-track rules.
            plain = (multimodal_generation_status is not None
                     and multimodal_generation_status.mode == "text"
                     and not multimodal_generation_status.has_placeholder
                     and input_ids is not None and input_ids.shape[0] == 1
                     and inputs_embeds is None and labels is None and use_cache
                     and isinstance(past_key_values, FrameworkCache)
                     and visual_inputs is None and audio_inputs is None
                     and all(v is None or v.numel() == 0 for v in (visual_ids, audio_ids, audio_text_ids))
                     and isinstance(logits_to_keep, int) and logits_to_keep in (0, 1))
            if plain:
                b, s = input_ids.shape
                if cache_position is None:
                    cache_position = torch.arange(past_key_values.framework_length,
                                                  past_key_values.framework_length + s, device=device)
                if position_ids is None:
                    position_ids = cache_position.view(1, -1)
                context = past_key_values.ngram_context
                n = config.emb_neighbor_num - 1
                context = input_ids.new_zeros((b, n)) if context is None else torch.nn.functional.pad(
                    context[:, -n:], (max(0, n - context.shape[1]), 0))
                logits = executor.call("text_step", input_ids, context, cache_position, position_ids,
                                       positions=cache_position.cpu().tolist())
                past_key_values.update_ngram_context(input_ids)
                past_key_values.framework_length += s
                output = longcat_model.LongcatNextForCausalLMOutputWithPast(
                    logits=logits, past_key_values=past_key_values, visual_ids=visual_ids, audio_ids=audio_ids)
                return output
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, cache_position=cache_position, logits_to_keep=logits_to_keep,
                visual_inputs=visual_inputs, visual_ids=visual_ids, audio_inputs=audio_inputs,
                audio_ids=audio_ids, audio_text_ids=audio_text_ids,
                multimodal_generation_status=multimodal_generation_status,
                visual_generation_config=visual_generation_config,
                # The outer HF decorator owns the caller's tuple preference.
                # Keep the inherited decorated forward's internal result an
                # output object, including when config.return_dict is False.
                audio_generation_config=audio_generation_config, return_dict=True, **kwargs)

    # Sampling/state transitions stay official; ordinary text's numerical
    # forward is one distributed call, media forward retains the HF method.
    model = GenerationAdapter().eval()
    return model, FrameworkCache


class CollectiveExecutor:
    def __init__(self, rank, world_size, device, handlers, control=None):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.handlers = handlers
        self.failed = False
        # CPU control traffic may wait while the owner runs a decoder/refiner.
        # Do not keep an HCCL device operation waiting through file processing.
        self.control = control if control is not None else dist.new_group(
            ranks=list(range(world_size)), backend="gloo", timeout=timedelta(hours=1))

    def _exchange(self, command=None, tensors=(), arguments=None):
        if self.rank == 0:
            tensors = tuple(t.detach().to(self.device).contiguous() for t in tensors)
            header = [
                command,
                [(tuple(t.shape), str(t.dtype).removeprefix("torch.")) for t in tensors],
                arguments or {},
            ]
        else:
            header = None
        payload = [header]
        dist.broadcast_object_list(payload, src=0, group=self.control)
        command, specs, arguments = payload[0]
        if command == "stop":
            if arguments.get("error"):
                raise RuntimeError("Controller failed: " + arguments["error"])
            return command, None
        if self.rank != 0:
            tensors = tuple(torch.empty(shape, dtype=getattr(torch, dtype), device=self.device)
                            for shape, dtype in specs)
        for tensor in tensors:
            if tensor.numel():
                dist.broadcast(tensor, src=0)
        error, result = None, None
        try:
            result = self.handlers[command](*tensors, **arguments)
        except Exception as exc:
            error = f"rank{self.rank} {command}: {type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self.control)
        if any(errors):
            self.failed = True
            raise RuntimeError("; ".join(e for e in errors if e))
        return command, result

    def call(self, command, *tensors, **arguments):
        if self.rank != 0:
            raise RuntimeError("Only rank0 may issue generation commands")
        return self._exchange(command, tensors, arguments)[1]

    def serve(self):
        while True:
            command, _ = self._exchange()
            if command == "stop":
                return

    def stop(self, error=None):
        if self.failed:
            # All ranks have already received the command failure and left
            # serve(); a second broadcast would have no receivers.
            return
        dist.broadcast_object_list([["stop", [], {"error": error}]], src=0, group=self.control)


# ---------------------------------------------------------------------------
# startup, request orchestration and result files
# ---------------------------------------------------------------------------

TASKS = {"text", "image_understanding", "image_generation", "audio_to_text",
         "audio_to_audio", "speech_synthesis"}
MODEL_DIR = Path(__file__).resolve().parents[1]  # models/longcat_next



def require_versions():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Python={sys.version.split()[0]}; required 3.12 for the validated CANN 9.1.0 environment."
        )
    for package, expected in (("torch", "2.8.0"), ("torch_npu", "2.8.0"), ("transformers", "4.57.6")):
        actual = version(package)
        # Ascend wheels may append .postN or a local build suffix.
        base = actual.split("+")[0].split(".post")[0]
        if base != expected:
            raise RuntimeError(
                f"{package}={actual}; required {expected}. Keep the specified CANN 9.1.0 environment."
            )


def _merge_generation_config(base, override):
    merged = copy.deepcopy(base)
    for name, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(name), dict):
            merged[name] = _merge_generation_config(merged[name], value)
        else:
            merged[name] = copy.deepcopy(value)
    return merged


def generation_options(config, request):
    options = config.model_config.custom_params["multimodal"]
    defaults = dict(options["generation"])
    for name in ("visual_generation_config", "audio_generation_config"):
        defaults[name] = _merge_generation_config(options[name], defaults.get(name, {}))
    settings = _merge_generation_config(defaults, request.get("generation", {}))
    settings.setdefault("max_new_tokens", config.scheduler_config.max_new_tokens)
    if settings["max_new_tokens"] > config.scheduler_config.max_new_tokens:
        raise ValueError("Request max_new_tokens exceeds scheduler cache reservation")
    forbidden = {"past_key_values", "inputs", "input_ids", "use_cache", "return_dict_in_generate",
                 "synced_gpus", "num_beams", "num_return_sequences", "prefill_chunk_size"}
    if forbidden & settings.keys():
        raise ValueError(f"Generation options cannot override adapter-owned fields: {forbidden & settings.keys()}")
    return settings


def run_multimodal(config, yaml_path):
    require_versions()
    options = config.model_config.custom_params["multimodal"]
    validate_config(config)
    world = config.parallel_config.world_size
    config.model_config.model_path = str(resolve_path(config.model_config.model_path))
    if not Path(config.model_config.model_path).is_dir():
        raise FileNotFoundError(config.model_config.model_path)
    requests = load_requests(resolve_path(options["request_file"]))
    # Check user-controlled generation settings before loading any weights.
    request_settings = [generation_options(config, request) for request in requests]
    template_settings = [chat_template_options(config, request) for request in requests]
    head_batches = head_graph_batches(requests, request_settings)
    tasks = {r["task"] for r in requests}
    visual = any("image" in task for task in tasks)
    audio = bool(tasks & {"audio_to_text", "audio_to_audio", "speech_synthesis"})
    visual_generation = "image_generation" in tasks
    audio_generation = bool(tasks & {"audio_to_audio", "speech_synthesis"})
    # Requests are serial. Reserve CFG only when image generation is requested,
    # and use the manifest's actual maximum output budget for paged KV sizing.
    config.scheduler_config.batch_size = 2 if visual_generation else 1
    config.scheduler_config.batch_size_per_dp_rank = config.scheduler_config.batch_size
    config.scheduler_config.max_new_tokens = max(s["max_new_tokens"] for s in request_settings)
    # The offline engine sizes KV storage from data_config.input_truncated_len
    # plus max_new_tokens.  A multimodal YAML commonly sets
    # scheduler_config.max_prefill_tokens higher than input_truncated_len; the
    # former is the advertised prompt budget, but it was previously ignored by
    # the offline capacity calculation.  That made a long prompt hit index
    # 4096 and surface the asynchronous error at the following visual-head
    # call.  Honor the larger prompt budget and leave room for the image
    # anyres prefix inserted by the official generation state machine.
    advertised_prefill = int(config.scheduler_config.max_prefill_tokens)
    prefix_margin = 64 if visual_generation else 0
    required_input_capacity = max(int(config.data_config.input_truncated_len), advertised_prefill) + prefix_margin
    if required_input_capacity != int(config.data_config.input_truncated_len):
        logger.info("Multimodal KV capacity: input_truncated_len %s -> %s "
                    "(max_prefill_tokens=%s, prefix_margin=%s)",
                    config.data_config.input_truncated_len, required_input_capacity,
                    advertised_prefill, prefix_margin)
        config.data_config.input_truncated_len = required_input_capacity
    if config.model_config.exe_mode == "npugraph_ex" and not config.model_config.enable_dynamic_graph:
        logger.warning("LongCat npugraph_ex uses changing SymInt[] attention lengths; enabling dynamic=True")
        config.model_config.enable_dynamic_graph = True
    rank = config.parallel_config.global_rank
    hf_config = load_official(config.model_config.model_path)
    validate_parallelism(config.parallel_config, hf_config)
    if getattr(hf_config, "quantization_config", None):
        raise ValueError("Initial multimodal adapter requires an unquantized checkpoint")
    # Runtime config overrides only; never rewrite checkpoint config/source files.
    if visual_generation:
        path = resolve_path(options["image_decoder_path"], Path(config.model_config.model_path))
        if not path.is_file():
            raise FileNotFoundError(path)
        hf_config.visual_config.visual_decoder_config.weight_path = str(path)
    if audio_generation:
        path = resolve_path(options["vocoder_path"], Path(config.model_config.model_path))
        if not path.is_file():
            raise FileNotFoundError(path)
        hf_config.audio_config.cosy24kvocoder_config.weight_path = str(path)
    # The official launcher owns WORK_DIR/RES_PATH and log_<rank>.log.
    result_root = Path(os.getenv("WORK_DIR", ".")) / os.getenv("RES_PATH", "")
    result_root.mkdir(parents=True, exist_ok=True)
    config.model_config.output_path = str(result_root)
    engine = ExecutionEngine(config)
    model_cls, config_cls = load_model_classes("longcat_next")
    engine.init(config_cls, model_cls)
    control = dist.new_group(ranks=list(range(world)), backend="gloo", timeout=timedelta(hours=1))
    # This entry warms its own text/features interfaces before measured requests;
    # it does not compile the normal executor's unrelated main_decode wrapper.
    tp = options.get("head_tp_size", 4)
    cm = engine.comm_manager
    cm.register_group("multimodal_head_tp_group", group_num=world // tp, group_size=tp)
    group = cm.get_group("multimodal_head_tp_group")
    head_rank = cm.get_rank("multimodal_head_tp_group")
    components, heads = {}, {}
    startup_error = None
    try:
        if tp > 1 or rank == 0:
            for enabled, modality in ((visual_generation, "visual"), (audio_generation, "audio")):
                if enabled:
                    with component_dtype(), torch.device(engine.device):
                        head = ParallelDepthHead(hf_config, modality, tp, head_rank, group)
                    heads[modality] = head
                    components[modality + "_head"] = head
        if rank == 0:
            # Empty parameters avoid a second random initialization and peak copies.
            # Buffers are constructed normally and retain their declared dtypes.
            with component_dtype(), init_empty_weights(include_buffers=False):
                if visual:
                    components["model.visual_tokenizer"] = (
                        modular_longcat_next_visual.LongcatNextVisualTokenizer(hf_config))
                if audio:
                    components["model.audio_tokenizer"] = (
                        modular_longcat_next_audio.LongcatNextAudioTokenizer(hf_config))
        load_components(config.model_config.model_path, components, engine.device, tuple(heads.values()))
    except Exception as exc:
        startup_error = f"rank{rank}: {type(exc).__name__}: {exc}"
    startup_errors = [None] * world
    dist.all_gather_object(startup_errors, startup_error, group=control)
    if any(startup_errors):
        raise RuntimeError("Multimodal initialization failed: " + "; ".join(e for e in startup_errors if e))
    session = BackboneSession(engine, hf_config)
    # Rank/interface isolation is essential for TP shards and graph buffers.
    session.compile_cache_root = result_root / "compile_cache" / f"rank_{rank}"
    head_runner = HeadGraphRunner(heads, config,
                                  session.compile_cache_root / "heads", head_batches)

    def head_call(hidden, ids, level, modality):
        if ids.ndim != 2 or type(level) is not int or not 0 <= level < ids.shape[1]:
            raise ValueError("Invalid codebook level")
        # The codebook lookup is a collective over the embedding TP group, so it
        # runs eagerly on every rank, outside every head graph: a head graph must
        # not capture it.
        embeddings = session.embedding(ids[:, :-1].contiguous())
        if tp == 1 and rank != 0:
            # Owner-only head graphs exist on rank 0 when head_tp_size == 1.
            return None
        if modality not in heads:
            raise ValueError(f"{modality} generation head not enabled by the request manifest")
        return head_runner(modality, hidden, embeddings, level)

    def head_warm_up():
        warm_up_heads(head_call, head_batches, hf_config, engine.device)
        torch.npu.synchronize()

    handlers = {
        "begin": session.begin, "warm_up": session.warm_up, "embedding": session.embedding,
        "head_warm_up": head_warm_up,
        "ngram": session.ngram_embeddings, "backbone": session.forward,
        "text_step": session.text_step,
        "lm_head": engine.main_worker.model.forward_lm_head,
        "visual_head": lambda hidden, ids, level: head_call(hidden, ids, level, "visual"),
        "audio_head": lambda hidden, ids, level: head_call(hidden, ids, level, "audio"),
    }
    executor = CollectiveExecutor(rank, world, engine.device, handlers, control=control)
    if rank != 0:
        try:
            with torch.inference_mode():
                executor.serve()
        finally:
            engine.profiler.current_profiler.stop()
            engine.profiler.enable_profiler = False
        return
    failure = None
    try:
        tokenizer = LongcatNextTokenizer.from_pretrained(config.model_config.model_path,
                                                        local_files_only=True, fix_mistral_regex=True)
        processor = LongcatNextProcessor.from_pretrained(config.model_config.model_path,
                                                        local_files_only=True)
        processor.tokenizer = tokenizer
        model, cache_cls = build_hf_adapter(hf_config, executor, components, engine.device, tokenizer)
        generation_file = Path(config.model_config.model_path) / "generation_config.json"
        if generation_file.is_file():
            model.generation_config = GenerationConfig.from_pretrained(config.model_config.model_path,
                                                                       local_files_only=True)
        with torch.inference_mode():
            warm_up_seconds = 0.0
            if engine.exe_mode != "eager":
                logger.info("Multimodal graph warm-up before requests (B1%s)", ", B2 CFG" if visual_generation else "")
                # Worker ranks execute the same complete warm-up command.
                warm_up_started = time.perf_counter()
                executor.call("warm_up", text=True, features=tasks != {"text"}, cfg=visual_generation)
                warm_up_seconds = time.perf_counter() - warm_up_started
                logger.info("Multimodal graph warm-up completed in %.2f s", warm_up_seconds)
            if head_runner.mode != "eager" and any(head_batches.values()):
                logger.info("Head graph warm-up: mode=%s batches=%s", head_runner.mode, head_batches)
                warm_up_started = time.perf_counter()
                executor.call("head_warm_up")
                head_seconds = time.perf_counter() - warm_up_started
                warm_up_seconds += head_seconds
                logger.info("Head graph warm-up completed in %.2f s", head_seconds)
            for request, settings, template_kwargs in zip(requests, request_settings, template_settings):
                executor.call("begin")
                seed = request.get("seed", config.data_config.seed if config.data_config.seed is not None else 42)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.npu.manual_seed_all(seed)
                target = result_root / "results" / request["id"]
                target.mkdir(parents=True, exist_ok=True)
                messages = build_messages(request)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       **template_kwargs)
                text_inputs, visual_inputs, audio_inputs = processor(text=prompt, return_tensors="pt")
                input_ids = text_inputs["input_ids"].to(engine.device)
                if input_ids.shape[0] != 1:
                    raise ValueError("Requests must be processed individually")
                torch.npu.synchronize()
                started = time.perf_counter()
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=text_inputs.get(
                        "attention_mask", torch.ones_like(input_ids)).to(engine.device),
                    visual_inputs=visual_inputs.to(engine.device) if visual_inputs is not None else None,
                    audio_inputs=audio_inputs.to(engine.device) if audio_inputs is not None else None,
                    past_key_values=cache_cls(hf_config), use_cache=True,
                    return_dict_in_generate=True, synced_gpus=False, **settings)
                torch.npu.synchronize()
                elapsed = time.perf_counter() - started
                token_ids = outputs.sequences[0, input_ids.shape[1]:].cpu().tolist()
                audio_text_ids = outputs.audio_text_ids[0].cpu().tolist() if outputs.audio_text_ids is not None else []
                text = tokenizer.decode(audio_text_ids or token_ids, skip_special_tokens=True)
                (target / "text.txt").write_text(text + "\n", encoding="utf-8")
                # Pass clones: official media decoding subtracts codebook offsets in place.
                visual_ids, audio_ids = outputs.visual_ids, outputs.audio_ids
                image_files, audio_files = [], []
                if visual_ids is not None and visual_ids.numel():
                    visual_options = settings["visual_generation_config"]["custom_params"]
                    expected = visual_options["token_h"] * visual_options["token_w"]
                    if visual_ids.shape[0] != expected:
                        raise ValueError(f"Incomplete image: {visual_ids.shape[0]} code groups, expected {expected}")
                    image_files = model.model.decode_visual_ids_and_save(
                        visual_ids.clone(), save_prefix=str(target / "image"), **visual_options)
                if audio_ids is not None and audio_ids.numel():
                    audio_files = model.model.decode_audio_ids_and_save(
                        audio_ids.clone(), save_prefix=str(target / "audio"),
                        **settings["audio_generation_config"]["custom_params"])
                record = {"id": request["id"], "task": request["task"], "text": text,
                          "image_files": image_files, "audio_files": audio_files,
                          "generation": settings, "seed": seed, "autoregressive_seconds": elapsed,
                          "chat_template_options": template_kwargs, "generated_sequence_tokens": len(token_ids),
                          "reached_token_budget": len(token_ids) >= settings["max_new_tokens"],
                          "head_exe_mode": head_runner.mode,
                          "timing_includes_control": True, "warm_up_outside_timing": True,
                          "run_warm_up_seconds": warm_up_seconds,
                          "timing_scope": "generate including control/heads; excluding warm-up and media decode"}
                (target / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("Request %s: outputs: %s", request["id"], text)
                logger.info("Request %s: generation time %.2f s (including control/heads); results: %s",
                            request["id"], elapsed, target)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            executor.stop(error=failure)
        finally:
            engine.profiler.current_profiler.stop()
            engine.profiler.enable_profiler = False


def resolve_path(value, base=MODEL_DIR):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_requests(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("requests"), list):
        raise ValueError("Manifest requires schema_version=1 and a requests array")
    if not document["requests"]:
        raise ValueError("Manifest must contain at least one request")
    seen = set()
    requests = []
    for source in document["requests"]:
        request = dict(source)
        request_id = request.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", request_id) or request_id in seen:
            raise ValueError(f"Invalid or duplicate request id: {request_id!r}")
        seen.add(request_id)
        task = request.get("task")
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task!r}; expected {sorted(TASKS)}")
        for name in ("prompt", "system"):
            if name in request and not isinstance(request[name], str):
                raise ValueError(f"{request_id}: {name} must be a string")
        if task in {"text", "image_generation", "speech_synthesis"} and not request.get("prompt"):
            raise ValueError(f"{request_id}: prompt is required")
        required_media = {
            "image_understanding": ("image",),
            "audio_to_text": ("audio",),
            "audio_to_audio": ("audio", "reference_audio"),
            "speech_synthesis": ("reference_audio",),
        }.get(task, ())
        media = request.get("media", {})
        if set(media) != set(required_media):
            raise ValueError(f"{request_id}: media keys must be {required_media}")
        resolved = {}
        for name in required_media:
            media_path = resolve_path(media[name], path.parent)
            if not media_path.is_file():
                raise FileNotFoundError(f"{request_id}: missing {name}: {media_path}")
            if "<longcat_" in str(media_path):
                raise ValueError("Media paths cannot contain LongCat control markers")
            resolved[name] = str(media_path)
        request["media"] = resolved
        requests.append(request)
    return requests


def build_messages(request):
    task, media = request["task"], request["media"]
    prompt = request.get("prompt", "")
    system = request.get("system", "")
    if "reference_audio" in media:
        system = ("Replicate the voice in the audio clip to formulate an answer:"
                  f"<longcat_audio_start>{media['reference_audio']}<longcat_audio_end>")
    if task == "image_understanding":
        prompt += f"<longcat_img_start>{media['image']}<longcat_img_end>"
    elif task == "image_generation":
        prompt += "<longcat_img_start>"
    elif task in {"audio_to_text", "audio_to_audio"}:
        prompt += f"<longcat_audio_start>{media['audio']}<longcat_audio_end>"
    if task in {"audio_to_audio", "speech_synthesis"}:
        prompt += "<longcat_audiogen_start>"
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def chat_template_options(config, request):
    """Forward the official template switch without losing media start suffixes."""
    options = config.model_config.custom_params["multimodal"]
    value = request.get("enable_thinking", options.get("enable_thinking"))
    if value is not None and not isinstance(value, bool):
        raise ValueError("enable_thinking must be true, false or null, outside generation")
    if "enable_thinking" in request.get("generation", {}):
        raise ValueError("Put enable_thinking at request top level, not inside generation")
    if request["task"] in {"image_generation", "audio_to_audio", "speech_synthesis"}:
        if request.get("enable_thinking") is not None:
            raise ValueError("Explicit enable_thinking would bypass the official media start suffix")
        return {}
    return {} if value is None else {"enable_thinking": value}


def head_graph_batches(requests, settings):
    """Reserve exactly the generation head B/CFG variants used by this manifest."""
    batches = {"visual": set(), "audio": set()}
    for request, generation in zip(requests, settings):
        if request["task"] == "image_generation":
            scale = generation["visual_generation_config"]["custom_params"]["cfg_scale"]
            batches["visual"].add(1 if scale == 1.0 else 2)
        elif request["task"] in {"audio_to_audio", "speech_synthesis"}:
            batches["audio"].add(1)
    return {name: sorted(values) for name, values in batches.items()}


def validate_parallelism(parallel, hf_config):
    """Checkpoint dimensions against the TP/EP sizes.

    Each entry mirrors an integer division in the model code.  An inexact
    division does not raise anything at run time: the shard simply loses rows,
    so the relation is checked before any weight is loaded.  Vocabulary rows and
    generation-head widths are absent on purpose: the loader rounds them up to a
    tp multiple (model_infer.py:149 for the vocabulary, model_infer.py:778 for
    the head widths), so they cannot be inexact.
    """
    tp, moe_tp = parallel.attn_tp_size, parallel.moe_tp_size
    relations = (
        ("hidden_size", "hidden_size", tp,
         "MLA q_b/kv_b/o_proj column and row shards (modeling_longcat_next.py:222/237/247)"),
        ("q_lora_rank", "q_lora_rank", tp,
         "q_a_proj / q_b_proj shards (modeling_longcat_next.py:220/222)"),
        ("kv_lora_rank", "kv_lora_rank", tp,
         "kv_a_proj / kv_b_proj shards (modeling_longcat_next.py:232/237)"),
        ("expert_ffn_hidden_size", "expert_ffn_hidden_size", moe_tp,
         "expert gate_up/down shards (modeling_longcat_next.py:831/840)"),
    )
    for label, attr, shard, why in relations:
        size = getattr(hf_config, attr, None)
        if size is None or shard <= 1:
            continue
        if size % shard != 0:
            raise ValueError(
                f"{label}={size} is not divisible by shard size {shard}: {why}")


def validate_config(config):
    """Check what the framework cannot know.

    ``world_size`` divisibility for every tp_size, ``num_experts % moe_ep_size``
    and ``num_attention_heads % attn_tp_size`` are already validated by
    ``executor/core/config/inference_config.py`` (ParallelConfig._validate and
    validate_model_config), so this function only covers adapter boundaries and
    the MoE relation the framework does not express.
    """
    options = config.model_config.custom_params["multimodal"]
    p = config.parallel_config
    # ---- boundaries of this entry point (functionality not implemented here)
    if p.attn_dp_size != 1:
        # Requests are driven by one rank-0 command channel over one DP group;
        # a second group would need its own request partition.
        raise ValueError(f"this adapter serves a single request-DP group; attn_dp_size={p.attn_dp_size}")
    if p.cp_size != 1:
        raise ValueError(f"context parallelism is not implemented; cp_size={p.cp_size}")
    if config.model_config.next_n != 0:
        raise ValueError("speculative decoding (next_n) is not implemented by this adapter")
    # ---- MoE runs experts either tensor-parallel or expert-parallel, never both:
    #      self.use_ep = (moe_tp_size == 1 and moe_ep_size > 1)  (modeling_longcat_next.py:941)
    if p.moe_tp_size > 1 and p.moe_ep_size > 1:
        raise ValueError(
            f"moe_tp_size={p.moe_tp_size} and moe_ep_size={p.moe_ep_size} cannot both exceed 1; "
            "set moe_tp_size=1 for expert parallelism or let it equal world_size for pure TP")
    if config.disagg_config.disaggregation_mode != "NONE":
        raise ValueError("Multimodal entry currently supports offline execution only")
    if config.scheduler_config.batch_size not in (1, 2):
        raise ValueError("Use batch_size 1 or 2; effective CFG capacity is resolved from the manifest")
    head_tp = options.get("head_tp_size", 4)
    # Each generation head is a ColumnParallelLinear with tp_size=head_tp_size
    # (model_infer.py:782-784) and its dimensions are checked by head_partition
    # (model_infer.py:828-832), so only the group size relation is checked here.
    if not isinstance(head_tp, int) or head_tp < 1 or p.world_size % head_tp:
        raise ValueError(f"head_tp_size={head_tp} must be a positive divisor of world_size={p.world_size}")
    if options.get("head_exe_mode", "eager") not in {"eager", "ge_graph", "npugraph_ex"}:
        raise ValueError("head_exe_mode must be eager, ge_graph or npugraph_ex")
    if not isinstance(options.get("head_enable_cache_compile", False), bool):
        raise ValueError("head_enable_cache_compile must be a YAML boolean")
    if config.model_config.force_eplb or config.model_config.custom_params.get("enable_afd", False):
        raise ValueError("force_eplb and AFD are not supported by this multimodal adapter")
    if not config.model_config.with_ckpt or config.model_config.dtype != "bfloat16":
        raise ValueError("Multimodal adapter requires real BF16 checkpoint weights")
