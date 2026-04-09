# coding=utf-8
# Adapted from
# https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/modules/hstu_processor.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import os
import builtins
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch_npu
from torchrec.sparse.jagged_tensor import JaggedTensor
from configs.hstu_config import HSTUConfig
from configs.inference_config import InferenceHSTUConfig
from modules.jagged_data import JaggedData, pad_jd_values
from modules.mlp import MLP
from modules.position_encoder import HSTUPositionalEncoder
from dataset.utils import RankingBatch

lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)


def fused_enabled(name: str) -> bool:
    return name in getattr(builtins, "ENABLED_FUSED_OPS", set())


def length_to_complete_offsets(length_tensor: torch.Tensor):
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(length_tensor)
    return offsets


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def jagged_2d_tensor_concat(
    values_list: List[torch.Tensor],
    offsets_list: List[torch.Tensor],
):
    if len(values_list) == 0 or len(offsets_list) == 0:
        raise ValueError("values_list and offsets_list cannot be empty")
    if len(values_list) != len(offsets_list):
        raise ValueError("values_list and offsets_list must have same length")
    dtype = values_list[0].dtype
    device = values_list[0].device
    hidden_dim = values_list[0].size(-1)
    for value in values_list:
        if value.dtype != dtype:
            raise ValueError("All values tensors must have the same dtype")
        if value.device != device:
            raise ValueError("All values tensors must be on the same device")
        if value.size(-1) != hidden_dim:
            raise ValueError("All values tensors must have the same hidden_dim")
    values_list = [switch_to_contiguous_if_needed(v) for v in values_list]
    offsets_list = [offset.to(device) for offset in offsets_list]

    batch_size = offsets_list[0].numel() - 1
    if batch_size <= 0:
        raise ValueError(f"Invalid batch_size={batch_size}, offsets.shape={offsets_list[0].shape}")
    if len(values_list) == 1:
        lengths = offsets_list[0][1:] - offsets_list[0][:-1]
        return values_list[0], lengths
    merged_offsets = offsets_list[0].clone()
    for offset in offsets_list[1:]:
        merged_offsets = merged_offsets + offset
    merged_lengths = merged_offsets[1:] - merged_offsets[:-1]
    total_length = sum(int(value.size(0)) for value in values_list)
    merged_values = values_list[0].new_empty((total_length, hidden_dim))

    prefix = torch.zeros((batch_size,), device=device, dtype=torch.long)
    batch_ids_base = torch.arange(batch_size, device=device, dtype=torch.long)

    for value, offset in zip(values_list, offsets_list):
        lengths = offset[1:] - offset[:-1]
        value_len = int(value.size(0))
        if value_len == 0:
            prefix = prefix + lengths
            continue
        row_ids = torch.arange(value_len, device=device, dtype=torch.long)
        try:
            batch_ids = torch.searchsorted(offset[1:], row_ids, right=True)
        except Exception:
            batch_ids = torch.repeat_interleave(batch_ids_base, lengths)
        start = offset.index_select(0, batch_ids)
        pos_in_batch = row_ids - start
        dst0 = merged_offsets.index_select(0, batch_ids)
        dstp = prefix.index_select(0, batch_ids)
        dst = dst0 + dstp + pos_in_batch

        merged_values.index_copy_(0, dst, value)
        prefix = prefix + lengths

    return merged_values, merged_lengths


def jagged_2d_tensor_concat_mxrec(
    values_list: List[torch.Tensor],
    offsets_list: List[torch.Tensor],
):
    if not values_list or not offsets_list:
        raise ValueError("Values_list and offsets_list cannot be empty")
    if len(values_list) != len(offsets_list):
        raise ValueError("values_list and offsets_list must have same length")
    
    value_tensors = [switch_to_contiguous_if_needed(tensor) for tensor in values_list]
    
    device = value_tensors[0].device
    offsets_dtype = offsets_list[0].dtype
    offset_tensors = [
        offset_tensor if (offset_tensor.device == device and offset_tensor.dtype == offsets_dtype)
        else offset_tensor.to(device=device, dtype=offsets_dtype)
        for offset_tensor in offsets_list
    ]

    concat_values = torch.ops.mxrec.concat_nd_jagged(
        1024,
        value_tensors,
        offset_tensors,
    )
    
    merged_offsets = offset_tensors[0]
    for offset_tensor in offset_tensors[1:]:
        merged_offsets = merged_offsets + offset_tensor
    concat_lengths = merged_offsets[1:] - merged_offsets[:-1]
    return concat_values, concat_lengths


def torch_split_2d_jagged(
    values: torch.Tensor,
    offsets_a: Optional[torch.Tensor] = None,
    offsets_b: Optional[torch.Tensor] = None,
    dense_size: int = 0,
    n_prefix_to_right: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if values.dim() != 2:
        raise ValueError(f"`values` must be 2D(L, D), got shape {tuple(values.shape)}")
    values = values.contiguous()
    values_len, hidden_dim = values.shape
    a_is_dense = offsets_a is None
    b_is_dense = offsets_b is None
    
    if n_prefix_to_right != 0 and (a_is_dense or b_is_dense):
        raise NotImplementedError(
            "n_prefix_to_right != 0 is only supported for jagged-jagged (offsets_a and offsets_b both not None)"
        )
    
    def _to_cpu_list_int(x: torch.Tensor) -> list:
        if x.numel() == 0:
            return []
        return x.detach().to(device="cpu", dtype=torch.int64).tolist()
    
    a_chunks = []
    b_chunks = []
    if (not a_is_dense) and (not b_is_dense):
        if offsets_a.dim() != 1 or offsets_b.dim() != 1:
            raise ValueError("offsets_a/offsets_b must be 1D (B+1, )")
        if offsets_a.numel() != offsets_b.numel():
            raise ValueError("offsets_a and offsets_b must have the same length (B+1, )")
        
        batch_size = offsets_a.numel() - 1
        offsets_a_cpu = _to_cpu_list_int(offsets_a)
        offsets_b_cpu = _to_cpu_list_int(offsets_b)

        expected_len = offsets_a_cpu[-1] + offsets_b_cpu[-1]
        if values_len != expected_len:
            raise ValueError(f"values length L={values_len} dose not match expected {expected_len}=sumA + sumB")
        
        for batch_idx in range(batch_size):
            a0, a1 = offsets_a_cpu[batch_idx], offsets_a_cpu[batch_idx + 1]
            b0, b1 = offsets_b_cpu[batch_idx], offsets_b_cpu[batch_idx + 1]
            len_a = a1 - a0
            len_b = b1 - b0
            start = a0 + b0
            seg = values[start:start + len_a + len_b]

            if n_prefix_to_right == 0:
                a_seg = seg[:len_a]
                b_seg = seg[len_a:]
            else:
                prefix = min(int(n_prefix_to_right), len_b)
                a_seg = seg[prefix:prefix + len_a]
                b_seg = torch.cat([seg[:prefix], seg[prefix + len_a:]], dim=0) if len_b > 0 else seg[:0]
            a_chunks.append(a_seg)
            b_chunks.append(b_seg)
        
        out_a = torch.cat(a_chunks, dim=0) if a_chunks else values[:0]
        out_b = torch.cat(b_chunks, dim=0) if b_chunks else values[:0]
        return out_a, out_b
    
    if a_is_dense and b_is_dense:
        raise ValueError("At least one of offsets_a / offsets_b must be provided (cannot both be dense)")
    
    if a_is_dense:
        if offsets_b is None:
            raise ValueError("offset_b must be provided when A is dense")
        if dense_size <= 0:
            raise ValueError("dense_size must be > 0 when A is dense")
        if offsets_b.dim() != 1:
            raise ValueError("offsets_b must be 1D (B+1,)")
        
        batch_size = offsets_b.numel() - 1
        offsets_b_cpu = _to_cpu_list_int(offsets_b)
        
        expected_len = batch_size * dense_size + offsets_b_cpu[-1]
        if values_len != expected_len:
            raise ValueError(f"values length L={values_len} dose not match expected {expected_len}=sumA + sumB")
        for batch_idx in range(batch_size):
            b0, b1 = offsets_b_cpu[batch_idx], offsets_b_cpu[batch_idx + 1]
            len_b = b1 - b0
            start = batch_idx * dense_size + b0
            seg = values[start:start + dense_size + len_b]
            a_chunks.append(a_seg[:dense_size])
            b_chunks.append(b_seg[dense_size:])
        out_a = torch.stack(a_chunks, dim=0).reshape(batch_size, dense_size, hidden_dim)
        out_b = torch.cat(b_chunks, dim=0) if b_chunks else values[:0]
        return out_a, out_b
    if offsets_a is None:
        raise ValueError("offsets_a must be provided when B is dense")
    if dense_size <= 0:
        raise ValueError("dense_size must be > 0 when B is dense")
    if offsets_a.dim() != 1:
        raise ValueError("off_sets_a must be 1D (B+1,)")
    
    batch_size = offsets_a.numel() - 1
    offsets_a_cpu = _to_cpu_list_int(offsets_a)
    expected_len = offsets_a_cpu[-1] + batch_size * dense_size
    if values_len != expected_len:
        raise ValueError(
            (
                f"values length L={values_len} does not match expected {expected_len}="
                "sumA + sumB * dense_size"
            )
        )
    for batch_idx in range(batch_size):
        a0, a1 = offsets_a_cpu[batch_idx], offsets_b_cpu[batch_idx + 1]
        len_a = a1 - a0
        start = a0 + batch_idx * dense_size
        seg = values[start:start + len_a + dense_size]
        a_chunks.append(seg[:len_a])
        b_chunks.append(seg[len_a:])
    
    out_a = torch.cat(a_chunks, dim=0) if a_chunks else values[:0]
    out_b = torch.stack(b_chunks, dim=0).reshape(batch_size, dense_size, hidden_dim)
    return out_a, out_b


def hstu_preprocess_embeddings(
    embeddings: Dict[str, JaggedTensor],
    batch: RankingBatch,
    is_inference: bool,
    item_mlp: Optional[MLP] = None,
    contextual_mlp: Optional[MLP] = None,
    dtype: Optional[torch.dtype] = None,
    scaling_seqlen: int = -1,
) -> JaggedData:
    """
    Preprocesses the embeddings for use in the HSTU architecture.

    This method performs the following steps:
    1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
                         During inference, action embeddings are only for the history sequence, and
                         they will be interleaved with item embeddings of the history part, while
                         the embeddings of candidates need no interleaving.
    2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample,
                          following the order specified in the batch.
                          During inference, we concatenate three parts:
                          1) contextual embeedings,
                          2) interleaved *item & action* history embeddings, and
                          3) (item) candidates embeddings
                          for each sample, following the order specified in the batch.

    Args:
        embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key
        corresponds to a feature name and the value is a jagged tensor.
        batch (RankingBatch): The batch of ranking data.
        is_inference (bool): Whether is for inference
        dtype (dtype, optional): The output data type of the embeddings.
    Returns:
        JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
    """
    item_jt = embeddings[batch.item_feature_name]  # history + candidate
    dtype = item_jt.values().dtype if dtype is None else dtype
    sequence_embeddings = item_jt.values().to(dtype)
    sequence_embeddings_lengths = item_jt.lengths()
    sequence_embeddings_lengths_offsets = item_jt.offsets()
    sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

    if batch.action_feature_name is not None:
        action_jt = embeddings[batch.action_feature_name]
        jagged_size = sequence_embeddings.size(0)
        embedding_dim = sequence_embeddings.size(1)

        if not is_inference:
            sequence_embeddings = torch.cat(
                [sequence_embeddings, action_jt.values().to(dtype)], dim=1
            ).view(2 * jagged_size, embedding_dim)
            sequence_embeddings_lengths = sequence_embeddings_lengths * 2
            sequence_embeddings_lengths_offsets = (
                sequence_embeddings_lengths_offsets * 2
            )
            sequence_max_seqlen = sequence_max_seqlen * 2
        else:
            action_offsets = action_jt.offsets()
            item_offsets = item_jt.offsets()
            candidates_indptr = item_offsets[: batch.batch_size] + action_jt.lengths()

            item_embs = item_jt.values().to(dtype)
            action_embs = action_jt.values().to(dtype)
            interleaved_embeddings = [
                (
                    torch.cat(
                        [
                            item_embs[item_offsets[idx]: candidates_indptr[idx]],
                            action_embs[action_offsets[idx]: action_offsets[idx + 1]],
                        ],
                        dim=1,
                    ).view(-1, embedding_dim),
                    item_embs[candidates_indptr[idx]: item_offsets[idx + 1]],
                )
                for idx in range(batch.batch_size)
            ]
            interleaved_embeddings = list(itertools.chain(*interleaved_embeddings))
            sequence_embeddings = torch.cat(interleaved_embeddings, dim=0).view(
                -1, embedding_dim
            )
            sequence_embeddings_lengths = item_jt.lengths() + action_jt.lengths()
            sequence_embeddings_lengths_offsets = (
                item_jt.offsets() + action_jt.offsets()
            )
            sequence_max_seqlen += batch.feature_to_max_seqlen[
                batch.action_feature_name
            ]
        if item_mlp is not None:
            sequence_embeddings = item_mlp(sequence_embeddings)

    if (
        batch.num_candidates is not None
        and batch.action_feature_name is not None
        and not is_inference
    ):
        num_candidates = batch.num_candidates * 2
        max_num_candidates = batch.max_num_candidates * 2
    else:
        num_candidates = batch.num_candidates
        max_num_candidates = batch.max_num_candidates

    contextual_max_seqlen = 0
    contextual_seqlen = None
    contextual_seqlen_offsets = None
    if len(batch.contextual_feature_names) > 0:
        contextual_max_seqlens = [
            batch.feature_to_max_seqlen[name] for name in batch.contextual_feature_names
        ]
        contextual_jts = [embeddings[name] for name in batch.contextual_feature_names]
        contextual_jts_values = [jt.values().to(dtype) for jt in contextual_jts]
        contextual_jts_offsets = [jt.offsets() for jt in contextual_jts]

        if fused_enabled("concat_nd_jagged") and hasattr(torch.ops.mxrec, "concat_nd_jagged"):
            (contextual_sequence_embeddings, contextual_seqlen) = jagged_2d_tensor_concat_mxrec(
                contextual_jts_values,
                contextual_jts_offsets,
            )
        else:
            (contextual_sequence_embeddings, contextual_seqlen) = jagged_2d_tensor_concat(
                contextual_jts_values,
                contextual_jts_offsets,
            )
        if contextual_mlp is not None:
            contextual_sequence_embeddings = contextual_mlp(
                contextual_sequence_embeddings
            )
        contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            contextual_seqlen
        )
        contextual_max_seqlen = max(
            len(batch.contextual_feature_names), sum(contextual_max_seqlens)
        )
        if fused_enabled("concat_nd_jagged") and hasattr(torch.ops.mxrec, "concat_nd_jagged"):
            (
                sequence_embeddings,
                sequence_embeddings_lengths,
            ) = jagged_2d_tensor_concat_mxrec(
                [contextual_sequence_embeddings, sequence_embeddings],
                [contextual_seqlen_offsets, sequence_embeddings_lengths_offsets],
            )
        else:
            (
                sequence_embeddings,
                sequence_embeddings_lengths,
            ) = jagged_2d_tensor_concat(
                [contextual_sequence_embeddings, sequence_embeddings],
                [contextual_seqlen_offsets, sequence_embeddings_lengths_offsets],
            )

        sequence_embeddings_lengths_offsets = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(sequence_embeddings_lengths)
        )
        sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

    return JaggedData(
        values=sequence_embeddings,
        seqlen=sequence_embeddings_lengths.to(
            torch.int32
        ),  # contextual + history + candidate
        seqlen_offsets=sequence_embeddings_lengths_offsets.to(torch.int32),
        max_seqlen=sequence_max_seqlen,
        max_num_candidates=max_num_candidates,
        num_candidates=num_candidates.to(torch.int32)
        if num_candidates is not None
        else None,
        num_candidates_offsets=length_to_complete_offsets(num_candidates).to(
            torch.int32
        )
        if num_candidates is not None
        else None,
        contextual_max_seqlen=contextual_max_seqlen,
        contextual_seqlen=contextual_seqlen.to(torch.int32)
        if contextual_seqlen is not None
        else None,
        contextual_seqlen_offsets=contextual_seqlen_offsets.to(torch.int32)
        if contextual_seqlen_offsets is not None
        else None,
        has_interleaved_action=batch.action_feature_name is not None,
        scaling_seqlen=scaling_seqlen,
    )


class HSTUBlockPreprocessor(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: Union[HSTUConfig, InferenceHSTUConfig],
        is_inference: bool,
    ):
        super().__init__()
        self.config = config
        self._training_dtype = torch.float32
        if config.bf16:
            self._training_dtype = torch.bfloat16
        if config.fp16:
            self._training_dtype = torch.float16
        if isinstance(config, HSTUConfig):
            self._sequence_parallel = config.sequence_parallel
        else:
            self._sequence_parallel = False
        self._tp_size = 1
        if is_inference:
            self._sequence_parallel = False

        self._item_mlp = None
        self._contextual_mlp = None
        if config.hstu_preprocessing_config is not None:
            if config.hstu_preprocessing_config.item_embedding_dim > 0:
                self._item_mlp = MLP(
                    in_size=config.hstu_preprocessing_config.item_embedding_dim,
                    layer_sizes=[config.hidden_size, config.hidden_size],
                    activation="relu",
                    bias=True,
                )
            if config.hstu_preprocessing_config.contextual_embedding_dim > 0:
                self._contextual_mlp = MLP(
                    in_size=config.hstu_preprocessing_config.contextual_embedding_dim,
                    layer_sizes=[config.hidden_size, config.hidden_size],
                    activation="relu",
                    bias=True,
                )

        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if config.position_encoding_config is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=config.position_encoding_config.num_position_buckets,
                num_time_buckets=config.position_encoding_config.num_time_buckets,
                embedding_dim=config.hidden_size,
                is_inference=is_inference,
                use_time_encoding=config.position_encoding_config.use_time_encoding,
                training_dtype=self._training_dtype,
            )
        self._is_inference = is_inference
        self._dropout_ratio = 0.0
        if not self._is_inference:
            if not isinstance(config, HSTUConfig):
                raise TypeError("Training config should be HSTUConfig")
            self._dropout_ratio = config.hidden_dropout
        self._scaling_seqlen = config.scaling_seqlen


    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: RankingBatch,
        seq_start_position: torch.Tensor = None,
    ) -> JaggedData:
        """
        Preprocesses the embeddings for use in the HSTU architecture.

        This method performs the following steps:
        1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
        2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample,
                              following the order specified in the batch.
        3. **Padding**: Pads the jagged length of JaggedData to the TP size if sequence parallel is enabled.
        4. **Position Encoding**: Applies position encoding to the concatenated embeddings.

        Args:
            embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds 
                                                  to a feature name and the value is a jagged tensor.
            batch (RankingBatch): The batch of ranking data.

        Returns:
            JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
        """
        device = torch.device("npu", torch_npu.npu.current_device())
        batch = batch.to(device)
        # Interleaving & concatenation
        jd = hstu_preprocess_embeddings(
            embeddings,
            batch,
            is_inference=self._is_inference,
            item_mlp=self._item_mlp,
            contextual_mlp=self._contextual_mlp,
            dtype=self._training_dtype,
            scaling_seqlen=self._scaling_seqlen,
        )
        if self._sequence_parallel:
            jd = pad_jd_values(jd, self._tp_size)
        if self._positional_encoder is not None:
            jd.values = self._positional_encoder(
                max_seq_len=jd.max_seqlen,
                seq_lengths=jd.seqlen,
                seq_offsets=jd.seqlen_offsets,
                seq_timestamps=None,
                seq_embeddings=jd.values,
                num_targets=jd.num_candidates,
                seq_start_position=seq_start_position,
            )

        jd.values = torch.nn.functional.dropout(
            jd.values,
            p=self._dropout_ratio,
            training=self.training,
        ).to(self._training_dtype)

        return jd


class HSTUBlockPostprocessor(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(self, is_inference: bool, sequence_parallel: bool = False):
        super().__init__()
        self._is_inference = is_inference
        self._sequence_parallel = sequence_parallel

        if self._is_inference:
            self._sequence_parallel = False


    def forward(self, jd: JaggedData) -> JaggedData:
        """
        Postprocess the output from the HSTU architecture.
        1. If max_num_candidates > 0, split and only keep last ``num_candidates`` embeddings as 
           candidates embedding for further processing.
        2, If sequence parallel is on, we need to gather the values back and remove the padding.
        3. Remove action embeddings if present. Only use item embedding for further processing.

        Args:
            jd (JaggedData): The jagged data output from the HSTU architecture that needs further processing.

        Returns:
            JaggedData: The postprocessed jagged data.
        """
        sequence_embeddings: torch.Tensor
        seqlen_offsets: torch.Tensor
        max_seqlen: int
        if jd.max_num_candidates > 0:
            seqlen_offsets = jd.num_candidates_offsets
            max_seqlen = jd.max_num_candidates
            _, sequence_embeddings = torch_split_2d_jagged(
                jd.values,
                offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                offsets_b=seqlen_offsets,
            )
        elif jd.contextual_max_seqlen > 0:
            seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
            max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
            _, sequence_embeddings = torch_split_2d_jagged(
                jd.values,
                offsets_a=jd.contextual_seqlen_offsets,
                offsets_b=seqlen_offsets,
            )
        else:
            sequence_embeddings = jd.values
            seqlen_offsets = jd.seqlen_offsets
            max_seqlen = jd.max_seqlen

        if jd.has_interleaved_action and not self._is_inference:
            sequence_embeddings = sequence_embeddings[0::2, ...]
            seqlen_offsets = seqlen_offsets // 2
            max_seqlen = max_seqlen // 2

        sequence_embeddings = sequence_embeddings / torch.linalg.norm(
            sequence_embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)

        return JaggedData(
            values=sequence_embeddings,
            seqlen=torch.diff(seqlen_offsets).to(jd.seqlen.dtype),
            seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
            max_seqlen=max_seqlen,
            has_interleaved_action=False,
            scaling_seqlen=jd.scaling_seqlen,
        )