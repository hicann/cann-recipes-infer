# coding=utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

import argparse
import threading
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm
import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import save_file


class HunyuanWeightSplitterMixedSharded:

    def __init__(
        self,
        model_path: str,
        output_path: str,
        tp_attn: int,
        tp_moe: int,
        ep: int,
        max_shard_size_gb: float = 5.0,
        max_workers: int = 32,
        quant_int8=False,
        config: Dict = None
    ):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.tp_attn = tp_attn
        self.tp_moe = tp_moe
        self.ep = ep
        self.max_shard_size = int(max_shard_size_gb * 1e9)
        self.quant_int8 = quant_int8

        self.world_size = max(tp_attn, tp_moe * ep)

        if self.world_size % tp_attn != 0:
            raise ValueError(f"world_size ({self.world_size}) must be divisible by tp_attn ({tp_attn})")
        if self.world_size % (tp_moe * ep) != 0:
            raise ValueError(f"world_size ({self.world_size}) must be divisible by (tp_moe * ep) ({tp_moe * ep})")

        if config is None:
            with open(self.model_path / "config.json", 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config

        with open(self.model_path / "model.safetensors.index.json", 'r') as f:
            self.index_data = json.load(f)
            self.weight_map = self.index_data['weight_map']

        self.hidden_size = self.config['hidden_size']
        self.num_heads = self.config['num_attention_heads']
        self.num_kv_heads = self.config['num_key_value_heads']
        self.head_dim = self.config['attention_head_dim']
        self.num_experts = self.config['num_experts']
        self.num_layers = self.config['num_hidden_layers']

        self.output_path.mkdir(parents=True, exist_ok=True)

        # Number of configured threads for IO-intensive operations (load_and_split, save_sharded_weights)
        # 16-32 for SSD and 4-8 for HDD
        self.max_workers = max_workers

        # Number of configured threads for compute-intensive operations (recognize_moe_weights)
        # Automatically obtaining the number of CPU threads
        self.max_moe_workers = torch.get_num_threads()

        # Initialization of rank_weights
        self.rank_weights = {rank: {} for rank in range(self.world_size)}

        # Thread Lock: Protecting concurrent writes to rank_weights
        self.rank_weights_lock = threading.Lock()

        # Thread Lock: For updating progress_bar
        self.pbar_lock = threading.Lock()

        self.validate_split()

    def validate_split(self):
        if self.tp_attn <= self.num_kv_heads:
            if self.num_kv_heads % self.tp_attn != 0:
                raise ValueError(f"KV heads ({self.num_kv_heads}) must be divisible by tp_attn ({self.tp_attn})")
            self.replica_factor = 1
            self.kv_heads_per_rank = self.num_kv_heads // self.tp_attn
        else:
            if self.tp_attn % self.num_kv_heads != 0:
                raise ValueError(f"tp_attn ({self.tp_attn}) must be a multiple of KV heads ({self.num_kv_heads})")
            self.replica_factor = self.tp_attn // self.num_kv_heads
            self.kv_heads_per_rank = 1

        if self.num_experts % self.ep != 0:
            raise ValueError(f"Experts ({self.num_experts}) must be divisible by EP ({self.ep})")

    def should_include_expert(self, expert_id: int, rank: int) -> bool:
        experts_per_rank = self.num_experts // self.ep
        ep_rank = rank % self.ep
        start_expert = ep_rank * experts_per_rank
        end_expert = start_expert + experts_per_rank
        return start_expert <= expert_id < end_expert

    @staticmethod
    def split_tensor(
        tensor: torch.Tensor,
        dim: int,
        tp: int,
        rank: int
    ) -> torch.Tensor:
        if tp == 1:
            return tensor.clone()

        size = tensor.shape[dim]
        if size % tp != 0:
            raise ValueError(f"Dimension {dim} size {size} must be divisible by TP={tp}")

        tp_rank = rank % tp
        chunk_size = size // tp
        start = tp_rank * chunk_size
        end = start + chunk_size

        if dim == 0:
            return tensor[start:end, ...].contiguous()
        elif dim == 1:
            return tensor[:, start:end, ...].contiguous()
        else:
            raise ValueError(f"Unsupported split dimension: {dim}")

    def split_qkv_tensor_interleaved(
        self,
        qkv_tensor: torch.Tensor,
        rank: int
    ) -> torch.Tensor:
        is_bias = (qkv_tensor.dim() == 1)
        num_q_per_kv = self.num_heads // self.num_kv_heads

        tp_rank = rank % self.tp_attn

        if self.replica_factor > 1:
            kv_group_idx = tp_rank // self.replica_factor
            q_offset_in_group = tp_rank % self.replica_factor
            q_heads_per_rank = num_q_per_kv // self.replica_factor

            if is_bias:
                total_size = qkv_tensor.shape[0]
                expected_size = self.num_kv_heads * (num_q_per_kv + 2) * self.head_dim
                if total_size != expected_size:
                    raise ValueError(f"total_size need equal to expected_size!")

                qkv = qkv_tensor.reshape(self.num_kv_heads, num_q_per_kv + 2, self.head_dim)
                kv_group = qkv[kv_group_idx]
                q_part = kv_group[:num_q_per_kv]
                kv_part = kv_group[num_q_per_kv:]

                start_q = q_offset_in_group * q_heads_per_rank
                end_q = start_q + q_heads_per_rank
                q_split = q_part[start_q:end_q]

                result = torch.cat([q_split, kv_part], dim=0)
                result = result.reshape(-1).contiguous()
                return result

            else:
                out_dim, in_dim = qkv_tensor.shape
                expected_out_dim = self.num_kv_heads * (num_q_per_kv + 2) * self.head_dim
                if out_dim != expected_out_dim:
                    raise ValueError(f"out_dim need equal to expected_out_dim!")

                qkv = qkv_tensor.reshape(self.num_kv_heads, num_q_per_kv + 2, self.head_dim, in_dim)
                kv_group = qkv[kv_group_idx]
                q_part = kv_group[:num_q_per_kv]
                kv_part = kv_group[num_q_per_kv:]

                start_q = q_offset_in_group * q_heads_per_rank
                end_q = start_q + q_heads_per_rank
                q_split = q_part[start_q:end_q]

                result = torch.cat([q_split, kv_part], dim=0)
                new_out_dim = (q_heads_per_rank + 2) * self.head_dim
                result = result.reshape(new_out_dim, in_dim).contiguous()
                return result
        else:
            start_kv = tp_rank * self.kv_heads_per_rank
            end_kv = start_kv + self.kv_heads_per_rank

            if is_bias:
                total_size = qkv_tensor.shape[0]
                expected_size = self.num_kv_heads * (num_q_per_kv + 2) * self.head_dim
                if total_size != expected_size:
                    raise ValueError(f"total_size need equal to expected_size!")

                qkv = qkv_tensor.reshape(self.num_kv_heads, num_q_per_kv + 2, self.head_dim)
                qkv_split = qkv[start_kv:end_kv]
                result = qkv_split.reshape(-1).contiguous()
                return result
            else:
                out_dim, in_dim = qkv_tensor.shape
                expected_out_dim = self.num_kv_heads * (num_q_per_kv + 2) * self.head_dim
                if out_dim != expected_out_dim:
                    raise ValueError(f"out_dim need equal to expected_out_dim!")

                qkv = qkv_tensor.reshape(self.num_kv_heads, num_q_per_kv + 2, self.head_dim, in_dim)
                qkv_split = qkv[start_kv:end_kv]
                new_out_dim = self.kv_heads_per_rank * (num_q_per_kv + 2) * self.head_dim
                result = qkv_split.reshape(new_out_dim, in_dim).contiguous()
                return result

    def split_gate_and_up_proj(
        self,
        weight_tensor: torch.Tensor,
        rank: int
    ) -> torch.Tensor:
        is_bias = (weight_tensor.dim() == 1)

        if is_bias:
            total_size = weight_tensor.shape[0]
            if total_size % 2 != 0:
                raise ValueError("gate_and_up bias size should be even")

            half_size = total_size // 2
            up_bias = weight_tensor[:half_size]
            gate_bias = weight_tensor[half_size:]

            up_split = self.split_tensor(up_bias.unsqueeze(0), dim=1, tp=self.tp_moe, rank=rank).squeeze(0)
            gate_split = self.split_tensor(gate_bias.unsqueeze(0), dim=1, tp=self.tp_moe, rank=rank).squeeze(0)

            result = torch.cat([up_split, gate_split], dim=0)
            return result.contiguous()

        else:
            out_dim, in_dim = weight_tensor.shape
            if out_dim % 2 != 0:
                raise ValueError(f"gate_and_up_proj out_dim {out_dim} should be even")

            half_dim = out_dim // 2
            up = weight_tensor[:half_dim, :]
            gate = weight_tensor[half_dim:, :]

            up_split = self.split_tensor(up, dim=0, tp=self.tp_moe, rank=rank)
            gate_split = self.split_tensor(gate, dim=0, tp=self.tp_moe, rank=rank)

            result = torch.cat([up_split, gate_split], dim=0)

            expected_out_dim = out_dim // self.tp_moe
            if result.shape[0] != expected_out_dim:
                raise ValueError(f"gate_and_up split out_dim mismatch: {result.shape[0]} vs {expected_out_dim}")

            return result.contiguous()

    @staticmethod
    def quant_weight(
        weight_tensor: torch.Tensor,
        dist_type=torch.int8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the weight to INT8
        Reuturn: (int8_weight, scale)
        """
        ori_weight = weight_tensor
        int8_weight, scale = torch_npu.npu_dynamic_quant(ori_weight.to("npu"))
        return int8_weight.to("cpu"), scale.to("cpu")

    @staticmethod
    def switch_up_and_gate(weight_tensor: torch.Tensor) -> torch.Tensor:
        is_bias = (weight_tensor.dim() == 1)

        if is_bias:
            total_size = weight_tensor.shape[0]
            if total_size % 2 != 0:
                raise ValueError("gate_and_up bias size should be even")

            half_size = total_size // 2
            gate_bias = weight_tensor[:half_size]
            up_bias = weight_tensor[half_size:]
            reordered = torch.cat([up_bias, gate_bias], dim=0)
        else:
            out_dim, in_dim = weight_tensor.shape
            if out_dim % 2 != 0:
                raise ValueError(f"gate_and_up_proj out_dim {out_dim} should be even")

            half_dim = out_dim // 2
            gate = weight_tensor[:half_dim, :]
            up = weight_tensor[half_dim:, :]
            reordered = torch.cat([up, gate], dim=0)

        return reordered

    def _process_single_rank(self, rank: int, weights: Dict[str, torch.Tensor]) -> tuple:
        """
        Processing the MoE weights of a single rank
        Return: (rank, success or not, error infomation)
        """
        try:
            experts_per_rank = self.num_experts // self.ep
            ep_rank = rank % self.ep
            start_expert = ep_rank * experts_per_rank
            end_expert = start_expert + experts_per_rank

            moe_weights_by_layer = {}
            keys_to_remove = []

            # Parsing and classifying weight names
            for weight_name in list(weights.keys()):
                if '.mlp.experts.' in weight_name and not weight_name.endswith('.wg.weight'):
                    parts = weight_name.split('.')

                    # Extract layer index
                    layer_idx = None
                    for i, p in enumerate(parts):
                        if p == 'layers' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                break
                            except ValueError:
                                continue

                    # Extract layer index
                    expert_idx = None
                    for i, p in enumerate(parts):
                        if p == 'experts' and i + 1 < len(parts):
                            try:
                                expert_idx = int(parts[i + 1])
                                break
                            except ValueError:
                                continue

                    if layer_idx is not None and expert_idx is not None:
                        # Determine type of weight
                        if 'gate_and_up_proj.weight' in weight_name:
                            weight_type = 'w13_weight'
                        elif 'down_proj.weight' in weight_name:
                            weight_type = 'w2_weight'
                        elif 'gate_and_up_proj.scale' in weight_name:
                            weight_type = 'w13_scale'
                        elif 'down_proj.scale' in weight_name:
                            weight_type = 'w2_scale'
                        else:
                            continue

                        # Initialize data structure of layer
                        if layer_idx not in moe_weights_by_layer:
                            moe_weights_by_layer[layer_idx] = {
                                'w13_weight': {},
                                'w2_weight': {},
                                'w13_scale': {},
                                'w2_scale': {}
                            }

                        # Store the weight and mark the item for deletion
                        try:
                            moe_weights_by_layer[layer_idx][weight_type][expert_idx] = weights[weight_name]
                            keys_to_remove.append(weight_name)
                        except KeyError:
                            logger.error(f"weight {weight_name} (rank {rank}) is not exist")

            # Stack expert tensors of each layer
            for layer_idx in sorted(moe_weights_by_layer.keys()):
                layer_data = moe_weights_by_layer[layer_idx]

                # Process w13_weight
                if layer_data['w13_weight']:
                    expert_tensors = []
                    for exp_id in sorted(layer_data['w13_weight'].keys()):
                        expert_tensors.append(layer_data['w13_weight'][exp_id])
                    stacked = torch.stack(expert_tensors, dim=0)
                    new_name = f"model.layers.{layer_idx}.mlp.experts.w13_weight"
                    weights[new_name] = stacked

                # Process w2_weight
                if layer_data['w2_weight']:
                    expert_tensors = []
                    for exp_id in sorted(layer_data['w2_weight'].keys()):
                        expert_tensors.append(layer_data['w2_weight'][exp_id])
                    stacked = torch.stack(expert_tensors, dim=0)
                    new_name = f"model.layers.{layer_idx}.mlp.experts.w2_weight"
                    weights[new_name] = stacked

                # Process scale (if enabled)
                if self.quant_int8:
                    if layer_data['w13_scale']:
                        expert_tensors = []
                        for exp_id in sorted(layer_data['w13_scale'].keys()):
                            expert_tensors.append(layer_data['w13_scale'][exp_id])
                        stacked = torch.stack(expert_tensors, dim=0)
                        new_name = f"model.layers.{layer_idx}.mlp.experts.w13_scale"
                        weights[new_name] = stacked

                    if layer_data['w2_scale']:
                        expert_tensors = []
                        for exp_id in sorted(layer_data['w2_scale'].keys()):
                            expert_tensors.append(layer_data['w2_scale'][exp_id])
                        stacked = torch.stack(expert_tensors, dim=0)
                        new_name = f"model.layers.{layer_idx}.mlp.experts.w2_scale"
                        weights[new_name] = stacked

            # Delete original expert weights
            for key in keys_to_remove:
                if key in weights:
                    del weights[key]

            return (rank, True, None)

        except Exception as e:
            error_msg = f"Rank {rank} processing failed: {str(e)}"
            logger.error(error_msg)
            return (rank, False, error_msg)

    def recognize_moe_weights(self) -> None:
        """
        Recognize and reorganize weights of MoE
        """
        # Initialize the progress bar
        pbar = tqdm(total=self.world_size, desc="Processing MOE weights by rank")

        # Prepare thread tasks: One task for each rank
        tasks = [(rank, self.rank_weights[rank]) for rank in range(self.world_size)]

        # Execute multithread tasks
        results = []
        with ThreadPoolExecutor(max_workers=self.max_moe_workers) as executor:
            # Submit the processing tasks for all ranks
            future_to_rank = {
                executor.submit(self._process_single_rank, rank, weights): rank
                for rank, weights in tasks
            }

            # Collect task results and update progress bar
            for future in as_completed(future_to_rank):
                rank = future_to_rank[future]
                try:
                    res = future.result()
                    results.append(res)
                    # Update progress bar with Lock (to prevent multiple threads from writing at the same time)
                    with self.pbar_lock:
                        pbar.update(1)
                    if not res[1]:
                        tqdm.write(f"❌ Rank {rank} failed: {res[2]}")
                except Exception as e:
                    error_msg = f"Rank {rank} task exception: {str(e)}"
                    results.append((rank, False, error_msg))
                    tqdm.write(f"❌ {error_msg}")
                    with self.pbar_lock:
                        pbar.update(1)
                        pbar.set_postfix({"Processed": f"Rank {rank}", "Success": False})

        # Close the progress bar
        pbar.close()

        # Statistics processing result
        total_ranks = self.world_size
        success_ranks = sum(1 for _, s, _ in results if s)
        failed_ranks = total_ranks - success_ranks

        # Output the statistics
        if failed_ranks > 0:
            logger.error(f"⚠️ MOE weights finished: {success_ranks}/{total_ranks} ranks were successfully processed, \
                         and {failed_ranks} ranks failed to be processed")
            failed_rank_list = [r for r, s, _ in results if not s]
            logger.error(f"   List of failed ranks: {failed_rank_list}")
        else:
            logger.success(f"✅ MOE weights finished: All {total_ranks} ranks were successfully processed")

    def _process_single_file(self, file_name: str) -> tuple:
        """
        Loading and processing a single weight file
        Return: (file_name, success or not, error infomation)
        """
        try:
            file_path = self.model_path / file_name

            # 1. Load files
            with safe_open(file_path, framework="pt", device="cpu") as f:
                weight_names = [name for name, fname in self.weight_map.items() if fname == file_name]

                # 2. Process all weights within current file
                for weight_name in weight_names:
                    weight_tensor = f.get_tensor(weight_name).contiguous()

                    # 3. Distributed to each rank
                    for rank in range(self.world_size):
                        result = self.process_weight(weight_name, weight_tensor, rank)

                        if result is not None:
                            # Write to rank_weights with Lock
                            with self.rank_weights_lock:
                                try:
                                    new_name, new_tensor, scale = result
                                    scale_name = new_name.replace("weight", "scale")
                                    self.rank_weights[rank][scale_name] = scale
                                except ValueError:
                                    new_name, new_tensor = result
                                self.rank_weights[rank][new_name] = new_tensor

            return (file_name, True, None)

        except Exception as e:
            return (file_name, False, str(e))

    def load_and_split(self) -> None:
        """
        Load and split weights
        """
        # Obtain the list of files to be processed
        file_list = sorted(set(self.weight_map.values()))
        total_files = len(file_list)

        # Progress bar
        pbar = tqdm(total=total_files, desc="Processing weight files")

        # 多线程处理文件
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_name): file_name
                for file_name in file_list
            }

            # Collect task results and update progress
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    res = future.result()
                    results.append(res)
                    if not res[1]:
                        tqdm.write(f"⚠️ Failed to process files {res[0]}: {res[2]}")
                except Exception as e:
                    tqdm.write(f"⚠️ File task error {file_name}: {str(e)}")
                    results.append((file_name, False, str(e)))
                pbar.update(1)

        pbar.close()

        # Statistics results
        success_files = sum(1 for _, s, _ in results if s)
        failed_files = total_files - success_files

        if failed_files > 0:
            logger.error(f"❌ Finished: {success_files}/{total_files} files were successfully processed, \
                         and {failed_files} files failed to be processed")
        else:
            logger.success(f"✅ All {total_files} files were successfully processed")

    def process_weight(
        self,
        weight_name: str,
        weight_tensor: torch.Tensor,
        rank: int
    ) -> Tuple[str, torch.Tensor]:

        if any(x in weight_name for x in ['layernorm', 'norm.weight', 'norm.bias']):
            result = weight_tensor.clone()
            return (weight_name, result)

        if any(x in weight_name.lower() for x in ['embed', 'embedding']):
            result = weight_tensor.clone()
            return (weight_name, result)

        if 'lm_head' in weight_name:
            result = weight_tensor.clone()
            return (weight_name, result)

        if 'self_attn' in weight_name:
            if 'qkv_proj.weight' in weight_name or 'qkv_proj.bias' in weight_name:
                result = self.split_qkv_tensor_interleaved(weight_tensor, rank)
                return (weight_name, result)
            elif 'o_proj.weight' in weight_name:
                result = self.split_tensor(weight_tensor, dim=1, tp=self.tp_attn, rank=rank)
                return (weight_name, result)
            else:
                result = weight_tensor.clone()
                return (weight_name, result)

        if '.mlp.' in weight_name:
            if '.gate.wg.weight' in weight_name:
                result = weight_tensor.clone()
                return (weight_name, result)

            if 'shared_mlp' in weight_name:
                if 'gate_and_up_proj' in weight_name:
                    result = self.switch_up_and_gate(weight_tensor)
                else:
                    result = weight_tensor.clone()
                return (weight_name, result)

            if '.experts.' in weight_name:
                parts = weight_name.split('.')

                layer_idx = None
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue

                expert_idx = None
                for i, p in enumerate(parts):
                    if p == 'experts' and i + 1 < len(parts):
                        try:
                            expert_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue

                if expert_idx is None:
                    return None

                if not self.should_include_expert(expert_idx, rank):
                    return None

                if 'gate_and_up_proj' in weight_name:
                    reordered = self.switch_up_and_gate(weight_tensor)
                    result = self.split_gate_and_up_proj(reordered, rank)
                    if self.quant_int8:
                        result, scale = self.quant_weight(result)
                        return (weight_name, result, scale)
                    return (weight_name, result)

                elif 'down_proj.weight' in weight_name:
                    result = self.split_tensor(weight_tensor, dim=1, tp=self.tp_moe, rank=rank)
                    if self.quant_int8:
                        result, scale = self.quant_weight(result)
                        return (weight_name, result, scale)
                    return (weight_name, result)

                else:
                    result = weight_tensor.clone()
                    return (weight_name, result)

        result = weight_tensor.clone()
        return (weight_name, result)

    def shard_weights(self, weights: Dict[str, torch.Tensor]) -> Tuple[List[Dict], Dict]:
        shards = []
        current_shard = {}
        current_size = 0
        weight_map = {}

        for name, tensor in weights.items():
            size = tensor.numel() * tensor.element_size()

            if current_size + size > self.max_shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0

            current_shard[name] = tensor
            current_size += size

            shard_idx = len(shards)
            weight_map[name] = f"model-{shard_idx+1:05d}-of-{{total:05d}}.safetensors"

        if current_shard:
            shards.append(current_shard)

        total_shards = len(shards)
        weight_map = {
            k: v.format(total=total_shards)
            for k, v in weight_map.items()
        }

        return shards, weight_map

    def save_config(self):
        new_config = self.config.copy()
        new_config['tp_attn'] = self.tp_attn
        new_config['tp_moe'] = self.tp_moe
        new_config['ep'] = self.ep
        new_config['world_size'] = self.world_size
        new_config['replica_factor'] = self.replica_factor
        new_config['parallel_strategy'] = f"TP{self.tp_attn}_TP{self.tp_moe}_EP{self.ep}"
        if self.replica_factor > 1:
            new_config['parallel_strategy'] += f"_KVReplica{self.replica_factor}"
        new_config['max_shard_size'] = f"{self.max_shard_size / 1e9:.1f}GB"

        config_path = self.output_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)

        for rank in range(self.world_size):
            rank_dir = self.output_path / f"rank-{rank:02d}"
            rank_config_path = rank_dir / "config.json"
            shutil.copy2(config_path, rank_config_path)

    def copy_additional_files(self):
        files_to_copy = [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json",
            "generation_config.json", "preprocessor_config.json", "model_index.json",
        ]

        copied_files = []

        for file_name in files_to_copy:
            src = self.model_path / file_name
            if src.exists():
                dst = self.output_path / file_name
                shutil.copy2(src, dst)
                copied_files.append(file_name)

        if copied_files:
            for rank in range(self.world_size):
                rank_dir = self.output_path / f"rank-{rank:02d}"
                for file_name in copied_files:
                    src = self.output_path / file_name
                    dst = rank_dir / file_name
                    shutil.copy2(src, dst)

    @staticmethod
    def _save_single_shard(shard: Dict[str, torch.Tensor], shard_path: Path) -> tuple:
        """
        Thread task function for saving a single shard file
        Return: (shard_path, success or not, error information)
        """
        try:
            save_file(shard, str(shard_path))  # Make sure that the path is a string and is compatible with safetensors
            return (str(shard_path), True, None)
        except Exception as e:
            return (str(shard_path), False, str(e))

    def save_sharded_weights(self) -> None:
        """
        Saving shard weights in batches
        """
        # 1. Traverse earch rank
        # (Operation at the rank-level is performed in serial mode to avoid competition for directory/index files)
        for rank in tqdm(range(self.world_size), desc="Saving weights"):
            rank_dir = self.output_path / f"rank-{rank:02d}"
            rank_dir.mkdir(parents=True, exist_ok=True)

            # 2. Obtain shards of current rank and weight maps
            shards, weight_map = self.shard_weights(self.rank_weights[rank])
            total_shards = len(shards)

            # 3. Save all shard files of current rank with multi-thread mode
            shard_tasks = []
            for shard_idx, shard in enumerate(shards):
                shard_filename = f"model-{shard_idx+1:05d}-of-{total_shards:05d}.safetensors"
                shard_path = rank_dir / shard_filename
                shard_tasks.append((shard, shard_path))

            # 4. Creat a Thread Pool to execute the shard saving tasks
            shard_results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit saving tasks of all shards
                future_to_path = {
                    executor.submit(self._save_single_shard, shard, shard_path): str(shard_path)
                    for shard, shard_path in shard_tasks
                }

                # Collect task results
                for future in as_completed(future_to_path):
                    shard_path = future_to_path[future]
                    try:
                        res = future.result()
                        shard_results.append(res)
                        if not res[1]:
                            tqdm.write(f"⚠️ Failed to save shard {res[0]}: {res[2]}")
                    except Exception as e:
                        tqdm.write(f"⚠️ Shard task error {shard_path}: {str(e)}")
                        shard_results.append((shard_path, False, str(e)))

            # Check if all shards of current rank have been successed saved
            failed_shards = [r for r in shard_results if not r[1]]
            if failed_shards:
                tqdm.write(f"❌ {len(failed_shards)} shard of rank {rank} failed to be saved")
            else:
                tqdm.write(f"✅ All {total_shards} shards of rank {rank} were successfully saved")

            # 5. Calculate the total size of current rank and generate the index file.
            # (Performed in serial mode to avoid competion)
            total_size = sum(
                t.numel() * t.element_size()
                for t in self.rank_weights[rank].values()
            )

            index_data = {
                "metadata": {
                    "total_size": total_size,
                    "format": "pt"
                },
                "weight_map": weight_map
            }

            index_path = rank_dir / "model.safetensors.index.json"
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)

        # 6. Save config files
        self.save_config()

    def run(self):
        logger.info(f"Starting weight splitting")
        logger.info(f"  TP_attn: {self.tp_attn}")
        logger.info(f"  TP_moe: {self.tp_moe}")
        logger.info(f"  EP: {self.ep}")
        logger.info(f"  World size: {self.world_size}")
        if self.replica_factor > 1:
            logger.info(f"  KV replica factor: {self.replica_factor}")

        self.load_and_split()

        self.recognize_moe_weights()

        self.save_sharded_weights()

        self.copy_additional_files()

        logger.success(f"Weight update is Completed!")


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanImage3.0 TP splitting with KV replica support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Standard mode (tp_attn <= num_kv_heads):
  python split.py --model-path /path/to/model --output-path /path/to/output \\
    --tp-attn 8 --tp-moe 8 --ep 8 --max-shard-size 5.0

  # KV replica mode (tp_attn > num_kv_heads):
  python split.py --model-path /path/to/model --output-path /path/to/output \\
    --tp-attn 16 --tp-moe 16 --ep 1 --max-shard-size 5.0
        """
    )
    parser.add_argument("--model-path", type=str, required=True, help="Original model path")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for split weights")
    parser.add_argument("--tp-attn", type=int, required=True, help="TP degree for attention layers")
    parser.add_argument("--tp-moe", type=int, required=True, help="TP degree for MoE layers")
    parser.add_argument("--ep", type=int, required=True, help="Expert Parallel degree")
    parser.add_argument("--max-shard-size", type=float, default=5.0, help="Max shard size in GB (default: 5.0)")
    parser.add_argument("--max-workers", type=int, default=32, help="""max_workers for IO-intensive operations
                        (default 32, 16-32 recommended for SSD and 4-8 recommended for HDD)""")
    parser.add_argument("--int8", action="store_true", help="int8 weight only quant, only quantize moe")

    args = parser.parse_args()

    if args.tp_attn <= 0:
        raise ValueError(f"tp_attn must be positive, got: {args.tp_attn}")
    if args.tp_moe <= 0:
        raise ValueError(f"tp_moe must be positive, got: {args.tp_moe}")
    if args.ep <= 0:
        raise ValueError(f"EP must be positive, got: {args.ep}")
    if 64 % args.ep != 0:
        raise ValueError(f"64 experts cannot be divided by EP={args.ep}")
    if args.max_shard_size <= 0:
        raise ValueError(f"max_shard_size must be positive, got: {args.max_shard_size}")

    splitter = HunyuanWeightSplitterMixedSharded(
        model_path=args.model_path,
        output_path=args.output_path,
        tp_attn=args.tp_attn,
        tp_moe=args.tp_moe,
        ep=args.ep,
        max_shard_size_gb=args.max_shard_size,
        quant_int8=args.int8
    )
    splitter.run()


if __name__ == "__main__":
    main()