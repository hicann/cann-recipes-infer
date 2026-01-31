# coding=utf-8
# Adapted from
# https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/inference/inference_gr_ranking.py
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
import argparse
import enum
import math
import os
import sys
import time
import logging
import gin
import torch
import torch.distributed as dist
import torch_npu
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from modules.metrics import get_multi_event_metric_module
from preprocessor import get_common_preprocessors
from utils import DatasetArgs, NetworkArgs, RankingArgs
from model.inference_ranking_gr import InferenceRankingGR
from dataset import get_data_loader
from dataset.inference_dataset import InferenceDataset
from dataset.sequence_dataset import get_dataset

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)


def recursive_traverse_dict(input_dict, key_prefix):
    ret = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            sub_dict = recursive_traverse_dict(value, key_prefix=f"{key_prefix}.{key}")
            ret.update(sub_dict)
        else:
            ret.update({f"{key_prefix}.{key}": value})
    return ret


def stringify_dict(input_dict, prefix="", sep=","):
    ret = recursive_traverse_dict(input_dict, prefix)
    output = ""
    for key, value in ret.items():
        if isinstance(value, torch.Tensor):
            value.float()
            if value.dim() != 0:
                raise ValueError(f"{key}: expected a scalar tensor (0-dim), got dim={value.dim()}")
            value = value.cpu().item()
            output += key + ": " + f"{value:6f}{sep}"
        elif isinstance(value, float):
            output += key + ": " + f"{value:6f}{sep}"
        elif isinstance(value, int):
            output += key + ": " + f"{value}{sep}"
        else:
            raise TypeError(f"stringify_dict does not support type {type(value)} for key={key}")
    # remove the ending sep
    pos = output.rfind(sep)
    return output[0:pos]


class RunningMode(enum.Enum):
    EVAL = "eval"
    SIMULATE = "simulate"

    def __str__(self):
        return self.value


def get_inference_dataset_and_embedding_configs(
    disable_contextual_features: bool = False,
):
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    HASH_SIZE = 10_000_000
    if dataset_args.dataset_name == "kuairand-1k":
        embedding_configs = [
            InferenceEmbeddingConfig(
                feature_names=["user_id"],
                table_name="user_id",
                vocab_size=1000,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["video_id"],
                table_name="video_id",
                vocab_size=HASH_SIZE,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["action_weights"],
                table_name="action_weights",
                vocab_size=233,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
        ]
        return (
            dataset_args,
            embedding_configs
            if not disable_contextual_features
            else embedding_configs[-2:],
        )

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


def get_inference_hstu_model(
    emb_configs,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    hccl_comm_dict=None,
    embed_tp_size=1
):
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
        static_max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
        scaling_seqlen=network_args.scaling_seqlen,
    )

    kvcache_args = {
        "blocks_in_primary_pool": 5120,
        "page_size": 32,
        "offload_chunksize": 1024,
        "max_batch_size": max_batch_size,
        "max_seq_len": math.ceil(total_max_seqlen / 32) * 32,
    }
    kv_cache_config = get_kvcache_config(**kvcache_args)

    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [128] + [i * 256 for i in range(1, 34)],
    }

    model = InferenceRankingGR(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=True,
        cudagraph_configs=hstu_cudagraph_configs,
        hccl_comm_dict=hccl_comm_dict,
        embed_tp_size=embed_tp_size
    )
    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()
    model.eval()

    return model


def create_tp_process_group(rank, world_size, tp_size):
    hccl_comm_dict = {}
    dist_ready = dist.is_available() and dist.is_initialized()
    tp_enabled = (tp_size > 1) and (world_size > 1)
    if (not dist_ready) or (not tp_enabled):
        logger.info("[Rank %s] TP disable, skipping group creation", rank)
        return {"embed_tp_group": None}

    if world_size % tp_size != 0:
        raise ValueError(f"World size {world_size} must be divided by TP size {tp_size}")

    num_tp_groups = world_size // tp_size
    for i in range(num_tp_groups):
        start_rank = i * tp_size
        end_rank = start_rank + tp_size
        ranks_in_group = list(range(start_rank, end_rank))
        group = dist.new_group(ranks=ranks_in_group, backend="hccl")
        
        if rank in ranks_in_group:
            hccl_comm_dict["embed_tp_group"] = group
            logger.info("[Rank %s] Joined TP group with ranks: %s", rank, ranks_in_group)
    
    return hccl_comm_dict


def ensure_dist_and_device():
    rank, world_size = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.npu.set_device(local_rank)
        logger.info(
            "Using existing Process Group: rank=%s, world_size=%s, device=%s",
            rank, world_size, local_rank
        )
        return rank, world_size, local_rank
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        logger.info(
            "Initialized Process Group: rank=%s, world_size=%s, device=%s",
            rank, world_size, local_rank
        )
        return rank, world_size, local_rank
    
    torch.npu.set_device(local_rank)
    logger.info("Not using distributed mode")
    return rank, world_size, local_rank


def run_ranking_gr_simulate(
    check_auc: bool = False,
    disable_contextual_features: bool = False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 1
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    logger.info("total_max_seqlen %s", total_max_seqlen)

    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
        )

        if check_auc:
            eval_module = get_multi_event_metric_module(
                num_classes=model._task_config.prediction_head_arch[-1],
                num_tasks=model._task_config.num_tasks,
                metric_types=model._task_config.eval_metrics,
            )

        dataset = InferenceDataset(
            seq_logs_file=dataproc._inference_sequence_file,
            batch_logs_file=dataproc._inference_batch_file,
            batch_size=max_batch_size,
            max_seqlen=dataset_args.max_sequence_length,
            item_feature_name=dataproc._item_feature_name,
            contextual_feature_names=dataproc._contextual_feature_names
            if not disable_contextual_features
            else [],
            action_feature_name=dataproc._action_feature_name,
            max_num_candidates=dataset_args.max_num_candidates,
            item_vocab_size=10_000_000,
            userid_name="user_id",
            date_name="date",
            sequence_endptr_name="interval_indptr",
            timestamp_names=["date", "interval_end_ts"],
        )

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        num_batches_ctr = 0
        start_time = time.time()
        cur_date = None
        while True:
            try:
                uids, dates, seq_endptrs = next(dataloader_iter)
                if dates[0] != cur_date:
                    if cur_date is not None:
                        eval_metric_dict = eval_module.compute()
                        logger.info(
                            "[eval]:\n    %s",
                            stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    "),
                        )
                    model.clear_kv_cache()
                    cur_date = dates[0]
                cached_start_pos, cached_len = model.get_user_kvdata_info(
                    uids, dbg_print=True
                )
                new_cache_start_pos = cached_start_pos + cached_len
                non_contextual_mask = new_cache_start_pos >= num_contextual_features
                contextual_mask = torch.logical_not(non_contextual_mask)
                seq_startptrs = (
                    torch.clip(new_cache_start_pos - num_contextual_features, 0) / 2
                ).int()

                batch_0 = dataset.get_input_batch(
                    uids[non_contextual_mask],
                    dates[non_contextual_mask],
                    seq_endptrs[non_contextual_mask],
                    seq_startptrs[non_contextual_mask],
                    with_contextual_features=False,
                    with_ranking_labels=True,
                )
                if batch_0 is not None:
                    logits = model.forward(
                        batch_0,
                        uids[non_contextual_mask].int(),
                        new_cache_start_pos[non_contextual_mask],
                    )
                    eval_module(logits, batch_0.labels)

                batch_1 = dataset.get_input_batch(
                    uids[contextual_mask],
                    dates[contextual_mask],
                    seq_endptrs[contextual_mask],
                    seq_startptrs[contextual_mask],
                    with_contextual_features=True,
                    with_ranking_labels=True,
                )
                if batch_1 is not None:
                    logits = model.forward(
                        batch_1,
                        uids[contextual_mask].int(),
                        new_cache_start_pos[contextual_mask],
                    )
                    eval_module(logits, batch_1.labels)

                num_batches_ctr += 1
            except StopIteration:
                break
        end_time = time.time()
        logger.info("Total #batch: %s", num_batches_ctr)
        logger.info("Total time(s): %s", end_time - start_time)


def run_ranking_gr_evaluate(
    disable_contextual_features: bool = False,
    enable_profiler: bool = False,
    embed_tp_size: int = 1,
):
    rank, world_size, local_rank = ensure_dist_and_device()
    device = torch.device(f"npu:{torch.npu.current_device()}")
    hccl_comm_dict = create_tp_process_group(rank, world_size, embed_tp_size)

    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 1
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    logger.info("total_max_seqlen %s", total_max_seqlen)

    def strip_candidate_action_tokens(batch, action_feature_name):
        kjt_dict = batch.features.to_dict()
        action_jagged_tensor = kjt_dict[action_feature_name]
        values = action_jagged_tensor.values()
        lengths = action_jagged_tensor.lengths()
        num_candidates = batch.num_candidates
        split_lengths = torch.stack(
            [lengths - num_candidates, num_candidates], dim=1
        ).reshape((-1,))
        stripped_value = torch.split(values, split_lengths.tolist())[::2]
        kjt_dict[action_feature_name] = JaggedTensor.from_dense(stripped_value)
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        return batch

    def strip_padding_batch(batch, unpadded_batch_size):
        batch.batch_size = unpadded_batch_size
        kjt_dict = batch.features.to_dict()
        for k in kjt_dict:
            kjt_dict[k] = JaggedTensor.from_dense_lengths(
                kjt_dict[k].to_padded_dense()[: batch.batch_size],
                kjt_dict[k].lengths()[: batch.batch_size].long(),
            )
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        batch.num_candidates = batch.num_candidates[: batch.batch_size]
        return batch
    
    def prepare_one_batch(dataloader_iter):
        batch = next(dataloader_iter)
        if model._task_config.num_tasks > 0:
            batch = strip_candidate_action_tokens(batch, dataproc._action_feature_name)

        device = torch.device(f"npu:{torch.npu.current_device()}")
        batch = batch.to(device)
        d = batch.features.to_dict()
        user_ids = d["user_id"].values().cpu().int()
        seq_startpos = torch.zeros_like(user_ids)

        if user_ids.shape[0] != batch.batch_size:
            batch = strip_padding_batch(batch, user_ids.shape[0])
        
        return batch, user_ids, seq_startpos

    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            hccl_comm_dict=hccl_comm_dict,
            embed_tp_size=embed_tp_size
        )

        eval_module = get_multi_event_metric_module(
            num_classes=model._task_config.prediction_head_arch[-1],
            num_tasks=model._task_config.num_tasks,
            metric_types=model._task_config.eval_metrics,
        )

        _, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_sequence_length=dataset_args.max_sequence_length,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=model._task_config.num_tasks,
            batch_size=max_batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        local_steps = len(dataloader)

        max_steps = local_steps
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            steps_tensor = torch.tensor([local_steps], dtype=torch.long, device=device)
            dist.all_reduce(steps_tensor, op=dist.ReduceOp.MIN)
            max_steps = steps_tensor.item()
        
        dataloader_iter = iter(dataloader)
        local_step_count = 0

        if not enable_profiler:
            for _ in range(max_steps):
                batch, user_ids, seq_startpos = prepare_one_batch(dataloader_iter)
                local_step_count += 1
                logits = model.forward(batch, user_ids, seq_startpos)
                eval_module(logits, batch.labels)

            logger.info(
                "[Rank %s] Finished inference.\n"
                "Processed %s valid batches out of %s loops",
                rank, local_step_count, max_steps
            )
            eval_metric_dict = eval_module.compute()
            if rank == 0:
                logger.info(
                    "[eval]:\n    %s",
                    stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    "),
                )

        else:
            profile_step = 20
            if local_steps < profile_step:
                raise RuntimeError(f"[Rank {rank}] has fewer steps {local_steps} than profile_step {profile_step}")
            
            prefetched = []
            for _ in range(profile_step):
                prefetched.append(prepare_one_batch(dataloader_iter))
            
            torch_npu.npu.synchronize()

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            )

            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                    ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10, repeat=1, skip_first=10),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_modules=False,
                with_flops=False,
                experimental_config=experimental_config
            ) as prof:
                for batch, user_ids, seq_startpos in prefetched:
                    logits = model.forward(batch, user_ids, seq_startpos)
                    torch_npu.npu.synchronize()
                    prof.step()
            
            remaining_steps = max_steps - profile_step
            local_step_count = profile_step
            for _ in range(remaining_steps):
                batch, user_ids, seq_startpos = prepare_one_batch(dataloader_iter)
                local_step_count += 1
                logits = model.forward(batch, user_ids, seq_startpos)
                eval_module(logits, batch.labels)

            logger.info(
                "[Rank %s] Finished inference.\n"
                "Processed %s valid batches out of %s loops",
                rank, local_step_count, max_steps
            )
            eval_metric_dict = eval_module.compute()
            if rank == 0:
                logger.info(
                    "[eval]:\n    %s",
                    stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    "),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument(
        "--mode", type=RunningMode, choices=list(RunningMode), required=True
    )
    parser.add_argument("--disable_auc", action="store_true")
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--enable_profiler", action="store_true")
    parser.add_argument("--embed_tp_size", type=int, default=1)

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    if args.mode == RunningMode.EVAL:
        if args.disable_auc:
            logger.info("disable_auc is ignored in Eval mode.")
        if args.disable_context:
            logger.info("disable_context is ignored in Eval mode.")
        run_ranking_gr_evaluate(
            enable_profiler=args.enable_profiler,
            embed_tp_size=args.embed_tp_size,
        )
    elif args.mode == RunningMode.SIMULATE:
        if args.enable_profiler:
            logger.info("enable_profiler is ignored in Simulate mode.")
        run_ranking_gr_simulate(
            check_auc=not args.disable_auc,
            disable_contextual_features=args.disable_context,
        )
