# Adapted from 
# https://github.com/ali-vilab/TeaCache,
# https://github.com/chengzeyi/ParaAttention.
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright (C) 2025 ali-vilab.
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
import os
import json
from typing import Dict, Any, Optional, List

from loguru import logger

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_config.json")


def load_cache_config(config_path=DEFAULT_CONFIG_PATH):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.loads(f.read(), parse_float=lambda x: float(x))
    except Exception as e:
        raise ValueError(f"File {config_path} not found!") from e
    _validate_config_keys(config)
    return config


def _validate_config_keys(config: dict):
    required_keys = ["cache_forward", "FBCache", "TeaCache", "NoCache"]
    missed_key = [k for k in required_keys if k not in config]
    if missed_key:
        raise ValueError(f"Missing required key(s): {','.join(missed_key)}")


class CacheManager():
    def __init__(self) -> None:
        self.cache_step = None
        self.config = None

    def from_config(self, config_path):
        self.config = load_cache_config(config_path)

        if self.config["cache_forward"] == "FBCache":
            self.cache_step = FBCache(self.config)
        elif self.config["cache_forward"] == "TeaCache":
            self.cache_step = TeaCache(self.config)
        else:
            self.cache_step = NoCache()
        logger.info(f"Apply dit cache method: {self.cache_step.cache_name}!")


cache_manager = CacheManager()


class StepCache():
    def __init__(self):
        self.num_steps = 0
        self.skip_cnt = 0
        self.previous_residual = None
        self.ori_latent = None
        self.should_skip = False

    def step_counter(self):
        self.num_steps += 1

    def print_statistics(self):
        raise NotImplementedError("need print_statistics")
    
    def reuse_cache(self) -> torch.Tensor:
        if self.ori_latent is None or self.previous_residual is None:
            raise ValueError("reuse_cache fail")
        return self.previous_residual + self.ori_latent
    
    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        raise NotImplementedError("need pre_cache_process")

    def post_cache_update(self, latent: torch.Tensor):
        raise NotImplementedError("need post_cache_update")


class FBCache(StepCache):
    def __init__(self, cache_config):
        super().__init__()
        self.prev_block = None
        self.diff_ratio = 0
        try:
            fb_cache_config = cache_config['FBCache']
            self.rel_l1_thresh_fbcache = fb_cache_config['rel_l1_thresh']
            self.cache_name = fb_cache_config['cache_name']
        except KeyError as e:
            missing_key = str(e).strip("'")
            if missing_key == 'FBCache':
                raise KeyError("The configuration file is missing the required 'FBCache' section.") from e
            else:
                raise KeyError(f"Missing config item in the 'FBCache' section: '{missing_key}'。") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while reading the cache configuration: {e}") from e

    def cache_update(self, current_block: torch.Tensor, current_latent: torch.Tensor):
        self.previous_residual = (current_latent - self.ori_latent).detach()

    def should_cache(self, current_block: torch.Tensor) -> bool:
        self.step_counter()

        if self.prev_block is None:
            self.prev_block = current_block.detach()
            return False
        
        mean_diff = torch.mean(torch.abs(current_block - self.prev_block))
        mean_current = torch.mean(torch.abs(current_block))
        self.diff_ratio = mean_diff / (mean_current + 1e-8)
        can_reuse = self.diff_ratio < self.rel_l1_thresh_fbcache

        if can_reuse:
            self.skip_cnt += 1
            self.should_skip = True
        else:
            self.prev_block = current_block.detach()
            self.should_skip = False
        return can_reuse

    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        latent = args["latent"]
        judge_input = args["judge_input"]
        self.ori_latent = latent.clone()
        can_reuse = self.should_cache(judge_input)
        should_calc = True

        if can_reuse and self.previous_residual is not None:
            should_calc = False
            latent = self.reuse_cache()
        
        return should_calc, latent

    def post_cache_update(self, latent: torch.Tensor):
        self.cache_update(current_block=self.ori_latent, current_latent=latent)        
    
    def print_statistics(self):
        skip_rate = self.skip_cnt / self.num_steps * 100 if self.num_steps > 0 else 0.0
        logger.info(
            f"cache strategy:FB // [total step]: {self.num_steps} // [skip rate]: {skip_rate}"
        )


class TeaCache(StepCache):
    def __init__(self, cache_config):
        super().__init__()
        self.prev_judge_input = None
        self.accumulated_rel_l1 = 0.0
        self.accumulated_rel_l1_distance = 0
        try:
            tea_cache_config = cache_config['TeaCache']
            self.rel_l1_thresh_fbcache = tea_cache_config['rel_l1_thresh']
            self.cache_name = tea_cache_config['cache_name']
            self.coefficients = tea_cache_config['coefficients']
        except KeyError as e:
            missing_key = str(e).strip("'")
            if missing_key == 'TeaCache':
                raise KeyError("The configuration file is missing the required 'TeaCache' section.") from e
            else:
                raise KeyError(f"Missing config item in the 'TeaCache' section: '{missing_key}'。") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while reading the cache configuration: {e}") from e
        self.rescale_func = np.poly1d(self.coefficients)

    def cache_update(self, current_judge_input: torch.Tensor, current_latent: torch.Tensor):
        self.previous_residual = current_latent - self.ori_latent.detach()
        self.prev_judge_input = current_judge_input.detach()

    def should_cache(self, judge_input: torch.Tensor) -> bool:
        self.step_counter()
        if self.prev_judge_input is None:
            self.prev_judge_input = judge_input.detach()
            return False
        
        abs_diff = torch.abs(judge_input - self.prev_judge_input)
        rel_l1 = abs_diff.mean() / (self.prev_judge_input.abs().mean() + 1e-8)
        scaled_rel_l1 = self.rescale_func(rel_l1.cpu().item())

        self.accumulated_rel_l1 += scaled_rel_l1
        can_reuse = self.accumulated_rel_l1 < self.rel_l1_thresh_teacache

        if can_reuse:
            self.skip_cnt += 1
            self.should_skip = True
        else:
            self.accumulated_rel_l1 = 0.0
            self.should_skip = False
        self.prev_judge_input = judge_input.detach()
        return can_reuse
    
    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        latent = args["latent"]
        judge_input_tea = args["judge_input"]

        if judge_input_tea is None:
            raise ValueError("need judge_input")
        self.ori_latent = latent.clone()
        can_reuse = self.should_cache(judge_input_tea)
        should_calc = not can_reuse

        if can_reuse and self.previous_residual is not None:
            latent = self.reuse_cache()
        
        return should_calc, latent

    def post_cache_update(self, latent: torch.Tensor):
        self.cache_update(current_judge_input=self.prev_judge_input, current_latent=latent)        
    
    def print_statistics(self):
        skip_rate = self.skip_cnt / self.num_steps * 100 if self.num_steps > 0 else 0.0
        logger.info(
            f"cache strategy:Tea // [total step]: {self.num_steps} // [skip rate]: {skip_rate}"
        )


class NoCache(StepCache):
    def __init__(self, cache_config=None):
        super().__init__()
        self.cache_name = "NoCache"
    
    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        latent = args["latent"]
        return True, latent

    def post_cache_update(self, latent: torch.Tensor):
        pass
    
    def print_statistics(self):
        logger.info("No Dit cache method applied")
