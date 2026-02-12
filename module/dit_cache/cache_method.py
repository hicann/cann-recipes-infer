# Adapted from 
# https://github.com/ali-vilab/TeaCache,
# https://github.com/chengzeyi/ParaAttention.
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
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
import math
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
    required_keys = ["cache_forward", "enable_separate_cfg", "FBCache", "TeaCache", "NoCache", "TaylorSeer"]
    missed_keys = [k for k in required_keys if k not in config]
    if missed_keys:
        raise ValueError(f"Missing required key(s): {','.join(missed_keys)}")


class CacheManager():
    def __init__(self) -> None:
        self.cache_method = None
        self.config = None
        self.cache_params = {}
        self.enable_separate_cfg = False

    def from_config(self, config_path, cache_params=None):
        self.config = load_cache_config(config_path)
        self.enable_separate_cfg = self.config.get("enable_separate_cfg", False)
        if cache_params is not None:
            self.cache_params.update(cache_params)
        if self.config["cache_forward"] == "FBCache":
            self.cache_method = FBCache(self.config, self.cache_params)
        elif self.config["cache_forward"] == "TeaCache":
            self.cache_method = TeaCache(self.config, self.cache_params)
        elif self.config["cache_forward"] == "TaylorSeer":
            self.cache_method = TaylorSeer(self.config, self.cache_params)
        else:
            self.cache_method = NoCache()
        logger.info(f"Apply dit cache method: {self.cache_method.cache_name}!")
        logger.info(f"Enable separate_cfg: {self.enable_separate_cfg}!")


cache_manager = CacheManager()


class BaseCache():
    def __init__(self):
        self.num_steps = 0
        self.skip_cnt = 0
        self.previous_residual = None
        self.ori_latent = None
        self.should_skip = False
        self.streams_init()
        self.copy_events = []
        self.cal_events = []

    def step_counter(self):
        self.num_steps += 1

    def print_statistics(self):
        raise NotImplementedError("Need print_statistics")
    
    def reuse_cache(self, is_cond: bool = True) -> torch.Tensor:
        idx = 1 if is_cond else 0
        return self.previous_residual[idx] + self.ori_latent

    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        raise NotImplementedError("Need pre_cache_process")

    def post_cache_update(self, latent: torch.Tensor):
        raise NotImplementedError("Need post_cache_update")

    def derivative_approximation(self, cache_dic: Dict, current: Dict, feature: torch.Tensor):
        """
        Compute derivative approximation
        :param cache_dic: Cache dictionary
        :param current: Information of the current step
        """
        if len(current.get('activated_steps', [])) < 2:
            return

        distance = current['activated_steps'][-1] - current['activated_steps'][-2]
        if distance == 0:
            raise ValueError("Invalid step difference")
        updated_taylor_factors = {}
        updated_taylor_factors[0] = feature
        
        if self.offload:
            self._offload_derivative_approximation(cache_dic, current, distance, updated_taylor_factors)

        else:
            updated_cache = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
            for i in range(cache_dic['max_order']):
                if (updated_cache.get(i, None) is not None) and (current['step'] > cache_dic['warmup'] - 2):
                    try:
                        updated_taylor_factors[i + 1] = (
                            updated_taylor_factors[i] - updated_cache[i].to('npu', non_blocking=True)) / distance
                    except KeyError as e:
                        raise KeyError("Missing taylor factor") from e
                else:
                    break
            utf = updated_taylor_factors 
            cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = utf

    def _offload_derivative_approximation(self, cache_dic: Dict, current: Dict, distance, utf):
        copy_stream = self.copy_stream
        updated_cache = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        for i in range(cache_dic['max_order']):
            if (updated_cache.get(i, None) is not None) and (current['step'] > cache_dic['warmup'] - 2):
                try:
                    utf[i + 1] = (utf[i] - updated_cache[i].to('npu', non_blocking=True)) / distance
                except KeyError as e:
                    raise KeyError("Missing taylor factor") from e
            else:
                break

        copy_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(copy_stream):
            if isinstance(utf, dict):
                utf = {k: (v.to('cpu', non_blocking=True) if torch.is_tensor(v) else v) for k, v in utf.items()}
            else: 
                utf = utf.to('cpu', non_blocking=True)

        torch.npu.current_stream().wait_stream(copy_stream)
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = utf
        

    def taylor_formula(self, cache_dic: Dict, current: Dict) -> torch.Tensor: 
        """
        Compute Taylor expansion error
        :param cache_dic: Cache dictionary
        :param current: Information of the current step
        """
        updated_cache = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        x = current['step'] - current['activated_steps'][-1]
        output = 0
        if self.offload:
            output = self._offload_taylor_formula(cache_dic, current, x, output)
        else:
            sorted_de = [updated_cache[key] for key in sorted(updated_cache.keys())]
            for i, factor in enumerate(sorted_de):
                output += (1 / math.factorial(i)) * factor * (x ** i)
        
        return output

    def _offload_taylor_formula(self, cache_dic, current, x, output, device='npu'):
        updated_cache = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        copy_num = len(updated_cache)
        self.events_init(copy_num)
        pre_feature = [None for i in range(copy_num)]
        with torch.npu.stream(self.copy_stream):
            for i in range(copy_num):
                if i > 0:
                    self.copy_stream.wait_event(self.cal_events[i - 1])
                pre_feature[i] = updated_cache[i].to(device, non_blocking=True)
                self.copy_events[i].record()

        with torch.npu.stream(self.cal_stream):
            for i in range(copy_num):
                self.cal_stream.wait_event(self.copy_events[i])
                output += (1 / math.factorial(i)) * pre_feature[i] * (x ** i)
                pre_feature[i] = None
                self.cal_events[i].record()

        torch.npu.current_stream().wait_stream(self.cal_stream)
        return output

    def streams_init(self):
        self.copy_stream = torch.npu.Stream()
        self.cal_stream = torch.npu.Stream()

    def events_init(self, copy_num=0):
        self.copy_events = [torch.npu.Event() for i in range(copy_num)]
        self.cal_events = [torch.npu.Event() for i in range(copy_num)]


class TaylorSeer(BaseCache):
    def __init__(self, cache_config, cache_params=None):
        super().__init__()
        self.cache_params = cache_params
        try:
            taylor_config = cache_config['TaylorSeer']
            self.cache_name = taylor_config['cache_name']
            self.n_derivatives = taylor_config['n_derivatives']
            self.skip_interval_steps = taylor_config['skip_interval_steps']
            self.warmup = taylor_config['warmup']
            self.cutoff_steps = taylor_config['cutoff_steps']
            self.offload = taylor_config['offload']
            self.n_de = taylor_config['cache_name']
            self.double_stream_layers = self.cache_params.get("double_stream_layers")
            self.single_stream_layers = self.cache_params.get("single_stream_layers")
            self.total_steps = self.cache_params.get("num_steps")
        except KeyError as e:
            missing_key = str(e).strip("'")
            if missing_key == 'TaylorSeer':
                raise KeyError("The configuration file is missing the required 'TaylorSeer' section.") from e
            else:
                raise KeyError(f"Missing config item in the 'TaylorSeer' section: '{missing_key}'。") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while reading the cache configuration: {e}") from e
        self.total_layers_per_step = self.double_stream_layers + self.single_stream_layers
        self.layer_counter = 0
        self.activated_steps = []
        self.total_compute_steps = 0
        self.total_predict_steps = 0
        self.should_skip = False
        self._init_taylor_cache()

        self.current = {
            "step": self.num_steps,
            "activated_steps": self.activated_steps,
            "layer": 0,
            "module": "",
            "stream": "",
            "type": self._get_step_type(self.num_steps),
            "warmup": self.warmup,
            "cutoff_steps": self.cutoff_steps,
            "num_steps": self.total_steps
        }

    def _init_taylor_cache(self):
        self.cache_dic = {
            "cache": {-1: {"double_stream": {}, "single_stream": {}}},
            "max_order": self.n_derivatives,
            "warmup": self.warmup,
            "cache_counter": 0
        }
        for layer in range(self.double_stream_layers):
            self.cache_dic['cache'][-1]['double_stream'][layer] = {
                'img_attn': {}, 'txt_attn': {}, 'img_mlp': {}, 'txt_mlp': {}
            }

        for layer in range(self.single_stream_layers):
            self.cache_dic['cache'][-1]['single_stream'][layer] = {
                'total': {}
            }


    def _get_step_type(self, step):
        is_warmup = step < self.warmup
        is_interval_full = (step - self.warmup) % self.skip_interval_steps == 0
        is_cutoff = (self.total_steps - step) <= self.cutoff_steps

        if is_warmup or is_interval_full or is_cutoff:
            if step not in self.activated_steps:
                self.activated_steps.append(step)
                self.total_compute_steps += 1
            logger.info(f"步数{step},FUll计算, 预热：{is_warmup},间隔={is_interval_full}")
            return "full"
        else:
            self.total_predict_steps += 1
            logger.info(f"步数{step},taylor预测") 
            return "taylor_cache"    

    def update_layer_counter(self):
        self.layer_counter += 1
        if self.layer_counter >= self.total_layers_per_step:
            self.num_steps += 1
            self.layer_counter = 0
            self.current["step"] = self.num_steps
            self.current["type"] = self._get_step_type(self.num_steps)
            self.current["activated_steps"] = self.activated_steps

    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        return True, args["latent"]


    def post_cache_update(self, latent: torch.Tensor):
        pass

    def print_statistics(self):
        predict_rate = (self.total_predict_steps / self.total_steps * 100)
        logger.info(
            f"\n cache strategy:TaylorSeer // [total step]: {self.num_steps} \n"
            f"// [predict_steps]: {self.total_predict_steps} //  [predict_rate]: {predict_rate:.2f}%\n"
            f"// [Derivative Order]: {self.n_derivatives} // [skip interval]: {self.skip_interval_steps}\n"
        )


class FBCache(BaseCache):
    def __init__(self, cache_config, cache_params=None):
        super().__init__()
        self.prev_block = [None, None]
        self.diff_ratio = 0
        self.previous_residual = [None, None]
        self.last_is_cond = False
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

    def cache_update(self, current_block: torch.Tensor, current_latent: torch.Tensor, is_cond: bool = True):
        idx = 1 if is_cond else 0
        residual = (current_latent - self.ori_latent).detach()
        self.previous_residual[idx] = residual
        self.prev_block[idx] = current_block.detach()

    def should_cache(self, current_block: torch.Tensor, is_cond: bool = True) -> bool:
        self.step_counter()
        idx = 1 if is_cond else 0
        if self.prev_block[idx] is None:
            self.prev_block[idx] = current_block.detach()
            return False
        prev_block = self.prev_block[idx]
        mean_diff = torch.mean(torch.abs(current_block - prev_block))
        mean_current = torch.mean(torch.abs(current_block))
        diff_ratio = mean_diff / (mean_current + 1e-8)
        can_reuse = diff_ratio < self.rel_l1_thresh_fbcache
        if can_reuse:
            self.skip_cnt += 1
            self.should_skip = True
        else:
            self.prev_block[idx] = current_block.detach()
            self.should_skip = False
        return can_reuse

    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        latent = args["latent"]
        judge_input = args["judge_input"]
        is_cond = args.get("is_cond", True)
        self.ori_latent = latent.clone()
        self.last_is_cond = is_cond
        can_reuse = self.should_cache(judge_input, is_cond)
        should_calc = True

        if can_reuse:
            latent = self.reuse_cache(is_cond)
            should_calc = False
        return should_calc, latent

    def post_cache_update(self, latent: torch.Tensor):
        self.cache_update(
            current_block=self.ori_latent, 
            current_latent=latent,
            is_cond=self.last_is_cond)        
    
    def print_statistics(self):
        skip_rate = self.skip_cnt / self.num_steps * 100 if self.num_steps > 0 else 0.0
        logger.info(
            f"cache strategy:FB // [total step]: {self.num_steps} // [skip rate]: {skip_rate}"
        )


class TeaCache(BaseCache):
    def __init__(self, cache_config, cache_params=None):
        super().__init__()
        self.prev_judge_input = [None, None]
        self.previous_residual = [None, None]
        self.accumulated_rel_l1 = [0.0, 0.0]
        self.accumulated_rel_l1_distance = 0
        try:
            tea_cfg_config = cache_config['TeaCache']
            self.rel_l1_thresh = tea_cfg_config['rel_l1_thresh']
            self.cache_name = tea_cfg_config['cache_name']
            self.coefficients = tea_cfg_config['coefficients']
            self.ret_steps = tea_cfg_config['warmup']
        except KeyError as e:
            missing_key = str(e).strip("'")
            if missing_key == 'TeaCache':
                raise KeyError("The configuration file is missing the required 'TeaCache' section.") from e
            else:
                raise KeyError(f"Missing config item in the 'TeaCache' section: '{missing_key}'。") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while reading the cache configuration: {e}") from e
        self.rescale_func = np.poly1d(self.coefficients)
        self.cache_params = cache_params
        self.total_steps = self.cache_params.get('num_steps')
        self.cutoff_steps = self.total_steps - self.ret_steps 
        self.cnt = 0
        self.last_is_cond = False

    def should_cache(self, judge_input: torch.Tensor, is_cond: bool) -> bool:
        self.step_counter()
        idx = 1 if is_cond else 0
        if self.cnt >= self.total_steps:
            self.cnt = 0
        current_cnt = self.cnt
        cutoff_condition = current_cnt >= self.cutoff_steps
        if current_cnt < self.ret_steps or cutoff_condition:
            logger.info("in warm_up or final_step, TeaCache Force compute")
            self.accumulated_rel_l1[idx] = 0.0
            self.cnt += 1
            return False

        if self.prev_judge_input[idx] is None:
            self.prev_judge_input[idx] = judge_input.detach()
            self.accumulated_rel_l1[idx] = 0.0
            self.cnt += 1
            return False
        
        prev_input = self.prev_judge_input[idx]
        accumulated_rel_l1 = self.accumulated_rel_l1[idx]
        abs_diff = torch.abs(judge_input - prev_input)
        rel_l1 = abs_diff.mean() / (prev_input.abs().mean() + 1e-8)
        scaled_rel_l1 = abs(self.rescale_func(rel_l1.cpu().item()))
        self.accumulated_rel_l1[idx] += scaled_rel_l1
        accumulated_rel_l1 = self.accumulated_rel_l1[idx]
        can_reuse = accumulated_rel_l1 < self.rel_l1_thresh

        if can_reuse:
            self.skip_cnt += 1
            self.should_skip = True 
        else:
            self.accumulated_rel_l1[idx] = 0.0
            self.should_skip = False
        self.prev_judge_input[idx] = judge_input.detach()
        self.cnt += 1
        return can_reuse


    def pre_cache_process(self, args: Dict[str, torch.Tensor]) -> (bool, torch.Tensor):
        latent = args["latent"]
        judge_input = args["judge_input"]
        is_cond = args.get("is_cond", True)
        if judge_input is None:
            raise ValueError("Need judge_input")
        self.ori_latent = latent.clone()
        can_reuse = self.should_cache(judge_input, is_cond)
        should_calc = not can_reuse
        if can_reuse:
            try:
                latent = self.reuse_cache(is_cond)
            except ValueError:
                should_calc = True
        self.last_is_cond = is_cond
        return should_calc, latent


    def post_cache_update(self, latent: torch.Tensor):
        if self.ori_latent is None:
            return
        idx = 1 if self.last_is_cond else 0
        residual = (latent - self.ori_latent).detach()
        self.previous_residual[idx] = residual

    def print_statistics(self):
        skip_rate = self.skip_cnt / self.num_steps * 100 if self.num_steps > 0 else 0.0
        logger.info(
            f"cache strategy:TeaCache // [total step]: {self.num_steps} // [skip rate]: {skip_rate}"
        )

    def cache_update(self, current_judge_input: torch.Tensor, current_latent: torch.Tensor):
        self.previous_residual = current_latent - self.ori_latent.detach()
        self.prev_judge_input = current_judge_input.detach()


class NoCache(BaseCache):
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