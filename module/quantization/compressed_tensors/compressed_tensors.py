# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from typing import Any, Dict, List, Optional, cast
from pydantic import BaseModel
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from module.quantization import QuantizationMethods, QuantizationConfig
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import CompressedTensorsMoEGMMMethod
from module.quantization.compressed_tensors.compressed_tensors_a8w8_int8 import CompressedTensorsW8A8Int8LinearMethod
from module.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from module.fuse_moe_gmm import FusedMoEGMM
from .utils import (find_matched_target, is_activation_quantization_format, should_ignore_layer)

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, Optional[dict[str, QuantizationArgs]]]


class CompressedTensorsConfig(QuantizationConfig):

    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        sparsity_scheme_map: Dict,
        sparsity_ignore_list: list[str],
        kv_cache_scheme: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.kv_cache_scheme = kv_cache_scheme
        self.sparsity_scheme_map = sparsity_scheme_map
        self.sparsity_ignore_list = sparsity_ignore_list
        self.config = config

    def get_linear_method(self) -> "CompressedTensorsLinearMethod":
        return CompressedTensorsLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    def get_name(self) -> QuantizationMethods:
        return "compressed-tensors"

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        elif isinstance(layer, FusedMoEGMM):
            moe_method = CompressedTensorsMoEGMMMethod.get_moe_method(self, layer)
            return moe_method
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompressedTensorsConfig":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(
            config=config)

        return cls(target_scheme_map=target_scheme_map,
                ignore=ignore,
                quant_format=quant_format,
                kv_cache_scheme=config.get("kv_cache_scheme"),
                sparsity_scheme_map=None,
                sparsity_ignore_list=None,
                config=config
        )

    @classmethod
    def _quantization_scheme_map_from_config(
            cls, config: dict[str, Any]) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """
        target_scheme_map: dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target][
                    "weights"] = QuantizationArgs.model_validate(
                        quant_config.get("weights"))

                target_scheme_map[target]["input_activations"] = None
                if is_activation_quantization_format(quant_format):
                    input_activations = quant_config.get("input_activations")
                    # The only case where we have activation quant supported
                    # but no input_activations provided in the config
                    # should be w8a16int8 w8a16int8 can also run for cases where
                    # there is an input_quant but it is ignored
                    if not input_activations:
                        assert target_scheme_map[target][
                            "weights"].type == QuantizationType.FLOAT
                    else:
                        target_scheme_map[target][
                            "input_activations"] = QuantizationArgs.model_validate(
                                quant_config.get("input_activations"))
        return target_scheme_map

    def is_dynamic_token_w8a8(self,
                               weight_quant: BaseModel,
                               input_quant: BaseModel,) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_token = (weight_strategy and input_quant.strategy
                    == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic

    def _get_scheme_from_parts(
            self,
            layer_name: str,
            weight_quant: BaseModel,
            input_quant: BaseModel) -> "CompressedTensorsScheme":

        if is_activation_quantization_format(self.quant_format):
            if self.is_dynamic_token_w8a8(weight_quant, input_quant):
                return CompressedTensorsW8A8Int8LinearMethod(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=False,
                    input_symmetric=input_quant.symmetric)

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found.")

    def get_scheme(self,
                   layer: torch.nn.Module,
                   layer_name: Optional[str] = None
                   ) -> Optional["CompressedTensorsScheme"]:
        """
        compressed-tensors supports non uniform in the following way:

        ignore: List of layer_names or nn.Module names to be ignored.
        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        We first check whether a layer is in the ignore group and use
        CompressedTensorsUnquantized (i.e. fp16/bf16) scheme for the layer

        We then detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for infernece.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # so we do not have to re-write these functions
        # need to make accelerate optional in ct to do this
        matched_target = find_matched_target(
            layer_name=layer_name,
            module=layer,
            targets=self.target_scheme_map.keys())

        # Find the quant_scheme
        scheme_dict = self.target_scheme_map[matched_target]

        scheme = self._get_scheme_from_parts(
            layer_name=layer_name,
            weight_quant=scheme_dict["weights"],
            input_quant=scheme_dict["input_activations"])
        return scheme


    def get_cache_scale(self, name: str) -> Optional[str]:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        # If no matches, return None
        return None


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              dynamic_scale: Optional = None,
              out_dtype: Optional = None):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias, dynamic_scale=dynamic_scale, out_dtype=out_dtype)

    def process_weights_after_loading(self, layer: torch.nn.Module,
                                      is_transpose: bool = True,
                                      is_nz: bool = True,
                                      scales_dtype: torch.dtype = None,
                                      ) -> None:
        layer.scheme.process_weights_after_loading(layer,
                                                   is_transpose=is_transpose,
                                                   is_nz=is_nz,
                                                   scales_dtype=scales_dtype
                                                   )
