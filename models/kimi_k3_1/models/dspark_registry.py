# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Resolve the DSpark checkpoint/runtime pair selected by draft_model_type."""

from __future__ import annotations


DRAFT_MODEL_NONE = "none"
DRAFT_MODEL_DSPARK_LEGACY = "dspark"
DRAFT_MODEL_DSPARK_GQA = "dspark_gqa"
DRAFT_MODEL_DSPARK_MLA = "dspark_mla"
DSPARK_DRAFT_MODEL_TYPES = (
    DRAFT_MODEL_DSPARK_GQA,
    DRAFT_MODEL_DSPARK_MLA,
)
SUPPORTED_DRAFT_MODEL_TYPES = (DRAFT_MODEL_NONE, *DSPARK_DRAFT_MODEL_TYPES)


def get_draft_model_type(settings: dict) -> str:
    """Return the normalized draft runtime selected by the YAML file."""
    model_config = settings.get("model_config", {})
    draft_model_type = str(
        model_config.get("draft_model_type", DRAFT_MODEL_NONE)
    ).strip().lower()
    if draft_model_type == DRAFT_MODEL_DSPARK_LEGACY:
        draft_model_type = DRAFT_MODEL_DSPARK_GQA
    if draft_model_type not in SUPPORTED_DRAFT_MODEL_TYPES:
        raise ValueError(
            "draft_model_type must be 'none', 'dspark' (legacy GQA), "
            f"'dspark_gqa', or 'dspark_mla', got {draft_model_type!r}"
        )
    return draft_model_type


def get_dspark_config_class(draft_model_type: str):
    """Import only the configuration class selected by the runtime."""
    if draft_model_type == DRAFT_MODEL_DSPARK_GQA:
        from .configuration_dspark import KimiK3DSparkConfig

        return KimiK3DSparkConfig
    if draft_model_type == DRAFT_MODEL_DSPARK_MLA:
        from .configuration_dspark_mla import KimiK3DSparkConfig

        return KimiK3DSparkConfig
    raise ValueError(f"unsupported draft_model_type={draft_model_type!r}")


def get_dspark_model_class(draft_model_type: str):
    """Import only the model implementation selected by the runtime."""
    if draft_model_type == DRAFT_MODEL_DSPARK_GQA:
        from .modeling_dspark import K3DSparkForCausalLM

        return K3DSparkForCausalLM
    if draft_model_type == DRAFT_MODEL_DSPARK_MLA:
        from .modeling_dspark_mla import K3DSparkForCausalLM

        return K3DSparkForCausalLM
    raise ValueError(f"unsupported draft_model_type={draft_model_type!r}")


__all__ = [
    "DRAFT_MODEL_DSPARK_GQA",
    "DRAFT_MODEL_DSPARK_LEGACY",
    "DRAFT_MODEL_DSPARK_MLA",
    "DRAFT_MODEL_NONE",
    "DSPARK_DRAFT_MODEL_TYPES",
    "SUPPORTED_DRAFT_MODEL_TYPES",
    "get_draft_model_type",
    "get_dspark_config_class",
    "get_dspark_model_class",
]
