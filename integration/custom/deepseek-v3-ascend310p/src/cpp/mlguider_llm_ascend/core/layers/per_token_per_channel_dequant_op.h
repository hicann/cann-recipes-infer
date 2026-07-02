/**
 * Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "tensor.h"

void PerTokenPerChannelDequant(aclrtStream& stream, void* qx, void* wscale, void* xscale, void* outputPtr, int64_t num_tokens, int64_t dimension);

void PerTokenPerChannelBatchDeQuantI32(aclrtStream& stream, void* qx, void* wscale, void* xscale, void* outputPtr, int64_t num_tokens, int64_t dimension, int64_t batchSize);

void PerTokenPerChannelGroupDeQuantI32(aclrtStream& stream, void* qx, void* wscale, void* xscale, void* groupList, void* outputPtr, int64_t num_tokens, int64_t dimension, int64_t cur_rank_experts);
