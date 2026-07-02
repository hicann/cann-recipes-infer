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

/**
 * @brief ScaledMaskedSoftmax
 * based: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/API/aolapi/context/aclnnMultinomial.md
 *
 * @param stream
 * @param input: shape of (M, N)
 * @param numsamples: An integer on the host side specifying how many samples are drawn from each multinomial distribution. It must be non-negative; when replacement is false, numsamples must not exceed C.
 * @param replacement: A boolean on the host side determining whether elements are sampled with replacement.
 * @param seed: Seed for the random-number generator that influences the generated sequence.
 * @param offset: Offset for the random-number generator that shifts the starting position of the generated sequence. After setting the offset, the sequence begins at the specified point.
 */
void Multinomial(aclrtStream& stream, void* input, void* output, int64_t m, int64_t n, int64_t numsamples=1, bool replacement=false, int64_t seed=1, int64_t offset=0);
