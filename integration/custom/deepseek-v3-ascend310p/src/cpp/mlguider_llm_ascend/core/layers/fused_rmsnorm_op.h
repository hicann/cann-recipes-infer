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

void RmsNorm(aclrtStream& stream, void* input, void* weight, void* output, int m, int n, float epsilon);

void ResRmsNorm(aclrtStream& stream, void* input, void* residual, void* weight, void* output, void* outputRes, int m, int n, float epsilon);

void DeepSeekV3RmsNormSplit(aclrtStream& stream, void* input, void* gammaQuery, void* gammaKNopeV, void* query, void* kNopeV, void* kPe, int m, int n, int n1, int n2, int n3, float epsilon);

void Res2RmsNorm(aclrtStream& stream, void* input, void* residual, void* residual2, void* weight, void* output, void* newResidual, int m, int n, float epsilon);
