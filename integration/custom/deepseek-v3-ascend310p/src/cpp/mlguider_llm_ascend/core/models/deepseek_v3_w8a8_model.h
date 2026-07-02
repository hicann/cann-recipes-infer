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
#include "base_llm.h"

class DeepSeekV3W8A8Model : public BaseLLM
{
public:
    DeepSeekV3W8A8Model(LLMConfig& config);

    bool Inference(InputContext& inputContext, OutputContext& outputContext, std::vector<void*>& kcache,
        std::vector<void*>& ccache, int maxPageNum, int pageLen, int batchSize, int pageIdLen, int tokenNum,
        aclrtStream& stream);

private:
    void LoadWeight(const std::string modelPath, const uint32_t rank);
    void InitParams(const nlohmann::json& modelParams);
    void UpdateRotaryPosEmb();
    void DeepSeekV3Attention(aclrtStream& stream, void* inputPtr, InputContext& inputContext, int layerId, void* kCache,
        void* cCache, void* compressedQKVQuantPtr, void* tokensScalePtr, void* compressedQKVW8a8Ptr,
        void* compressedQKVDequantPtr, void* compressedQPtr, void* compressedKNopeVPtr, void* kPePtr,
        void* queryNopeQuantPtr, void* queryNopeW8a8Ptr, void* queryNopeDequantPtr, void* qPeQuantPtr, void* qPeW8a8Ptr,
        void* qPeDequantPtr, void* queryNopeI8Ptr, void* queryNopeTPtr, void* qNopeInputW8a8Ptr,
        void* qNopeInputDequantPtr, void* qNopeNeoPtr, void* outAttenPtr, void* contextResPtr, void* contextQuantPtr,
        void* contextW8a8Ptr, void* contextDequantPtr, void* contextTPtr, void* attnOutQuantPtr, void* attnOutW8a8Ptr,
        void* attnOutDequantPtr, int batchSize, int pageIdLen, int maxPageNum, int pageLen, int tokenNum, int tpSize);
    void DeepSeekV3MLP(aclrtStream& stream, int layerId, void* inputPtr, void* ffn1OutputQuantPtr, void* tokensScalePtr,
        void* ffn1OutputW8a8Ptr, void* ffn1OutputDequantPtr, void* siluPtr, void* ffn2OutputQuantPtr,
        void* ffn2OutputW8a8Ptr, void* ffn2OutputDequantPtr, int tokenNum, int tpSize);
    void DeepSeekV3MoE(aclrtStream& stream, int layerId, void* inputPtr, InputContext& inputContext, void* castFP32Ptr,
        void* outputFP32Ptr, void* topkExpertIdsPerTokenPtr, void* topkExpertScoresPerTokenPtr,
        void* encodedTokenIdsPtr, void* gatherTokensPtr, void* tokensPerExpertPtr, void* des2srcPtr,
        void* expertFfn1OutputQuantPtr, void* tokensScalePtr, void* expertFfn1OutputDequantPtr,
        void* expertFfn1OutputDequantPtr1, void* moeSiluPtr, void* expertFfn2OutputQuantPtr,
        void* expertFfn2OutputDequantPtr, void* expertFfn2OutputDequantPtr1, void* routingPtr, int tokenNum);
    void DeepSeekV3MoEGate(aclrtStream& stream, int layerId, void* inputPtr, int64_t curRankBeginExpertId,
        int64_t curRankEndExpertId, void* castFP32Ptr, void* outputFP32Ptr, void* topkExpertIdsPerTokenPtr,
        void* topkExpertScoresPerTokenPtr, void* encodedTokenIdsPtr, int tokenNum);
    void DeepSeekV3MoEInfer(aclrtStream& stream, int layerId, void* inputPtr, void* topkExpertIdsPerTokenPtr,
        void* topkExpertScoresPerTokenPtr, void* encodedTokenIdsPtr, const int64_t curRankBeginExpertId,
        const int64_t curRankEndExpertId, void* gatherTokensPtr, void* tokensPerExpertPtr, void* des2srcPtr,
        void* expertFfn1OutputQuantPtr, void* tokensScalePtr, void* expertFfn1OutputDequantPtr,
        void* expertFfn1OutputDequantPtr1, void* moeSiluPtr, void* expertFfn2OutputQuantPtr,
        void* expertFfn2OutputDequantPtr, void* expertFfn2OutputDequantPtr1, void* routingPtr, int tokenNum,
        int epSize);

private:
    int64_t mFirstKDenseReplace;
    int64_t mHiddenSize;
    int64_t mIntermediateSize;
    int64_t mKvLoraRank;
    int64_t mMaxPositionEmbeddings;
    int64_t mMoeIntermediateSize;
    int64_t mNGroup;
    int64_t mNRoutedExperts;
    int64_t mNSharedExperts;
    int64_t mNumAttentionHeads;
    int64_t mNumExpertsPerTok;
    int64_t mNumHiddenLayers;
    int64_t mNumKeyValueHeads;
    int64_t mNumNextnPredictLayers;
    int64_t mQLoraRank;
    int64_t mQKNopeHeadDim;
    int64_t mQkRopeHeadDim;
    float mRmsNormEps;
    float mRopeTheta;
    float mRoutedScalingFactor;
    int64_t mTopkGroup;
    int64_t mVHeadDim;
    int64_t mVocabSize;
    int64_t mHeadDim;

    int64_t mBetaFast;
    int64_t mBetaSlow;
    int64_t mFactor;
    float mScale;
    int64_t mOriginalMaxPositionEmbeddings;

    float mSoftmaxScale;
};
