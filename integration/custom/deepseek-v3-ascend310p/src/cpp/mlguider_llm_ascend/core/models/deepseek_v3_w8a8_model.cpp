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
#include "deepseek_v3_w8a8_model.h"

#include "batch_matmul_op.h"
#include "cast_op.h"
#include "comm_domain_manager.h"
#include "deepseek_v3_mqa_op.h"
#include "deepseek_v3_utils.h"
#include "embedding.h"
#include "expert_token_dispatcher_op.h"
#include "fused_quant_linear_dequant.h"
#include "fused_rmsnorm_op.h"
#include "fused_rotary_op.h"
#include "gated_silu_op.h"
#include "insert_kv_op.h"
#include "linear_op.h"
#include "moe_routing_op.h"
#include "quant_op.h"
#include "sigmoid_group_topk_op.h"
#include "group_linear_dequant_op.h"
#include "scaled_softmax.h"
#include "scaled_op.h"
#include "multinomial.h"
#include "per_token_per_channel_dequant_op.h"
#include "per_token_quant_op.h"
#include "div_op.h"
#include "config.h"
#include "logger.h"
#include "nlohmann/json.hpp"
#include "tensor.h"
#include "file_utils.h"

// Init params
int32_t vocabSize = 129280;
int32_t hiddenStateDim = 7168;
int32_t qLoraRank = 1536;
int32_t headNum = 128;
int32_t qkNopeHeadDim = 128;
int32_t qkRopeHeadDim = 64;
int32_t kvLoraRank = 512;
int32_t maxPosLen = 163840;
int32_t vHeadDim = 128;
int32_t intermediateSize = 18432;
int32_t moeIntermediateSize = 2048;
int32_t nGroup = 8;
int32_t topkGroup = 4;
int32_t numExpertsPerTok = 8;
float routedScalingFactor = 2.5;

int t = 0; // Step counter

#define DYNAMIC_QUANT 1

DeepSeekV3W8A8Model::DeepSeekV3W8A8Model(LLMConfig& config)
{
    // Init param
    InitParams(config.modelConfig.modelParams);

    // load weight
    LoadWeight(config.modelConfig.modelPath, config.parallelConfig.rank);
}

void DeepSeekV3W8A8Model::LoadWeight(const std::string modelPath, const uint32_t rank)
{
    std::string fileName = modelPath + "mi_" + std::to_string(rank) + ".safetensors";
    LOG_INFO("Loading model from %s to ASCEND", fileName.c_str());
    weight.LoadFromFile(fileName);
    LOG_DEBUG("Load weight done");
}

void DeepSeekV3W8A8Model::InitParams(const nlohmann::json& modelParams)
{
    mFirstKDenseReplace = modelParams["first_k_dense_replace"];
    mHiddenSize = modelParams["hidden_size"];
    mIntermediateSize = modelParams["intermediate_size"];
    mKvLoraRank = modelParams["kv_lora_rank"];
    mMaxPositionEmbeddings = modelParams["max_position_embeddings"];
    mMoeIntermediateSize = modelParams["moe_intermediate_size"];
    mNGroup = modelParams["n_group"];
    mNRoutedExperts = modelParams["n_routed_experts"];
    mNSharedExperts = modelParams["n_shared_experts"];
    mNumAttentionHeads = modelParams["num_attention_heads"];
    mNumExpertsPerTok = modelParams["num_experts_per_tok"];
    mNumHiddenLayers = modelParams["num_hidden_layers"];
    mNumKeyValueHeads = modelParams["num_key_value_heads"];
    mNumNextnPredictLayers = modelParams["num_nextn_predict_layers"];
    mQLoraRank = modelParams["q_lora_rank"];
    mQKNopeHeadDim = modelParams["qk_nope_head_dim"];
    mQkRopeHeadDim = modelParams["qk_rope_head_dim"];
    mRmsNormEps = modelParams["rms_norm_eps"];
    mRopeTheta = modelParams["rope_theta"];
    mRoutedScalingFactor = modelParams["routed_scaling_factor"];
    mTopkGroup = modelParams["topk_group"];
    mVHeadDim = modelParams["v_head_dim"];
    mVocabSize = modelParams["vocab_size"];
    mBetaFast = modelParams["beta_fast"];
    mBetaSlow = modelParams["beta_slow"];
    mFactor = modelParams["factor"];
    mScale = modelParams["mscale"];
    mOriginalMaxPositionEmbeddings = modelParams["original_max_position_embeddings"];

    // ===========================================================================
    // cos_cache, sin_cache compute
    // ===========================================================================
    UpdateRotaryPosEmb();

    // softmax_scale
    float mscaleAllDim = modelParams["mscale_all_dim"];
    mSoftmaxScale = std::sqrt(1.0f / (mQkRopeHeadDim + mQKNopeHeadDim));
    if (mscaleAllDim)
    {
        mScale = YarnGetMscale(mFactor, mscaleAllDim);
        mSoftmaxScale = mSoftmaxScale * mScale * mScale;
    }
    LOG_INFO("Init param success");
}

void DeepSeekV3W8A8Model::UpdateRotaryPosEmb()
{
    std::vector<float> inv_freq(mQkRopeHeadDim / 2); // there is no need to compute the last half of the freq
    for (int i = 0; i < mQkRopeHeadDim / 2; ++i) // only compute the first half of the freq
    {
        inv_freq[i] = 1.0f / std::pow(mRopeTheta, static_cast<float>(i * 2) / mQkRopeHeadDim); // freqs
    }
    if (this->mMaxPositionEmbeddings > mOriginalMaxPositionEmbeddings)
    {
        auto [low, high]
            = FindCorrectionRange(mBetaFast, mBetaSlow, mQkRopeHeadDim, mRopeTheta, mOriginalMaxPositionEmbeddings);
        auto smooth = LinearRampFactor(low, high, mQkRopeHeadDim / 2);
        for (int i = 0; i < mQkRopeHeadDim / 2; ++i) // only compute the first half of the freq
        {
            smooth[i] = 1 - smooth[i];
            inv_freq[i] = inv_freq[i] / mFactor * (1 - smooth[i]) + inv_freq[i] * smooth[i];
        }
    }
    std::vector<std::vector<float>> sin(this->mMaxPositionEmbeddings), cos(this->mMaxPositionEmbeddings);
    for (int i = 0; i < this->mMaxPositionEmbeddings; i++)
    {
        sin[i].resize(mQkRopeHeadDim / 2); // there is no need to compute the last half of the freq
        cos[i].resize(mQkRopeHeadDim / 2); // there is no need to compute the last half of the freq
        for (int j = 0; j < inv_freq.size(); ++j)
        {
            sin[i][j] += std::sin(static_cast<float>(i) * inv_freq[j]);
            cos[i][j] += std::cos(static_cast<float>(i) * inv_freq[j]);
        }
    }
    std::vector<float> fsin, fcos;
    for (int i = 0; i < sin.size(); ++i)
    {
        fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
        fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
    }

    WeightData cacheCos;
    WeightData cacheSin;
    aclrtMalloc(&cacheCos.dataPtr, this->mMaxPositionEmbeddings * mQkRopeHeadDim / 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // cache_cos
    aclrtMemcpy(cacheCos.dataPtr, this->mMaxPositionEmbeddings * mQkRopeHeadDim / 2 * sizeof(float), fcos.data(),
        this->mMaxPositionEmbeddings * mQkRopeHeadDim / 2 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE); // cache_cos

    aclrtMalloc(&cacheSin.dataPtr, this->mMaxPositionEmbeddings * mQkRopeHeadDim / 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // cache_sin
    aclrtMemcpy(cacheSin.dataPtr, this->mMaxPositionEmbeddings * mQkRopeHeadDim / 2 * sizeof(float), fsin.data(),
        this->mMaxPositionEmbeddings * mQkRopeHeadDim / 2 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE); // cache_sin
    weight.weight.emplace("cache_cos", cacheCos);
    weight.weight.emplace("cache_sin", cacheSin);
}

std::vector<int32_t> LogitsProcess1(
    aclrtStream& stream, HcclComm& comm,
    void* inputlogitsPtr,
    void* outputlogitsPtr,
    void* outputlogitsTPtr,
    void* softmaxOutputPtr,
    void* outputIdPtr,
    std::vector<int32_t>& cuSeqlen,
    int M, int N, int tpSize,
    double temperature=0.6)
{
    uint32_t tpRankId;
    auto ret = HcclGetRankId(comm, &tpRankId);
    if (ret != HCCL_SUCCESS)
    {
        LOG_DEBUG("HcclGetRankId failed, ret = %d", ret);
    }
    LOG_DEBUG("tpSize: %d, tpRankId: %d", tpSize, tpRankId);

    if (tpSize > 1)
    {
        ret = HcclAllGather(inputlogitsPtr, outputlogitsPtr, M*N, HCCL_DATA_TYPE_FP16, comm, stream);
        if (ret != HCCL_SUCCESS)
        {
            LOG_DEBUG("HcclAllGather failed, ret = %d", ret);
        }
        TransposeFP16(stream, outputlogitsPtr, outputlogitsTPtr, {tpSize, M, N}, {1, 0, 2});
    }

    std::vector<int32_t> outputIdsHostVec(M, 0);
    std::vector<int64_t> outputIdsHostVecI64(M, 0);
    std::vector<int16_t> softmaxOutputVec(M*N*tpSize, 0);

    if(temperature > 0.01 && M < 100){ // when M < 100, enable sampling
        MultinomialSampling(stream, outputlogitsTPtr, softmaxOutputPtr, outputIdPtr, M, N*tpSize, 1.0 / temperature); // take the reciprocal of temperature

        auto retas = aclrtMemcpyAsync(outputIdsHostVecI64.data(),
            M*8,
            outputIdPtr,
            M*8,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream
        );

        aclrtSynchronizeStream(stream);
        for (size_t i = 0; i < M; ++i) {
            outputIdsHostVec[i] = static_cast<int32_t>(outputIdsHostVecI64[i]);
        }

    }
    else
    {
        Argmax(stream, outputlogitsTPtr, outputIdPtr, M, N*tpSize);

        auto retas = aclrtMemcpyAsync(outputIdsHostVec.data(),
            M*4,
            outputIdPtr,
            M*4,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream
        );
        aclrtSynchronizeStream(stream);
    }

    std::vector<int32_t> outputIdsVec;

    for (int i = 1; i < cuSeqlen.size(); ++i)
    {
        int32_t token = outputIdsHostVec[cuSeqlen[i] - 1]; // last token of each sequence
        outputIdsVec.push_back(token);
    }

    return outputIdsVec;
}

bool DeepSeekV3W8A8Model::Inference(
    InputContext& inputContext, OutputContext& outputContext,
    std::vector<void*> &kcache, std::vector<void*> &ccache,
    int maxPageNum, int pageLen,
    int batchSize, int pageIdLen,
    int tokenNum,
    aclrtStream& stream)
{
    int32_t tpSize = 8;  // Tensor parallel
    int32_t epSize = 32; // Expert parallel

    void* embedding_ptr = nullptr;

    void* rmsnormptr1 = nullptr;
    void* rmsnormptr2 = nullptr;
    void* compressedQKVQuantPtr = nullptr;
    void* tokensScalePtr = nullptr;
    void* compressedQKVW8a8Ptr = nullptr;
    void* compressedQKVDequantPtr = nullptr; // dequant ptr
    void* queryNopeQuantPtr = nullptr;
    void* queryNopeW8a8Ptr = nullptr;
    void* queryNopeDequantPtr = nullptr;
    void* compressedQPtr = nullptr;
    void* compressedKNopeVPtr = nullptr;
    void* kPePtr = nullptr;
    void* qPeQuantPtr=nullptr;
    void* qPeW8a8Ptr=nullptr;
    void* qPeDequantPtr=nullptr;
    void* queryNopeI8Ptr = nullptr;
    void* queryNopeTPtr = nullptr;
    void* qNopeInputW8a8Ptr = nullptr;
    void* qNopeInputDequantPtr = nullptr;
    void* qNopeNeoPtr = nullptr;
    void* outAttenPtr = nullptr;
    void* contextResPtr = nullptr;
    void* contextQuantPtr = nullptr;
    void* contextW8a8Ptr = nullptr;
    void* contextDequantPtr = nullptr;
    void* contextTPtr = nullptr;
    void* attnOutQuantPtr = nullptr;
    void* attnOutW8a8Ptr = nullptr;
    void* attnOutDequantPtr = nullptr;

    void* rmsnormptr3 = nullptr;
    void* rmsnormptr4 = nullptr;

    void* ffn1OutputQuantPtr = nullptr;
    void* ffn1Output_w8a8_ptr1 = nullptr;
    void* ffn1Output_dequant_ptr1 = nullptr;
    void* ffn1Output_w8a8_ptr2 = nullptr;
    void* ffn1Output_dequant_ptr2 = nullptr;
    void* siluPtr1 = nullptr;
    void* ffn2Output_quant_ptr1 = nullptr;
    void* ffn2OutputW8a8Ptr = nullptr;
    void* ffn2OutputDequantPtr = nullptr;
    void* siluPtr2 = nullptr;
    void* ffn2Output_quant_ptr2 = nullptr;
    void* castFP32Ptr = nullptr;
    void* outputFP32Ptr = nullptr;
    void* topkExpertIdsPerTokenPtr = nullptr;
    void* topkExpertScoresPerTokenPtr = nullptr;
    void* encodedTokenIdsPtr = nullptr;
    void* gatherTokensPtr = nullptr;
    void* tokensPerExpertPtr = nullptr;
    void* des2srcPtr = nullptr;
    void* expertFfn1OutputQuantPtr=nullptr;
    void* expertFfn1OutputDequantPtr =nullptr;
    void* expertFfn1OutputDequantPtr1 = nullptr;
    void* moeSiluPtr = nullptr;
    void* expertFfn2OutputQuantPtr = nullptr;
    void* expertFfn2OutputDequantPtr = nullptr;
    void* expertFfn2OutputDequantPtr1 = nullptr;
    void* routingPtr = nullptr;

    void* outputFP16Ptr = nullptr;

    void* outputlogitsPtr = nullptr;
    void* outputlogitsTPtr = nullptr;
    void* softmaxOutputPtr = nullptr;
    void* outputIdPtr = nullptr;

    void* normWeightPrt = nullptr;
    void* temperaturePrt = nullptr;
    void* scaledNormWeightPrt = nullptr;

    auto ret = aclrtMalloc(&embedding_ptr, tokenNum * hiddenStateDim* sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&rmsnormptr1, tokenNum*hiddenStateDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&rmsnormptr2, tokenNum*hiddenStateDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&compressedQKVQuantPtr, tokenNum*hiddenStateDim, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&tokensScalePtr, tokenNum*headNum/tpSize*8*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // Align for 64 bytes
    ret = aclrtMalloc(&compressedQKVW8a8Ptr, tokenNum*2112*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // 2112 the sum of QKV dim
    ret = aclrtMalloc(&compressedQKVDequantPtr, tokenNum*2112*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST); // 2112 the sum of QKV dim
    ret = aclrtMalloc(&compressedQPtr, tokenNum*qLoraRank*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&compressedKNopeVPtr, tokenNum*512*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST); // 512 KNopeV dim
    ret = aclrtMalloc(&kPePtr, tokenNum*64*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST); // 64 kPe dim
    ret = aclrtMalloc(&queryNopeQuantPtr, tokenNum*qLoraRank, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&queryNopeW8a8Ptr, tokenNum*headNum*qkNopeHeadDim/tpSize*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&queryNopeDequantPtr, tokenNum*headNum*qkNopeHeadDim/tpSize*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&qPeQuantPtr, tokenNum*headNum*qLoraRank/tpSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&qPeW8a8Ptr, tokenNum*headNum*qkRopeHeadDim/tpSize*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&qPeDequantPtr, tokenNum*headNum*qkRopeHeadDim/tpSize*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&queryNopeI8Ptr, tokenNum*headNum*qkNopeHeadDim/tpSize*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&queryNopeTPtr, tokenNum*headNum/tpSize*qkNopeHeadDim*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&qNopeInputW8a8Ptr, tokenNum*headNum/tpSize*kvLoraRank*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&qNopeInputDequantPtr, tokenNum*headNum/tpSize*kvLoraRank*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&qNopeNeoPtr, tokenNum*headNum/tpSize*kvLoraRank*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&outAttenPtr, tokenNum*headNum/tpSize*kvLoraRank*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&contextResPtr, tokenNum*headNum/tpSize*kvLoraRank*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&contextQuantPtr, tokenNum*headNum/tpSize*kvLoraRank, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&contextW8a8Ptr, tokenNum*headNum/tpSize*vHeadDim*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&contextDequantPtr, tokenNum*headNum/tpSize*vHeadDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&contextTPtr, tokenNum*headNum/tpSize*vHeadDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&attnOutQuantPtr, tokenNum*headNum/tpSize*vHeadDim, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&attnOutW8a8Ptr, tokenNum*hiddenStateDim*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&attnOutDequantPtr, tokenNum*hiddenStateDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclrtMalloc(&rmsnormptr3, tokenNum*hiddenStateDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&rmsnormptr4, tokenNum*hiddenStateDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclrtMalloc(&ffn1OutputQuantPtr, tokenNum*hiddenStateDim, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&ffn1Output_w8a8_ptr1, tokenNum*intermediateSize*2/tpSize*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // 2 for fusing GateSilu
    ret = aclrtMalloc(&ffn1Output_dequant_ptr1, tokenNum*intermediateSize*2/tpSize*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST); // 2 for fusing GateSilu
    ret = aclrtMalloc(&ffn1Output_w8a8_ptr2, tokenNum*moeIntermediateSize*2*sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST); // 2 for fusing GateSilu
    ret = aclrtMalloc(&ffn1Output_dequant_ptr2, tokenNum*moeIntermediateSize*2*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);  // 2 for fusing GateSilu
    ret = aclrtMalloc(&siluPtr1, tokenNum*intermediateSize*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&ffn2Output_quant_ptr1, tokenNum*intermediateSize/tpSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&ffn2OutputW8a8Ptr, tokenNum*hiddenStateDim*sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&ffn2OutputDequantPtr, tokenNum*hiddenStateDim*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&siluPtr2, tokenNum*moeIntermediateSize*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&ffn2Output_quant_ptr2, tokenNum*moeIntermediateSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclrtMalloc(&castFP32Ptr, tokenNum*hiddenStateDim*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&outputFP32Ptr, tokenNum*256*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // 256 for expert number

    ret = aclrtMalloc(&gatherTokensPtr, tokenNum*hiddenStateDim*numExpertsPerTok*sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&tokensPerExpertPtr, 256 / epSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // 256 for expert number
    ret = aclrtMalloc(&des2srcPtr, tokenNum*numExpertsPerTok*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclrtMalloc(&expertFfn1OutputQuantPtr, tokenNum * numExpertsPerTok * hiddenStateDim, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&expertFfn1OutputDequantPtr, tokenNum * numExpertsPerTok * moeIntermediateSize * 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // 2 for fusing GateSilu
    ret = aclrtMalloc(&expertFfn1OutputDequantPtr1, tokenNum * numExpertsPerTok * moeIntermediateSize * 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // 2 for fusing GateSilu
    ret = aclrtMalloc(&moeSiluPtr, tokenNum * numExpertsPerTok * moeIntermediateSize* sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&expertFfn2OutputQuantPtr, tokenNum * numExpertsPerTok * moeIntermediateSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&expertFfn2OutputDequantPtr, tokenNum * numExpertsPerTok * hiddenStateDim* sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&expertFfn2OutputDequantPtr1, tokenNum * numExpertsPerTok * hiddenStateDim* sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&routingPtr, tokenNum * hiddenStateDim* sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclrtMalloc(&outputFP16Ptr, tokenNum * vocabSize / tpSize *sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclrtMalloc(&outputlogitsPtr, tokenNum * vocabSize * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&outputlogitsTPtr, tokenNum * vocabSize * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&softmaxOutputPtr, tokenNum * vocabSize * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&outputIdPtr, tokenNum * 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST); // 8 for tpSize

    ret = aclrtMalloc(&normWeightPrt, hiddenStateDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&temperaturePrt, hiddenStateDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&scaledNormWeightPrt, hiddenStateDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    Embedding(stream, inputContext.inputIds, weight["model.embed_tokens.weight"].data(), embedding_ptr, tokenNum, hiddenStateDim, vocabSize);
    LOG_DEBUG("Embedding operator completed");

    for (size_t layerId = 0; layerId < mNumHiddenLayers; layerId++)
    {
        LOG_DEBUG("============== layer %d ==============", layerId);

        if (layerId == 0)
        {
            RmsNorm(stream, embedding_ptr, weight["model.layers.0.input_layernorm.weight"].data(), rmsnormptr1,
                tokenNum, hiddenStateDim, mRmsNormEps);
            LOG_DEBUG("RmsNorm operator completed for layer %d", layerId);

            aclrtMalloc(&topkExpertIdsPerTokenPtr, tokenNum*mNumExpertsPerTok*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc(&topkExpertScoresPerTokenPtr, tokenNum*mNumExpertsPerTok*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc(&encodedTokenIdsPtr, tokenNum*mNumExpertsPerTok*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        }
        else if (layerId <= mFirstKDenseReplace)
        {
            ResRmsNorm(stream, ffn2OutputDequantPtr, rmsnormptr3,
                weight["model.layers." + std::to_string(layerId) + ".input_layernorm.weight"].data(), rmsnormptr1,
                rmsnormptr2, tokenNum, hiddenStateDim, mRmsNormEps);
            LOG_DEBUG("ResRmsNorm operator completed for layer %d", layerId);
        }
        else
        {

            Res2RmsNorm(stream, rmsnormptr3, ffn2OutputDequantPtr, routingPtr,
                weight["model.layers." + std::to_string(layerId) + ".input_layernorm.weight"].data(), rmsnormptr1,
                rmsnormptr2, tokenNum, hiddenStateDim, mRmsNormEps);
        }

        DeepSeekV3Attention(stream, rmsnormptr1, inputContext, layerId,
            kcache[layerId], ccache[layerId],
            compressedQKVQuantPtr, tokensScalePtr, compressedQKVW8a8Ptr, compressedQKVDequantPtr,
            compressedQPtr, compressedKNopeVPtr, kPePtr,
            queryNopeQuantPtr, queryNopeW8a8Ptr, queryNopeDequantPtr,
            qPeQuantPtr, qPeW8a8Ptr, qPeDequantPtr,
            queryNopeI8Ptr,
            queryNopeTPtr,
            qNopeInputW8a8Ptr, qNopeInputDequantPtr,
            qNopeNeoPtr,
            outAttenPtr,
            contextResPtr,
            contextQuantPtr, contextW8a8Ptr, contextDequantPtr,
            contextTPtr,
            attnOutQuantPtr, attnOutW8a8Ptr, attnOutDequantPtr,
            batchSize, pageIdLen,
            maxPageNum, pageLen,
            tokenNum, tpSize
        );
        LOG_DEBUG("DeepSeekV3Attention operator completed for layer %d", layerId);

        if (layerId == 0)
        {
            ResRmsNorm(stream, attnOutDequantPtr, embedding_ptr,
                weight["model.layers." + std::to_string(layerId) + ".post_attention_layernorm.weight"].data(), rmsnormptr1,
                rmsnormptr3, tokenNum, hiddenStateDim, mRmsNormEps
            );
        }
        else
        {
            ResRmsNorm(stream, attnOutDequantPtr, rmsnormptr2,
                weight["model.layers." + std::to_string(layerId) + ".post_attention_layernorm.weight"].data(), rmsnormptr1,
                rmsnormptr3, tokenNum, hiddenStateDim, mRmsNormEps
            );
        }
        LOG_DEBUG("ResRmsNorm operator completed for layer %d", layerId);
        if (layerId < mFirstKDenseReplace)
        {
            DeepSeekV3MLP(stream, layerId, rmsnormptr1,
                ffn1OutputQuantPtr, tokensScalePtr, ffn1Output_w8a8_ptr1, ffn1Output_dequant_ptr1,
                siluPtr1,
                ffn2Output_quant_ptr1, ffn2OutputW8a8Ptr, ffn2OutputDequantPtr,
                tokenNum, tpSize
            );
            LOG_DEBUG("DeepSeekV3MLP operator completed for layer %d", layerId);
        }
        else
        {
            DeepSeekV3MoE(stream, layerId, rmsnormptr1, inputContext,
                castFP32Ptr,
                outputFP32Ptr,
                topkExpertIdsPerTokenPtr, topkExpertScoresPerTokenPtr, encodedTokenIdsPtr,
                gatherTokensPtr, tokensPerExpertPtr, des2srcPtr,
                expertFfn1OutputQuantPtr, tokensScalePtr, expertFfn1OutputDequantPtr, expertFfn1OutputDequantPtr1,
                moeSiluPtr,
                expertFfn2OutputQuantPtr, expertFfn2OutputDequantPtr, expertFfn2OutputDequantPtr1,
                routingPtr,
                tokenNum
            );
            DeepSeekV3MLP(stream, layerId, rmsnormptr1,
                ffn1OutputQuantPtr, tokensScalePtr, ffn1Output_w8a8_ptr2, ffn1Output_dequant_ptr2,
                siluPtr2,
                ffn2Output_quant_ptr2, ffn2OutputW8a8Ptr, ffn2OutputDequantPtr,
                tokenNum, tpSize
            );
            LOG_DEBUG("DeepSeekV3MoE operator completed for layer %d", layerId);
        }
    }

    t++;
    if(inputContext.isPrefill)
    {
        Res2RmsNorm(stream, rmsnormptr3, ffn2OutputDequantPtr, routingPtr,
            weight["model.norm.weight"].data(), rmsnormptr1, rmsnormptr2, tokenNum,
            hiddenStateDim, mRmsNormEps);

        AclnnCastFP16toFP32(stream, weight["model.norm.weight"].data(), normWeightPrt, 1, hiddenStateDim);
        std::vector<float> temperatureVec;
        for(int i = 0; i < hiddenStateDim; i ++)
        {
            temperatureVec.push_back(0.3f); // 0.3f for recomanded value
        }
        aclrtMemcpyAsync(temperaturePrt,
            hiddenStateDim*sizeof(float),
            temperatureVec.data(),
            hiddenStateDim*sizeof(float),
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream
        );
        AclnnDiv1(stream, normWeightPrt, temperaturePrt, scaledNormWeightPrt, 1, hiddenStateDim);

        AclnnCastFP32toFP16(stream, scaledNormWeightPrt, weight["model.norm.weight"].data(), 1, hiddenStateDim);
        aclrtSynchronizeStream(stream);
    }
    else
    {
        Res2RmsNorm(stream, rmsnormptr3, ffn2OutputDequantPtr, routingPtr,
            weight["model.norm.weight"].data(), rmsnormptr1, rmsnormptr2, tokenNum,
            hiddenStateDim, mRmsNormEps);
    }
    LOG_DEBUG("Res2RmsNorm operator completed for last norm");

    LinearFP16(stream, rmsnormptr1, weight["lm_head.weight"].data(), outputFP16Ptr, tokenNum,
        vocabSize / tpSize, hiddenStateDim);
    LOG_DEBUG("Linear operator completed for logits");

    auto& tpComm = CommDomainManager::GetHcclComm(CommDomainManager::CommType::TP);
    std::vector<int32_t> outputIds = LogitsProcess1(stream, tpComm,
        outputFP16Ptr,
        outputlogitsPtr,
        outputlogitsTPtr,
        softmaxOutputPtr,
        outputIdPtr,
        inputContext.cuSeqlensQueryHost,
        tokenNum, vocabSize / tpSize, tpSize,
        0.6 // recommand value of temperature
    );

    LOG_DEBUG("outputIds");
    for (int i = 0; i < outputIds.size(); i++)
    {
        LOG_DEBUG("%d", outputIds[i]);
    }

    outputContext.outputIds = outputIds;

    aclrtFree(embedding_ptr);
    aclrtFree(rmsnormptr1);
    aclrtFree(rmsnormptr2);
    aclrtFree(compressedQKVQuantPtr);
    aclrtFree(tokensScalePtr);
    aclrtFree(compressedQKVW8a8Ptr);
    aclrtFree(compressedQKVDequantPtr); // dequant ptr
    aclrtFree(queryNopeQuantPtr);
    aclrtFree(queryNopeW8a8Ptr);
    aclrtFree(queryNopeDequantPtr);
    aclrtFree(compressedQPtr);
    aclrtFree(compressedKNopeVPtr);
    aclrtFree(kPePtr);
    aclrtFree(qPeQuantPtr);
    aclrtFree(qPeW8a8Ptr);
    aclrtFree(qPeDequantPtr);
    aclrtFree(queryNopeI8Ptr);
    aclrtFree(queryNopeTPtr);
    aclrtFree(qNopeInputW8a8Ptr);
    aclrtFree(qNopeInputDequantPtr);
    aclrtFree(qNopeNeoPtr);
    aclrtFree(outAttenPtr);
    aclrtFree(contextResPtr);
    aclrtFree(contextTPtr);
    aclrtFree(attnOutQuantPtr);
    aclrtFree(attnOutW8a8Ptr);
    aclrtFree(attnOutDequantPtr);
    aclrtFree(rmsnormptr3);
    aclrtFree(rmsnormptr4);

    aclrtFree(ffn1OutputQuantPtr);
    aclrtFree(ffn1Output_w8a8_ptr1);
    aclrtFree(ffn1Output_dequant_ptr1);
    aclrtFree(ffn1Output_w8a8_ptr2);
    aclrtFree(ffn1Output_dequant_ptr2);
    aclrtFree(siluPtr1);
    aclrtFree(ffn2Output_quant_ptr1);
    aclrtFree(ffn2OutputW8a8Ptr);
    aclrtFree(ffn2OutputDequantPtr);
    aclrtFree(siluPtr2);
    aclrtFree(ffn2Output_quant_ptr2);
    aclrtFree(castFP32Ptr);
    aclrtFree(outputFP32Ptr);
    aclrtFree(topkExpertIdsPerTokenPtr);
    aclrtFree(topkExpertScoresPerTokenPtr);
    aclrtFree(encodedTokenIdsPtr);
    aclrtFree(gatherTokensPtr);
    aclrtFree(tokensPerExpertPtr);
    aclrtFree(des2srcPtr);
    aclrtFree(expertFfn1OutputQuantPtr);
    aclrtFree(expertFfn1OutputDequantPtr);
    aclrtFree(expertFfn1OutputDequantPtr1);
    aclrtFree(moeSiluPtr);
    aclrtFree(expertFfn2OutputQuantPtr);
    aclrtFree(expertFfn2OutputDequantPtr);
    aclrtFree(expertFfn2OutputDequantPtr1);
    aclrtFree(routingPtr);
    aclrtFree(outputFP16Ptr);
    aclrtFree(outputlogitsPtr);
    aclrtFree(outputlogitsTPtr);
    aclrtFree(softmaxOutputPtr);
    aclrtFree(outputIdPtr);
    aclrtFree(normWeightPrt);
    aclrtFree(temperaturePrt);
    aclrtFree(scaledNormWeightPrt);

    return true;
}

void DeepSeekV3W8A8Model::DeepSeekV3Attention(
    aclrtStream& stream, void* inputPtr, InputContext& inputContext, int layerId,
    void* kCache, void* cCache,
    void* compressedQKVQuantPtr, void* tokensScalePtr, void* compressedQKVW8a8Ptr, void* compressedQKVDequantPtr,
    void* compressedQPtr, void* compressedKNopeVPtr, void* kPePtr,
    void* queryNopeQuantPtr, void* queryNopeW8a8Ptr, void* queryNopeDequantPtr,
    void* qPeQuantPtr, void* qPeW8a8Ptr, void* qPeDequantPtr,
    void* queryNopeI8Ptr,
    void* queryNopeTPtr,
    void* qNopeInputW8a8Ptr, void* qNopeInputDequantPtr,
    void* qNopeNeoPtr,
    void* outAttenPtr,
    void* contextResPtr,
    void* contextQuantPtr, void* contextW8a8Ptr, void* contextDequantPtr,
    void* contextTPtr,
    void* attnOutQuantPtr, void* attnOutW8a8Ptr, void* attnOutDequantPtr,
    int batchSize, int pageIdLen,
    int maxPageNum, int pageLen,
    int tokenNum, int tpSize)
{
#ifndef DYNAMIC_QUANT
    FusedQuantLinearDequant(stream, inputPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.qkv_down_proj.weight"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.qkv_down_proj.quant_scale"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.qkv_down_proj.dequant_scale"].data(),
        tokenNum, 2112, hiddenStateDim,
        compressedQKVQuantPtr, compressedQKVW8a8Ptr, compressedQKVDequantPtr
    ); // 2112 for fusing QKV dim
#else
    FusedQuantLinearDequantDynamic(stream, inputPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.qkv_down_proj.weight"].data(),
        tokensScalePtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.qkv_down_proj.dequant_scale"].data(),
        tokenNum, 2112, hiddenStateDim,
        compressedQKVQuantPtr, compressedQKVW8a8Ptr, compressedQKVDequantPtr
    ); // 2112 for fusing QKV dim

#endif
    LOG_DEBUG("Run FusedQuantLinearDequant layer success!");

    DeepSeekV3RmsNormSplit(stream, compressedQKVDequantPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_a_layernorm.weight"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.kv_a_layernorm.weight"].data(), compressedQPtr,
        compressedKNopeVPtr, kPePtr, tokenNum, 2112, qLoraRank, 512, 64, mRmsNormEps); // 512 for fusing QKV dim; and 512 for NOPE and 64
#ifndef DYNAMIC_QUANT
    FusedQuantLinearDequant(stream, compressedQPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_nope_proj.weight"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_nope_proj.quant_scale"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_nope_proj.dequant_scale"].data(),
        tokenNum, headNum*qkNopeHeadDim/tpSize, qLoraRank,
        queryNopeQuantPtr, queryNopeW8a8Ptr, queryNopeDequantPtr
    );
#else
    FusedQuantLinearDequantDynamic(stream, compressedQPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_nope_proj.weight"].data(),
        tokensScalePtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_nope_proj.dequant_scale"].data(),
        tokenNum, headNum*qkNopeHeadDim/tpSize, qLoraRank,
        queryNopeQuantPtr, queryNopeW8a8Ptr, queryNopeDequantPtr
    );
#endif

#ifndef DYNAMIC_QUANT
    FusedQuantLinearDequant(stream, compressedQPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_rope_proj.weight"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_rope_proj.quant_scale"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_rope_proj.dequant_scale"].data(),
        tokenNum, headNum*qkRopeHeadDim/tpSize, qLoraRank,
        qPeQuantPtr, qPeW8a8Ptr, qPeDequantPtr
    );
#else
    FusedQuantLinearDequantDynamic(stream, compressedQPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_rope_proj.weight"].data(),
        tokensScalePtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.q_rope_proj.dequant_scale"].data(),
        tokenNum, headNum*qkRopeHeadDim/tpSize, qLoraRank,
        qPeQuantPtr, qPeW8a8Ptr, qPeDequantPtr
    );

#endif

#ifndef DYNAMIC_QUANT
    QuantizeFP16ToInt8(stream, queryNopeDequantPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.k_b_proj.quant_scale"].data(), queryNopeI8Ptr,
        tokenNum, headNum*qkNopeHeadDim/tpSize);
    TransposeI8(stream, queryNopeI8Ptr, queryNopeTPtr,
        {tokenNum, headNum/tpSize, qkNopeHeadDim}, {1, 0, 2});
    LOG_DEBUG("queryNope transpose success!");
#else
    TransposeFP16(stream, queryNopeDequantPtr, queryNopeTPtr, {tokenNum, headNum/tpSize, qkNopeHeadDim}, {1, 0, 2});
    PerTokenQuant(stream, queryNopeTPtr, queryNopeI8Ptr, tokensScalePtr, tokenNum*headNum/tpSize, qkNopeHeadDim);
#endif

#ifndef DYNAMIC_QUANT
    BatchMatmulInt8(stream, queryNopeTPtr, weight["model.layers." + std::to_string(layerId) + ".self_attn.k_b_proj.weight"].data(),
        qNopeInputW8a8Ptr, tokenNum, kvLoraRank, qkNopeHeadDim, headNum/tpSize);
    BatchDequantizeInt32ToFp16(stream, qNopeInputW8a8Ptr, weight["model.layers." + std::to_string(layerId) + ".self_attn.k_b_proj.dequant_scale"].data(),
        qNopeInputDequantPtr, tokenNum, kvLoraRank, headNum/tpSize);
#else
    BatchMatmulInt8(stream, queryNopeI8Ptr, weight["model.layers." + std::to_string(layerId) + ".self_attn.k_b_proj.weight"].data(),
        qNopeInputW8a8Ptr, tokenNum, kvLoraRank, qkNopeHeadDim, headNum/tpSize);
    PerTokenPerChannelBatchDeQuantI32(stream, qNopeInputW8a8Ptr, weight["model.layers." + std::to_string(layerId) + ".self_attn.k_b_proj.dequant_scale"].data(),
        tokensScalePtr, qNopeInputDequantPtr, tokenNum, kvLoraRank, headNum/tpSize);
#endif
    LOG_DEBUG("Run BatchMatmulDequant layer success!");

    DeepSeekV3ApplyRoPeV2Inplace(stream, qPeDequantPtr, kPePtr, inputContext.positionIds,
        weight["cache_cos"].data(), weight["cache_sin"].data(), tokenNum, qkRopeHeadDim, headNum / tpSize,
        maxPosLen);

    bool isChunkPrefill = inputContext.cuSeqlensQueryHost[1] - inputContext.cuSeqlensQueryHost[0] != 1;

    TransposeFP16(stream, qNopeInputDequantPtr, qNopeNeoPtr,
        {headNum/tpSize, tokenNum, kvLoraRank}, {1, 0, 2});

    DeepSeekV3InsertCacheV3(stream, compressedKNopeVPtr, kPePtr, kCache, cCache,
        inputContext.slotMapping, tokenNum, 512, 64,
        128, 256); // 128 for page num; 256 for page len

    DeepSeekV3MQA(stream, qNopeNeoPtr, qPeDequantPtr, cCache, kCache, inputContext.pageIds,
        inputContext.cuSeqlensQuery, inputContext.seqlensKV, outAttenPtr, tokenNum,
        kvLoraRank, qkRopeHeadDim, headNum/tpSize, 128, 256,
        batchSize, pageIdLen, mSoftmaxScale, isChunkPrefill); //  128 for page num and 256 for page len

    LOG_DEBUG("Run DeepSeekV3MQA layer success!");

    TransposeFP16(stream, outAttenPtr, contextResPtr,
        {tokenNum, headNum / tpSize, kvLoraRank}, {1, 0, 2});

#ifndef DYNAMIC_QUANT
    BatchQuantizeFP16ToInt8(stream, contextResPtr, weight["model.layers." + std::to_string(layerId) + ".self_attn.v_b_proj.quant_scale"].data(),
        contextQuantPtr, tokenNum, kvLoraRank, headNum / tpSize);
    BatchMatmulInt8(stream, contextQuantPtr, weight["model.layers." + std::to_string(layerId) + ".self_attn.v_b_proj.weight"].data(),
        contextW8a8Ptr, tokenNum, vHeadDim, kvLoraRank, headNum / tpSize);
    BatchDequantizeInt32ToFp16(stream, contextW8a8Ptr, weight["model.layers." + std::to_string(layerId) + ".self_attn.v_b_proj.dequant_scale"].data(),
        contextDequantPtr, tokenNum, vHeadDim, headNum / tpSize);
#else
    PerTokenQuant(stream, contextResPtr, contextQuantPtr, tokensScalePtr, tokenNum*headNum/tpSize, kvLoraRank);
    BatchMatmulInt8(stream, contextQuantPtr, weight["model.layers." + std::to_string(layerId) + ".self_attn.v_b_proj.weight"].data(),
        contextW8a8Ptr, tokenNum, vHeadDim, kvLoraRank, headNum / tpSize);
    PerTokenPerChannelBatchDeQuantI32(stream, contextW8a8Ptr, weight["model.layers." + std::to_string(layerId) + ".self_attn.v_b_proj.dequant_scale"].data(),
        tokensScalePtr, contextDequantPtr, tokenNum, vHeadDim, headNum/tpSize);
#endif
    LOG_DEBUG("Run FusedBatchQuantBatchMatmulBatchDequant layer success!");
    TransposeFP16(
        stream, contextDequantPtr, contextTPtr, {headNum / tpSize, tokenNum, vHeadDim}, {1, 0, 2});
    LOG_DEBUG("ContextT transpose success!");

#ifndef DYNAMIC_QUANT
    FusedQuantLinearDequant(stream, contextTPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.o_proj.weight"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.o_proj.quant_scale"].data(),
        weight["model.layers." + std::to_string(layerId) + ".self_attn.o_proj.dequant_scale"].data(),
        tokenNum, hiddenStateDim, headNum / tpSize * vHeadDim,
        attnOutQuantPtr, attnOutW8a8Ptr, attnOutDequantPtr
    );
#else
    FusedQuantLinearDequantDynamic(stream, contextTPtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.o_proj.weight"].data(),
        tokensScalePtr,
        weight["model.layers." + std::to_string(layerId) + ".self_attn.o_proj.dequant_scale"].data(),
        tokenNum, hiddenStateDim, headNum / tpSize * vHeadDim,
        attnOutQuantPtr, attnOutW8a8Ptr, attnOutDequantPtr
    );

#endif
    LOG_DEBUG("Run FusedQuantLinearDequant layer success!");

    auto& tpComm = CommDomainManager::GetHcclComm(CommDomainManager::CommType::TP);
    HcclAllReduce(
        attnOutDequantPtr, attnOutDequantPtr, tokenNum*hiddenStateDim, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, tpComm, stream);
}

void DeepSeekV3W8A8Model::DeepSeekV3MLP(aclrtStream& stream, int layerId, void* inputPtr,
    void* ffn1OutputQuantPtr, void* tokensScalePtr, void* ffn1OutputW8a8Ptr, void* ffn1OutputDequantPtr,
    void* siluPtr,
    void* ffn2OutputQuantPtr, void* ffn2OutputW8a8Ptr, void* ffn2OutputDequantPtr,
    int tokenNum, int tpSize
)
{
    int _intermediateSize = layerId < 3 ? intermediateSize * 2 / tpSize : moeIntermediateSize * 2;
#ifndef DYNAMIC_QUANT
    FusedQuantLinearDequant(stream, inputPtr,
        weight["model.layers." + std::to_string(layerId) + ".ffn1_proj.weight"].data(),
        weight["model.layers." + std::to_string(layerId) + ".ffn1_proj.quant_scale"].data(),
        weight["model.layers." + std::to_string(layerId) + ".ffn1_proj.dequant_scale"].data(),
        tokenNum, _intermediateSize, hiddenStateDim,
        ffn1OutputQuantPtr, ffn1OutputW8a8Ptr, ffn1OutputDequantPtr
    );
#else
    FusedQuantLinearDequantDynamic(stream, inputPtr,
        weight["model.layers." + std::to_string(layerId) + ".ffn1_proj.weight"].data(),
        tokensScalePtr,
        weight["model.layers." + std::to_string(layerId) + ".ffn1_proj.dequant_scale"].data(),
        tokenNum, _intermediateSize, hiddenStateDim,
        ffn1OutputQuantPtr, ffn1OutputW8a8Ptr, ffn1OutputDequantPtr
    );
#endif

    GatedSiLu(stream, ffn1OutputDequantPtr, siluPtr, tokenNum, _intermediateSize);
    LOG_DEBUG("DeepSeekV3MLP GatedSiLu activation completed for layer %d", layerId);

#ifndef DYNAMIC_QUANT
    FusedQuantLinearDequant(stream, siluPtr,
        weight["model.layers." + std::to_string(layerId) + ".ffn2_proj.weight"].data(),
        weight["model.layers." + std::to_string(layerId) + ".ffn2_proj.quant_scale"].data(),
        weight["model.layers." + std::to_string(layerId) + ".ffn2_proj.dequant_scale"].data(),
        tokenNum, hiddenStateDim, _intermediateSize / 2, // divided 2 for gated silu
        ffn2OutputQuantPtr, ffn2OutputW8a8Ptr, ffn2OutputDequantPtr
    );
#else
    FusedQuantLinearDequantDynamic(stream, siluPtr,
        weight["model.layers." + std::to_string(layerId) + ".ffn2_proj.weight"].data(),
        tokensScalePtr,
        weight["model.layers." + std::to_string(layerId) + ".ffn2_proj.dequant_scale"].data(),
        tokenNum, hiddenStateDim, _intermediateSize / 2, // divided 2 for gated silu
        ffn2OutputQuantPtr, ffn2OutputW8a8Ptr, ffn2OutputDequantPtr
    );
#endif

    LOG_DEBUG("DeepSeekV3MLP ffn2_proj completed for layer %d", layerId);

    if (layerId < 3) // layers without MOE
    {
        auto& tpComm = CommDomainManager::GetHcclComm(CommDomainManager::CommType::TP);
        HcclAllReduce(ffn2OutputDequantPtr, ffn2OutputDequantPtr, tokenNum*hiddenStateDim, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM,
            tpComm, stream);

        LOG_DEBUG("DeepSeekV3MLP AllReduce completed for layer %d", layerId);
    }
}

void DeepSeekV3W8A8Model::DeepSeekV3MoEGate(aclrtStream& stream, int layerId,
    void* inputPtr, int64_t curRankBeginExpertId, int64_t curRankEndExpertId,
    void* castFP32Ptr,
    void* outputFP32Ptr,
    void* topkExpertIdsPerTokenPtr, void* topkExpertScoresPerTokenPtr, void* encodedTokenIdsPtr,
    int tokenNum
)
{
    AclnnCastFP16toFP32(stream, inputPtr, castFP32Ptr, tokenNum, hiddenStateDim);
    LOG_DEBUG("DeepSeekV3MoEGate cast to float32 completed for layer %d", layerId);

    LinearFP32(stream, castFP32Ptr, weight["model.layers." + std::to_string(layerId) + ".mlp.gate.weight"].data(),
        outputFP32Ptr, tokenNum, 256, hiddenStateDim); // 256 for expert number;

    LOG_DEBUG("DeepSeekV3MoEGate gate linear completed for layer %d", layerId);

    DeepSeekV3SigmoidGroupTopkV2(stream, outputFP32Ptr,
        weight["model.layers." + std::to_string(layerId) + ".mlp.gate.bias"].data(), topkExpertIdsPerTokenPtr,
        topkExpertScoresPerTokenPtr, encodedTokenIdsPtr, tokenNum, 256, nGroup, topkGroup,
        numExpertsPerTok, curRankBeginExpertId, curRankEndExpertId, routedScalingFactor); // 256 for expert number;

    LOG_DEBUG("DeepSeekV3MoEGate TopK expert selection completed for layer %d", layerId);
}

void DeepSeekV3W8A8Model::DeepSeekV3MoEInfer(aclrtStream& stream, int layerId, void* inputPtr,
    void* topkExpertIdsPerTokenPtr, void* topkExpertScoresPerTokenPtr, void* encodedTokenIdsPtr,
    const int64_t curRankBeginExpertId, const int64_t curRankEndExpertId,
    void* gatherTokensPtr, void* tokensPerExpertPtr, void* des2srcPtr,
    void* expertFfn1OutputQuantPtr, void* tokensScalePtr, void* expertFfn1OutputDequantPtr, void* expertFfn1OutputDequantPtr1,
    void* moeSiluPtr,
    void* expertFfn2OutputQuantPtr, void* expertFfn2OutputDequantPtr, void* expertFfn2OutputDequantPtr1,
    void* routingPtr,
    int tokenNum, int epSize
)
{
    ExpertTokenDispatcher(stream, inputPtr, topkExpertIdsPerTokenPtr, encodedTokenIdsPtr,
        gatherTokensPtr, tokensPerExpertPtr, des2srcPtr, tokenNum, hiddenStateDim,
        numExpertsPerTok, curRankEndExpertId - curRankBeginExpertId + 1);
#ifndef DYNAMIC_QUANT
    QuantizeFP16ToInt8(stream, gatherTokensPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn1_proj.quant_scale"].data(),
        expertFfn1OutputQuantPtr, tokenNum * numExpertsPerTok, hiddenStateDim);
    GroupLinearDequant(stream, expertFfn1OutputQuantPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn1_proj.weight"].data(),
        tokensPerExpertPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn1_proj.dequant_scale"].data(),
        expertFfn1OutputDequantPtr,
        tokenNum*numExpertsPerTok, moeIntermediateSize*2, hiddenStateDim, 256 / epSize); // 256 for expert number;
#else
    PerTokenQuant(stream, gatherTokensPtr, expertFfn1OutputQuantPtr, tokensScalePtr, tokenNum * numExpertsPerTok, hiddenStateDim);
    GroupLinearDequant(stream, expertFfn1OutputQuantPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn1_proj.weight"].data(),
        tokensPerExpertPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn1_proj.dequant_scale"].data(),
        expertFfn1OutputDequantPtr,
        tokenNum*numExpertsPerTok, moeIntermediateSize*2, hiddenStateDim, 256 / epSize); // 256 for expert number;

    AclnnCastFP16toFP32(stream, expertFfn1OutputDequantPtr, expertFfn1OutputDequantPtr1, tokenNum*numExpertsPerTok, moeIntermediateSize*2);
    AclnnDiv2(stream, expertFfn1OutputDequantPtr1, tokensScalePtr, expertFfn1OutputDequantPtr, tokenNum*numExpertsPerTok, moeIntermediateSize*2);
    AclnnCastFP32toFP16(stream, expertFfn1OutputDequantPtr, expertFfn1OutputDequantPtr1, tokenNum*numExpertsPerTok, moeIntermediateSize*2);
#endif
    LOG_DEBUG("DeepSeekV3MoEInfer expert ffn1_proj completed for layer %d", layerId);
    GatedSiLu(stream, expertFfn1OutputDequantPtr1, moeSiluPtr, tokenNum*numExpertsPerTok, moeIntermediateSize*2);

    LOG_DEBUG("DeepSeekV3MoEInfer GatedSiLu activation completed for layer %d", layerId);

#ifndef DYNAMIC_QUANT
    GroupQuantizeFP16ToInt8(stream, moeSiluPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn2_proj.quant_scale"].data(),
        tokensPerExpertPtr, expertFfn2OutputQuantPtr, tokenNum * numExpertsPerTok, moeIntermediateSize, 256 / epSize); // 256 for expert number;
    GroupLinearDequant(stream, expertFfn2OutputQuantPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn2_proj.weight"].data(),
        tokensPerExpertPtr,
        weight["model.layers." + std::to_string(layerId) + ".experts.ffn2_proj.dequant_scale"].data(),
        expertFfn2OutputDequantPtr, tokenNum * numExpertsPerTok, hiddenStateDim, moeIntermediateSize, 256 / epSize); // 256 for expert number;
#else
    PerTokenQuant(stream, moeSiluPtr, expertFfn2OutputQuantPtr, tokensScalePtr, tokenNum * numExpertsPerTok, moeIntermediateSize);
    GroupLinearDequant(stream, expertFfn2OutputQuantPtr, weight["model.layers." + std::to_string(layerId) + ".experts.ffn2_proj.weight"].data(),
        tokensPerExpertPtr,
        weight["model.layers." + std::to_string(layerId) + ".experts.ffn2_proj.dequant_scale"].data(),
        expertFfn2OutputDequantPtr, tokenNum * numExpertsPerTok, hiddenStateDim, moeIntermediateSize, 256 / epSize); // 256 for expert number;

    AclnnCastFP16toFP32(stream, expertFfn2OutputDequantPtr, expertFfn2OutputDequantPtr1, tokenNum * numExpertsPerTok, hiddenStateDim);
    AclnnDiv2(stream, expertFfn2OutputDequantPtr1, tokensScalePtr, expertFfn2OutputDequantPtr, tokenNum * numExpertsPerTok, hiddenStateDim);
    AclnnCastFP32toFP16(stream, expertFfn2OutputDequantPtr, expertFfn2OutputDequantPtr1, tokenNum * numExpertsPerTok, hiddenStateDim);

#endif
    LOG_DEBUG("DeepSeekV3MoEInfer expert ffn2_proj completed for layer %d", layerId);

    MoeRoutingV1(stream, expertFfn2OutputDequantPtr1, topkExpertScoresPerTokenPtr, des2srcPtr,
        tokensPerExpertPtr, routingPtr, tokenNum * numExpertsPerTok, hiddenStateDim,
        numExpertsPerTok, 256 / epSize);
    LOG_DEBUG("DeepSeekV3MoEInfer expert routing completed for layer %d", layerId);

    auto& epComm = CommDomainManager::GetHcclComm(CommDomainManager::CommType::EP);
    HcclAllReduce(routingPtr, routingPtr, tokenNum * hiddenStateDim, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, epComm, stream);
    LOG_DEBUG("DeepSeekV3MoEInfer AllReduce completed for layer %d", layerId);
}

void DeepSeekV3W8A8Model::DeepSeekV3MoE(
    aclrtStream& stream, int layerId, void* inputPtr, InputContext& inputContext,
    void* castFP32Ptr,
    void* outputFP32Ptr,
    void* topkExpertIdsPerTokenPtr, void* topkExpertScoresPerTokenPtr, void* encodedTokenIdsPtr,
    void* gatherTokensPtr, void* tokensPerExpertPtr, void* des2srcPtr,
    void* expertFfn1OutputQuantPtr, void* tokensScalePtr, void* expertFfn1OutputDequantPtr, void* expertFfn1OutputDequantPtr1,
    void* moeSiluPtr,
    void* expertFfn2OutputQuantPtr, void* expertFfn2OutputDequantPtr, void* expertFfn2OutputDequantPtr1,
    void* routingPtr,
    int tokenNum
)
{
    uint32_t epSize, epRank, dpSize, dpRank;
    auto& epComm = CommDomainManager::GetHcclComm(CommDomainManager::CommType::EP);
    HcclResult ret = HcclGetRankSize(epComm, &epSize);
    if (ret != HCCL_SUCCESS)
    {
        LOG_ERROR("HcclGetRankSize failed, ret = %d", ret);
    }
    ret = HcclGetRankId(epComm, &epRank);
    if (ret != HCCL_SUCCESS)
    {
        LOG_ERROR("HcclGetRankId failed, ret = %d", ret);
    }
    LOG_DEBUG("DeepSeekV3MoE got EP rank %d of size %d for layer %d", epRank, epSize, layerId);

    auto& dpComm = CommDomainManager::GetHcclComm(CommDomainManager::CommType::DP);
    ret = HcclGetRankSize(dpComm, &dpSize);
    if (ret != HCCL_SUCCESS)
    {
        LOG_ERROR("HcclGetRankSize failed, ret = %d", ret);
    }
    ret = HcclGetRankId(dpComm, &dpRank);
    if (ret != HCCL_SUCCESS)
    {
        LOG_ERROR("HcclGetRankId failed, ret = %d", ret);
    }

    int numExpertPerRank = mNRoutedExperts / epSize;
    int64_t curRankBeginExpertId = epRank * numExpertPerRank;
    int64_t curRankEndExpertId = (epRank + 1) * numExpertPerRank - 1;

    DeepSeekV3MoEGate(stream, layerId, inputPtr, curRankBeginExpertId, curRankEndExpertId,
            castFP32Ptr,
            outputFP32Ptr,
            topkExpertIdsPerTokenPtr, topkExpertScoresPerTokenPtr, encodedTokenIdsPtr,
            tokenNum
        );

    DeepSeekV3MoEInfer(stream, layerId, inputPtr, topkExpertIdsPerTokenPtr,
        topkExpertScoresPerTokenPtr, encodedTokenIdsPtr, curRankBeginExpertId, curRankEndExpertId,
        gatherTokensPtr, tokensPerExpertPtr, des2srcPtr,
        expertFfn1OutputQuantPtr, tokensScalePtr, expertFfn1OutputDequantPtr, expertFfn1OutputDequantPtr1,
        moeSiluPtr,
        expertFfn2OutputQuantPtr, expertFfn2OutputDequantPtr, expertFfn2OutputDequantPtr1,
        routingPtr,
        tokenNum, epSize
    );
    LOG_DEBUG("DeepSeekV3MoE expert inference completed for layer %d", layerId);
}
