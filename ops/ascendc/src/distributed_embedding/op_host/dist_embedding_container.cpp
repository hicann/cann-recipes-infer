/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "dist_embedding_container.h"

// The host direct-invocation translation unit is compiled with -xasc. Include
// the kernel definitions here so template instantiation and <<<>>> launch
// resolution happen in the same Ascend C compilation unit.
#include "../op_kernel/distributed_hashtable.cpp"

namespace {
constexpr uint64_t K_LOCAL_MEM_SIZE = 1024UL * 1024UL * 1024UL;
constexpr uint32_t K_SHMEM_TIMEOUT = 300;

uint64_t PointerToAddress(uint32_t* pointer)
{
    static_assert(sizeof(uint64_t) >= sizeof(pointer), "uint64_t cannot hold a pointer value");
    uint64_t address = 0;
    const int ret = memcpy_s(&address, sizeof(address), &pointer, sizeof(pointer));
    if (ret != EOK) {
        throw std::runtime_error("memcpy_s failed while converting pointer address");
    }
    return address;
}
}

template <typename Tkey, typename Tvalue>
DistEmbeddingContainer<Tkey, Tvalue>::DistEmbeddingContainer(
    const DistEmbeddingOptions& options)
    : tableSize_(options.tableSize),
      myPe_(options.myPe),
      nPes_(options.nPes),
      deviceOffset_(options.deviceOffset),
      maxKeysPerPe_(options.maxKeysPerPe),
      ipPort_(options.ipPort),
      tableDevice_(nullptr),
      recvBuffer_(nullptr),
      sendCount_(nullptr),
      defaultFlagUid_{},
      stream_(nullptr),
      realCoreNum_(0)
{
    unusedKey_ = std::numeric_limits<Tkey>::max();
    unusedValue_ = std::numeric_limits<Tvalue>::max();
    const int memsetRet = memset_s(&defaultFlagUid_, sizeof(defaultFlagUid_), 0, sizeof(defaultFlagUid_));
    if (memsetRet != EOK) {
        throw std::runtime_error("memset_s failed in DistEmbeddingContainer");
    }
    defaultFlagUid_.version = ACLSHMEM_UNIQUEID_VERSION;
    defaultFlagUid_.my_pe = options.myPe;
    defaultFlagUid_.n_pes = options.nPes;
}

template <typename Tkey, typename Tvalue>
DistEmbeddingContainer<Tkey, Tvalue>::~DistEmbeddingContainer()
{
    Finalize();
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::Init()
{
    std::cout << "DistEmbeddingContainer::Init: myPe=" << myPe_ << ", nPes=" << nPes_
              << ", deviceOffset=" << deviceOffset_ << '\n';
    aclError ret = aclrtSetDevice(myPe_ + deviceOffset_);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSetDevice failed in Init: " + std::to_string(ret));
    }
    std::cout << "DistEmbeddingContainer::Init: aclrtSetDevice done" << '\n';

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtCreateStream failed in Init: " + std::to_string(ret));
    }
    std::cout << "DistEmbeddingContainer::Init: aclrtCreateStream done" << '\n';

    const auto& platformInfoMgr = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platformInfoMgr != nullptr) {
        realCoreNum_ = platformInfoMgr->GetCoreNumAiv();
        if (realCoreNum_ <= 0) {
            realCoreNum_ = defaultAivNum;
        }
    }
    std::cout << "DistEmbeddingContainer::Init: realCoreNum=" << realCoreNum_ << '\n';

    InitShmem();
    std::cout << "DistEmbeddingContainer::Init: InitShmem done" << '\n';
    AllocSymmetricMemory();
    std::cout << "DistEmbeddingContainer::Init: AllocSymmetricMemory done" << '\n';
    InitHashTable();
    std::cout << "DistEmbeddingContainer::Init: InitHashTable done" << '\n';

    if (nPes_ > 1) {
        InitRecvBuffer();
        InitSendCount();
    }
    std::cout << "DistEmbeddingContainer::Init: InitRecvBuffer done" << '\n';
    aclshmem_barrier_all();
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::InitShmem()
{
    aclshmemx_init_attr_t attributes;
    const int memsetRet = memset_s(&attributes, sizeof(attributes), 0, sizeof(attributes));
    if (memsetRet != EOK) {
        throw std::runtime_error("memset_s failed in InitShmem");
    }
    const size_t ipLen = std::min(strlen(ipPort_), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN - 1));
    const int memcpyRet = memcpy_s(attributes.ip_port, sizeof(attributes.ip_port), ipPort_, ipLen);
    if (memcpyRet != EOK) {
        throw std::runtime_error("memcpy_s failed in InitShmem");
    }
    attributes.ip_port[ipLen] = '\0';
    attributes.my_pe = myPe_;
    attributes.n_pes = nPes_;
    attributes.local_mem_size = K_LOCAL_MEM_SIZE;
    int attrVersion = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes.option_attr = {attrVersion, ACLSHMEM_DATA_OP_MTE, K_SHMEM_TIMEOUT, K_SHMEM_TIMEOUT,
                              K_SHMEM_TIMEOUT};
    attributes.comm_args = reinterpret_cast<void*>(&defaultFlagUid_);

    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (ret != 0) {
        throw std::runtime_error("aclshmemx_init_attr failed: " + std::to_string(ret));
    }
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::AllocSymmetricMemory()
{
    size_t tableBytes = tableSize_ * sizeof(pair_type);
    tableDevice_ = (pair_type*)aclshmemx_malloc(tableBytes, DEVICE_SIDE);
    if (tableDevice_ == nullptr) {
        throw std::runtime_error("aclshmem_malloc failed");
    }
    // 多卡申请 recvBuffer
    if (nPes_ > 1) {
        size_t recvBytes = nPes_ * maxKeysPerPe_ * sizeof(pair_type);
        recvBuffer_ = (pair_type*)aclshmemx_malloc(recvBytes, DEVICE_SIDE);
        if (recvBuffer_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc failed");
        }
    }
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::InitHashTable()
{
    uint32_t blockNum;
    uint32_t threadNum;
    CalBlockDim(tableSize_, &blockNum, &threadNum);
    // static_assert(std::is_same_v<Tkey, uint32_t> && std::is_same_v<Tvalue, uint32_t>,
    //               "distributed embedding host launch currently supports uint32 key/value");
    unordered_hashtable_init_kernel<Tkey, Tvalue><<<blockNum, 0, stream_>>>(
        reinterpret_cast<uint8_t*>(tableDevice_), unusedKey_, unusedValue_,
        static_cast<int64_t>(tableSize_), threadNum);
    auto ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed in InitHashTable: " + std::to_string(ret));
    }
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::InitRecvBuffer()
{
    uint64_t recvBufSize = static_cast<uint64_t>(nPes_) * maxKeysPerPe_;
    uint32_t blockNum;
    uint32_t threadNum;
    CalBlockDim(recvBufSize, &blockNum, &threadNum);
    distributed_hashtable_init_recvbuff_kernel<Tkey, Tvalue><<<blockNum, 0, stream_>>>(
        recvBuffer_, unusedKey_, unusedValue_, recvBufSize, threadNum);
    auto ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed in InitRecvBuffer: " + std::to_string(ret));
    }
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::InitSendCount()
{
    auto ret = aclrtMalloc((void**)&sendCount_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed in InitSendCount: " + std::to_string(ret));
    }
    uint32_t zero = 0;
    ret = aclrtMemcpy(sendCount_, sizeof(uint32_t), &zero, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMemcpy failed in InitSendCount: " + std::to_string(ret));
    }
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::Insert(
    const Tkey* keys, const Tvalue* values, size_t nums)
{
    uint32_t blockNum;
    uint32_t threadNum;
    CalBlockDim(nums, &blockNum, &threadNum);
    if (nPes_ == 1) {
        UnorderdHashTableTilingData tiling;
        tiling.threadNum = threadNum;
        tiling.keyNum = nums;
        tiling.tableSize = tableSize_;
        // static_assert(std::is_same_v<Tkey, uint32_t> && std::is_same_v<Tvalue, uint32_t>,
        //               "distributed embedding host launch currently supports uint32 key/value");
        unordered_hashtable_insert_kernel<Tkey, Tvalue><<<blockNum, 0, stream_>>>(
            tableDevice_, keys, values, tiling, unusedKey_);
    } else {
        DisHashTableTilingData tiling;
        tiling.threadNum = threadNum;
        tiling.keyNum = nums;
        tiling.tableSize = tableSize_;
        tiling.nPes = nPes_;
        tiling.myPe = myPe_;
        tiling.maxKeysPerPe = maxKeysPerPe_;
        tiling.sendCountAddr = PointerToAddress(sendCount_);

        uint32_t zero = 0;
        auto ret = aclrtMemcpy(sendCount_, sizeof(uint32_t), &zero, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy failed in Insert: " + std::to_string(ret));
        }
        distributed_hashtable_dispatch_kernel<Tkey, Tvalue><<<blockNum, 0, stream_>>>(
            tableDevice_, recvBuffer_, keys, values, tiling, unusedKey_, unusedValue_);

        aclshmemx_barrier_all_on_stream(stream_);

        uint64_t recvBufSize = static_cast<uint64_t>(nPes_) * maxKeysPerPe_;
        CalBlockDim(recvBufSize, &blockNum, &threadNum);
        tiling.threadNum = threadNum;
        distributed_hashtable_local_insert_kernel<Tkey, Tvalue><<<blockNum, 0, stream_>>>(
            tableDevice_, recvBuffer_, tiling, unusedKey_, unusedValue_);

        aclshmemx_barrier_all_on_stream(stream_);
    }
    auto ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed in Insert: " + std::to_string(ret));
    }
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::Search(
    const Tkey* keys, Tvalue* values, size_t nums)
{
    uint32_t blockNum;
    uint32_t threadNum;
    CalBlockDim(nums, &blockNum, &threadNum);
    if (nPes_ == 1) {
        UnorderdHashTableTilingData tiling;
        tiling.threadNum = threadNum;
        tiling.keyNum = nums;
        tiling.tableSize = tableSize_;
        // static_assert(std::is_same_v<Tkey, uint32_t> && std::is_same_v<Tvalue, uint32_t>,
        //               "distributed embedding host launch currently supports uint32 key/value");
        unordered_hashtable_search_kernel<Tkey, Tvalue><<<blockNum, 0, stream_>>>(
            tableDevice_, keys, values, tiling, unusedKey_, unusedValue_);
    } else {
        DisHashTableTilingData tiling;
        tiling.threadNum = threadNum;
        tiling.keyNum = nums;
        tiling.tableSize = tableSize_;
        tiling.nPes = nPes_;
        tiling.myPe = myPe_;
        tiling.maxKeysPerPe = maxKeysPerPe_;
        distributed_hashtable_search_kernel<Tkey, Tvalue><<<blockNum, 0, stream_>>>(
            tableDevice_, keys, values, tiling, unusedKey_, unusedValue_);
    }
    auto ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtSynchronizeStream failed in Search: " + std::to_string(ret));
    }
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::Finalize()
{
    if (tableDevice_ != nullptr) {
        aclshmemx_free(tableDevice_, DEVICE_SIDE);
        tableDevice_ = nullptr;
    }
    if (recvBuffer_ != nullptr) {
        aclshmemx_free(recvBuffer_, DEVICE_SIDE);
        recvBuffer_ = nullptr;
    }
    if (sendCount_ != nullptr) {
        aclrtFree(sendCount_);
        sendCount_ = nullptr;
    }
    aclshmem_finalize();
    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    aclrtResetDevice(myPe_ + deviceOffset_);
}

template <typename Tkey, typename Tvalue>
void DistEmbeddingContainer<Tkey, Tvalue>::CalBlockDim(
    size_t nums, uint32_t* blockNum, uint32_t* threadNumPerBlock)
{
    *blockNum = realCoreNum_;
    *threadNumPerBlock = (nums + *blockNum - 1) / *blockNum;
    if (*threadNumPerBlock > maxThreadCount) {
        *threadNumPerBlock = maxThreadCount;
    }
    *blockNum = (nums + *threadNumPerBlock - 1) / *threadNumPerBlock;
    if (*blockNum > maxBlockCount) {
        *blockNum = maxBlockCount;
    }
    if (*blockNum == 0) {
        *blockNum = 1;
    }
    if (*threadNumPerBlock == 0) {
        *threadNumPerBlock = 1;
    }
}

template class DistEmbeddingContainer<unsigned int, size_t>;
template class DistEmbeddingContainer<long long, size_t>;
