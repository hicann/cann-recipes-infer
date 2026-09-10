
#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/device_sync_functions.h"
#include "shmem.h"
#include "distributed_hashtable.h"

using namespace AscendC;

namespace
{
    constexpr uint32_t THREAD_COUNT = 2048;
}

// 类型安全的shmem 单标量远程写入辅助函数
template <typename T>
__simt_callee__ inline void ShmemPutScalar(__gm__ T *dst, T value, int32_t pe)
{
    if constexpr (sizeof(T) == sizeof(uint32_t))
    {
        simt::aclshmem_uint32_p(reinterpret_cast<__gm__ uint32_t *>(dst), value, pe);
    } else if constexpr (sizeof(T) == sizeof(uint64_t))
    {
        simt::aclshmem_uint64_p(reinterpret_cast<__gm__ uint64_t *>(dst), value, pe);
    } else {
        static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t),
                      "ShmemPutScalar only supports uint32_t and uint64_t");
    }
}

// 类型安全的shmem 单标量远程读取辅助函数
template <typename T>
__simt_callee__ inline T ShmemGetScalar(__gm__ T *src, int32_t pe)
{
    if constexpr (sizeof(T) == sizeof(uint32_t))
    {
        return simt::aclshmem_uint32_g(reinterpret_cast<__gm__ uint32_t *>(src), pe);
    } else if constexpr (sizeof(T) == sizeof(uint64_t))
    {
        return simt::aclshmem_uint64_g(reinterpret_cast<__gm__ uint64_t *>(src), pe);
    } else {
        static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t),
                      "ShmemGetScalar only supports uint32_t and uint64_t");
    }
}

// The device atomic CAS overloads use the fixed-width C types. In particular,
// on this platform `long long` is a distinct type from `int64_t` even though
// both are 64 bits wide, so convert through the supported overload while
// preserving the key bit pattern.
template <typename Tkey>
__simt_callee__ inline Tkey AtomicCasKey(__gm__ Tkey *address, Tkey compare, Tkey value)
{
    static_assert(sizeof(Tkey) == sizeof(uint32_t) || sizeof(Tkey) == sizeof(uint64_t),
                  "AtomicCasKey supports only 32-bit and 64-bit keys");
    if constexpr (sizeof(Tkey) == sizeof(uint32_t)) {
        return static_cast<Tkey>(Simt::AtomicCas(
            reinterpret_cast<__gm__ uint32_t *>(address), static_cast<uint32_t>(compare),
            static_cast<uint32_t>(value)));
    } else {
        return static_cast<Tkey>(Simt::AtomicCas(
            reinterpret_cast<__gm__ int64_t *>(address), static_cast<int64_t>(compare),
            static_cast<int64_t>(value)));
    }
}

// MurMerHash3函数
template <typename Tkey>
__simt_callee__ inline uint32_t MurmurHash3(const __gm__ Tkey *key, int len, uint32_t seed)
{
    static_assert(sizeof(Tkey) == sizeof(uint32_t) || sizeof(Tkey) == sizeof(uint64_t),
                  "MurmurHash3 supports only 32-bit and 64-bit keys");

    // Keep the block loads scalar for reliable device-code generation. A 64-bit
    // key is hashed as two little-endian 32-bit MurmurHash3 blocks.
    constexpr uint32_t c1 = 0xcc9e2d51U;
    constexpr uint32_t c2 = 0x1b873593U;
    constexpr uint32_t blockCount = sizeof(Tkey) / sizeof(uint32_t);
    uint64_t keyBits = static_cast<uint64_t>(*key);
    uint32_t hash = seed;

    for (uint32_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
        uint32_t block = static_cast<uint32_t>(keyBits >> (blockIdx * 32U));
        block *= c1;
        block = (block << 15) | (block >> 17);
        block *= c2;
        hash ^= block;
        hash = (hash << 13) | (hash >> 19);
        hash = hash * 5U + 0xe6546b64U;
    }

    hash ^= static_cast<uint32_t>(len);
    hash ^= hash >> 16;
    hash *= 0x85ebca6bU;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35U;
    hash ^= hash >> 16;
    return hash;
}

template <typename VectorType, typename PairType>
__simt_callee__ inline void StorePairVectorizedAs(__gm__ PairType* const ptr, const PairType val)
{
    union PairVector {
        VectorType vecVal;
        PairType pairVal;
    };
    PairVector converter{};
    converter.pairVal = val;
    *reinterpret_cast<__gm__ VectorType*>(ptr) = converter.vecVal;
}

// 向量化写入一个pair到GM
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_callee__ inline void StorePairVectorized(__gm__ pair_type* const ptr, const pair_type val)
{
    if (sizeof(uint4) == sizeof(pair_type)) {
        StorePairVectorizedAs<uint4>(ptr, val);
    } else if (sizeof(uint2) == sizeof(pair_type)) {
        StorePairVectorizedAs<uint2>(ptr, val);
    } else if (sizeof(int) == sizeof(pair_type)) {
        StorePairVectorizedAs<int>(ptr, val);
    } else if (sizeof(short) == sizeof(pair_type)) {
        StorePairVectorizedAs<short>(ptr, val);
    } else {
        ptr->key = val.key;
        ptr->value = val.value;
    }
}

// 向量化读取一个pair从GM
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_callee__ inline pair_type LoadPairVectorized(__gm__ pair_type *const ptr)
{
    if (sizeof(uint4) == sizeof(pair_type)) {
        union pair_type2vec_type
        {
            uint4 vecVal;
            pair_type pairVal;
        };
        pair_type2vec_type converter = {0, 0, 0, 0};
        converter.vecVal = *reinterpret_cast<__gm__ uint4 *>(ptr);
        return converter.pairVal;
    } else if (sizeof(uint2) == sizeof(pair_type)) {
        union pair_type2vec_type
        {
            uint2 vecVal;
            pair_type pairVal;
        };
        pair_type2vec_type converter = {0, 0};
        converter.vecVal = *reinterpret_cast<__gm__ uint2 *>(ptr);
        return converter.pairVal;
    } else if (sizeof(int) == sizeof(pair_type)) {
        union pair_type2vec_type
        {
            int vecVal;
            pair_type pairVal;
        };
        pair_type2vec_type converter = {0};
        converter.vecVal = *reinterpret_cast<__gm__ int *>(ptr);
        return converter.pairVal;
    } else if (sizeof(short) == sizeof(pair_type)) {
        union pair_type2vec_type
        {
            short vecVal;
            pair_type pairVal;
        };
        pair_type2vec_type converter = {0};
        converter.vecVal = *reinterpret_cast<__gm__ short *>(ptr);
        return converter.pairVal;
    } else {
        pair_type pair;
        pair.key = ptr->key;
        pair.value = ptr->value;
        return pair;
    }
}

// 初始化hash表，所有的bucket填充为 {unusedKey, unusedValue}
template <typename Tkey, typename Tvalue>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtInitUnorderdHashTable(__gm__ uint8_t *hashtbl, Tkey key,
                                                                                  Tvalue value, int64_t hashtblSize,
                                                                                  uint32_t blockIdx, uint32_t blockNum)
{
    using pair_type = BucketPair<Tkey, Tvalue>;
    uint32_t threadIdx = static_cast<uint32_t>(Simt::GetThreadIdx());
    uint32_t threadNum = static_cast<uint32_t>(Simt::GetThreadNum());
    for (uint32_t idx = blockIdx * threadNum + threadIdx; idx < hashtblSize; idx += blockNum * threadNum)
    {
        if (idx > hashtblSize)
        {
            return;
        }
        pair_type pair = {key, value};
        StorePairVectorized<Tkey, Tvalue>((__gm__ pair_type *)hashtbl + idx, pair);
    }
}

// kernel入口
template <typename Tkey, typename Tvalue>
__global__ __vector__ void unordered_hashtable_init_kernel(
    __gm__ uint8_t *hashtbl, Tkey key, Tvalue value, int64_t hashtblSize, int64_t threadNum)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<SimtInitUnorderdHashTable<Tkey, Tvalue>>(cce::dim3(threadNum), hashtbl, key, value, hashtblSize,
                                                           blockIdx, blockNum);
}

// 开放寻址法插入 key-value pair 到 hash 表
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void InsertCompute(
    __gm__ pair_type *tableHandle, const __gm__ Tkey *keys, const __gm__ Tvalue *values,
    uint32_t keyNum, uint64_t tableSize, uint32_t blockNum, uint32_t blockIdx, Tkey unusedKey)
{
    uint32_t threadIdx = static_cast<uint32_t>(Simt::GetThreadIdx());
    uint32_t threadNum = static_cast<uint32_t>(Simt::GetThreadNum());
    for (uint32_t idx = blockIdx * threadNum + threadIdx; idx < keyNum; idx += blockNum * threadNum)
    {
        Tkey insertKey = keys[idx];
        Tvalue insertValue = values[idx];
        size_t currIdx = static_cast<size_t>(MurmurHash3(keys + idx, sizeof(Tkey), 0) % tableSize);
        __gm__ pair_type *pCurrBucket = tableHandle + currIdx;
        size_t counts = 0;
        bool insertSuccess = false;
        while (!insertSuccess)
        {
            if (counts++ >= tableSize)
            {
                return;
            }
            __gm__ Tkey &storedKey = pCurrBucket->key;
            __gm__ Tvalue &storedValue = pCurrBucket->value;
            Tkey oldKey = AtomicCasKey(&storedKey, unusedKey, insertKey);
            if (oldKey == unusedKey || oldKey == insertKey)
            {
                storedValue = insertValue;
                insertSuccess = true;
                break;
            }
            currIdx = (currIdx + 1) % tableSize;
            pCurrBucket = tableHandle + currIdx;
        }
        assert(insertSuccess && "insert failed");
    }
}

// kernel入口
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__global__ __vector__ void unordered_hashtable_insert_kernel(
    __gm__ pair_type *tableHandle, const __gm__ Tkey *keys, const __gm__ Tvalue *values,
    UnorderdHashTableTilingData tilingData, Tkey unusedKey)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<InsertCompute<Tkey, Tvalue>>(cce::dim3(tilingData.threadNum), tableHandle, keys, values,
                                               tilingData.keyNum, tilingData.tableSize, blockNum, blockIdx,
                                               unusedKey);
}

// 线性探测搜索 hash 表
template <typename Tkey, typename Tvalue>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtUnorderedHashTableSearch(
    __gm__ BucketPair<Tkey, Tvalue> *hashTable, const __gm__ Tkey *keys, __gm__ Tvalue *values,
    int64_t totalNum, int64_t hashTblSize, uint32_t blockIdx, uint32_t blockNum,
    Tkey unusedkey, Tvalue unusedValue)
{
    for (int64_t i = AscendC::Simt::GetThreadIdx<0>() + blockIdx * AscendC::Simt::GetThreadNum<0>();
         i < totalNum; i += blockNum * AscendC::Simt::GetThreadNum<0>())
    {
        Tkey key = keys[i];
        uint32_t hashIdx = static_cast<size_t>(MurmurHash3(keys + i, sizeof(Tkey), 0) % hashTblSize);
        int64_t startIdx = hashIdx;
        Tvalue outVal = (Tvalue)unusedValue;
        while (true)
        {
            Tkey storedKey = hashTable[hashIdx].key;
            if (storedKey == unusedkey)
            {
                break;
            }
            if (storedKey == key)
            {
                outVal = hashTable[hashIdx].value;
                break;
            }
            hashIdx = (hashIdx + 1) % hashTblSize;
            if (hashIdx == startIdx)
            {
                break;
            }
        }
        values[i] = outVal;
    }
}

// kernel入口
template <typename Tkey, typename Tvalue>
__global__ __vector__ void unordered_hashtable_search_kernel(
    __gm__ BucketPair<Tkey, Tvalue> *hashTable, const __gm__ Tkey *keys, __gm__ Tvalue *values,
    UnorderdHashTableTilingData tilingData, Tkey unusedKey, Tvalue unusedValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<SimtUnorderedHashTableSearch<Tkey, Tvalue>>(
        cce::dim3(tilingData.threadNum), hashTable, keys, values, tilingData.keyNum,
        tilingData.tableSize, blockIdx, blockNum, unusedKey, unusedValue);
}

// 统计 hash 表中已占用的 bucket 数量
template <typename Tkey, typename Tvalue>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtUnorderedHashTableSize(
    __gm__ BucketPair<Tkey, Tvalue> *hashTable, __gm__ uint64_t *containerSize,
    int64_t hashTableSize, uint32_t blockNum, uint32_t blockIdx, Tkey unusedKey)
{
    uint64_t localSize = 0;
    for (int64_t i = AscendC::Simt::GetThreadIdx<0>() + blockIdx * AscendC::Simt::GetThreadNum<0>();
         i < hashTableSize; i += blockNum * AscendC::Simt::GetThreadNum<0>())
    {
        if (hashTable[i].key != unusedKey)
        {
            localSize++;
        }
    }
    if (localSize > 0)
    {
        Simt::AtomicAdd(containerSize, localSize);
    }
}

// kernel入口
template <typename Tkey, typename Tvalue>
__global__ __vector__ void
unordered_hashtable_size_kernel(
    __gm__ BucketPair<Tkey, Tvalue> *hashTable, __gm__ uint64_t *containerSize,
    int64_t hashTableSize, int64_t threadNum, Tkey unusedKey)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<SimtUnorderedHashTableSize<Tkey, Tvalue>>(
        cce::dim3(threadNum), hashTable, containerSize, hashTableSize, blockNum, blockIdx, unusedKey);
}

// 插入 key 并返回对应的唯一递增ID
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void GetInsertCompute(
    __gm__ pair_type *tableHandle, const __gm__ Tkey *keys, __gm__ Tvalue *values,
    uint32_t keyNum, __gm__ Tvalue *dcount, uint64_t tableSize, uint32_t blockNum,
    uint32_t blockIdx, Tkey unusedKey, Tvalue unusedValue)
{
    uint32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    uint32_t threadNum = AscendC::Simt::GetThreadNum<0>();
    for (uint64_t idx = blockIdx * threadNum + threadIdx; idx < keyNum; idx += blockNum * threadNum) {
        Tkey insertKey = keys[idx];
        Tvalue insertVal = values[idx];
        size_t currIdx = static_cast<size_t>(MurmurHash3(keys + idx, sizeof(Tkey), 0) % tableSize);
        __gm__ pair_type* pCurrBucket = tableHandle + currIdx;
        size_t counts = 0;
        bool isSucc = false;
        while (!isSucc) {
            if (counts++ >= tableSize) {
                return;
            }
            __gm__ Tkey& existKey = pCurrBucket->key;
            __gm__ volatile Tvalue& existVal = pCurrBucket->value;
            Tkey oldKey = AtomicCasKey(&existKey, unusedKey, insertKey);
            if (oldKey == unusedKey) {
                existVal = (Tvalue)(asc_atomic_inc(dcount));
                break;
            } else if (insertKey == oldKey) {
                while (existKey == unusedKey) {
                    // continue;
                }
                break;
            }
            currIdx = (currIdx + 1) % tableSize;
            pCurrBucket = tableHandle + currIdx;
        }
        values[idx] = pCurrBucket->value;
    }
}

// kernel入口
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__global__ __vector__ void unordered_hashtable_get_insert_kernel(
    __gm__ pair_type *tableHandle, const __gm__ Tkey *keys, __gm__ Tvalue *values,
    __gm__ Tvalue *dcount, UnorderdHashTableTilingData tilingData,
    Tkey unusedKey, Tvalue unusedValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<GetInsertCompute<Tkey, Tvalue>>(
        cce::dim3(tilingData.threadNum), tableHandle, keys, values, tilingData.keyNum,
        dcount, tilingData.tableSize, blockNum, blockIdx, unusedKey, unusedValue);
}

// 遍历 hash 表，将所有有效 key-value pair 导出到输出数组
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtDumpUnorderedHashTable(
    __gm__ Tkey *d_key, __gm__ Tvalue *d_value, __gm__ pair_type *tableHandle,
    __gm__ uint32_t *dDumpCounter, __ubuf__ Tkey *pUbKey, __ubuf__ Tvalue *pUbValue,
    __ubuf__ uint32_t *blockAcc, __ubuf__ uint32_t *globalAcc, const size_t offset,
    const size_t searchLength, uint32_t blockIdx, Tkey unusedKey)
{
    uint32_t threadIdx = static_cast<uint32_t>(AscendC::Simt::GetThreadIdx<>());
    uint32_t threadNum = static_cast<uint32_t>(AscendC::Simt::GetThreadNum<>());
    uint64_t idx = static_cast<uint64_t>(blockIdx) * threadNum + threadIdx;

    if (threadIdx == 0) { *blockAcc = 0; }
    if (idx >= searchLength) { return; }
    asc_syncthreads();

    pair_type pair = LoadPairVectorized<Tkey, Tvalue>(tableHandle + offset + idx);
    if (pair.key != unusedKey) {
        uint32_t localIndex = Simt::AtomicAdd(blockAcc, static_cast<uint32_t>(1));
        pUbKey[localIndex] = pair.key;
        pUbValue[localIndex] = pair.value;
    }
    asc_syncthreads();

    if (threadIdx == 0) {
        *globalAcc = Simt::AtomicAdd(dDumpCounter, *blockAcc);
    }
    asc_syncthreads();

    if (threadIdx < *blockAcc) {
        d_key[*globalAcc + threadIdx] = pUbKey[threadIdx];
        d_value[*globalAcc + threadIdx] = pUbValue[threadIdx];
    }
}

// kernel入口
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__global__ __vector__ void unordered_hashtable_dump_kernel(
    __gm__ Tkey *d_key, __gm__ Tvalue *d_value, __gm__ pair_type *tableHandle,
    __gm__ uint32_t *dDumpCounter, const size_t offset, const size_t searchLength,
    uint32_t threadNum, Tkey unusedKey)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    // UBuf is an Ascend C compiler-provided symbol for the kernel UB scratch area.
    extern __ubuf__ char UBuf[];  // NOLINT(G.EXP.05)
    __ubuf__ Tkey* pUbKey = (__ubuf__ Tkey*)UBuf;
    __ubuf__ Tvalue* pUbValue = (__ubuf__ Tvalue*)&(pUbKey[threadNum]);
    __ubuf__ uint32_t* blockAcc = (__ubuf__ uint32_t*)&(pUbValue[threadNum]);
    __ubuf__ uint32_t* globalAcc = blockAcc + 1;

    for (uint32_t i = blockIdx; i < (searchLength + blockNum - 1) / blockNum; i += blockNum) {
        Simt::VF_CALL<SimtDumpUnorderedHashTable<Tkey, Tvalue>>(
            cce::dim3(threadNum), d_key, d_value, tableHandle, dDumpCounter, pUbKey,
            pUbValue, blockAcc, globalAcc, offset, searchLength, i, unusedKey);
    }
}

// 以下是 nPes > 1 时的 kernel 入口函数

// 初始化 recvBuff, 所有 slot 填充为 {unusedKey, unusedValue}
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtInitRecvBuffer(
    __gm__ pair_type *recvBuffer, Tkey unusedKey, Tvalue unusedValue, uint64_t bufSize,
    uint32_t blockIdx, uint32_t blockNum)
{
    uint32_t threadIdx = static_cast<uint32_t>(AscendC::Simt::GetThreadIdx<>());
    uint32_t threadNum = static_cast<uint32_t>(AscendC::Simt::GetThreadNum<>());

    for (uint64_t idx = blockIdx * threadNum + threadIdx; idx < bufSize; idx += blockNum * threadNum) {
        pair_type pair = {unusedKey, unusedValue};
        StorePairVectorized<Tkey, Tvalue>(recvBuffer + idx, pair);
    }
}

// kernel入口
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__global__ __vector__ void distributed_hashtable_init_recvbuff_kernel(
    __gm__ pair_type *recvBuffer, Tkey unusedKey, Tvalue unusedValue, uint64_t bufSize,
    int64_t threadNum)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<SimtInitRecvBuffer<Tkey, Tvalue>>(
        cce::dim3(threadNum), recvBuffer, unusedKey, unusedValue, bufSize, blockIdx, blockNum);
}

// 按 hash 值将 key 分发到 PE
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtDispatchKeys(
    __gm__ pair_type *localHashTable, __gm__ pair_type *recvBuffer,
    const __gm__ Tkey *keys, const __gm__ Tvalue *values, uint32_t keyNum,
    uint64_t tableSize, uint32_t nPes, uint32_t myPe, uint32_t maxKeysPerPe,
    __gm__ uint32_t *sendCount, Tkey unusedKey, Tvalue unusedValue,
    uint32_t blockIdx, uint32_t blockNum)
{
    uint32_t threadIdx = static_cast<uint32_t>(AscendC::Simt::GetThreadIdx<>());
    uint32_t threadNum = static_cast<uint32_t>(AscendC::Simt::GetThreadNum<>());
    for (uint64_t idx = blockIdx * threadNum + threadIdx; idx < keyNum; idx += blockNum * threadNum) {
        Tkey insertKey = keys[idx];
        Tvalue insertValue = values[idx];

        size_t hashVal = static_cast<size_t>(MurmurHash3<Tkey>(keys + idx, sizeof(Tkey), 0));
        uint32_t targetPe = static_cast<uint32_t>(hashVal % nPes);

        if (targetPe == myPe) {
            size_t currIdx = hashVal % tableSize;
            __gm__ pair_type *pCurrBucket = localHashTable + currIdx;
            size_t counts = 0;
            bool isSucc = false;
            while (!isSucc) {
                if (counts++ >= tableSize) { return; }
                __gm__ Tkey& existKey = pCurrBucket->key;
                __gm__ Tvalue& existValue = pCurrBucket->value;
                Tkey oldKey = AtomicCasKey(&existKey, unusedKey, insertKey);
                if (oldKey == unusedKey || oldKey == insertKey) {
                    existValue = insertValue;
                    isSucc = true;
                    break;
                }
                currIdx = (currIdx + 1) % tableSize;
                pCurrBucket = localHashTable + currIdx;
            }
        } else {
            uint32_t slotIdx = Simt::AtomicAdd(sendCount, static_cast<uint32_t>(1));
            uint64_t offset = static_cast<uint64_t>(myPe) * maxKeysPerPe + slotIdx;
            ShmemPutScalar<Tkey>(&recvBuffer[offset].key, insertKey, (int32_t)targetPe);
            ShmemPutScalar<Tvalue>(&recvBuffer[offset].value, insertValue, (int32_t)targetPe);
        }
    }
}

// kernel入口
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__global__ __vector__ void distributed_hashtable_dispatch_kernel(
    __gm__ pair_type *localHashTable, __gm__ pair_type *recvBuffer,
    const __gm__ Tkey *keys, const __gm__ Tvalue *values,
    DisHashTableTilingData tilingData, Tkey unusedKey, Tvalue unusedValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<SimtDispatchKeys<Tkey, Tvalue>>(
        cce::dim3(tilingData.threadNum), localHashTable, recvBuffer, keys, values,
        tilingData.keyNum, tilingData.tableSize, tilingData.nPes, tilingData.myPe,
        tilingData.maxKeysPerPe, reinterpret_cast<__gm__ uint32_t *>(tilingData.sendCountAddr),
        unusedKey, unusedValue, blockIdx, blockNum);
}

// 多卡 Insert 第二阶段: 从recvBuffer 中提取属于本卡的 key 并插入本地 hash 表
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtLocalInsertFromRecv(
    __gm__ pair_type *localHashTable, __gm__ pair_type *recvBuffer,
    uint32_t nPes, uint32_t myPe, uint32_t maxKeysPerPe, uint64_t tableSize,
    Tkey unusedKey, Tvalue unusedValue, uint32_t blockIdx, uint32_t blockNum)
{
    uint32_t threadIdx = static_cast<uint32_t>(AscendC::Simt::GetThreadIdx<>());
    uint32_t threadNum = static_cast<uint32_t>(AscendC::Simt::GetThreadNum<>());
    for (uint32_t pe = 0; pe < nPes; pe++) {
        uint64_t base = static_cast<uint64_t>(pe) * maxKeysPerPe;
        for (uint32_t slot = blockIdx * threadNum + threadIdx; slot < maxKeysPerPe; slot += blockNum * threadNum) {
            uint64_t idx = base + slot;
            pair_type pair = LoadPairVectorized<Tkey, Tvalue>(recvBuffer + idx);
            if (pair.key == unusedKey) { continue; }

            uint32_t hashVal = MurmurHash3<Tkey>(&recvBuffer[idx].key, sizeof(Tkey), 0);
            uint32_t targetPe = hashVal % nPes;
            if (targetPe != myPe) { continue; }

            Tkey insertKey = pair.key;
            Tvalue insertValue = pair.value;
            size_t currIdx = hashVal % tableSize;
            __gm__ pair_type *pCurrBucket = localHashTable + currIdx;
            size_t counts = 0;
            bool isSucc = false;
            while (!isSucc) {
                if (counts++ >= tableSize) { return; }
                __gm__ Tkey& existKey = pCurrBucket->key;
                __gm__ Tvalue& existValue = pCurrBucket->value;
                Tkey oldKey = AtomicCasKey(&existKey, unusedKey, insertKey);
                if (oldKey == unusedKey || oldKey == insertKey) {
                    existValue = insertValue;
                    isSucc = true;
                    break;
                }
                currIdx = (currIdx + 1) % tableSize;
                pCurrBucket = localHashTable + currIdx;
            }
        }
    }
}

// kernel入口
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__global__ __vector__ void distributed_hashtable_local_insert_kernel(
    __gm__ pair_type *localHashTable, __gm__ pair_type *recvBuffer,
    DisHashTableTilingData tilingData, Tkey unusedKey, Tvalue unusedValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<SimtLocalInsertFromRecv<Tkey, Tvalue>>(
        cce::dim3(tilingData.threadNum), localHashTable, recvBuffer,
        tilingData.nPes, tilingData.myPe, tilingData.maxKeysPerPe,
        tilingData.tableSize, unusedKey, unusedValue, blockIdx, blockNum);
}

// 根据 key 的 hash 值确定目标 PE, 本地或远程搜索
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void SimtDistHashTableSearch(
    __gm__ pair_type *hashTable, const __gm__ Tkey *keys, __gm__ Tvalue *values,
    int64_t totalNum, int64_t hashTblSize, uint32_t nPes, uint32_t myPe,
    uint32_t blockIdx, uint32_t blockNum, Tkey unusedKey, Tvalue unusedValue)
{
    uint32_t threadNum = static_cast<uint32_t>(AscendC::Simt::GetThreadNum<0>());

    for (int64_t i = AscendC::Simt::GetThreadIdx<0>() + blockIdx * threadNum; i < totalNum; i += blockNum * threadNum) {
        Tkey key = keys[i];
        uint32_t hashVal = MurmurHash3<Tkey>(keys + i, sizeof(Tkey), 0);

        uint32_t targetPe = hashVal % nPes;
        uint32_t hashIdx = hashVal % hashTblSize;
        int64_t startIdx = hashIdx;
        Tvalue outVal = (Tvalue)unusedValue;

        if (targetPe == myPe) {
            while (true) {
                Tkey storedKey = hashTable[hashIdx].key;
                if (storedKey == unusedKey) { break; }
                if (storedKey == key) {
                    outVal = hashTable[hashIdx].value;
                    break;
                }
                hashIdx++;
                if (hashIdx >= hashTblSize) { hashIdx = 0; }
                if (hashIdx == startIdx) { break; }
            }
        } else {
            while (true) {
                Tkey remoteKey = ShmemGetScalar<Tkey>(&hashTable[hashIdx].key, (int32_t)targetPe);
                if (remoteKey == unusedKey) { break; }
                if (remoteKey == key) {
                    outVal = ShmemGetScalar<Tvalue>(&hashTable[hashIdx].value, (int32_t)targetPe);
                    break;
                }
                hashIdx++;
                if (hashIdx >= hashTblSize) { hashIdx = 0; }
                if (hashIdx == startIdx) { break; }
            }
        }
        values[i] = outVal;
    }
}

// kernel入口
template <typename Tkey, typename Tvalue, typename pair_type = BucketPair<Tkey, Tvalue>>
__global__ __vector__ void distributed_hashtable_search_kernel(
    __gm__ pair_type *hashTable, const __gm__ Tkey *keys, __gm__ Tvalue *values,
    DisHashTableTilingData tilingData, Tkey unusedKey, Tvalue unusedValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    Simt::VF_CALL<SimtDistHashTableSearch<Tkey, Tvalue>>(
        cce::dim3(tilingData.threadNum), hashTable, keys, values, tilingData.keyNum,
        tilingData.tableSize, tilingData.nPes, tilingData.myPe, blockIdx, blockNum,
        unusedKey, unusedValue);
}
