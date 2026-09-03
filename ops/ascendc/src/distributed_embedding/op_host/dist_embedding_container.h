#ifndef DIST_EMBEDDING_CONTAINER_H
#define DIST_EMBEDDING_CONTAINER_H

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "shmem.h"
#include "../op_kernel/distributed_hashtable.h"

struct DistEmbeddingOptions {
    uint64_t tableSize;
    int32_t myPe;
    int32_t nPes;
    int32_t deviceOffset;
    uint32_t maxKeysPerPe;
    const char* ipPort;
};

template <typename Tkey, typename Tvalue>
class DistEmbeddingContainer {
public:
    using pair_type = BucketPair<Tkey, Tvalue>;

    explicit DistEmbeddingContainer(const DistEmbeddingOptions& options);

    template <typename... Args>
    explicit DistEmbeddingContainer(Args... args)
        : DistEmbeddingContainer(DistEmbeddingOptions{args...})
    {
    }

    ~DistEmbeddingContainer();

    void Init();
    void Insert(const Tkey* keys, const Tvalue* values, size_t nums);
    void Search(const Tkey* keys, Tvalue* values, size_t nums);
    void Finalize();

    int32_t GetMyPe() const { return myPe_; }
    int32_t GetNPes() const { return nPes_; }
    uint64_t GetTableSize() const { return tableSize_; }

private:
    void InitShmem();
    void AllocSymmetricMemory();
    void InitHashTable();
    void InitRecvBuffer();
    void InitSendCount();
    void CalBlockDim(size_t nums, uint32_t* blockNum, uint32_t* threadNumPerBlock);

    uint64_t tableSize_;
    int32_t myPe_;
    int32_t nPes_;
    int32_t deviceOffset_;
    uint32_t maxKeysPerPe_;
    const char* ipPort_;
    static constexpr float loadFactor = 0.75f;

    pair_type* tableDevice_;
    pair_type* recvBuffer_;
    uint32_t* sendCount_;

    aclshmemx_uniqueid_t defaultFlagUid_;
    aclrtStream stream_;
    uint32_t realCoreNum_;
    Tkey unusedKey_;
    Tvalue unusedValue_;

    static constexpr uint32_t maxThreadCount = 2048;
    static constexpr uint32_t maxBlockCount = 65535;
    static constexpr uint32_t defaultAivNum = 64;
};

#endif  // DIST_EMBEDDING_CONTAINER_H
