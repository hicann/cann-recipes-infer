
#ifndef DISTRIBUTED_HASHTABLE_H
#define DISTRIBUTED_HASHTABLE_H

#include <cstdint>

struct UnorderdHashTableTilingData {
    uint32_t threadNum;
    uint32_t keyNum;
    uint64_t tableSize;
};

struct DisHashTableTilingData {
    uint32_t threadNum;
    uint32_t keyNum;
    uint32_t tableSize;
    uint32_t nPes;
    uint32_t myPe;
    uint32_t maxKeysPerPe;
    uint64_t sendCountAddr;
};

template <typename Tkey, typename Tvalue>
struct BucketPair {
    Tkey key;
    Tvalue value;
};

#endif  // DISTRIBUTED_HASHTABLE_H
