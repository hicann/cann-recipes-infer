/*
 * Two-PE functional test for DistEmbeddingContainer.
 *
 * One process owns each PE. The launcher assigns rank 0 to NPU 5 and rank 1
 * to NPU 6. Each rank inserts keys for both partitions, then searches the
 * complete key set so both remote puts and remote gets are exercised.
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"

#include "../op_host/dist_embedding_container.h"
#include "device_buffer.h"

namespace {

using Key = unsigned int;
using Value = size_t;

constexpr int K_WORLD_SIZE = 2;
constexpr uint64_t K_TABLE_SIZE = 1024;
constexpr uint32_t K_MAX_KEYS_PER_PE = 32;
constexpr int K_MAX_DEVICE_ID = 1023;
constexpr int K_NUMBER_BASE = 10;
constexpr uint32_t K_MURMUR_ROTATE_LEFT = 15;
constexpr uint32_t K_MURMUR_ROTATE_RIGHT = 17;
constexpr uint32_t K_MURMUR_MIX_SHIFT = 13;
constexpr uint32_t K_MURMUR_MIX_SHIFT_RIGHT = 19;
constexpr uint32_t K_MURMUR_FINAL_SHIFT = 16;
constexpr uint32_t K_MURMUR_MULTIPLIER = 5;
constexpr int K_ENTRIES_PER_TARGET = 2;

struct Entry {
    Key key;
    Value value;
};

int ReadEnvInt(const char* name, int minValue, int maxValue)
{
    const char* envValue = std::getenv(name);
    if (envValue == nullptr || *envValue == '\0') {
        throw std::runtime_error(std::string("missing ") + name);
    }
    const std::string value(envValue);
    size_t parsedChars = 0;
    long parsed = 0;
    try {
        parsed = std::stol(value, &parsedChars, K_NUMBER_BASE);
    } catch (const std::exception&) {
        parsedChars = 0;
    }
    if (parsedChars != value.size() || parsed < minValue || parsed > maxValue) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

void CheckAcl(aclError ret, const char* operation)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(ret));
    }
}

uint32_t MurmurHash3(Key key)
{
    constexpr uint32_t c1 = 0xcc9e2d51U;
    constexpr uint32_t c2 = 0x1b873593U;
    uint32_t block = key * c1;
    block = (block << K_MURMUR_ROTATE_LEFT) | (block >> K_MURMUR_ROTATE_RIGHT);
    block *= c2;
    uint32_t hash = block;
    hash = (hash << K_MURMUR_MIX_SHIFT) | (hash >> K_MURMUR_MIX_SHIFT_RIGHT);
    hash = hash * K_MURMUR_MULTIPLIER + 0xe6546b64U;
    hash ^= sizeof(Key);
    hash ^= hash >> K_MURMUR_FINAL_SHIFT;
    hash *= 0x85ebca6bU;
    hash ^= hash >> K_MURMUR_MIX_SHIFT;
    hash *= 0xc2b2ae35U;
    hash ^= hash >> K_MURMUR_FINAL_SHIFT;
    return hash;
}

std::vector<Entry> MakeRankEntries(int rank)
{
    std::vector<Entry> entries;
    Key candidate = static_cast<Key>(1000 + rank * 1000);

    for (int targetPe = 0; targetPe < K_WORLD_SIZE; ++targetPe) {
        for (int index = 0; index < K_ENTRIES_PER_TARGET; ++index) {
            while (MurmurHash3(candidate) % K_WORLD_SIZE != static_cast<uint32_t>(targetPe)) {
                ++candidate;
            }
            entries.push_back({candidate, static_cast<Value>(rank * 100 + targetPe * 10 + index)});
            ++candidate;
        }
    }
    return entries;
}

std::vector<Entry> MakeExpectedEntries()
{
    std::vector<Entry> entries;
    for (int rank = 0; rank < K_WORLD_SIZE; ++rank) {
        const std::vector<Entry> rankEntries = MakeRankEntries(rank);
        entries.insert(entries.end(), rankEntries.begin(), rankEntries.end());
    }
    return entries;
}

}  // namespace

void ValidateTargets(const std::vector<Entry>& entries, int rank)
{
    bool hasLocalTarget = false;
    bool hasRemoteTarget = false;
    for (const Entry& entry : entries) {
        hasLocalTarget |= MurmurHash3(entry.key) % K_WORLD_SIZE == static_cast<uint32_t>(rank);
        hasRemoteTarget |= MurmurHash3(entry.key) % K_WORLD_SIZE != static_cast<uint32_t>(rank);
    }
    if (!hasLocalTarget || !hasRemoteTarget) {
        throw std::runtime_error("test data does not cover both local and remote partitions");
    }
}

void TransferAndValidate(DistEmbeddingContainer<Key, Value>& container,
                         const std::vector<Entry>& localEntries,
                         const std::vector<Entry>& expectedEntries)
{
    std::vector<Key> insertKeys;
    std::vector<Value> insertValues;
    std::vector<Key> searchKeys;
    std::vector<Value> searchValues(expectedEntries.size(), 0);
    insertKeys.reserve(localEntries.size());
    insertValues.reserve(localEntries.size());
    searchKeys.reserve(expectedEntries.size());
    for (const Entry& entry : localEntries) {
        insertKeys.push_back(entry.key);
        insertValues.push_back(entry.value);
    }
    for (const Entry& entry : expectedEntries) {
        searchKeys.push_back(entry.key);
    }

    const size_t insertKeyBytes = insertKeys.size() * sizeof(Key);
    const size_t insertValueBytes = insertValues.size() * sizeof(Value);
    const size_t searchKeyBytes = searchKeys.size() * sizeof(Key);
    const size_t searchValueBytes = searchValues.size() * sizeof(Value);
    DeviceBuffer deviceInsertKeys;
    DeviceBuffer deviceInsertValues;
    DeviceBuffer deviceSearchKeys;
    DeviceBuffer deviceSearchValues;
    deviceInsertKeys.Allocate(insertKeyBytes, "aclrtMalloc(deviceInsertKeys)");
    deviceInsertValues.Allocate(insertValueBytes, "aclrtMalloc(deviceInsertValues)");
    deviceSearchKeys.Allocate(searchKeyBytes, "aclrtMalloc(deviceSearchKeys)");
    deviceSearchValues.Allocate(searchValueBytes, "aclrtMalloc(deviceSearchValues)");
    CheckAcl(aclrtMemcpy(deviceInsertKeys.Get(), insertKeyBytes, insertKeys.data(), insertKeyBytes,
        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(insertKeys)");
    CheckAcl(aclrtMemcpy(deviceInsertValues.Get(), insertValueBytes, insertValues.data(), insertValueBytes,
        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(insertValues)");
    CheckAcl(aclrtMemcpy(deviceSearchKeys.Get(), searchKeyBytes, searchKeys.data(), searchKeyBytes,
        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(searchKeys)");
    container.Insert(static_cast<const Key*>(deviceInsertKeys.Get()),
                     static_cast<const Value*>(deviceInsertValues.Get()), insertKeys.size());
    container.Search(static_cast<const Key*>(deviceSearchKeys.Get()),
                     static_cast<Value*>(deviceSearchValues.Get()), searchKeys.size());
    CheckAcl(aclrtMemcpy(searchValues.data(), searchValueBytes, deviceSearchValues.Get(), searchValueBytes,
        ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(searchValues)");

    for (size_t index = 0; index < expectedEntries.size(); ++index) {
        if (searchValues[index] != expectedEntries[index].value) {
            throw std::runtime_error("unexpected value for key=" + std::to_string(expectedEntries[index].key) +
                                     ", expected=" + std::to_string(expectedEntries[index].value) +
                                     ", actual=" + std::to_string(searchValues[index]));
        }
    }
}

void RunRankOperations(int rank, int deviceId, const char* ipPort)
{
    DistEmbeddingOptions options{K_TABLE_SIZE, rank, K_WORLD_SIZE, deviceId - rank,
                                 K_MAX_KEYS_PER_PE, ipPort};
    DistEmbeddingContainer<Key, Value> container(options);
    container.Init();
    const std::vector<Entry> localEntries = MakeRankEntries(rank);
    const std::vector<Entry> expectedEntries = MakeExpectedEntries();
    ValidateTargets(localEntries, rank);
    TransferAndValidate(container, localEntries, expectedEntries);
    std::cout << "PASS: rank=" << rank << " validated " << expectedEntries.size()
              << " keys across both partitions\n";
}

int main()
{
    bool aclInitialized = false;
    try {
        const int rank = ReadEnvInt("RANK_ID", 0, K_WORLD_SIZE - 1);
        const int deviceId = ReadEnvInt("DEVICE_ID", 0, K_MAX_DEVICE_ID);
        const char* ipPort = std::getenv("SHMEM_IP_PORT");
        if (ipPort == nullptr || *ipPort == '\0') {
            throw std::runtime_error("missing SHMEM_IP_PORT");
        }
        std::cout << "test_rank2: rank=" << rank << ", device=" << deviceId
                  << ", ipPort=" << ipPort << '\n';
        CheckAcl(aclInit(nullptr), "aclInit");
        aclInitialized = true;
        RunRankOperations(rank, deviceId, ipPort);
        (void)aclFinalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        if (aclInitialized) {
            (void)aclFinalize();
        }
        return 1;
    }
}
