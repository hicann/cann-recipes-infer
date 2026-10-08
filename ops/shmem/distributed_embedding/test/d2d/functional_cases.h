/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */
#ifndef DISTRIBUTED_EMBEDDING_D2D_FUNCTIONAL_CASES_H
#define DISTRIBUTED_EMBEDDING_D2D_FUNCTIONAL_CASES_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dist_embedding_container.h"
#include "tiling/platform/platform_ascendc.h"
#include "../device_buffer.h"

namespace d2d_test {
using Value = size_t;
static_assert(sizeof(Value) == 8, "D2D boundary cases require 64-bit size_t");
constexpr Value K_MISSING = std::numeric_limits<Value>::max();

// Named constants for the host oracle, bucket layout and boundary coverage.
constexpr size_t HASH_WORD_BYTES = sizeof(uint32_t);
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint32_t HASH_MIX_LEFT = 15U;
constexpr uint32_t HASH_MIX_RIGHT = 17U;
constexpr uint32_t HASH_ACC_LEFT = 13U;
constexpr uint32_t HASH_ACC_RIGHT = 19U;
constexpr uint32_t HASH_FOLD_TOP = 16U;
constexpr uint32_t HASH_FOLD_LOW = 13U;
constexpr uint32_t KEY_SIGN_BIT = 31U;
constexpr uint32_t UPDATE_VALUE_BIT = 40U;
constexpr uint64_t COLLISION_BUCKET = 7U;
constexpr size_t UPDATED_ENTRIES_PER_PE = 2U;
constexpr size_t DUPLICATE_ENTRIES = 8U;
constexpr int COLLISION_STEPS = 3;
constexpr int KEY_BITS_32 = 32;
constexpr int KEY_BITS_64 = 64;
constexpr uint64_t PARITY_MODULUS = 2U;
constexpr uint64_t LOCAL_MEM_SIZE = 128ULL * 1024 * 1024;

inline void CheckAcl(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

// Host-side routing oracle. Hash explicit little-endian bytes, independently of
// the device helper's scalar-word loads. Expected values come only from inputs.
template <typename Key>
uint32_t Hash(Key key)
{
    using UnsignedKey = typename std::make_unsigned<Key>::type;
    const auto bits = static_cast<UnsignedKey>(key);
    uint32_t hash = 0;
    for (size_t offset = 0; offset < sizeof(Key); offset += HASH_WORD_BYTES) {
        uint32_t word = 0;
        for (size_t byte = 0; byte < HASH_WORD_BYTES; ++byte) {
            const uint32_t shift = static_cast<uint32_t>((offset + byte) * BITS_PER_BYTE);
            word |= static_cast<uint32_t>((bits >> shift) & 0xffU) << (byte * BITS_PER_BYTE);
        }
        word *= 0xcc9e2d51U;
        word = (word << HASH_MIX_LEFT) | (word >> HASH_MIX_RIGHT);
        word *= 0x1b873593U;
        hash ^= word;
        hash = (hash << HASH_ACC_LEFT) | (hash >> HASH_ACC_RIGHT);
        hash = hash * 5U + 0xe6546b64U;
    }
    hash ^= sizeof(Key);
    hash ^= hash >> HASH_FOLD_TOP;
    hash *= 0x85ebca6bU;
    hash ^= hash >> HASH_FOLD_LOW;
    hash *= 0xc2b2ae35U;
    hash ^= hash >> HASH_FOLD_TOP;
    return hash;
}

template <typename Key>
class Cases {
    struct Entry {
        Key key;
        Value value;
    };
    using Batch = std::vector<std::vector<Entry>>;
    using EmbeddingTable = DistEmbeddingContainer<Key, Value>;

public:
    Cases(DistEmbeddingContainer<Key, Value>& table, int rank, int peers, size_t cores, size_t maxInput)
        : table_(table), rank_(rank), peers_(peers), cores_(cores), maxInput_(maxInput)
    {
        boundaryKeys_ = {Key{0}, Key{1}, static_cast<Key>(std::numeric_limits<Key>::max() - 1)};
        if constexpr (sizeof(Key) == 4) {
            boundaryKeys_.push_back(static_cast<Key>(uint32_t{1} << KEY_SIGN_BIT));
        } else {
            boundaryKeys_.insert(boundaryKeys_.end(), {
                std::numeric_limits<Key>::min(), Key{-1}, Key{1} << 32,
                (Key{1} << 32) + 1, (Key{2} << 32) + 1});
        }
        used_.insert(boundaryKeys_.begin(), boundaryKeys_.end());
    }

    void Run()
    {
        const auto missing = Routed(2);
        const auto missingKeys = Keys(missing);
        Check(missingKeys, "empty_table_query");
        RunBasicCases(missingKeys);
        RunRoundCases(missingKeys);
        RunBoundaryCases(missingKeys);
        RunCapacityCases(missingKeys);
        CheckEmptySearch();
    }

private:
    void RunBasicCases(const std::vector<Key>& missingKeys)
    {
        // Run on the otherwise empty table: the first key occupies the initial
        // bucket; subsequent single-key rounds must probe and wrap at table end.
        CollisionCase(COLLISION_BUCKET, "hash_collision");
        CollisionCase(table_.GetTableSize() - 1, "probe_wraparound");

        auto initial = Routed(8);
        Insert(initial);
        CheckAll(missingKeys, "hits_and_misses");
        auto updates = initial;
        for (auto& entries : updates) {
            entries.resize(UPDATED_ENTRIES_PER_PE);
            for (auto& entry : entries) {
                entry.value = (Value{1} << UPDATE_VALUE_BIT) + static_cast<Value>(entry.key);
            }
        }
        Insert(updates);
        CheckAll(missingKeys, "existing_key_update");
        // Same key/value repeated within each PE and across source PEs; no
        // assertion depends on which concurrent writer wins.
        Batch duplicates(peers_);
        const Entry duplicate = updates.front().front();
        for (auto& entries : duplicates) {
            entries.assign(DUPLICATE_ENTRIES, duplicate);
        }
        Insert(duplicates);
        CheckAll(missingKeys, "duplicate_key_same_value");
    }

    void RunRoundCases(const std::vector<Key>& missingKeys)
    {
        Insert(Routed(1));
        CheckAll(missingKeys, "multiple_insert_rounds");
        Insert(Batch(peers_));
        CheckAll(missingKeys, "empty_insert_preserves_table");
        if (peers_ > 1) {
            auto uneven = Routed(3);
            uneven[0].clear();
            Insert(uneven);
            CheckAll(missingKeys, "one_empty_pe");
        }
    }

    void RunBoundaryCases(const std::vector<Key>& missingKeys)
    {
        Batch boundaries(peers_);
        const std::vector<Value> values = {
            0, 1, (Value{1} << 32) - 1, Value{1} << 32,
            (Value{1} << 32) + 1, Value{1} << 63, K_MISSING - 1};
        // Cycle all value boundaries across every key boundary, including 64-bit
        // keys with equal low words, negative keys, and the largest usable key.
        for (size_t round = 0; round < values.size(); ++round) {
            for (auto& entries : boundaries) {
                entries.clear();
            }
            for (size_t i = 0; i < boundaryKeys_.size(); ++i) {
                boundaries[i % peers_].push_back({boundaryKeys_[i], values[(i + round) % values.size()]});
            }
            Insert(boundaries);
            CheckAll(missingKeys, "key_value_boundaries_round_" + std::to_string(round));
        }
    }

    void RunCapacityCases(const std::vector<Key>& missingKeys)
    {
        // Generate once and reuse prefixes so capacity remains bounded. Each
        // source contributes keys alternating between all owner PEs.
        const auto large = Routed(maxInput_);
        std::vector<size_t> sizes = {0, 1, 31, 32, 33, 2047, 2048, 2049,
            cores_ - 1, cores_, cores_ + 1,
            cores_ * 2048 - 1, cores_ * 2048, cores_ * 2048 + 1};
        std::sort(sizes.begin(), sizes.end());
        sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
        for (size_t count : sizes) {
            Batch batch(peers_);
            for (int source = 0; source < peers_; ++source) {
                batch[source].assign(large[source].begin(), large[source].begin() + count);
            }
            Insert(batch);
            const auto keys = Keys(batch);
            if (!keys.empty()) {
                // Use per-source query sizes as well as the global verification.
                std::vector<Key> localKeys;
                for (const auto& entry : batch[rank_]) {
                    localKeys.push_back(entry.key);
                }
                Check(localKeys, "query_size_" + std::to_string(count));
            }
            CheckAll(missingKeys, "insert_size_" + std::to_string(count));
        }
    }

    Key Next(int owner, uint64_t bucket = std::numeric_limits<uint64_t>::max())
    {
        for (; candidate_ < static_cast<uint64_t>(std::numeric_limits<Key>::max()); ++candidate_) {
            const Key key = static_cast<Key>(candidate_);
            const uint32_t hash = Hash(key);
            if (hash % peers_ == static_cast<uint32_t>(owner) &&
                (bucket == std::numeric_limits<uint64_t>::max() || hash % table_.GetTableSize() == bucket) &&
                used_.insert(key).second) {
                ++candidate_;
                return key;
            }
        }
        throw std::runtime_error("exhausted key candidates");
    }

    Batch Routed(size_t count)
    {
        Batch batch(peers_);
        for (int source = 0; source < peers_; ++source) {
            for (size_t i = 0; i < count; ++i) {
                const Key key = Next(static_cast<int>(i % peers_));
                batch[source].push_back({key, static_cast<Value>(i + 100 + source)});
            }
        }
        return batch;
    }

    static std::vector<Key> Keys(const Batch& batch)
    {
        std::vector<Key> keys;
        for (const auto& entries : batch) {
            for (const auto& entry : entries) {
                keys.push_back(entry.key);
            }
        }
        return keys;
    }

    void CollisionCase(uint64_t bucket, const std::string& label)
    {
        std::vector<Key> queries;
        for (int owner = 0; owner < peers_; ++owner) {
            for (int step = 0; step < COLLISION_STEPS; ++step) {
                const Key key = Next(owner, bucket);
                Batch batch(peers_);
                const int source = (owner + step) % peers_;
                batch[source].push_back({key, static_cast<Value>(500 + owner * 10 + step)});
                Insert(batch);
                queries.push_back(key);
                Check(queries, label + "_owner_" + std::to_string(owner) + "_step_" + std::to_string(step));
            }
            queries.push_back(Next(owner, bucket));
        }
        Check(queries, label + "_missing_after_collision");
    }

    void Insert(const Batch& batch)
    {
        const auto& local = batch[rank_];
        std::vector<Key> keys;
        std::vector<Value> values;
        for (const auto& entry : local) {
            keys.push_back(entry.key);
            values.push_back(entry.value);
        }
        DeviceBuffer deviceKeys;
        DeviceBuffer deviceValues;
        // Non-null pointers for zero-length inputs, without zero-byte ACL calls.
        deviceKeys.Allocate(std::max<size_t>(keys.size(), 1) * sizeof(Key), "allocate insert keys");
        deviceValues.Allocate(std::max<size_t>(values.size(), 1) * sizeof(Value), "allocate insert values");
        if (!keys.empty()) {
            CheckAcl(aclrtMemcpy(deviceKeys.Get(), keys.size() * sizeof(Key), keys.data(),
                keys.size() * sizeof(Key), ACL_MEMCPY_HOST_TO_DEVICE), "copy insert keys");
            CheckAcl(aclrtMemcpy(deviceValues.Get(), values.size() * sizeof(Value), values.data(),
                values.size() * sizeof(Value), ACL_MEMCPY_HOST_TO_DEVICE), "copy insert values");
        }
        table_.Insert(static_cast<const Key*>(deviceKeys.Get()),
                      static_cast<const Value*>(deviceValues.Get()), keys.size());
        aclshmem_barrier_all();
        for (const auto& entries : batch) {
            for (const auto& entry : entries) {
                expected_[entry.key] = entry.value;
            }
        }
    }

    void Check(const std::vector<Key>& keys, const std::string& label)
    {
        if (keys.empty()) {
            throw std::runtime_error("use CheckEmptySearch for zero-length queries");
        }
        DeviceBuffer deviceKeys;
        DeviceBuffer deviceValues;
        deviceKeys.Allocate(keys.size() * sizeof(Key), "allocate query keys");
        deviceValues.Allocate(keys.size() * sizeof(Value), "allocate query values");
        CheckAcl(aclrtMemcpy(deviceKeys.Get(), keys.size() * sizeof(Key), keys.data(),
            keys.size() * sizeof(Key), ACL_MEMCPY_HOST_TO_DEVICE), "copy query keys");
        // Poison output so an unwritten zero-valued result cannot pass by chance.
        CheckAcl(aclrtMemset(deviceValues.Get(), keys.size() * sizeof(Value), 0xa5,
            keys.size() * sizeof(Value)), "poison query output");
        table_.Search(static_cast<const Key*>(deviceKeys.Get()), static_cast<Value*>(deviceValues.Get()), keys.size());
        std::vector<Value> actual(keys.size());
        CheckAcl(aclrtMemcpy(actual.data(), actual.size() * sizeof(Value), deviceValues.Get(),
            actual.size() * sizeof(Value), ACL_MEMCPY_DEVICE_TO_HOST), "read query values");
        for (size_t i = 0; i < keys.size(); ++i) {
            const auto found = expected_.find(keys[i]);
            const Value expected = found == expected_.end() ? K_MISSING : found->second;
            if (actual[i] != expected) {
                throw std::runtime_error(label + ": rank=" + std::to_string(rank_) +
                    " key=" + std::to_string(keys[i]) + " expected=" + std::to_string(expected) +
                    " actual=" + std::to_string(actual[i]));
            }
        }
        // All readers finish before any rank begins the next update or teardown.
        aclshmem_barrier_all();
        std::cout << "PASS: " << label << " key_bits=" << (sizeof(Key) * BITS_PER_BYTE)
                  << " rank=" << rank_ << '\n';
    }

    void CheckAll(const std::vector<Key>& missing, const std::string& label)
    {
        std::vector<Key> keys = missing;
        for (const auto& entry : expected_) {
            keys.push_back(entry.first);
        }
        std::sort(keys.begin(), keys.end());
        Check(keys, label);
    }

    void CheckEmptySearch()
    {
        DeviceBuffer keys;
        DeviceBuffer values;
        keys.Allocate(sizeof(Key), "allocate empty query key");
        values.Allocate(sizeof(Value), "allocate empty query guard");
        const Value guard = 1234567;
        CheckAcl(aclrtMemcpy(values.Get(), sizeof(Value), &guard, sizeof(Value), ACL_MEMCPY_HOST_TO_DEVICE),
                 "initialize empty query guard");
        std::cout << "CASE: empty_search key_bits=" << (sizeof(Key) * BITS_PER_BYTE)
                  << " rank=" << rank_ << std::endl;
        table_.Search(static_cast<const Key*>(keys.Get()), static_cast<Value*>(values.Get()), 0);
        Value actual = 0;
        CheckAcl(aclrtMemcpy(&actual, sizeof(Value), values.Get(), sizeof(Value), ACL_MEMCPY_DEVICE_TO_HOST),
                 "read empty query guard");
        if (actual != guard) {
            throw std::runtime_error("empty_search modified output");
        }
        aclshmem_barrier_all();
        std::cout << "PASS: empty_search\n";
    }

    EmbeddingTable& table_;
    int rank_;
    int peers_;
    size_t cores_;
    size_t maxInput_;
    uint64_t candidate_ = 1024;
    std::vector<Key> boundaryKeys_;
    std::unordered_set<Key> used_;
    std::unordered_map<Key, Value> expected_;
};

template <typename Key>
void Run(int rank, int peers, int device, const char* ipPort)
{
    CheckAcl(aclrtSetDevice(device), "aclrtSetDevice");
    const auto& platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const size_t cores = platform != nullptr && platform->GetCoreNumAiv() > 0 ? platform->GetCoreNumAiv() : 64;
    const size_t maxInput = cores * 2048 + 1;
    // Leave ample room even if every input hashes to one PE. An odd bucket count
    // permits both PE owners to hash to the same final bucket in two-PE tests.
    uint64_t requested = 2 * maxInput * peers + 1024;
    while (static_cast<uint64_t>(static_cast<double>(requested) / 0.75f) % PARITY_MODULUS == 0) {
        ++requested;
    }
    DistEmbeddingContainer<Key, Value> table(requested, 0, rank, peers, device - rank,
        static_cast<uint32_t>(maxInput), ipPort, LOCAL_MEM_SIZE);
    table.Init();
    if (aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED) {
        throw std::runtime_error("SHMEM is not initialized");
    }
    std::cout << "D2D cases: key_bits=" << (sizeof(Key) * BITS_PER_BYTE) << " rank=" << rank
              << " cores=" << cores << " buckets=" << table.GetTableSize()
              << " max_input=" << maxInput << std::endl;
    Cases<Key>(table, rank, peers, cores, maxInput).Run();
    // Destructor owns the single Finalize call, after all device buffers die.
}
}  // namespace d2d_test
#endif
