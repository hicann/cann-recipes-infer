/*
 * Single-PE lifecycle test for DistEmbeddingContainer.
 *
 * Host and kernel implementations are built as separate targets. This file
 * only exercises the public host container API.
 */

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "acl/acl.h"

#include "../op_host/dist_embedding_container.h"

namespace {

using Key = unsigned int;
using Value = size_t;

constexpr int K_MAX_DEVICE_ID = 1023;
constexpr int K_NUMBER_BASE = 10;
constexpr int K_SINGLE_PE_COUNT = 1;
constexpr int K_DEVICE_OFFSET = 0;
constexpr uint64_t K_TABLE_SIZE = 1024;
constexpr uint32_t K_MAX_KEYS_PER_PE = 16;
constexpr size_t K_KEY_COUNT = 3;
constexpr size_t K_INSERT_COUNT = 2;

int ReadEnvInt(const char* name, int fallback)
{
    const char* envValue = std::getenv(name);
    if (envValue == nullptr || *envValue == '\0') {
        return fallback;
    }
    const std::string value(envValue);
    size_t parsedChars = 0;
    long parsed = 0;
    try {
        parsed = std::stol(value, &parsedChars, K_NUMBER_BASE);
    } catch (const std::exception&) {
        parsedChars = 0;
    }
    if (parsedChars != value.size() || parsed < 0 || parsed > K_MAX_DEVICE_ID) {
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

void FreeDevice(void*& pointer)
{
    if (pointer != nullptr) {
        (void)aclrtFree(pointer);
        pointer = nullptr;
    }
}

void FreeDeviceBuffers(void*& deviceKeys, void*& deviceValues,
                       void*& deviceSearchKeys, void*& deviceSearchValues)
{
    FreeDevice(deviceKeys);
    FreeDevice(deviceValues);
    FreeDevice(deviceSearchKeys);
    FreeDevice(deviceSearchValues);
}

}  // namespace

void RunContainerOperations(DistEmbeddingContainer<Key, Value>& container,
                            void*& deviceKeys, void*& deviceValues,
                            void*& deviceSearchKeys, void*& deviceSearchValues)
{
    constexpr size_t keyBytes = K_KEY_COUNT * sizeof(Key);
    constexpr size_t valueBytes = K_KEY_COUNT * sizeof(Value);
    const std::array<Key, K_KEY_COUNT> keys = {11, 22, 99};
    const std::array<Value, K_KEY_COUNT> values = {1011, 2022, 9099};
    std::array<Value, K_KEY_COUNT> searchValues = {0, 0, 0};
    const std::array<Key, K_INSERT_COUNT> insertKeys = {keys[0], keys[1]};
    const std::array<Value, K_INSERT_COUNT> insertValues = {values[0], values[1]};
    constexpr size_t insertKeyBytes = K_INSERT_COUNT * sizeof(Key);
    constexpr size_t insertValueBytes = K_INSERT_COUNT * sizeof(Value);

    CheckAcl(aclrtMalloc(&deviceKeys, insertKeyBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(deviceKeys)");
    CheckAcl(aclrtMalloc(&deviceValues, insertValueBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(deviceValues)");
    CheckAcl(aclrtMalloc(&deviceSearchKeys, keyBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(deviceSearchKeys)");
    CheckAcl(aclrtMalloc(&deviceSearchValues, valueBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(deviceSearchValues)");
    CheckAcl(aclrtMemcpy(deviceKeys, insertKeyBytes, insertKeys.data(), insertKeyBytes,
        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(keys)");
    CheckAcl(aclrtMemcpy(deviceValues, insertValueBytes, insertValues.data(), insertValueBytes,
        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(values)");
    CheckAcl(aclrtMemcpy(deviceSearchKeys, keyBytes, keys.data(), keyBytes,
        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(searchKeys)");

    container.Insert(static_cast<const Key*>(deviceKeys), static_cast<const Value*>(deviceValues),
                     insertKeys.size());
    container.Search(static_cast<const Key*>(deviceSearchKeys), static_cast<Value*>(deviceSearchValues),
                     K_KEY_COUNT);
    CheckAcl(aclrtMemcpy(searchValues.data(), valueBytes, deviceSearchValues, valueBytes,
        ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(searchValues)");

    const Value missingValue = std::numeric_limits<Value>::max();
    if (searchValues != std::array<Value, K_KEY_COUNT>{values[0], values[1], missingValue}) {
        throw std::runtime_error("Search returned unexpected values");
    }
}

int RunSinglePeTest(int deviceId, const char* ipPort)
{
    bool shmemInitAttempted = false;
    bool containerInitialized = false;
    bool containerFinalized = false;
    std::unique_ptr<DistEmbeddingContainer<Key, Value>> container;
    void* deviceKeys = nullptr;
    void* deviceValues = nullptr;
    void* deviceSearchKeys = nullptr;
    void* deviceSearchValues = nullptr;
    int result = 1;
    try {
        std::cout << "test_distributed_embedding: deviceId=" << deviceId
                  << ", ipPort=" << ipPort << '\n';
        DistEmbeddingOptions options{K_TABLE_SIZE, deviceId, K_SINGLE_PE_COUNT, K_DEVICE_OFFSET,
                                     K_MAX_KEYS_PER_PE, ipPort};
        container = std::make_unique<DistEmbeddingContainer<Key, Value>>(options);
        shmemInitAttempted = true;
        container->Init();
        containerInitialized = true;
        const int status = aclshmemx_init_status();
        if (status != ACLSHMEM_STATUS_IS_INITIALIZED) {
            throw std::runtime_error("SHMEM status is not initialized: " + std::to_string(status));
        }
        RunContainerOperations(*container, deviceKeys, deviceValues, deviceSearchKeys, deviceSearchValues);
        FreeDeviceBuffers(deviceKeys, deviceValues, deviceSearchKeys, deviceSearchValues);
        container->Finalize();
        containerFinalized = true;
        std::cout << "PASS: Init, Insert, Search, and Finalize succeeded (device=" << deviceId
                  << ", my_pe=0, n_pes=1, status=" << status << ")\n";
        result = 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    FreeDeviceBuffers(deviceKeys, deviceValues, deviceSearchKeys, deviceSearchValues);
    if (container != nullptr && containerInitialized && !containerFinalized) {
        container->Finalize();
        containerFinalized = true;
    } else if (shmemInitAttempted && !containerFinalized) {
        const int ret = aclshmem_finalize();
        if (ret != 0) {
            std::cerr << "WARN: aclshmem_finalize returned " << ret << '\n';
            result = 1;
        }
    }
    return result;
}

int main()
{
    const int deviceId = ReadEnvInt("DEVICE_ID", 0);
    const char* ipPort = std::getenv("SHMEM_IP_PORT");
    if (ipPort == nullptr || *ipPort == '\0') {
        ipPort = "tcp://127.0.0.1:8998";
    }

    bool aclInitialized = false;
    try {
        CheckAcl(aclInit(nullptr), "aclInit");
        aclInitialized = true;
        const int result = RunSinglePeTest(deviceId, ipPort);
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        if (aclInitialized) {
            (void)aclFinalize();
        }
        return 1;
    }
}
