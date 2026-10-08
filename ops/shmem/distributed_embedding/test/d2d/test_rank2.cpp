/*
 * Two-PE D2D functional tests. Every PE queries the global golden table,
 * exercising both local and remote lookups for each selected key type.
 */
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "functional_cases.h"

namespace d2d_test {

int ReadEnvInt(const char* name, int fallback, int minValue, int maxValue)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        if (fallback < minValue) {
            throw std::runtime_error(std::string("missing ") + name);
        }
        return fallback;
    }
    const std::string value(raw);
    size_t consumed = 0;
    long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed < minValue || parsed > maxValue) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

int Main(int peers)
{
    bool aclInitialized = false;
    try {
        const int rank = peers == 1 ? 0 : ReadEnvInt("RANK_ID", -1, 0, peers - 1);
        const int device = ReadEnvInt("DEVICE_ID", peers == 1 ? 0 : -1, 0, 1023);
        const int bits = ReadEnvInt("D2D_KEY_BITS", KEY_BITS_32, KEY_BITS_32, KEY_BITS_64);
        if (bits != KEY_BITS_32 && bits != KEY_BITS_64) {
            throw std::runtime_error("D2D_KEY_BITS must be 32 or 64");
        }
        const char* ipPort = std::getenv("SHMEM_IP_PORT");
        if (ipPort == nullptr || *ipPort == '\0') {
            if (peers != 1) {
                throw std::runtime_error("missing SHMEM_IP_PORT");
            }
            ipPort = "tcp://127.0.0.1:8998";
        }
        CheckAcl(aclInit(nullptr), "aclInit");
        aclInitialized = true;
        if (bits == KEY_BITS_32) {
            Run<unsigned int>(rank, peers, device, ipPort);
        } else {
            Run<long long>(rank, peers, device, ipPort);
        }
        CheckAcl(aclFinalize(), "aclFinalize");
        std::cout << "PASS: D2D functional suite key_bits=" << bits << " rank=" << rank << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        if (aclInitialized) {
            (void)aclFinalize();
        }
        return 1;
    }
}

}  // namespace d2d_test

int main()
{
    const int peers = 2;
    return d2d_test::Main(peers);
}
