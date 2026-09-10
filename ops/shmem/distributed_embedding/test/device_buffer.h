/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

#ifndef DEVICE_BUFFER_H
#define DEVICE_BUFFER_H

#include <cstddef>
#include <stdexcept>
#include <string>

#include "acl/acl.h"

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            (void)aclrtFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void Allocate(size_t bytes, const char* name)
    {
        const aclError ret = aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(std::string(name) + " failed: " + std::to_string(ret));
        }
    }

    void* Get() const { return data_; }

private:
    void* data_ = nullptr;
};

#endif  // DEVICE_BUFFER_H
