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

#include "file_utils.h"

#include "logger.h"

#include <fcntl.h>
#include <fstream>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1)
    {
        LOG_ERROR("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0)
    {
        LOG_ERROR("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open())
    {
        LOG_ERROR("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf* buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0)
    {
        LOG_ERROR("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize)
    {
        LOG_ERROR("file(%s) size(%ld) is larger than buffer size(%ld)", filePath.c_str(), size, bufferSize);
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char*>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr)
    {
        LOG_ERROR("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0)
    {
        LOG_ERROR("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    const char* bufPtr = static_cast<const char*>(buffer);
    size_t bytesRemaining = size;
    ssize_t writeSize;

    while (bytesRemaining > 0)
    {
        writeSize
            = write(fd, bufPtr, std::min(bytesRemaining, static_cast<size_t>(std::numeric_limits<ssize_t>::max())));
        if (writeSize < 0)
        {
            LOG_ERROR("Write file failed.");
            close(fd);
            return false;
        }
        bytesRemaining -= writeSize;
        bufPtr += writeSize;
    }
    (void) close(fd);
    return true;
}
