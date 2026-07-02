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
#include "model_utils.h"
#include "logger.h"
#include "progress_bar.hpp"
#include "deepseek_v3_w8a8_model.h"
#include <boost/iostreams/device/mapped_file.hpp>

std::unique_ptr<BaseLLM> CreateLLMModel(LLMConfig& config)
{
    std::string modelType = config.modelConfig.modelType;
    if (modelType == "deepseek_v3-w8a8")
    {
        LOG_INFO("Load %s model ...", modelType.c_str());
        return std::make_unique<DeepSeekV3W8A8Model>(config);
    }

    throw std::runtime_error("Unsupported model type: " + modelType);
}

std::string get_vector_shape_str(const std::vector<int64_t>& shape)
{
    std::string shape_str;
    shape_str += "(";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        shape_str += std::to_string(shape[i]);
        if (i != shape.size() - 1)
        {
            shape_str += ", ";
        }
    }
    shape_str += ")";
    return shape_str;
}

struct TensorInfo
{
    std::string dtype;
    std::vector<int64_t> shape;
    size_t offset;
    size_t size;
    size_t bytes;

    TensorInfo()
        : offset(0)
        , size(0)
        , bytes(0)
    {
    }

    void Print() const
    {
        printf("TensorInfo(dtype=%s, shape=%s, offset=%zu, size=%zu, bytes=%zu)\n", dtype.c_str(),
            get_vector_shape_str(shape).c_str(), offset, size, bytes);
    }
};

class SafeTensorsLoader
{
public:
    explicit SafeTensorsLoader(const std::string& file_path)
    {
        if (!LoadFile(file_path))
        {
            throw std::runtime_error("Failed to load file: " + file_path);
        }
    }

    bool LoadFile(const std::string& file_path)
    {
        try
        {
            file.open(file_path);
            if (!file.is_open())
            {
                return false;
            }

            const char* data = file.data();
            size_t pos = 0;

            // get header
            uint64_t header_len = *reinterpret_cast<const uint64_t*>(data);
            pos += 8;  // skip header length

            std::string header_str(data + pos, header_len);
            json header = json::parse(header_str);
            pos += header_len;

            // get data in tensor
            for (const auto& [key, value] : header.items())
            {
                if (key == "__metadata__")
                    continue;
                TensorInfo info;
                info.dtype = value["dtype"];
                info.shape = value["shape"].get<std::vector<int64_t>>();
                info.offset = value["data_offsets"][0].get<size_t>() + pos;
                info.size = std::accumulate(info.shape.begin(), info.shape.end(),
                    1ull, // using 1ull avoid overflow
                    std::multiplies<size_t>());
                info.bytes = get_dtype_bytes(info.dtype) * info.size;
                tensors[key] = info;
            }
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error loading file: " << e.what() << std::endl;
            return false;
        }
    }

    static size_t get_dtype_bytes(const std::string& dtype)
    {
        if (dtype == "F16")
        {
            return 2;
        }
        if (dtype == "F32")
        {
            return 4;
        }
        if (dtype == "F64")
        {
            return 8;
        }
        if (dtype == "I8")
        {
            return 1;
        }
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }

    template <typename T>
    const T* get_tensor(const std::string& name) const
    {
        auto it = tensors.find(name);
        if (it == tensors.end())
        {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return reinterpret_cast<const T*>(file.data() + it->second.offset);
    }

public:
    boost::iostreams::mapped_file_source file;
    std::unordered_map<std::string, TensorInfo> tensors;
};

void WeightMap::LoadFromFile(const std::string& fileName)
{
    SafeTensorsLoader loader(fileName);
    LOG_DEBUG("LoadFromFile: %s", fileName.c_str());

    progresscpp::ProgressBar progressBar(loader.tensors.size(), 70); // 70 is the width of the progress bar
    for (auto& [tensorName, tensorInfo] : loader.tensors)
    {
        WeightData wd;
        auto ret = aclrtMalloc(&wd.dataPtr, tensorInfo.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            throw std::runtime_error(tensorName+" Malloc failed! ");
        }
        ret = aclrtMemcpy(wd.dataPtr, tensorInfo.bytes, (uint8_t*) loader.file.data() + tensorInfo.offset, tensorInfo.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
        {
            throw std::runtime_error(tensorName+" Memcpy failed! ");
        }
        weight.emplace(tensorName, wd);

        ++progressBar;
        progressBar.Display();
    }
    progressBar.Done();
}

WeightData& WeightMap::operator[](const std::string& key)
{
    auto it = weight.find(key);
    if (it == weight.end())
    {
        LOG_ERROR("Tensor with key '%s' not found in WeightMap", key.c_str());
    }
    return it->second;
}
