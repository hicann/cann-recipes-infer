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

#include "comm_domain_manager.h"
#include <stdexcept>

#define HCCLCHECK(ret)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (ret != HCCL_SUCCESS)                                                                                       \
        {                                                                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);                        \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

MPI_Comm CommDomainManager::mGlobalCommMpi;
MPI_Comm CommDomainManager::mTpCommMpi;
MPI_Comm CommDomainManager::mEpCommMpi;
MPI_Comm CommDomainManager::mDpCommMpi;
HcclComm CommDomainManager::mGlobalCommHccl;
HcclComm CommDomainManager::mTpCommHccl;
HcclComm CommDomainManager::mEpCommHccl;
HcclComm CommDomainManager::mDpCommHccl;

/**
 * @brief Initialize an HCCL communicator based on an MPI communicator.
 * @param comm       MPI communicator handle.
 * @param hcclComm   HCCL communicator handle to be initialized.
 * @param rank       Rank of the current process within the communicator.
 * @param commName   Name of the communicator.
 * @details The function performs the following steps:
 * Obtain the size of the MPI communicator.
 * Retrieve root-node information from rank 0 in the MPI communicator.
 * Broadcast the root-node information to all processes.
 * Initialize HCCL communicator configurations.
 * Create the HCCL communicator.
*/
int InitializeHcclCommFromMpi(MPI_Comm& mpiComm, HcclComm& hcclComm, int rank, std::string commName)
{
    // Get comm size
    int commSize;
    MPI_Comm_size(mpiComm, &commSize);

    HcclRootInfo rootInfo;
    if (rank == 0)
    {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
    }

    // broadcast root_info
    MPI_Bcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, 0, mpiComm);
    MPI_Barrier(mpiComm);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclBufferSize = 50; // default value
    strcpy(config.hcclCommName, commName.c_str());

    HCCLCHECK(HcclCommInitRootInfoConfig(commSize, &rootInfo, rank, &config, &hcclComm));

    return 0;
}

void CommDomainManager::Initialize(ParallelConfig& config, MPI_Comm workerGlobalComm)
{
    uint32_t rank = config.rank;
    uint32_t worldSize = config.worldSize;
    uint32_t tpSize = config.tensorParallelSize;
    uint32_t epSize = config.expertParallelSize;
    uint32_t ppSize = config.pipelineParallelSize;
    uint32_t dpSize = config.dataParallelSize;

    int ret;

    // init global MPI and HCCL
    mGlobalCommMpi = workerGlobalComm;

    // global comm
    InitializeHcclCommFromMpi(mGlobalCommMpi, mGlobalCommHccl, rank, "global_comm");

    // tp comm
    if (tpSize > worldSize)
    {
        throw std::runtime_error("Tensor parallel size cannot be larger than world size");
    }
    if (worldSize % tpSize != 0)
    {
        throw std::runtime_error("World size must be divisible by tensor parallel size");
    }
    if (tpSize == worldSize)
    {
        mTpCommMpi = mGlobalCommMpi;
        mTpCommHccl = mGlobalCommHccl;
    }
    else
    {
        int tpColor = rank / tpSize;
        ret = MPI_Comm_split(mGlobalCommMpi, tpColor, rank, &mTpCommMpi);
        if (ret != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Comm_split tp_comm failed.");
        }

        InitializeHcclCommFromMpi(mTpCommMpi, mTpCommHccl, rank % tpSize, "tp_comm");
    }

    // EP comm
    if (epSize > worldSize)
    {
        throw std::runtime_error("Expert parallel size cannot be larger than world size");
    }
    if (worldSize % epSize != 0)
    {
        throw std::runtime_error("World size must be divisible by expert parallel size");
    }
    if (config.expertParallelSize == worldSize)
    {
        mEpCommMpi = mGlobalCommMpi;
        mEpCommHccl = mGlobalCommHccl;
    }
    else
    {
        int epColor = rank / epSize;
        ret = MPI_Comm_split(mGlobalCommMpi, epColor, rank, &mEpCommMpi);
        if (ret != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Comm_split ep_comm failed.");
        }
        InitializeHcclCommFromMpi(mEpCommMpi, mEpCommHccl, rank % epSize, "ep_comm");
    }

    // DP comm
    if (dpSize > worldSize)
    {
        throw std::runtime_error("data parallel size cannot be larger than world size");
    }
    if (tpSize * dpSize != epSize && dpSize > 1)
    {
        throw std::runtime_error("tensor parallel size * data parallel size must equal to expert parallel size");
    }
    else
    {
        int dpColor = rank % (epSize / dpSize);
        ret = MPI_Comm_split(mGlobalCommMpi, dpColor, rank, &mDpCommMpi);
        if (ret != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Comm_split ep_comm failed.");
        }
        InitializeHcclCommFromMpi(mDpCommMpi, mDpCommHccl, rank / (epSize / dpSize), "dp_comm");
    }
}

void CommDomainManager::Finalize()
{
    // free hccl comm
    if (mGlobalCommHccl != nullptr)
    {
        HcclCommDestroy(mGlobalCommHccl);
    }
    if (mTpCommHccl != nullptr && mTpCommHccl != mGlobalCommHccl)
    {
        HcclCommDestroy(mTpCommHccl);
    }
    if (mEpCommHccl != nullptr && mEpCommHccl != mGlobalCommHccl)
    {
        HcclCommDestroy(mEpCommHccl);
    }
    if (mDpCommHccl != nullptr && mDpCommHccl != mGlobalCommHccl)
    {
        HcclCommDestroy(mDpCommHccl);
    }

    // free mpi comm
    if (mTpCommMpi != mGlobalCommMpi)
    {
        MPI_Comm_free(&mTpCommMpi);
    }
    if (mEpCommMpi != mGlobalCommMpi)
    {
        MPI_Comm_free(&mEpCommMpi);
    }
    if (mDpCommMpi != mGlobalCommMpi)
    {
        MPI_Comm_free(&mDpCommMpi);
    }
}
