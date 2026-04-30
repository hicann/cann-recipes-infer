# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
function launch()
{
    check_launch
    get_rank $1
    check_env_vars $1
    set_hccl
    launch_infer_task $1
}

function check_launch()
{
    if pgrep -f "python.*infer.py" > /dev/null; then
        echo "A Python process executing infer.py was detected to be running, and the script was interrupted and exited."
        exit 1
    else
        echo "No Python process running infer.py was detected."
    fi
}

# Resolve PD_ROLE from the explicit argument or infer it from the local IP.
# Sets and exports PD_ROLE; exits with an error if the role cannot be determined.
# Usage: resolve_pd_role [prefill|decode|""]
function resolve_pd_role()
{
    local role="$1"
    local local_host
    local_host=$(hostname -I | awk '{print $1}')

    local in_prefill=0 in_decode=0
    for pip in "${PREFILL_IPS[@]}"; do
        [ "$local_host" = "$pip" ] && in_prefill=1 && break
    done
    for dip in "${DECODE_IPS[@]}"; do
        [ "$local_host" = "$dip" ] && in_decode=1 && break
    done

    if [ "$role" = "prefill" ]; then
        if [ "$in_prefill" -ne 1 ]; then
            echo "Error: --role prefill given but local IP ${local_host} not in PREFILL_IPS=(${PREFILL_IPS[*]})."
            echo "Hint: role only needs to be passed when PREFILL_IPS and DECODE_IPS overlap (single-server PD); otherwise omit it and let it be inferred from the local IP."
            exit 1
        fi
        export PD_ROLE="prefill"
    elif [ "$role" = "decode" ]; then
        if [ "$in_decode" -ne 1 ]; then
            echo "Error: --role decode given but local IP ${local_host} not in DECODE_IPS=(${DECODE_IPS[*]})."
            echo "Hint: role only needs to be passed when PREFILL_IPS and DECODE_IPS overlap (single-server PD); otherwise omit it and let it be inferred from the local IP."
            exit 1
        fi
        export PD_ROLE="decode"
    elif [ "$in_prefill" -eq 1 ] && [ "$in_decode" -eq 1 ]; then
        echo "Error: local IP ${local_host} appears in both PREFILL_IPS and DECODE_IPS (single-server PD); cannot infer PD_ROLE."
        echo "Please pass 'prefill' or 'decode' explicitly."
        exit 1
    elif [ "$in_prefill" -eq 1 ]; then
        echo "Inferred PD_ROLE=prefill from local IP ${local_host}"
        export PD_ROLE="prefill"
    elif [ "$in_decode" -eq 1 ]; then
        echo "Inferred PD_ROLE=decode from local IP ${local_host}"
        export PD_ROLE="decode"
    else
        echo "Error: local IP ${local_host} not found in PREFILL_IPS or DECODE_IPS."
        echo "Please pass 'prefill' or 'decode' explicitly."
        exit 1
    fi

    # Single-server PD: prefill and decode share the host, so the two roles
    # must be pinned to disjoint NPU sets — otherwise both grab NPU 0 and OOM.
    if [ "$in_prefill" -eq 1 ] && [ "$in_decode" -eq 1 ]; then
        if [ -z "${ASCEND_RT_VISIBLE_DEVICES+x}" ]; then
            echo "Error: single-server PD (local IP ${local_host} is in both PREFILL_IPS and DECODE_IPS) requires ASCEND_RT_VISIBLE_DEVICES to be set explicitly."
            echo "Hint: split the NPUs between roles, e.g. ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 for prefill and =4,5,6,7 for decode."
            exit 1
        fi
    fi
}

function get_rank()
{
    mode=$1
    if [ "$mode" = "online" ]; then
        resolve_pd_role "${PD_ROLE}"
        if [ "${PD_ROLE}" = "prefill" ]; then
            export YAML="${P_YAML}"
            IPs=("${PREFILL_IPS[@]}")
        else
            export YAML="${D_YAML}"
            IPs=("${DECODE_IPS[@]}")
        fi
    fi
    filename=$(basename "$YAML")
    world_size=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['world_size'])")
    platform_version=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['model_config'].get('platform_version'))")
    if [ "$platform_version" = "950" ]; then
        chip_num=8
    else
        chip_num=16
    fi
    offset=$(expr $chip_num - 1)
    if [ -n "$world_size" ]; then
        export WORLD_SIZE=$world_size
        echo "world_size is: $WORLD_SIZE"
        SERVER_NUM=$(((WORLD_SIZE + $offset) / $chip_num ))
        echo "server_num is: $SERVER_NUM"

        if [ "$SERVER_NUM" -eq 1 ]; then
            LOCAL_HOST=$(hostname -I|awk -F " " '{print$1}')
            export IPs=($LOCAL_HOST)
        else
            export IPs=(${IPs[@]:0:$SERVER_NUM})
        fi
    else
        echo "Cannot find world_size in '$filename'"
        exit 1
    fi
}

function check_env_vars()
{
    mode=$1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`       # Obtain current server's IP
    if [[ ${ON_CLOUD} -eq 0 ]]; then
        export HCCL_SOCKET_IFNAME=enp                     # Socket prefix, to obtain Host IP for HCCL and HCCL group; modify to enp/eth accordingly
        MA_NUM_HOSTS=${#IPs[@]}                           # Number of servers
        export MASTER_ADDR=${IPs[0]}                      # IP of the master server
        export MASTER_PORT=6038                           # Port of the master server
        VC_TASK_INDEX=0                                   # Task index of the current server
        # obtain the task index of each server
        for i in "${!IPs[@]}";
        do
            echo "LOCAL_HOST=${LOCAL_HOST}, IPs[${i}]=${IPs[$i]}"
            if [ "$LOCAL_HOST" == "${IPs[$i]}" ]; then
                echo "Node Rank : ${i}"
                VC_TASK_INDEX=$i
                break
            fi
        done
    else
        echo "Python version >>>" `python3 -V`
        export HCCL_SOCKET_IFNAME=eth0
        export MASTER_ADDR=`echo $VC_WORKER_HOSTS|awk -F "," '{print $1}'`
        export MASTER_PORT=6138                            # Port of the master server
    fi
    echo "VC_TASK_INDEX >>>" $VC_TASK_INDEX
    export MA_NUM_GPUS=$chip_num  # Number of devices on each server. Should be the same for each server
    if [ "$WORLD_SIZE" -lt "$MA_NUM_GPUS" ]; then
        MA_NUM_GPUS=$WORLD_SIZE
    fi
    export LOCAL_WORLD_SIZE=${MA_NUM_GPUS}
    export RANK_OFFSET=`expr $VC_TASK_INDEX \* ${MA_NUM_GPUS}`

    # check world size
    if [ $MA_NUM_HOSTS ]; then
        export DEVICE_SIZE=$(($MA_NUM_GPUS*$MA_NUM_HOSTS))  # Total number of devices across all servers
        if [ ${DEVICE_SIZE} -ge  ${WORLD_SIZE} ]; then
            echo "[INFO] total ranks is ${DEVICE_SIZE}, and use ${WORLD_SIZE} ranks in actual!"
        else
            echo "[ERROR] total ranks is ${DEVICE_SIZE}, but use ${WORLD_SIZE} ranks in actual!"
            exit 1
        fi
    fi

    DATE=`date +%Y%m%d`
    # set result path
    DIR_PREFIX="res"
    MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['model_name'])")
    PREFIX=$(basename "$YAML")
    PREFIX="${PREFIX%.*}"
    NAME=${MODEL_NAME}_${PREFIX}
    export CASE_NAME=$NAME

    if [[ ${ON_CLOUD} -eq 0 ]]; then
        if [ "$mode" = "online" ]; then
            export RES_PATH="${DIR_PREFIX}/${DATE}/${NAME}/${PD_ROLE}_node${VC_TASK_INDEX}"
        else
            export RES_PATH="${DIR_PREFIX}/${DATE}/${NAME}"
        fi
        WORK_DIR=`pwd`
        DUMP_PRECISION_PATH=${WORK_DIR}'/'${RES_PATH}'/dump_data'
        mkdir -p ${WORK_DIR}'/'${RES_PATH}
        mkdir -p ${DUMP_PRECISION_PATH}
    else
        export RES_PATH="${DIR_PREFIX}/${DATE}/${NAME}/${VC_TASK_INDEX}"
        WORK_DIR='/home/ma-user/modelarts/outputs/train_url_0'
        DUMP_PRECISION_PATH=${WORK_DIR}'/'${RES_PATH}'/dump_data'
        mkdir -p ${DUMP_PRECISION_PATH}
    fi

    SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    PARENT_PARENT_DIR=$(cd "$SCRIPT_PATH/../.." &>/dev/null && pwd)
    echo "==================================>"
}

function set_hccl()
{
    export HCCL_IF_IP=$LOCAL_HOST
    export HCCL_IF_BASE_PORT=23456

    micro_batch_mode=$(python3 -c "import yaml; \
        print(yaml.safe_load(open('$YAML')).get('model_config').get('micro_batch_mode', 0))")

    # 910c needs enable HCCL aiv
    if [ "$platform_version" != "950" ]; then
         export HCCL_OP_EXPANSION_MODE=AIV
    fi

    if [[ ${micro_batch_mode} -eq 1 ]]; then
        unset HCCL_OP_EXPANSION_MODE
    fi

    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=1200
}


function launch_infer_task()
{
    mode=$1
    if [ "$mode" = "online" ]; then
        SERVER_PATH=${PARENT_PARENT_DIR}/executor/online/server.py
        if [ "${PD_ROLE}" = "decode" ]; then
            export MASTER_PORT=6239
            export HCCL_IF_BASE_PORT=23556
        fi

        # Router: prefill node 0 only. Reads both YAMLs to compute leader strides.
        if [ "${PD_ROLE}" = "prefill" ] && [ "${VC_TASK_INDEX}" = "0" ]; then
            P_WS=$(python3 -c "import yaml; print(yaml.safe_load(open('${P_YAML}'))['parallel_config']['world_size'])")
            D_WS=$(python3 -c "import yaml; print(yaml.safe_load(open('${D_YAML}'))['parallel_config']['world_size'])")
            P_STEP=$(( P_WS / MA_NUM_GPUS )); [ "$P_STEP" -lt 1 ] && P_STEP=1
            prefill_leaders=()
            for ((i=0; i<${#PREFILL_IPS[@]}; i+=P_STEP)); do
                prefill_leaders+=("${PREFILL_IPS[$i]}")
            done
            D_STEP=$(( D_WS / MA_NUM_GPUS )); [ "$D_STEP" -lt 1 ] && D_STEP=1
            decode_leaders=()
            for ((i=0; i<${#DECODE_IPS[@]}; i+=D_STEP)); do
                decode_leaders+=("${DECODE_IPS[$i]}")
            done

            python3 "${SERVER_PATH}" \
                --role router \
                --prefill-addrs "${prefill_leaders[@]}" \
                --decode-addrs "${decode_leaders[@]}" \
                2>&1 > "${WORK_DIR}/${RES_PATH}/log_router.log" &
        fi

        # Both roles receive PREFILL_IPS and DECODE_IPS (flat cross-instance)
        # so server.py can derive the memfabric store host (PREFILL_IPS[0])
        # symmetrically on prefill and decode without per-role CLI or env plumbing.
        server_cmd=(
            python3 "${SERVER_PATH}"
            --role "${PD_ROLE}"
            --yaml-file-path "${YAML}"
            --node-index "${VC_TASK_INDEX}"
            --devices-per-node "${MA_NUM_GPUS}"
            --ips "${IPs[@]}"
            --prefill-ips "${PREFILL_IPS[@]}"
            --decode-ips "${DECODE_IPS[@]}"
        )

        "${server_cmd[@]}" 2>&1 > "${WORK_DIR}/${RES_PATH}/log_server.log" &
        return
    fi

    # Offline mode
    INFER_PATH=${PARENT_PARENT_DIR}/models/${MODEL_DIR}/infer.py
    if [ ! -f "${INFER_PATH}" ]; then
        INFER_PATH="${PARENT_PARENT_DIR}/executor/offline/infer.py"
    fi
    EXTRA_ARGS=()

    cores=`cat /proc/cpuinfo|grep "processor" |wc -l`
    avg_core_per_rank=`expr $cores \/ $MA_NUM_GPUS`
    core_gap=`expr $avg_core_per_rank \- 1`
    for((i=0; i<${MA_NUM_GPUS}; i++))
    do
        echo $i
        start=`expr $i \* $avg_core_per_rank`
        end=`expr $start \+ $core_gap`
        cmdopt=$start"-"$end
        export LOCAL_RANK=$i
        export RANK_ID=$(expr $i + $RANK_OFFSET)
        cmd=(taskset -c "$cmdopt" python3 "${INFER_PATH}" --yaml_file_path="${YAML}" "${EXTRA_ARGS[@]}")
        if [ $i -eq 0 ] && [[ $LAUNCH_MODE -ne 1 ]];then
            "${cmd[@]}" 2>&1 | tee ${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log &
        else
            "${cmd[@]}" &> ${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log &
        fi
    done
    wait
}

function check_result()
{
    file=${WORK_DIR}/${RES_PATH}/log_0.log
    echo "check" $file

    if [ ! -f "$file" ];then
        echo "ERROR: log "$file" not exist."
        exit 1
    fi
    result_str=$(grep "Inference decode result: " $file -A 1 2>/dev/null)
    if [ -n "$result_str" ];then
        echo "CASE" ${CASE_NAME} "inference result is " $result_str
    fi
    error_str=$(grep "ERROR" $file 2>/dev/null)
    if [ -n "$error_str" ];then
        echo "CASE" ${CASE_NAME} "found ERROR, plz check."
    fi
}

function save_key_info()
{
    if [ ${ON_CLOUD} -eq 1 ]; then
        mv ./extra-info ${WORK_DIR}/extra-info_${VC_TASK_INDEX}
        mv /root/ascend/atrace ${WORK_DIR}/atrace_${VC_TASK_INDEX}
    fi
    last_worker_index=`expr $MA_NUM_HOSTS \- 1`
    if [ ${ON_CLOUD} -eq 1 ] && [ ${VC_TASK_INDEX} -eq ${last_worker_index} ]; then
        echo "===================start to save key infos"
        cur_dir=`pwd`
        key_info_dir=${WORK_DIR}/info/

        cann_info_dir=${key_info_dir}/cann/
        log_info_dir=${key_info_dir}/log/
        prof_info_dir=${key_info_dir}/prof/
        dump_info_dir=${key_info_dir}/dump/
        code_info_dir=${key_info_dir}/code/

        mkdir -p ${cann_info_dir}
        mkdir -p ${log_info_dir}
        mkdir -p ${prof_info_dir}
        mkdir -p ${dump_info_dir}
        mkdir -p ${code_info_dir}

        cp -r ${cur_dir}/../../../../ma-pre-start.sh ${cann_info_dir}/
        cat /usr/local/Ascend/CANN*/*/version.info |grep timestamp > ${cann_info_dir}/timestamp.txt
        pip3 show torch_npu >> ${cann_info_dir}/timestamp.txt
        cp ${cur_dir}/../config/output.yaml ${key_info_dir}/
        cp ${WORK_DIR}/${RES_PATH}/log_*.log ${log_info_dir}/
        cp -r ${PROFILING_PATH} ${prof_info_dir}/
        cp -r ${DUMP_PRECISION_PATH} ${dump_info_dir}/
        cp -r ${cur_dir}/../../../../inference/ ${code_info_dir}/
        rm -rf ${code_info_dir}/inference/moe/deepseek/scripts/models/DeepseekV2ForCausalLM*
    fi
}
