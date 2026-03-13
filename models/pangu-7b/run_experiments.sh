#!/bin/bash

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd "${SCRIPT_PATH}/../.." &>/dev/null && pwd)

# --- Configuration ---
YAML_PATH="${SCRIPT_PATH}/config/openpangu_v5_7b_mxfp8.yaml"
CMD_DIR="${SCRIPT_PATH}"
BATCH_SIZES=(1 2 32 64)
LOG_FILE="${SCRIPT_PATH}/experiment_results_$(date +%Y%m%d_%H%M%S).log"

echo "Starting experiments at $(date)" | tee -a "$LOG_FILE"
echo "[INFO] script dir: ${SCRIPT_PATH}" | tee -a "$LOG_FILE"
echo "[INFO] repo root: ${REPO_ROOT}" | tee -a "$LOG_FILE"

# --- Loop through Batch Sizes ---
for BS in "${BATCH_SIZES[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "----------------------------------------------------" | tee -a "$LOG_FILE"
    echo "[$TIMESTAMP] Testing Batch Size: $BS" | tee -a "$LOG_FILE"
    echo "----------------------------------------------------" | tee -a "$LOG_FILE"

    # 1. Update the YAML file using sed
    # Matches 'batch_size: [number]' and replaces it with the current $BS
    echo "[INFO] current dir: $(pwd)" | tee -a "$LOG_FILE"
    sed -i "s/batch_size: [0-9]*/batch_size: $BS/" "$YAML_PATH"

    # 2. Run the inference script
    # Capture both stdout and stderr into the log file
    # We use a subshell to prefix the specific BS output for easier searching later
    {
        echo "--- START OUTPUT FOR BS=$BS ---"
        cd "$CMD_DIR" || exit 1
        bash ./infer.sh
        echo "--- END OUTPUT FOR BS=$BS ---"
        echo -e "\n"
    } >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$BS] Finished successfully." | tee -a "$LOG_FILE"
    else
        echo "[$BS] Failed. Check log for details." | tee -a "$LOG_FILE"
    fi
done

echo "All experiments completed. Logs saved to: $LOG_FILE"
