import os
import argparse

import torch
import torch_npu
import torch.nn.functional as F

from offline_profiling import get_cumulative_coverage_of_different_sparsity, \
                        get_sparsity_of_target_cumulative_coverage, save_expected_sparsity


def parse_args():
    parser = argparse.ArgumentParser(
        description="offline profiling配置"
    )

    parser.add_argument(
        "--qk_dir_path",
        type=str,
        help="存放qk的文件路径"
    )

    parser.add_argument(
        "--target_dir_path",
        type=str,
        help="保存稀疏度的文件路径"
    )

    parser.add_argument(
        "--global_layer_num",
        type=int,
        default=60,
        help="全局层数"
    )

    # 浮点参数
    parser.add_argument(
        "--head_num",
        type=int,
        default=24,
        help="head数"
    )

    # 布尔开关参数
    parser.add_argument(
        "--target_coverage",
        type=float,
        default=0.95,
        help="目标CAC"
    )

    parser.add_argument(
        "--step_start",
        type=int,
        default=15,
        help="开始稀疏的step层数"
    )

    parser.add_argument(
        "--step_end",
        type=int,
        default=50,
        help="开始稀疏的step层数"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="设备id"
    )

    return parser.parse_args()

args = parse_args()
scene_list = []
step_start = args.step_start
step_end = args.step_end
for idx in range(step_start, step_end):
    scene_list.append(f"step-{idx}")

global_layer_num = args.global_layer_num
head_num = args.head_num
target_coverage = args.target_coverage
sparsity_list = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
device = f"npu:{args.device}"

for scene in scene_list:
    dir_path = os.path.join(args.qk_dir_path, scene)
    target_dir_path = os.path.join(args.target_dir_path, scene)
    all_layer_cumulative_coverage = get_cumulative_coverage_of_different_sparsity(dir_path, global_layer_num,
                                        sparsity_list, device=device)
    sparsity_of_target_coverage = get_sparsity_of_target_cumulative_coverage(all_layer_cumulative_coverage,
                                    global_layer_num, head_num, sparsity_list, target_coverage, device=device)
    save_expected_sparsity(target_dir_path, sparsity_of_target_coverage, target_coverage=target_coverage)