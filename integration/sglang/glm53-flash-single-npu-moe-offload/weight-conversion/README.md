# 权重转换

厂商发布的 GLM-5.3-Flash 是 FP8 E4M3（`weight_block_size=[128,128]`）。
这条部署需要两份派生权重，INT8 W8A8 用于 die 上的计算，MXFP4 用于主机侧的路由专家。

| 文件 | 来源 | 作用 |
|---|---|---|
| `pipeline_fp8_to_int8.py` | cann-recipes-infer | FP8 直接转 INT8 W8A8，不落 BF16 中间态 |
| `fp8_to_mxfp4_moe.py` | cann-recipes-infer | FP8 转 MXFP4 |
| `tools/fp8_to_bf16.py` | sglang，Apache-2.0 | FP8 反量化到 BF16，被上面第一个脚本按张量调用 |
| `tools/bf16_to_int8_ct.py` | sglang，Apache-2.0 | BF16 量化到 compressed-tensors W8A8 |

`tools/` 下两个文件取自 `sgl-project/sglang` 的
`docs/docs/glm53_npu_support/tools/`，提交 `97c69783`。算法主体保持不变；只补充了本仓要求的
版权与 Apache-2.0 许可证头，并把示例中的固定路径改为占位路径。
该路径在 `cann-recipes-infer` 声明的 sglang 基线 `5aab054e` 上已经不存在，上游把 NPU 文档
重组到了 `docs/docs/hardware-platforms/ascend-npus/`，所以随 `cann-recipes-infer` 一起提供。
许可证见同级配方目录的 [`LICENSE.txt`](../LICENSE.txt)。

`pipeline_fp8_to_int8.py` 按 `Path(__file__).parent / "tools"` 定位这两个文件，
所以这个目录结构不能改。

用法见[部署指南第 4 步](../README.md#4-权重)。
