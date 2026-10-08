#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

at::Tensor sparse_output(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() > 0 && x.size(-1) > 0 && x.size(-1) % 2 == 0,
                "x must have a positive even last dimension");
    auto shape = x.sizes().vec();
    shape.back() /= 2;
    return at::empty(shape, x.options());
}

at::Tensor npu_situ_and_mul_sparse_npu(
    const at::Tensor& x, const at::Tensor& expert_tokens,
    double beta = 1.0, double alpha = 1.0, bool high_precision = false)
{
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be FLOAT16 or BFLOAT16");
    TORCH_CHECK(expert_tokens.scalar_type() == at::kLong,
                "expert_tokens must be INT64");
    TORCH_CHECK(beta > 0.0 && alpha > 0.0, "beta and alpha must be positive");
    auto y = sparse_output(x);
    EXEC_NPU_CMD_V1(aclnnSituAndMulSparse, x, expert_tokens, beta, alpha, high_precision, y);
    return y;
}

at::Tensor npu_situ_and_mul_sparse_meta(
    const at::Tensor& x, const at::Tensor& expert_tokens,
    double beta = 1.0, double alpha = 1.0, bool high_precision = false)
{
    (void)expert_tokens; (void)beta; (void)alpha; (void)high_precision;
    return sparse_output(x);
}
}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_situ_and_mul_sparse", &custom::npu_situ_and_mul_sparse_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_situ_and_mul_sparse", &custom::npu_situ_and_mul_sparse_meta);
}
