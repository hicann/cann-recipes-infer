# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Compare both SiTU custom operators with an independent CPU PyTorch golden.

Run on Ascend 950 after installing both custom operators:
    python3 test_situ_custom_ops.py

The sparse operator is compared directly. The fused operator's E4M3 data is
dequantized with its E8M0 scales and compared with the same SiTU golden, allowing
the additional MXFP8 rounding/saturation error. No custom or torch_npu operator
is used to calculate the golden or the error budget. Unwritten sparse tail rows
are excluded from both comparisons. Any failed comparison exits nonzero.
"""

import torch

EPR = 3072  # moe_intermediate_size
GROUP_SIZE = 32
# Allow dtype rounding and FP32 exp/tanh implementation differences. FP8 error
# is accounted for separately below, rather than loosening the SiTU tolerance.
SITU_TOLERANCES = {
    torch.bfloat16: (1e-2, 1e-4),
    torch.float16: (2e-3, 1e-5),
}


def situ_golden(x, beta, alpha, high_precision):
    """SiTU(gate) * bounded_up, using native PyTorch on CPU only.

    high_precision=False rounds each branch to the input dtype before the
    product; True keeps both branches in FP32. Both modes round the output.
    """
    if x.device.type != "cpu":
        raise ValueError("SiTU golden must run on CPU")
    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = alpha * torch.tanh(up / alpha)
    if not high_precision:
        gate = gate.to(x.dtype).float()
        up = up.to(x.dtype).float()
    return (gate * up).to(x.dtype).float()


def mxfp8_error_budget(golden):
    """E4M3 error bounds derived from golden, never from the operator output.

    OCP MX uses scale=2**(floor(log2(group_amax))-8), clamped to E8M0.
    Normal rounding is bounded by |x|/16, subnormal rounding by scale/1024.
    Values exceeding 448*scale saturate and need a separate clipping bound.
    Tests use finite, normal-range SiTU inputs (plus exact zero).
    """
    groups = golden.reshape(golden.shape[0], EPR // GROUP_SIZE, GROUP_SIZE)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    exponent = (torch.floor(torch.log2(amax.clamp_min(2.0**-126))) - 8).clamp(-127, 127)
    scale = torch.exp2(exponent)
    rounding = torch.maximum(groups.abs() / 16, scale / 1024)
    clipping = (groups.abs() - 448 * scale).clamp_min(0)
    return torch.maximum(rounding, clipping).reshape_as(golden)


def dequantize_mxfp8(y, scale, valid):
    """Decode E8M0 bytes on CPU; code 255 is NaN, code 0 is 2**-127."""
    if y.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
        raise AssertionError(f"Unexpected quantized dtypes: {y.dtype}, {scale.dtype}")
    if y.shape[1:] != (EPR,) or scale.shape != (y.shape[0], EPR // 64, 2):
        raise AssertionError(f"Unexpected quantized shapes: {y.shape}, {scale.shape}")
    codes = scale[:valid].view(torch.uint8).cpu().reshape(valid, EPR // GROUP_SIZE)
    if (codes == 255).any():
        raise AssertionError("NaN E8M0 scale for finite input")
    scales = torch.exp2(codes.float() - 127)
    values = y[:valid].view(torch.uint8).cpu().view(torch.float8_e4m3fn).float()
    return values * scales.repeat_interleave(GROUP_SIZE, dim=-1)


def compare(name, actual, golden, budget):
    if actual.shape != golden.shape or golden.numel() == 0:
        raise AssertionError(f"Invalid comparison shapes: {actual.shape}, {golden.shape}")
    error = (actual - golden).abs()
    bad = ~torch.isfinite(actual) | ~torch.isfinite(golden) | (error > budget)
    ok = not bad.any().item()
    rmse = (actual - golden).square().mean().sqrt().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} vs torch golden: "
          f"max_abs={error.max().item():.6g} rmse={rmse:.6g} "
          f"out_of_tolerance={bad.sum().item()}/{golden.numel()}")
    return ok


def run_case(name, rows, dtype, beta, alpha, high_precision, expert_tokens_list,
             pattern="random"):
    x_cpu = torch.randn(rows, 2 * EPR, dtype=dtype)
    if pattern == "zeros":
        x_cpu.zero_()
    elif pattern == "boundary":
        # All gate/up sign combinations, zero, near-zero and saturating inputs.
        pairs = [(1e4, 1e4), (-1e4, -1e4), (1e4, -1e4), (-1e4, 1e4),
                 (0.0, 0.0), (1e-3, -1e-3), (beta, alpha), (-beta, alpha)]
        for row, (gate, up) in enumerate(pairs):
            x_cpu[row, :EPR] = gate
            x_cpu[row, EPR:] = up
    valid = min(sum(expert_tokens_list), rows)
    if valid <= 0 or any(count < 0 for count in expert_tokens_list):
        raise ValueError("Precision cases need nonnegative counts and at least one valid row")
    print(f"{name}: rows={rows} dtype={dtype} beta={beta} alpha={alpha} "
          f"high_precision={high_precision} valid={valid}", flush=True)
    golden = situ_golden(x_cpu[:valid], beta, alpha, high_precision)
    x = x_cpu.npu()
    expert_tokens = torch.tensor(expert_tokens_list, dtype=torch.int64).npu()
    act = torch.ops.custom.npu_situ_and_mul_sparse(
        x, expert_tokens, beta=beta, alpha=alpha, high_precision=high_precision)
    y, scale = torch.ops.custom.grouped_situ_mx_quant(
        x, expert_tokens, beta=beta, alpha=alpha, high_precision=high_precision)
    torch.npu.synchronize()
    if act.shape != (rows, EPR) or act.dtype != dtype or y.shape != (rows, EPR):
        raise AssertionError("Unexpected SiTU output shape or dtype")

    rtol, atol = SITU_TOLERANCES[dtype]
    situ_budget = atol + rtol * golden.abs()
    sparse_ok = compare("npu_situ_and_mul_sparse", act[:valid].float().cpu(),
                        golden, situ_budget)
    fused_ok = compare("grouped_situ_mx_quant (dequantized)",
                       dequantize_mxfp8(y, scale, valid), golden,
                       situ_budget + mxfp8_error_budget(golden))
    return sparse_ok and fused_ok


def main():
    import torch_npu  # noqa: F401 - registers the NPU backend
    import custom_ops  # noqa: F401 - registers torch.ops.custom.*

    torch.manual_seed(0)
    cases = [
        dict(name="dense", rows=128, beta=1.0, alpha=1.0, expert_tokens_list=[128]),
        dict(name="parameters", rows=64, beta=2.5, alpha=0.7, expert_tokens_list=[64]),
        dict(name="kimi_k3", rows=128, beta=4.0, alpha=25.0, expert_tokens_list=[128]),
        dict(name="sparse", rows=128, beta=4.0, alpha=25.0,
             expert_tokens_list=[30, 0, 40, 20]),
        dict(name="tail", rows=33, beta=1.0, alpha=1.0, expert_tokens_list=[0, 1, 16]),
        dict(name="zeros", rows=1, beta=4.0, alpha=25.0,
             expert_tokens_list=[1], pattern="zeros"),
        dict(name="boundary", rows=8, beta=4.0, alpha=25.0,
             expert_tokens_list=[8], pattern="boundary"),
    ]
    ok = True
    for dtype in (torch.bfloat16, torch.float16):
        for hp in (False, True):
            for case in cases:
                ok &= run_case(dtype=dtype, high_precision=hp, **case)
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
