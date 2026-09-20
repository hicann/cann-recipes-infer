import importlib.util

_GOLDEN_PATH = (
    "/home/wx00445270/cannbot-skills/kv_compress_epilog_v2/test/"
    "main_validation/kv_compress_epilog_v2_golden.py"
)
_SPEC = importlib.util.spec_from_file_location("kcev2_main_golden", _GOLDEN_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)


# TTK 按算子属性名（camelCase，与 ACLNN 接口一致）以关键字方式传参，
# 属性经 **kwargs 透传；blockStride 由 4D cache 推导，golden 不消费。
_QUANT_DEFAULTS = {
    "quantGroupSize": 32,
    "quantMode": 2,
    "roundScale": True,
    "xScale": 1.0,
}


def kv_compress_epilog_v2_aclnn(cache_ref, x, slot_mapping, **kwargs):
    for name, default in _QUANT_DEFAULTS.items():
        kwargs.setdefault(name, default)
    kwargs.pop("blockStride", None)
    return _BASE.kv_compress_epilog_v2_aclnn(cache_ref, x, slot_mapping, **kwargs)


# TTK golden 框架按模块属性 __golden__ 发现入口，经 globals() 注册以规避双下划线命名告警。
globals()["__golden__"] = {
    "aclnn": {
        "aclnnKvCompressEpilogV2": "kv_compress_epilog_v2_aclnn",
    },
}
