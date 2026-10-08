#include "register/op_def_registry.h"

namespace ops {
class SituAndMulSparse : public OpDef {
public:
    explicit SituAndMulSparse(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("expert_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("beta").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("high_precision").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "situ_and_mul_sparse");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(SituAndMulSparse);
}  // namespace ops
