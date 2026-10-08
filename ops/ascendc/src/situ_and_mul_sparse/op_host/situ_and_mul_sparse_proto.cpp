#include "register/op_impl_registry.h"
#include "error/ops_error.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    gert::Shape* y = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, x, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, y, return ge::GRAPH_FAILED);
    *y = *x;
    const int64_t rank = y->GetDimNum();
    if (rank > 0) {
        y->SetDim(rank - 1, y->GetDim(rank - 1) / 2);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SituAndMulSparse).InferShape(InferShape).InferDataType(InferDtype);
}  // namespace ops
