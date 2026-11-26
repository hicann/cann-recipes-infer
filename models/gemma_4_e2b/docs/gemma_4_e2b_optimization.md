# Gemma-4-E2B 语言路径适配

## 配置与逐层输入

独立注册gemma_4_e2b，从checkpoint嵌套text_config读取全部结构尺寸。当前权重35层、hidden1536、head8/KV1、滑窗head256/full head512、window512、逐层输入维度256。非法维度、层排列、共享来源、激活、量化和不支持的bidirectional attention会失败。

主embedding乘sqrt(hidden_size)，专用逐层embedding乘sqrt(per_layer_input_dim)。主隐藏状态经per_layer_model_projection投影为各层输入，乘hidden_size^-0.5，再按per_layer_input_dim做RMSNorm；与专用embedding按2^-0.5融合。每个Decoder用自身gate/projection/归一化将逐层输入加回残差，保持官方计算顺序。

RMSNorm在FP32求均方与scale后转回原dtype，使用直接weight而非1+weight。MLP采用GELU(tanh)门控：前15层6144、共享区20层12288。每层layer_scalar必须从checkpoint加载，最后输出头与embedding绑定，应用final_logit_softcapping=30。

RoPE 频率表在初始化时预计算，以避免图模式 Decode 出现 `.item()` 或动态构图。频率积使用与 Transformers 5.5 官方实现相同的 batched matmul；不要改写为数学等价的 `torch.outer`，NPU BF16 实测会在特定位置的 `sin` 上产生一个量化级差异，并可能翻转接近并列的输出候选。full attention 仅计算 `partial_rotary_factor` 覆盖的通道，再将未旋转通道原样拼回，以保留紧凑表和图模式性能。

短 Prefill 使用与参考端相同的 BF16 QK matmul、FP32 softmax 和 BF16 PV matmul。A2 实测中，fused attention 从第 0 层产生一个 BF16 量化级偏差，在候选分数仅差 0.125 时会改变 greedy token；显式路径使 10 个固定样本的 Prefill logits 逐元素一致。该路径只处理不超过 64 token 的输入，避免长上下文的二次方 attention 矩阵扩大临时空间；128/512/1024/2048 性能矩阵继续使用 fused attention，Decode 路径保持不变。

短 Prefill 的显式掩码同时屏蔽未来位置和滑窗之外的历史位置。sliding 层仅允许 `0 <= query_position - key_position < sliding_window`，full 层只应用因果约束；小于 64 的滑窗配置同样遵循此约束。

## KV来源与物理所有权

来源从layer_types与num_kv_shared_layers推导，不能将当前13/14硬编码到通用配置算法中。前15层各自生成并写入K/V；后20层使用自身Q，只读取最后一个同类型来源层。当前sliding消费者来源13、full消费者来源14。

框架要求每个逻辑层有非空CacheEntry：15源层使用needs_block=True（共30个K/V物理条目），20消费者保留needs_block=False描述。来源tensor_setter同时将消费者模块k_cache/v_cache及entry.tensor绑定到同一存储。所有逻辑层与实际attention类型保持一致，预算不重复计入消费者，框架single-type managers唯一管理页块及释放。

Prefill使用本次forward完整来源K/V，避免长度大于window后来源仅保留滑窗尾部导致消费者丢失本次上下文；仅需要共享的源13/14临时保留引用，forward结束即释放。Decode从统一执行器slot_mapping/block_table读取历史页；消费者不执行scatter写缓存。滑窗FIA sparse_mode4、pre_tokens=window-1；full Prefill因果路径与Decode分页路径分别使用已验证布局。

每次请求结束，统一Scheduler调用KVCacheManager.free再触发完成回调。验证脚本只观察这一回调，断言该请求所有owner blocks为空、各manager free-block数恢复启动预算，不另建调度或释放逻辑。20条不同内容连续请求与新进程基线逐token一致。

## 权重审计及初始化

严格检查语言weight/buffer名称、shape、重复/绑定参数一致性；缺失必要语言项直接失败。只按明确视觉/音频前缀跳过范围外键，审计保存每个键及处置；不得笼统strict=False。

CheckpointLinear与ScaledEmbedding跳过随后必被checkpoint覆盖的随机初始化。只有完整加载成功才设置weights_ready=True；加载缺失或未知语言键后不能forward。模型RoPE的必要非持久buffer按配置构造，layer_scalar等checkpoint buffer仍严格加载。

## 执行模式与范围

构造、forward、load_weights、get_cache_info沿用统一契约。合法范围为BF16、单逻辑设备/1TP、batch1、Offline、eager/npugraph_ex；多模态、量化、MTP、Online及变长batch不支持。非法长度或上下文超限在参数分配前拒绝。自然语言由官方tokenizer编码，超过YAML输入上限按执行器截断。

base官方tokenizer没有chat template时，适配器仅还原单user原始文本，特殊token/BOS由原tokenizer负责；多轮、多模态或thinking请求报错。没有instruction prompt注入。官方参考使用相同原始completion协议。

公共改动仅support_models与tokenizer_registry新增lazy映射；全部原注册项与基线静态比较不变。没有修改旧MoE模型、公共缓存/调度、算子或推理执行器。模型按 A2 任务书要求使用独立 `models/gemma_4_e2b/` 目录和独立注册名。

开发实测、已知差异及正式门禁见[自验报告](gemma_4_e2b_validation.md)。
