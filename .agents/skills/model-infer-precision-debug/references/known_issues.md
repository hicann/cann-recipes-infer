# 已知精度问题

尚无对应专题 reference 的历史排查沉淀，进入定位流程前先对照一轮。命中即按修复列处理，再按主流程 1.3 的折返路径验证。某一域的条目积累到可独立成篇时，迁到该域的 reference。

数值为当时实测的量级参考，未记录硬件与版本条件，只作现象比对用，不作判据。

| 问题 | 症状 | 根因 | 修复 |
|------|------|------|------|
| **MoE 专家权重未加载** | MoE 输出全零或接近零，cosine~0.4-0.8 | Checkpoint 用 packed tensor 但 load_weights 用 per-expert mapping，权重未正确拷贝 | 添加 packed expert weight loader，按 EP rank 切片后 copy_ 到参数 |
| **Zero expert 处理错误** | MoE 输出偏差或切换路由算子后精度骤降至 ~40-50% | 零专家 identity 贡献公式错（应用 original_input 而非 MoE 输出），或路由算子 v1 无 `active_expert_range` 导致零专家 ID 污染 token 排列 | 路由前用 mask 分离零专家，路由后加回 `original_input * weight` 的 identity 贡献 |
| **DP+EP token 不同步 → NaN** | Decode 若干步后出现 NaN，之后全部 NaN | EP all_to_all 有浮点非确定性，DP 各 rank token 选择分叉 → KV cache 累积偏差 | 每步 Decode 后 `dist.broadcast(next_token, src=0)` 强制所有 DP rank 使用相同 token |
| **跳过训练组件的 scale 缺失** | 输出语义正确但严重重复/退化 | 模型训练时有额外组件，跳过后 embedding 幅度不匹配 | 保留训练时的 normalization factor |
| **W8A8 精度从某层开始放大** | 逐层偏差累积 | dequant_scale 与权重不匹配 | 检查 per-channel/per-tensor scale |
| **Mamba cache 索引回零** | sglang 集成路径下，用 Mamba 状态缓存的模型精度下降 | 内部索引 `mamba_cache_indices` 为 0 时产生错误结果，该索引在 cache 容量内循环，容量默认值通常为 2，因此频繁回零 | 调大启动参数 `max-mamba-cache-size`，可取 512。属降低触发概率不是根治，并带来性能开销 |
