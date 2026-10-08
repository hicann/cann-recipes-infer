# D2D 功能用例

`test_rank1.cpp` 和 `test_rank2.cpp` 共用 `functional_cases.h`，分别覆盖单 PE 和双 PE。
两个程序都编译 `unsigned int / long long` 两种 Key，Value 为 `size_t`（64 位）。
`D2D_KEY_BITS=32` 或 `64` 选择运行类型；两个运行脚本默认分两次进程运行两种类型，任一失败最终返回非零。

在算子目录仅编译，不运行：

```bash
bash test/build.sh d2d_rank1 d2d_rank2
```

运行入口：

```bash
bash test/d2d/run_rank1.sh
bash test/d2d/run_rank2.sh
# 仅运行 64 位 Key：
D2D_KEY_BITS=64 bash test/d2d/run_rank2.sh
```

双卡日志为 `test/build/d2d_rank2_logs/key32_rank0.log` 等四个文件，保留原有设备、地址和超时环境变量。

## 覆盖范围

| 用例标签 | 输入和检查 | 分支目标 |
|---|---|---|
| `empty_table_query` | Init 后直接查询，全部返回未命中标记 | 本地/远端查询遇空桶退出 |
| `hash_collision` | 同一 PE 的 3 个不同 key 哈希到桶 7，逐轮插入；追加同桶缺失 key | 空桶 CAS、冲突重试、碰撞链命中和未命中 |
| `probe_wraparound` | 同一 PE 的 3 个 key 哈希到实际表最后一个桶，逐轮插入 | 探测越过表尾回到桶 0；查询同样绕回 |
| `hits_and_misses` | 混合已有和从未插入的 key | 查询命中/未命中 |
| `existing_key_update` | 大批插入后，用小批更新已有 key，再验证整个表 | CAS 返回相同 key、旧数据保留、接收缓冲区清理 |
| `duplicate_key_same_value` | 同批重复 key/value，双卡还跨源 PE 重复 | 并发重复写入，不依赖不同 value 的竞争顺序 |
| `multiple_insert_rounds` | 多轮新增、更新、重复后再次新增 | 累积结果、发送计数复位 |
| `empty_insert_preserves_table` | 所有 PE 输入数为 0，再查全表 | 零次插入循环、空接收槽、旧记录不重放 |
| `one_empty_pe` | 双卡下一方 0 条、一方有输入 | 不等长输入和同步 |
| `key_value_boundaries_round_*` | 每个边界 key 轮流写入全部边界 value | Key 哈希高位、32/64 位 CAS、Value 高位和零值 |
| `insert_size_* / query_size_*` | 见下方规模集合，逐轮验证全表和缺失 key | 非整齐尾部、线程数上限前后、`maxKeysPerPe` 等于上限 |
| `empty_search` | Search 数量为 0，输出哨兵必须保持原值；放在最后 | 零长度查询及历史除零问题回归 |

双卡每个 PE 查询完整 golden 表，碰撞组分别覆盖两个 owner；逐轮切换输入来源，使本地插入、远端发送及接收插入均被覆盖。
碰撞生成使用 `GetTableSize()` 的实际桶数；申请容量调整为奇数桶，使双卡的两个 owner 都能落在指定桶。
预期结果由 CPU 上的输入映射维护，不读取被测表作为 golden。查询输出预填充非零字节，避免遗漏写入时零值误通过。
每次查询完成后执行 PE barrier，避免下一轮更新与另一 PE 的查询重叠。

Key 边界包括：0、1、最大可用值；32 位额外包含最高位为 1 的值；64 位额外包含最小负数、-1、2^32，以及低 32 位相同但高位不同的 key。
Key 最大值是空桶标记，不作为有效 key。Value 覆盖 0、1、2^32-1、2^32、2^32+1、2^63、最大值-1；最大值保留为未命中标记。

设实际 AIV 核数为 C，输入规模为去重后的：

- 0、1、31、32、33、2047、2048、2049；
- C-1、C、C+1；
- C×2048-1、C×2048、C×2048+1。

`maxKeysPerPe` 为 C×2048+1。大批输入复用同一组 key 的前缀，避免多轮测试累积到满表。
这些用例不覆盖满表、输入超限异常、所有 Host 错误分支或未接入容器接口的 kernel。

## 空查询回归

`Search` 与 `Insert` 均使用 `CalBlockDim(std::max<size_t>(nums, 1), ...)` 计算启动参数，避免零输入导致线程数为零和除零。传给 kernel 的实际输入数量仍为 0。
`empty_search` 检查调用正常返回，且输出哨兵值不变。
