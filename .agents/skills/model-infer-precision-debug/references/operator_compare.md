# 算子级取值与比对

逐级插点与基准比对、逐算子替换扫描、单算子参考实现重算、内存类问题的低扰动观测时使用。

---

## 记录项

每个点位按下列各项记录，缺一项就无法比对：

- 位置：算子名、层号、生成步
- 侧别：输入侧还是输出侧
- 次序：在执行序列里排第几
- 数据：张量本身，以及 shape 与 dtype

两侧用同一份插桩，各项按同一规则记录，否则比对时点位对不上。

### 样例

**普通点位**一行：

```python
torch.save(x.detach().float().cpu(), f"{DUMP}/03_L{self.layer_idx}_S{step}_fa_out.pt")
```

零填充序号让字典序等于执行顺序，省掉额外的次序记录。层号与生成步必须进文件名——一个 `DecoderLayer` 类里的插桩对所有层每步都生效，名字写死会让后写的覆盖先写的，只剩末层末步。多卡另加 rank。

**原地写回的算子**要存两侧。这类算子没有独立的输出参数，直接把结果写回某个输入张量，识别特征是函数名带尾部下划线、API 说明标注原地更新、或没有返回值：

```python
torch.save(cache.clone().float().cpu(), f"{DUMP}/04_L{self.layer_idx}_S{step}_cache_in.pt")
npu_scatter_nd_update_(cache, indices, updates)
torch.save(cache.clone().float().cpu(), f"{DUMP}/04_L{self.layer_idx}_S{step}_cache_out.pt")
```

两侧都要 clone。只在调用后取张量拿到的是输出，输入已被覆盖；只存引用不克隆，下一次调用会覆盖同一块内存。

还有一类**假原地算子**，声明为非原地但实际原地写回，靠命名识别不出来，判定见 `graph_mode_debug.md` 的分层 backend 排查。

### 不适用场景

**图模式**：在 forward 内落盘会破坏图捕获，取值方式见 `graph_mode_debug.md`。

**内存类问题**：每次落盘都触发一次设备到主机的拷贝，这个同步会改变现象。改用「低扰动异步探针」的写法。

### 低扰动异步探针

用于内存类问题。与图模式无关，eager 下同样适用。

预分配 CPU 张量并锁页，在 forward 中做非阻塞拷贝，推理结束后或每步之后读取。这样对执行时序的扰动小，适合内存类问题这种观测即扰动的场景。

```python
import torch

# Allocate once, outside the graph.
probe_cpu = torch.zeros(probe_shape, dtype=probe_dtype, pin_memory=True)


class DecoderLayer(torch.nn.Module):
    def forward(self, hidden_states, ...):
        ...
        probe_cpu.copy_(self.kv_norm.weight, non_blocking=True)
        ...


# After each step or after inference completes: synchronize before reading,
# otherwise the non-blocking copy may not have landed yet.
torch.npu.synchronize()
print(probe_cpu)
```

每个观测点需要一个独立的预分配张量，shape 与 dtype 在图外确定，不能在 forward 里创建。读取前必须等拷贝完成，用设备同步或记录事件后等待该事件，否则拿到的可能是上一步的值。

> 该写法在本仓代码中无对应实现，未经实测，使用前先在目标环境确认可用。

---

## 取点位置

起点不明确时，先在块边界插两个点定位到层，再在该区间内加点。插点数随怀疑范围收窄逐轮增加，不要一次铺满。

Transformer Block 内按执行顺序的常用位置：

1. Block 输入
2. Attention Norm 后
3. QKV 投影后
4. RoPE 后
5. KVCache 写入后
6. FA 输出
7. Attention 残差后
8. Block 输出

---

## 比对方法

```python
a, g = actual.astype(np.float32), golden.astype(np.float32)
if not (np.isfinite(a) & np.isfinite(g)).all():
    raise AssertionError("有效位出现 NaN 或 Inf")
pass_ratio = np.isclose(a, g, rtol=rtol, atol=atol).sum() / a.size
```

**参数取值**：`rtol` 按被测数据的精度格式与算子性质定，`atol` 必须给——只给 rtol 时基准侧的零值区会让相对偏差发散。

**判定**：看通过率而非逐元素全等。通过率门槛与 rtol 无关，单独定——本仓单算子测试用 `0.99`。有效位出现 NaN 或 Inf 即判失败，不按占比容忍。这是张量级口径，整网层面的判定口径由调用方定下，不要混用。

**容差参考**：`ops/ascendc/examples/` 下各单算子测试的 `rtol` 从 `1e-5` 到 `0.01` 不等，`atol` 从 `1e-8` 到 `0.01`，个别量化测试放到 `1`；`ops/pypto/examples/` 用到 `rtol=0.005` 与 `0.03`。按被测算子选，不套一个数。

**cache 类张量**：只取有效位。无效位可能本身带 NaN 或残留脏数据，整块比对会误判。

**两侧位置对不上时**：batch 组成不同、跨框架、跨硬件、单卡基准对多卡切分张量都会这样。先确认对齐关系——切分张量按切分维拼接后再比，其余按请求内位置或语义位置对齐，不要按存储顺序硬配。

**没有基准侧时**：按算法定义在参考后端重算该算子，只跑这一个，不重算整网，拿它的输出当基准。本仓已有可直接参照的写法——`ops/pypto/examples/test_lightning_indexer_prolog.py` 与 `ops/pypto/examples/test_deepseek_indexer_attention.py` 构造 golden 后逐张量比对，`ops/ascendc/examples/` 下的单算子测试同理。新写参考实现时对齐这些例子的组织方式。

**同侧多张量**：某一侧全部达标才算这一侧一致；取最差的一条作为该侧结论。
