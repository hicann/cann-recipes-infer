# DSpark：完整 100K RULER V2 请求

使用 NVIDIA NeMo-Skills 官方 RULER V2 `mk_niah_basic` 生成器生成的一条完整请求，随机种子 42。不是从 128K 请求截断，也没有额外重复填充。

- Kimi K3 默认 tokenizer：正文 102199 tokens；加 chat template 后 **102287 tokens**。
- 配置 `input_max_len: 102400` 可完整保留正文、答案线索和末尾问题。
- 标准答案：`4083993832`（不作为单独答案输入模型）。
- 这是同一公开评测任务的新生成样本，**不是 RadixArk 公布结果所用的原始请求**；本样本未实测接受长度。
- 基础检索题的回答可能很短，不适合据此推断长时间 decode 性能或整个数据集平均接受长度。

## 使用

请求保存在 `models/kimi_k3/examples/dspark_rulerv2_100k/default_prompt.json`。`models/kimi_k3/infer.sh` 在 launch 前将它复制到公共 `dataset/default_prompt.json`，沿用 `dataset: default` 加载流程。测试时修改本目录的示例文件即可，启动时会同步到公共文件；其他使用公共 default 的模型也会读取复制后的内容。

沿用原 DSpark 启动方式，在现有 YAML 的数据配置位置设置：

```yaml
dataset: default
input_max_len: 102400
max_new_tokens: 256
temperature: 0.0
```

保留真实 prefill。默认数据加载器会将该请求重复用于 batch，各项性能对比应保持 batch、生成上限和其他配置一致。

在日志目录查看：

```bash
grep -h 'The speculation accept' log_*.log
```

当前实现 accept length 包含每轮额外的一个 target token。单条请求的结果不能直接当作公开数据集平均值。需要换请求时，直接替换本目录 `default_prompt.json` 的 `text` 内容即可。JSON 保存原始用户文本，chat template 仍由现有推理流程添加。

`input_max_len` 请保持为 102400 或更大；较小的输入预算仍会触发现有截断逻辑，无法保证答案线索保留。

## 来源与生成参数

- [RadixArk DSpark 模型卡](https://huggingface.co/RadixArk/Kimi-K3-DSpark)：RULER V2 1M 总体接受长度 4.2553；MK 4.4658。公开结果是 1M 配置，不保证本 100K 样本达到 4。
- [RULER V2 官方说明](https://github.com/NVIDIA/RULER/tree/rulerv2-ns)
- [官方生成器](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/prepare_niah.py)
- [官方任务参数](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/ruler2/prepare.py)

在提供该模块的 NeMo-Skills 环境中，可使用以下参数生成；依赖包含 `wonderwords==2.2.0`、nltk、numpy、transformers、tiktoken、tenacity。

```bash
mkdir -p /tmp/rulerv2_100k
python -m nemo_skills.dataset.ruler2.prepare_niah \
  --output_folder /tmp/rulerv2_100k \
  --tokenizer_path /path/to/Kimi-K3 \
  --max_seq_length 102312 --num_samples 1 --random_seed 42 \
  --num_needle_k 1 --num_needle_v 1 --num_needle_q 1 \
  --type_haystack needle --type_needle_k words \
  --type_needle_v numbers --num_digits_v 10
```

102312 是正文预算，为本 tokenizer 的 chat template 预留 88 tokens。生成器按完整句子调整上下文，实际长度可以略低于预算。将输出 test.jsonl 第一行的 `question` 转为 `{"text": question}` 即为本仓库 default 请求格式。

## 校验

已在 CPU 上使用默认 Kimi K3 tokenizer 复现模型 encode/decode/chat-template 流程，确认无截断，问题和答案线索完整。模型加载器对缺少 CLS/SEP 的 `cls_sep` tokenizer 设置按现有实现切换为 `none`。未进行 NPU 推理或接受长度实测。
