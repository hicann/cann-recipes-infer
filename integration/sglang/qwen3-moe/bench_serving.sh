# 手动配置测试的B、S和输出长度
BatchSize=256
SeqLen=5120
OutputLen=500
NumPrompts=300
DatasetJsonPath=ShareGPT_V3_unfiltered_cleaned_split.json

python3 -m sglang.bench_serving --base-url http://127.0.0.1:30002 \
--dataset-path $DatasetJsonPath \
--dataset-name=random \
--random-range-ratio 1 \
--random-input $SeqLen \
--random-output $OutputLen \
--max-concurrency $BatchSize \
--num-prompts $NumPrompts