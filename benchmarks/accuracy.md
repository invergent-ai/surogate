# Accuracy test for recipes
We studied the impact of the recipes supported by Surogate using a custom version of the `gsm8k` dataset, specifically the `ro_gsm8k` dataset which is a Romanian translation of the original dataset. 

`Qwen/Qwen3-0.6B` was chosen as a reference model. The measured accuracy of the original model on the `ro_gsm8k` dataset is close to 0, so this provides a good way to see how fine-tuning will teach the model this new dataset.

## Summary table
| Precision / Config | accuracy | Stderr |
| :----------------- | :------- | :----- |
| BF16               | 0.2085   | 0.0095 |
| FP8                | 0.1888   | 0.0108 |
| FP4                | 0.1880   | 0.0108 |
| QBnB               | 0.0940   | 0.0080 |
| QFP8 + fp8-hybrid  | 0.1531   | 0.0099 |
| QFP8 + bf16        | 0.1698   | 0.0103 |
| QFP4               | 0.1600   | 0.0101 |


## Loss charts
### BF16
![BF16](./loss-charts/training_plot_bf16.png)
### FP8
![FP8](./loss-charts/training_plot_fp8.png)
### FP4
![FP4](./loss-charts/training_plot_fp4.png)
### QBnB
![FP4](./loss-charts/training_plot_qbnb.png)
### QFP8
![FP4](./loss-charts/training_plot_qfp8.png)
### QFP4
![FP4](./loss-charts/training_plot_qfp4.png)

## Config used:

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
warmup_steps: 20
weight_decay: 0.001
lr_scheduler_type: linear
lora_dropout: 0
lora_rank: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### Commands used:

```shell
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen3-0.6B --max-model-len 2048 --max-lora-rank 64 --enable-lora --lora-modules adapter=/home/densemax2/work/flavius/surogate/output/benchmark_sft_bf16/adapter/ --port 8001
```

```shell
lm-eval --model local-completions --model_args model=adapter,base_url=http://localhost:8001/v1/completions,num_concurrent=50,max_retries=3,tokenized_requests=False,tokenizer=Qwen/Qwen3-0.6B --task gsm8k --num_fewshot 0 --output_path ./base
```

```shell
curl -X POST http://localhost:8001/v1/load_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "adapter", "lora_path": "/home/densemax2/work/flavius/surogate/output/benchmark_sft_qfp8/adapter"}'
```

```shell
curl -X POST http://localhost:8001/v1/unload_lora_adapter -H "Content-Type: application/json" -d '{"lora_name": "adapter"}'
```