# Teach a model to reverse the prompt

In this example we demostrate how to train `Qwen3-0.6B` to reverse a small chunk of text. We will use a SFT warmup to learn the skill of text reversal on longer documents and then a quick RL run to reverse smaller chunks of text in the `reverse-text` environment.

This is of course just a quick vibe check and no full-fledged evaluation, but we can see that the model struggles with this task. In this specific instance, we got an **average reward of ~0.05** across the 20x3 rollouts. Let's do some training!

## Supervised Fine-Tuning
We will fine-tune `PrimeIntellect/Qwen3-0.6B` ([HF](https://huggingface.co/PrimeIntellect/Qwen3-0.6B)), which is a clone of `Qwen/Qwen3-0.6B` ([HF](https://huggingface.co/Qwen/Qwen3-0.6B)) with a chat template suitable for multi-turn RL, on `willcb/R1-reverse-wikipedia-paragraphs-v1-1000` ([HF](https://huggingface.co/datasets/willcb/R1-reverse-wikipedia-paragraphs-v1-1000)) which contains 1K examples of reversals of small paragraphs.

To train on a single GPU, run:

```shell
surogate sft examples/sft/reverse-text.yaml
```

After training is complete, the fine-tuned model will be saved in the `././reverse-fft` folder.

## Reinforcement Learning with GRPO
For the RL we will only do 20 steps at 8x16 rollouts, for a total batch size of 128 and sequence length 128. Because of the small context, training should be extremely quick.

Also, we will use the [Co-locate mode](../../guides/rl-training.md#co-locate-mode-details) of Surogate, which will start a single process for vLLM, orchestrator and trainer, and everybody will share the same weights to reduce GPU memory usage.

Run the following command:

```shell
surogate grpo --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml
```

After training is complete, the LoRA adapter will be saved in the `./outputs/final_adapter` folder. 

You can now serve the model and trained adapter with vLLM:

```shell
vllm serve ./reverse-fft --enable-lora --lora-modules adapter=./outputs/final_adapter
```

And run the evaluation:

```shell
uv run scripts/chat.py --model adapter --system-prompt "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
```

Point your browser to [http://localhost:7860](http://localhost:7860) and prompt the model something. 

Additionall, you can run `vf-eval` to properly evaluate the trained adapter against the `reverse-text`environment:

```shell
vf-eval reverse-text -m adapter -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```