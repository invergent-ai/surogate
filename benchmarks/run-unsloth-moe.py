from unsloth import FastLanguageModel

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="openai/gpt-oss-20b",
    max_seq_length=512,  # Context length - can be longer, but uses more memory
    load_in_4bit=True,  # 4bit uses much less memory
    load_in_fp8=False,  # A bit more accurate, uses 2x memory
    full_finetuning=False,  # We have full finetuning now!
    float8_kv_cache=False,
    fast_inference=False,
    use_exact_model_name=True
    # token = "hf_...",      # use one if using gated models
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=32,  # Best to choose alpha = rank or rank*2
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

dataset = load_dataset("OpenLLM-Ro/ro_gsm8k", split="train")


def format_prompts(examples):
    texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
        
    return {"text": texts}


dataset = dataset.map(format_prompts, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,  # Can set up evaluation!
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=20,
        num_train_epochs = 2, # Set this for 1 full training run.
        # max_steps=30,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",  # Use TrackIO/WandB etc
        packing=True,
        max_length=512
    ),
)
trainer_stats = trainer.train()
model.save_pretrained("./output_unsloth_bf16")
