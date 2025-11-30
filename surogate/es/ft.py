import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
import time
import gc

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
args = parser.parse_args()


# Hyperparameters for ES
NUM_ITERATIONS = 1000             # Number of ES iterations (generations)
POPULATION_SIZE = 30              # Population size (number of perturbations per iteration)
SIGMA = 0.001                     # Standard deviation for weight perturbations (noise scale)
ALPHA = 0.0005                    # Learning rate
max_new_tokens = 100              # Maximum number of tokens allowed to be generated
do_sample = False                 # Whether sampling is allowed in generating tokens, default to be not allowed (greedy decoding for ES)
initial_seed = 33                 # Initial random seed


# --- Dummy Dataset and Reward Function ---
# In practice, define a set of input reasoning tasks with desired targets.
dataset = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]

def compute_reward(generated_text, target_text):
    # Negative absolute difference in length
    return -abs(len(generated_text) - len(target_text))

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def evaluate_model(model, tokenizer, input_text, target_text, device, seed_idx=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards for batch texts
    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]

    if return_text:
        return rewards, generated_texts
    else:
        return rewards

def process_seed(seed_idx, seed, model, tokenizer, device, verbose=False):
    """Function to process a single seed"""
    if verbose:
        print(f"Processing seed {seed_idx} (value: {seed})")

    # Apply perturbation to weights
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, device,
                           seed_idx=seed_idx, verbose=verbose, return_text=False)
    total_reward = sum(rewards)

    # Restore original weights (direct inplace modification)
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    average_reward = total_reward / len(dataset)

    force_memory_cleanup()

    if verbose:
        print(f"Completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward


# --- Main Evolution Strategies Loop ---
def main():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
    print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")

    # Load model
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    print(f"Loading model {model_name}...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=hf_cache_dir,
        device_map={"": device},
        torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)

    print("Model loaded successfully")

    model.eval()  # Turn off dropout, etc.

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()

    np.random.seed(initial_seed)

    for iteration in range(NUM_ITERATIONS):
        # Record iteration start time
        iter_start_time = time.time()

        # Force garbage collection
        force_memory_cleanup()

        if args.verbose:
            print(f"Starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate seeds
        seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()

        if args.verbose:
            print(f"Generated {len(seeds)} seeds")

        # Process all seeds sequentially
        rewards = []
        for seed_idx, seed in enumerate(seeds):
            _, reward = process_seed(seed_idx, seed, model, tokenizer, device, verbose=args.verbose)
            rewards.append(reward)

        force_memory_cleanup()

        # Convert rewards to a tensor and normalize.
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Aggregate perturbations and update model weights
        if args.verbose:
            print(f"Updating model weights")
        for name, param in model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))

                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        # Synchronize to ensure weight updates are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

    total_time = time.time() - training_start_time

    # Save the fine-tuned model weights.
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    question_num = len(dataset)
    save_dir = f"finetuned_{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_question_num{question_num}_correct"
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved successfully.")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()