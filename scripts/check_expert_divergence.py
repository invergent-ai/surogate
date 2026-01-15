import torch
from transformers import AutoModelForCausalLM

"""
Newly upcycled MoE models may have experts that are still very similar to each other. This script helps
analyze the divergence between experts by comparing their weights.

When you first upcycled the model, the Cosine Similarity was 1.0000 (perfect clones).

- Similarity > 0.9999: The router has nothing to work with yet. The high aux_loss is essentially "screaming" at a wall because the experts haven't moved enough to provide different outputs.
- Similarity < 0.9990: This is the tipping point. Once experts diverge even slightly, the router will start to see that Expert A handles a specific token better than Expert B. This is when your aux_loss of 6.5 should finally start to drop.
"""
def check_expert_divergence(model_path):
    print(f"Loading model from {model_path}...")
    # Load in bf16 to match your training precision
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    # Target a specific layer (e.g., the middle layer)
    layer_idx = model.config.num_hidden_layers // 2
    
    # Qwen3-MoE path: model.layers[i].mlp.experts[j]
    # We compare the gate_proj as it often specializes first
    expert0_weights = model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.data
    expert7_weights = model.model.layers[layer_idx].mlp.experts[7].gate_proj.weight.data

    # Calculate metrics
    dist = torch.norm(expert0_weights - expert7_weights).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        expert0_weights.flatten(), 
        expert7_weights.flatten(), 
        dim=0
    ).item()

    print(f"\n--- Analysis for Layer {layer_idx} ---")
    print(f"Euclidean Distance: {dist:.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    
    if cos_sim > 0.9999:
        print("Status: 🔴 Experts are still nearly identical clones.")
    elif cos_sim > 0.99:
        print("Status: 🟡 Experts are beginning to diverge.")
    else:
        print("Status: 🟢 Experts have successfully specialized!")

if __name__ == "__main__":
    # Update this to your latest checkpoint directory
    check_expert_divergence("./moe")