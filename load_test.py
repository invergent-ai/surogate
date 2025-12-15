import os
os.environ["HF_HOME"] = "/work/.cache/huggingface"

from surogate.core.model.kernels.utils import get_lora_parameters
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-FP8", device_map="cuda")
lora_config = LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
peft_model = get_peft_model(model, lora_config)

for idx, layer in enumerate(peft_model.model.model.layers):
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj

    QW, QW_quant, QA, QB, QS = get_lora_parameters(q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(v_proj)

    if QW_quant is None or KW_quant is None or VW_quant is None:
        raise ValueError("Lora parameters not found in the model layers.")

print("Lora parameters successfully found in all model layers.")