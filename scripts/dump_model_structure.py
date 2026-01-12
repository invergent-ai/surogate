import argparse

from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="HF MoE Upcycling Script")
    parser.add_argument("--model", type=str, required=True, 
                        help="Hugging Face model ID (e.g., 'HuggingFaceTB/SmolLM2-135M-Instruct') or Path")
    args = parser.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    print(model)
    
    
if __name__ == "__main__":
    main()