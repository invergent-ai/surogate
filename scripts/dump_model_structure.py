import argparse

import torch
from transformers import AutoConfig, AutoModel


def main():
    parser = argparse.ArgumentParser(description="HF Transformers Model Structure Dumper")
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model ID (e.g., 'HuggingFaceTB/SmolLM2-135M-Instruct') or Path")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModel.from_config(config, trust_remote_code=True)
    print(model)


if __name__ == "__main__":
    main()
