---
title: "Common configuration options"
---
Surogate is based on a set of common configuration options that can be used across different modules and components. These options help standardize the configuration process and ensure consistency throughout the application.

## Common Configuration Options
- `model`: HuggingFace model name or path
- `tokenizer`: Path to a tokenizer config file if you want to use a custom tokenizer
- `seed`: Random seed for reproducibility

## Dataset Options
- `type`: Type of dataset. Supported values are:
  - `text`: The dataset is a textual dataset usually suitable for pretraining or language modeling tasks. It must have a column containing raw text data.
  - `instruction`: The dataset is an instruction-following dataset suitable for fine-tuning language models on instruction-based tasks. It must have columns for instructions, inputs (optional), and outputs.
  - `conversation`: The dataset is a conversational dataset suitable for training chat models. It must have columns for user messages and assistant responses.

### Text Dataset Options
- `text_column`: The name of the column in the dataset that contains the raw text data

### Instruction Dataset Options
- `instruction_column`: The name of the column in the dataset that contains the instructions
- `input_column`: The name of the column in the dataset that contains the inputs (optional
- `output_column`: The name of the column in the dataset that contains the outputs
- `format`: Python string template for formatting the prompt. Available variables: `{instruction}`: value of the instruction field, `{input}`: value of the input field 

### Conversation Dataset Options
- `thinking_key`: the key used by the chat template expects to reference the reasoning trace.
- `field_messages`: The name of the column in your dataset that contains the `messages` field.
- `field_system_prompt`: The name of the column in your dataset that contains the System Prompt
- `chat_template`: The name of the chat template to use. The following values are supported: 
  - `tokenizer_default` (default): Uses the chat template that is available in the tokenizer_config.json. If the chat template is not available in the tokenizer, it will raise an error.
  - `jinja`: Uses a custom jinja template for the chat template. The custom jinja template should be provided in the `chat_template_jinja` field.
- `chat_template_jinja`: Custom jinja chat template or path to jinja file. Used only if `chat_template: jinja`.
