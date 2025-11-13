---
title: "Datasets in Surogate"
---

This guide explains the dataset formats supported by Surogate and how to use them in your projects.

Dataset formats define how data is structured for training and inference with LLMs. Surogate supports three main formats: **conversational**, **instruction-based**, and **text-based**. Each format is suited for different types of tasks and model architectures.

To properly understand how to work with datasets in Surogate, it is essential to grasp the concept of Chat Templates.

## Chat Templates
At their core, Large Language Models (LLMs) are text completion engines. They do not natively understand the concept of a "conversation" or distinct "messages." They simply predict the next token in a string.

A **Chat Template** is a formatting rule (often a [Jinja2](https://jinja.palletsprojects.com/en/stable/templates/) template) that converts a list of conversational messages into a single, specific string format that the model was trained to understand.

### Why are Chat Templates needed?
If you interact with a model like Llama 3 or Mistral, you provide a structured list of messages (User, Assistant, System). However, the model requires a raw string. The chat template acts as the **bridge** between structured application code and the raw text input the model requires.

> [!IMPORTANT]
Using the wrong template or formatting the string manually without precision can significantly degrade model performance for your downstream task. If the syntax doesn't match what the model saw during training, it may hallucinate, refuse to answer, or output garbage.

### Anatomy of a Chat Template
A chat template handles some critical components:
1. **Control Tokens**: Special characters that indicate the start (e.g. `<s>`, `<|im_start|>`) and end (`</s>`, `<|im_end|>`) of turns.
2. **Roles**: Labels that define who is speaking:
    - **System**: Instructions on how the model should behave (e.g., "You are a helpful coding assistant").
    - **User**: The human input.
    - **Assistant**: The model's previous responses (for conversation history).
3. **Whitespace**: Specific newlines or spacing required between turns.
4. **Tool Calls**: If the model supports tool calling, the template must define how to format tool call requests and responses.

In the Hugging Face ecosystem, chat templates are stored within the `tokenizer_config.json` file. You generally do not need to write these raw strings manually.

### Examples
To understand how chat templates format data, look at how the same conversation is formatted for two different popular model architectures:

Consider the following conversation:
```python
messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "Hello!"}
]
```
**1. ChatML Format (e.g., Qwen, Yi, Microsoft models)**
```text
<|im_start|>system
You are a helpful AI.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

**2. Llama 2 Format (e.g., Llama, Mistral models)**
```text
<s>[INST] <<SYS>>
You are a helpful AI.
<</SYS>>

Hello! [/INST]
```

### Generation vs. Training
When preparing a prompt for generation (inference), the template must leave the string "open" for the assistant to complete, typically by ending with the assistant's start token.

Example: `<|im_start|>assistant` or `[/INST]`

When preparing data for training (fine-tuning), the template usually closes the assistant's turn with an End-of-Sequence (EOS) token.

Example: `<|im_start|>assistant Paris is the capital.<|im_end|>`

### Common Pitfalls
- **Hard-coding Prompts**: Avoid manually concatenating strings (e.g., `prompt = "User: " + input`). If you switch models later, your code will break. Always use the tokenizer's template.
- **Missing System Prompts**: Some models (like Gemma) were not trained with "System" roles. Injecting a system prompt into a model that doesn't support it will cause an error.
- **EOS Token Confusion**: Ensure your template implementation doesn't accidentally add an "End of Sequence" token at the very end of your prompt, or the model will think the conversation is already over and output nothing.
