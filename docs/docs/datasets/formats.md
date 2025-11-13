---
title: "Datasets Formats"
---
Surogate handles datasets in a different way compared to traditional machine learning libraries. Instead of relying on fixed dataset implementations, Surogate uses flexible data structures that can accommodate various dataset formats. This approach allows users to easily adapt their datasets for different tasks and model architectures.

The main objective of Surogate's dataset handling is to provide seamless integration with Large Language Models (LLMs) while ensuring that the data is formatted correctly according to the model's requirements. This is achieved through the use of [Chat Templates](./intro.md#chat-templates), which define how conversational data should be structured for different models.

## Supported Dataset Formats
Textual LLM datasets can be broadly categorized into three main formats:
- **Conversational**: Datasets designed for chat-based interactions.
- **Instruction-based**: Datasets focused on following specific instructions or prompts.
- **Text-based**: Datasets that involve plain text processing or generation.

### Conversational datasets
Conversational datasets are used for tasks involving dialogues or chat interactions between users and assistants. They contain sequences of messages where each message has a role (e.g., `user` or `assistant`) and content (the message text).

```python
messages = [
    # 1. System Prompt sets the stage
    {"role": "system", "content": "You are a technical support bot."},
    
    # 2. Turn 1
    {"role": "user", "content": "My internet is not working."},
    {"role": "assistant", "content": "I can help with that. Is the light on your router blinking or solid?"},
    
    # 3. Turn 2 (References Turn 1)
    {"role": "user", "content": "It is blinking red."},
    {"role": "assistant", "content": "A blinking red light usually indicates a connection error. Please try restarting it."}
]
```
The above example illustrates a simple conversation between a user and an assistant, with a system prompt that defines the assistant's role.

To ensure your conversational dataset is properly formatted, follow these guidelines to obtain the best results:

1. **Maintain the Alternating Pattern**: The standard flow for ChatML is: `System` → `User` → `Assistant` → `User` → `Assistant` ...
2. ***Avoid Consecutive Roles**: Most chat templates and training libraries expect strictly alternating roles:
    - **Bad**: `User` → `User` → `Assistant` (The model might get confused about who is speaking).
    - **Good**: If a user sends two messages in a row, combine them into one content block: `"content": "Wait. I also forgot to mention..."`
3. **For training, end with the Assistant**: For Training and Supervised Fine-Tuning (SFT), the last message in the list is typically the response you want the model to learn to generate. If the list ends with `User`, the model has nothing to learn for that final step (unless you are doing inference/testing).

#### Tool Calling
Some chat templates support tool calling, which allows the model to interact with external functions—referred to as tools—during generation. This extends the conversational capabilities of the model by enabling it to output a `tool_calls` field instead of a standard `content` message whenever it decides to invoke a tool.

After the assistant initiates a tool call, the tool executes and returns its output. The assistant can then process this output and continue the conversation accordingly.

Here’s a simple example of a tool-calling interaction:

```python
messages = [
    {"role": "user", "content": "Turn on the living room lights."},
    {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {
            "name": "control_light",
            "arguments": {"room": "living room", "state": "on"}
        }}]
    },
    {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
    {"role": "assistant", "content": "Done!"}
]
```

When preparing datasets for Supervised Fine-Tuning (SFT) with tool calling, it is important that your dataset includes an additional column named tools. This column contains the list of available tools for the model, which is usually used by the chat template to construct the system prompt.

The tools must be specified in a codified JSON schema format. You can automatically generate this schema from Python function signatures using the `get_json_schema` utility:

```python
from transformers.utils import get_json_schema

def control_light(room: str, state: str) -> str:
    """
    Controls the lights in a room.

    Args:
        room: The name of the room.
        state: The desired state of the light ("on" or "off").

    Returns:
        str: A message indicating the new state of the lights.
    """
    return f"The lights in {room} are now {state}."

# Generate JSON schema
json_schema = get_json_schema(control_light)
```

The generated schema would look like:
```json
{
    "type": "function",
    "function": {
        "name": "control_light",
        "description": "Controls the lights in a room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "description": "The name of the room."},
                "state": {"type": "string", "description": "The desired state of the light ('on' or 'off')."}
            },
            "required": ["room", "state"]
        },
        "return": {"type": "string", "description": "str: A message indicating the new state of the lights."}
    }
}
```

A complete dataset entry for SFT might look like:
```json
{"messages": messages, "tools": [json_schema]}
```

For more detailed information on tool calling, refer to the [Tool Calling section in the transformers documentation](https://huggingface.co/docs/transformers/chat_extras#tools-and-rag) and the blog post [Tool Use, Unified](https://huggingface.co/blog/unified-tool-use).

### Instruction datasets
Instruction datasets are the fuel used to train Large Language Models (LLMs) to move from simply predicting the next word (completion) to actually following user commands (instruction tuning).

The most common format for these datasets—popularized by Stanford's **Alpaca** dataset—breaks a training example into three distinct fields:

- **Instruction**: The directive or task the user wants the model to perform. It describes what needs to be done.
- **Input** (optional): Additional **context** or **information** that the model may need to complete the instruction. This field is not always present.
- **Output**: The expected response or result that the model should generate after following the instruction.

```python
data = [
    {
        "instruction": "Translate the following English text to French:",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment ça va?"
    },
    {
        "instruction": "Name the capital of France.",
        "output": "Paris"
    }
]
```

#### The Specific Purpose of "Input"
Think of the Input field as the **variable data** that the Instruction acts upon. Its primary purpose is to separate the task from the context.

1. **Providing Context**: In many tasks, an instruction cannot be completed without source material. The Input provides that source material. Example:
    - Instruction: "Summarize the following article."
    - Input: "Article text goes here..."
2. **Enabling Reusability (The Function Analogy)**: Think of the Instruction as a computer function (e.g., `translate_to_french()`) and the Input as the argument passed to that function (e.g., `Hello World`). By keeping them separate during training, the model learns that the Instruction is a reusable tool that can be applied to _any_ Input.
3. **Handling "Empty" Inputs**: Not all instructions require an input. If the instruction is self-contained (e.g., open-ended knowledge questions), the Input field is left empty (or missing). 

If we didn't have an "Input" field, we would have to mash the instruction and the data together into one long string. While LLMs can handle that, separating them in the dataset offers two distinct advantages:
1. **Structured Learning**: It teaches the model to distinguish between the command (the fixed part) and the data (the variable part).
2. **Prompt Engineering**: It mirrors how we use prompts in real life (e.g., System Prompt vs. User Prompt), making the model better at handling "Zero-shot" or "Few-shot" prompting tasks later on.

#### Difference from Conversational Datasets
While both datasets are used to "fine-tune" LLMs, the primary difference lies in **structure** (Single-turn vs. Multi-turn) and **intent** (Task Execution vs. Interaction):

1. **Structure & Format**: 
    - **Instruction Datasets** are **single-turn** and have a rigid structure (Instruction, Input, Output). The model does not need to remember what happened three questions ago. 
    - **Conversational Datasets** are **multi-turn** and consist of a sequence of messages with roles. The model learns to look at previous messages in the list to understand the current context (e.g., pronouns like "it" or "that").
2. **The Training Goal**: 
    - **Instruction Tuning (Obedience)**: The goal is to make the model a **tool**. It optimizes for **Precision** (following constraints exactly: "Answer in JSON format only"), **Factuality** (providing the correct answer immediately) and **Conciseness** (getting straight to the point)
    - **Conversation Tuning (Chattiness)**: The goal is to make the model a **persona**. It optimizes for **Context Retention** (remembering names, numbers, or topics mentioned earlier in the chat), **Tone** (Being polite, conversational, or adopting a specific character style) and **Flow** (Handling follow-up questions and clarifications naturally)

> [!NOTE]
You can format an instruction dataset as a single-turn conversation.


Here is a quick comparison table:

| Feature          | Instruction Dataset                  | Conversation Dataset                 |
|------------------|--------------------------------------|:-------------------------------------|
| Primary Unit     | A single task (Prompt/Response)      | A dialogue session (History)         |
| Context Window   | Limited to current task              | Long (accumulates over time)         |
| Best For         | Math, Code generation, Summarization | Customer support, Roleplay, Chatbots |
| Popular Examples | Alpaca, Dolly, FLAN                  | ShareGPT, OpenAssistant, UltraChat   |


### Text datasets
Text datasets are used for tasks that involve processing or generating plain text. These datasets typically have a text field that contains the input text for the model.

```python
data = [
    {"text": "Once upon a time in a land far, far away..."},
    {"text": "In the beginning, there was light."}
]
```