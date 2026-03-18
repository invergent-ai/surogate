from datasets import load_dataset

import verifiers as vf


def load_environment(
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL",
    dataset_split: str = "train",
    system_prompt: str
    | None = "Reverse the text character-by-character. Put your answer between <reversed_text></reversed_text> tags. Here is an example: PROMPT: hello world <reversed_text>dlrow olleh</reversed_text>",
) -> vf.Environment:
    train_dataset = load_dataset(dataset_name, split=dataset_split).map(
        lambda x: {
            "question": x["prompt"],
            "answer": x["prompt"][::-1],
            "info": {},
            "task": "reverse-text",
        }
    )
    train_dataset = train_dataset.remove_columns(["prompt"])

    parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")

    def lcs_reward_func(completion, answer, **kwargs) -> float:
        """
        LCS ratio of the reversed prompt and the parsed completion.
        Falls back to text after unclosed <reversed_text> tag.
        """
        import re
        from difflib import SequenceMatcher

        def lcs_ratio(x: str, y: str) -> float:
            return SequenceMatcher(None, x, y).ratio()

        # Try closed tags first
        response = parser.parse_answer(completion)

        # Fallback: unclosed <reversed_text> tag
        if response is None:
            if isinstance(completion, list):
                text = " ".join(
                    m.get("content", "") or ""
                    for m in completion
                    if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"
                )
            else:
                text = str(completion)
            m = re.search(r"<reversed_text>\s*(.*)", text, re.DOTALL)
            if m:
                # Strip a trailing </reversed_text> if partially present
                response = re.sub(r"</reversed_text>.*", "", m.group(1)).strip()

        return lcs_ratio(response or "", answer)

    def format_reward_func(completion, **kwargs) -> float:
        """Reward for producing the expected XML format."""
        if isinstance(completion, list):
            text = " ".join(
                m.get("content", "") or ""
                for m in completion
                if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant"
            )
        else:
            text = str(completion)
        has_open = "<reversed_text>" in text
        has_close = "</reversed_text>" in text
        if has_open and has_close:
            return 1.0
        elif has_open:
            return 0.5
        return 0.0

    rubric = vf.Rubric(
        funcs=[
            lcs_reward_func,
            format_reward_func,
        ],
        weights=[0.8, 0.2],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env