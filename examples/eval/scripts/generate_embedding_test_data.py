# examples/eval/scripts/generate_embedding_test_data.py

import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

test_texts = [
    "The cat sat on the mat",
    "A dog played in the park",
    "Machine learning is fascinating"
]

test_cases = []

for text in test_texts:
    # Get embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    test_case = {
        "input": text,
        "expected_output": "",  # Not used for embeddings
        "metadata": {
            "expected_embedding": embedding
        }
    }

    test_cases.append(test_case)

# Save to JSONL
with open('../datasets/embedding_test.jsonl', 'w') as f:
    for case in test_cases:
        f.write(json.dumps(case) + '\n')

print(f"Generated {len(test_cases)} test cases with real embeddings")