"""ARPA backoff, BOS/EOS, and unknown-token semantics without NeMo installed."""

import json
import math
import subprocess
from pathlib import Path

import pytest
from safetensors.numpy import load_file

from surogate.cli.serve import _resolve_binary

ARPA = """\\data\\
ngram 1=5
ngram 2=5
ngram 3=3

\\1-grams:
-3\t<unk>\t0
0\t<s>\t-0.4
-0.2\t</s>\t0
-0.5\td\t-0.1
-0.6\te\t-0.2

\\2-grams:
-0.3\te d\t-0.5
-0.25\td </s>
-0.2\td e\t-0.4
-0.8\t<s> </s>
-0.1\t<s> d\t-0.3

\\3-grams:
-0.06\td e d
-0.09\td e </s>
-0.04\t<s> d e

\\end\\
"""


def convert(tmp_path, text, vocab=5):
    binary = _resolve_binary("stt")
    converter = Path(binary).with_name("surogate-stt-lm") if binary else None
    if converter is None or not converter.is_file():
        pytest.skip("build surogate-stt-lm first")
    path = tmp_path / "test.arpa"
    path.write_bytes(text.encode() if isinstance(text, str) else text)
    return subprocess.run(
        [str(converter), str(path), str(tmp_path), str(vocab)], capture_output=True, text=True, timeout=10
    )


def test_arpa_histories_and_backoff(tmp_path):
    result = convert(tmp_path, ARPA)
    assert result.returncode == 0, result.stderr
    data = load_file(tmp_path / "lm.safetensors")
    assert json.loads((tmp_path / "lm.json").read_text())["max_order"] == 3

    def advance(state, token):
        score = 0
        while True:
            begin, end = data["start_end_arcs"][state]
            for index in range(begin, end):
                if data["ilabels"][index] == token:
                    return int(data["to_states"][index]), score + data["arcs_weights"][index]
            assert state != 0
            score += data["backoff_weights"][state]
            state = data["backoff_to_states"][state]

    d, score = advance(1, 0)  # <s> d
    assert score == pytest.approx(-0.1 * math.log(10))
    de, score = advance(d, 1)  # <s> d e, successor history d e
    assert score == pytest.approx(-0.04 * math.log(10))
    assert data["final_weights"][de] == pytest.approx(-0.09 * math.log(10))
    ed, score = advance(de, 0)
    assert score == pytest.approx(-0.06 * math.log(10))
    assert data["backoff_weights"][ed] == pytest.approx(-0.5 * math.log(10))
    unknown, score = advance(de, 4)
    assert unknown == 0
    assert score == pytest.approx(-3.6 * math.log(10) - math.log(3))
    assert int(data["start_end_arcs"][0, 1]) == 5


@pytest.mark.parametrize(
    "text",
    [
        ARPA.replace("\\end\\", ""),
        ARPA.replace("ngram 3=3", "ngram 3=4"),
        ARPA.replace("d e d", "d d e"),  # missing prefix
        ARPA.replace("-0.6\te", "-0.6\tword"),
        ARPA.replace("-0.6\te", "nan\te"),
        ARPA.replace("-0.6\te", "-0.6\td"),
    ],
)
def test_bad_arpa_is_rejected(tmp_path, text):
    result = convert(tmp_path, text)
    assert result.returncode != 0
    assert "speech language model:" in result.stderr


def test_invalid_utf8_cannot_alias_a_valid_token_id(tmp_path):
    invalid = b"\xc0\xa0\x80"
    text = (
        b"\\data\\\nngram 1=4\nngram 2=1\n\n\\1-grams:\n"
        b"-3\t<unk>\t0\n0\t<s>\t0\n-1\t</s>\n-2\t" + invalid + b"\t0\n"
        b"\n\\2-grams:\n-1\t<s> " + invalid + b"\n\n\\end\\\n"
    )
    assert convert(tmp_path, text, vocab=2048).returncode != 0
