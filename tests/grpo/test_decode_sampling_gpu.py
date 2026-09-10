"""GPU filtering, inverse-CDF selection, policy scores and row error isolation."""

import numpy as np
import pytest
import torch

from surogate.grpo.shared_model import sample_logits

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

DEFAULTS = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    top_logprobs=0,
    logit_bias={},
)


@pytest.mark.parametrize("vocabulary", [7, 257, 32771, 131075])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gpu_sampling_matches_cpu_constraints_and_unfiltered_policy_scores(vocabulary, dtype):
    from surogate import _surogate as ext

    rng = np.random.default_rng(76)
    logits = torch.tensor(rng.normal(size=(8, vocabulary)), dtype=dtype, device="cuda")
    counts = rng.integers(0, 4, size=(8, vocabulary), dtype=np.int32)
    parameters = [
        {},
        dict(temperature=0.0, top_logprobs=20),
        dict(top_k=3, top_p=0.7),
        dict(top_p=0.15),
        dict(min_p=0.3, temperature=0.7),
        dict(min_p=1.0),
        dict(
            top_k=0,
            repetition_penalty=1.3,
            presence_penalty=0.4,
            frequency_penalty=0.2,
            logit_bias={0: 3.0, 1: -5.0},
            blocked_tokens=[0],
        ),
        dict(
            temperature=1.3,
            repetition_penalty=0.8,
            frequency_penalty=-0.3,
            presence_penalty=-0.2,
            top_p=0.8,
            top_k=300,
            min_p=0.01,
            top_logprobs=12,
        ),
    ]
    requests, expected = [], []
    host = logits.float().cpu().numpy()
    for row, overrides in enumerate(parameters):
        params = DEFAULTS | overrides
        history = np.repeat(np.arange(vocabulary), counts[row])
        expected.append(
            sample_logits(
                host[row], history, params, params.get("blocked_tokens", []), np.random.default_rng(200 + row)
            )
        )
        requests.append(params | dict(uniform=float(np.random.default_rng(200 + row).random())))
    actual = ext._decode_sample(logits, torch.tensor(counts, device="cuda"), requests)
    for result, reference in zip(actual, expected, strict=True):
        assert result["status"] == 0
        assert result["token"] == reference["token"]
        assert result["top_ids"] == reference["top_ids"]
        np.testing.assert_allclose(result["logprob"], reference["logprob"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(result["top_logprobs"], reference["top_logprobs"], atol=1e-9, rtol=0)
    # Row placement has no effect on seeded selection, including a sorted row
    # batched with requests that do not use top-k or top-p.
    for row in (0, 2, 7):
        single = ext._decode_sample(
            logits[row : row + 1], torch.tensor(counts[row : row + 1], device="cuda"), [requests[row]]
        )
        assert single[0] == actual[row]


def test_sampling_ties_extreme_logits_and_per_row_errors():
    from surogate import _surogate as ext

    logits = torch.zeros((5, 263), device="cuda")
    logits[2, 0] = float("nan")
    logits[4, :] = -1e4
    logits[4, -1] = 1e4
    counts = torch.zeros_like(logits, dtype=torch.int32)
    requests = [
        dict(top_k=1, top_logprobs=3),
        dict(blocked_tokens=list(range(263))),
        {},
        dict(enabled=False),
        dict(uniform=np.nextafter(1.0, 0.0)),
    ]
    actual = ext._decode_sample(logits, counts, requests)
    assert actual[0]["token"] == 0 and actual[0]["top_ids"] == [0, 1, 2]
    assert actual[1]["status"] == 2 and actual[1]["token"] == -1
    assert actual[2]["status"] == 1 and actual[2]["token"] == -1
    assert actual[3]["status"] == 0 and actual[3]["token"] == -1
    assert actual[4]["status"] == 0 and actual[4]["token"] == 262
    assert actual[4]["logprob"] == 0


@pytest.mark.parametrize(
    "invalid",
    [
        dict(top_p=0),
        dict(uniform=1),
        dict(temperature=float("nan")),
        dict(repetition_penalty=0),
        dict(top_logprobs=21),
        dict(blocked_tokens=[0, 0]),
        dict(logit_bias={7: 1.0}),
    ],
)
def test_invalid_sampling_parameters_are_rejected(invalid):
    from surogate import _surogate as ext

    with pytest.raises(ValueError):
        ext._decode_sample(
            torch.zeros((1, 7), device="cuda"), torch.zeros((1, 7), device="cuda", dtype=torch.int32), [invalid]
        )


@pytest.mark.parametrize("top_k,top_p", [(-1, 1.0), (-1, 0.5), (17, 0.75)])
def test_large_uniform_vocabulary_preserves_ties_and_cdf_endpoints(top_k, top_p):
    from surogate import _surogate as ext

    vocabulary = 131075
    logits = torch.zeros(3, vocabulary, device="cuda", dtype=torch.bfloat16)
    counts = torch.zeros_like(logits, dtype=torch.int32)
    uniforms = [0.0, 0.37, np.nextafter(1.0, 0.0)]
    requests = [dict(top_k=top_k, top_p=top_p, uniform=u) for u in uniforms]
    results = ext._decode_sample(logits, counts, requests)
    kept = int(np.ceil((vocabulary if top_k < 0 else top_k) * top_p))
    assert [r["token"] for r in results] == [0, int(0.37 * kept), kept - 1]
    assert all(r["status"] == 0 for r in results)
    np.testing.assert_allclose([r["logprob"] for r in results], -np.log(vocabulary), atol=1e-10, rtol=0)
