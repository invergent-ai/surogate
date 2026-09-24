# Decisions API

`POST /api/alpha/decisions` (OpenRouter's path; aliases `POST /v1/decisions` and
`POST /api/v1/decisions`) answers several single-token
questions about one shared state in a single request, with the field names of OpenRouter's
decisions API. It is the serving form of the decision protocol the decision models were
trained and benchmarked with: each question is rendered as a two-turn chat (a fixed system
prompt, then the state, the question and lettered options), thinking is off, and the answer
is read at the first generated position as a softmax over the option-letter logits alone,
divided by the server's [calibration temperature](#calibration-temperature) (1, meaning
unchanged, unless the server was started with `--decision-temperature`). Nothing is sampled
and nothing is rounded.

The shared state is prefilled once. Every question then continues from that GPU state with
its own suffix, in waves of `min(64, --max-num-seqs)` rows, so a forty-question request costs
one prefill of the state plus forty short suffixes rather than forty prompts.

## Request

```json
{
  "model": "jev-gemma4-26b-a4b",
  "state": "Customer wrote: the package arrived late and damaged, I want my money back.",
  "questions": {
    "sentiment": {
      "type": "choice",
      "instructions": "What is the customer's sentiment?",
      "criteria": {"positive": "The message is positive", "neutral": "Neither", "negative": "The message is negative"}
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the customer ask for a refund?",
      "criteria": {"true": "A refund is requested", "false": "No refund is requested"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]
    }
  }
}
```

| Field | Meaning |
|---|---|
| `model` | Required. A served model id or adapter name, routed like every other endpoint. |
| `state` | Required. A string, object or array. Rendered as JSON text with Python's `json.dumps` conventions (`, ` and `: ` separators, keys in the order sent, non-ASCII raw, floats such as `1.0` and `1e+16`, integers of any width written back as sent), because that text is what the model was trained on. `NaN` and `Infinity`, which Python's parser accepts, are refused as invalid JSON. |
| `questions` | Required, at least one. An object of question name to question; the order is kept. |
| `provider`, `session_id`, `user`, `trace` | Accepted and ignored. |
| `images` | Our extension, optional: an array of data URLs (or `{"url": ...}` objects) attached to the user turn ahead of the text, in order, for every question. Needs a server started with `--vision`. |
| Any other field | Accepted and ignored. There is deliberately no per-request temperature; see [Calibration temperature](#calibration-temperature). |

A question is one of:

- `{"type": "choice", "instructions": ..., "criteria": {key: description, ...}}` with 2 to 255
  keys. The keys are the answer vocabulary, in the order sent: the first key is option `A`.
- `{"type": "noul", "instructions": ..., "criteria": {"true": ..., "false": ...}}`. Option `A` is
  the `false` description and option `B` the `true` one, whichever order they were sent in.
- `{"type": "score", "instructions": ..., "criteria": [level 0, level 1, ...]}` with 2 to 255
  levels, in order.

`instructions` and every description should be strings. A non-string value is accepted and
rendered as JSON text the same way the state is, but strings are the form the models saw.

Key order is meaningful: the questions object decides the answer order and a choice
question's criteria object decides which option gets which letter. The body is parsed with
that order preserved; a question named twice or a criteria key sent twice is refused.

## Response

```json
{
  "id": "dec-5f0a9c1e2b7d4a13",
  "model": "jev-gemma4-26b-a4b",
  "provider": "surogate",
  "answers": {
    "sentiment": {"type": "choice", "choice": "negative", "confidence": 0.93, "probabilities": {"positive": 0.01, "neutral": 0.04, "negative": 0.95}},
    "refund": {"type": "noul", "noul": 0.98},
    "urgency": {"type": "score", "score": 1.71, "confidence": 0.62, "legend": {"0": "Not urgent", "1": "Somewhat urgent", "2": "Very urgent"}, "probabilities": {"0": 0.03, "1": 0.23, "2": 0.74}}
  },
  "usage": {"input_tokens": 412, "output_tokens": 3, "cost": 0}
}
```

- `probabilities` is the softmax over the question's option logits at the answer position,
  divided by the server's calibration temperature `T` (1 by default) and computed in double,
  keyed by option key in option order. Values are written at full precision. Every other
  number below is computed from this distribution.
- A choice answer's `choice` is the most probable key (the first on a tie; the temperature
  does not change it) and its `confidence` is the peak probability rescaled from uniform to
  certainty, `(max - 1/n) / (1 - 1/n)`.
- A noul answer's `noul` is the probability of `true`.
- A score answer's `score` is the expected level index, `sum(i * p_i)`, and its `confidence`
  measures concentration around the modal level (TypeSafe's `score_confidence`, clamped at
  zero). `legend` maps each index to the level as sent.
- `usage.input_tokens` counts what was actually prefilled: the shared prefix once plus every
  question's suffix. `output_tokens` is the number of questions. `cost` is always 0.

## Protocol details

For each question the chat is `[{"role": "system", ...}, {"role": "user", ...}]` rendered by
the model's own chat template with the generation prompt and thinking disabled. The user
text is:

```
SHARED STATE (JSON string):
<json.dumps(state)>

QUESTION:
<instructions>
OPTIONS:
A: <description of the first option>
B: <description of the second option>
Answer with one option letter only.
```

Up to 26 options are labelled `A`..`Z`. Beyond that the labels come from the tokenizer's
codebook (`A`..`Z` then `AA`, `AB`, ... keeping only codes that are a single token which
decodes back to itself), the closing line asks for "one option code", and the system prompt
says "code" instead of "letter". Every label is verified to be exactly one continuation
token after the rendered prompt, tokenised in context rather than on its own; a template or
tokenizer for which that does not hold is refused with HTTP 400.

The shared prefix is the longest common token prefix of all questions' prompts, capped at
the part rendered before `QUESTION:` and at one token less than the shortest prompt. A
request mixing questions with 26 or fewer options and questions with more has two different
system prompts, so its shared prefix stops before the state; that is the protocol, not a
defect.

Each question is then either a whole prompt or at least 47 tokens beyond the shared prefix,
47 being the engine's MoE wide-prefill width: a prefill step of fewer tokens is dispatched to
the small-T kernels, a different GEMM arrangement whose op-level rounding, amplified through
the layers, can move a near-even decision (measured: one 41-token suffix moved a probability
from 0.56 to 0.44). The benchmark prefilled whole prompts, so the endpoint pins its readouts
to that route: the shared prefix gives tokens back until every suffix is at least 47 tokens,
and when the prefix itself would be shorter (the mixed case above, or a very short state)
nothing is shared and each question runs as the whole prompt a plain request would, with the
prefix cache off. A one-question request and a request with images always run that way (the
runtime keeps saved GPU prefix states for text prompts only). This is a reproducibility pin,
not an accuracy claim, and it has two limits: a whole prompt longer than the engine's prefill
chunk (2,048 tokens by default) is chunked by the engine on its own terms (tail chunks of 10,
30, 47 and 200 tokens and a 4,116-token prompt showed no divergence from the whole-prompt
readout), and with a LoRA adapter bound the wide route is never taken, so on adapter-routed
requests the floor guarantees nothing. `input_tokens` counts what was actually prefilled.

Requests are recorded like every other protocol: a console line and, with
`--request-log-jsonl`, `request_start` / `request_done` / `request_rejected` records with
`protocol: "decisions"`, the question count, the shared prefix, the calibration temperature
(`decisions.temperature`, on start and done records) and the token usage. Console rejection lines echo the question name
or label they refer to, never the state.

A `--decision-temperature` that is not a finite number greater than zero (out-of-range text
such as `1e999` or a subnormal included) stops the server at startup with the usage text; it
never reaches a request.

Errors use the server's standard `{"error": {...}}` envelope: HTTP 400 for a malformed body
(missing or mistyped fields, an unknown question type, fewer than 2 or more than 255
options, a repeated question name or option key, an empty question set, a prompt over the
model context), `vision_disabled` for images on a server without `--vision`, 404 for an
unknown model, and the usual 429/503 when the queue is full.

## Calibration temperature

A model can rank the options well and still be overconfident: its peak probability runs
ahead of how often it is right. `--decision-temperature T` corrects that on the server (see
[the CLI](cli.md#decisions-calibration)). With `z` the option-label logits of one question
at the answer position, the answer is read from

```
p_i = exp((z_i - max z) / T) / sum_k exp((z_k - max z) / T)        (T > 0, finite)
```

which is the renormalised candidate distribution tempered once: the candidate
log-probabilities `log q_i = z_i - logsumexp(z)`, divided by `T` and renormalised, give the
same `p`, because any per-question constant cancels. `T = 1` is the untempered readout bit
for bit (dividing by one is exact), `T > 1` flattens the distribution towards uniform and
`T < 1` sharpens it towards the argmax. Subtracting the maximum first keeps every exponent at
or below zero, so no temperature overflows, divides by zero or produces a NaN; probabilities
too small for a double are exactly 0.

What changes and what does not:

- `probabilities`, a choice or score `confidence`, `noul` and the expected `score` are all
  computed from the tempered `p`, so they stay consistent with each other.
- `choice` does not change. Dividing by `T > 0` keeps the order of the logits, so the most
  probable option at any `T` is the most probable at `T = 1`. Where rounding makes two
  options tie on one side only, the choice is the first option most probable on both: an
  extreme `T` that rounds a near-tie to equal probabilities keeps the untempered choice, and
  the choice is always a maximum of the returned `probabilities`. (A `T` below 1 can
  separate options whose logits were within about 5e-17, which tied by rounding at `T = 1`;
  that is the only case where the choice can differ from the untempered one.) The benchmark
  accuracy of a calibrated server is therefore unchanged.
- It is applied exactly once per question, in one place, after the readout: the same way
  for choice, noul and score questions, for option letters and for the codebook codes used
  past 26 options, for questions answered on the shared GPU prefix and for whole prompts, and
  on `/api/alpha/decisions`, `/v1/decisions` and `/api/v1/decisions` alike.
- It applies to every decisions request the process answers, including those for models
  added with `--model` and for LoRA adapters; models whose fitted temperatures differ belong
  in separate processes. Chat, completions and every other endpoint are unaffected;
  `--temperature` and request sampling fields never reach this readout.

The value is part of the deployment, not of the request. The request schema accepts and
ignores unknown fields, so a per-request temperature would be silently dropped by any
OpenRouter-compatible server that does not implement it, returning uncalibrated
probabilities without an error. The server records the value in its `server_start` record
(`server.decision_temperature`) and in every decisions `request_start` and `request_done`
record (`decisions.temperature`), and prints it at startup when it is not 1. A value that is
not a finite, normal number greater than zero is refused when the server starts.

### Choosing the temperature

`T` belongs to a model (and to an adapter), and is fitted once:

1. Hold out a calibration split with known answers, separate from any data the model will
   be evaluated on. Answer it on a server at `T = 1`.
2. Fit `T` on that split, typically by minimising the mean negative log-likelihood of the
   right option (a convex problem in `1/T`), then confirm it with the expected calibration
   error and the share of answers above 95% confidence. A server run is not needed per
   candidate `T`: the full-precision `probabilities` of the `T = 1` answers determine every
   tempered answer, since `softmax(log p / T)` equals `softmax(z / T)` (up to options whose
   probability was already 0 at `T = 1`).
3. Publish the fitted value with the model: the model card should state the recommended
   `--decision-temperature`, and the serving command should use it.

Refit when the model, the adapter or the quantization changes. The chosen option is the same
at every `T`, so accuracy does not move; the stated probabilities do, and with them a score
question's expected level.

## Example

```bash
curl -s http://127.0.0.1:8080/api/alpha/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-gemma4-26b-a4b",
    "state": {"ticket": "Order 8812 arrived late and the box was crushed. I want a refund."},
    "questions": {
      "refund": {"type": "noul", "instructions": "Is a refund requested?",
                 "criteria": {"true": "A refund is requested", "false": "No refund is requested"}},
      "tone": {"type": "choice", "instructions": "What is the tone?",
               "criteria": {"calm": "Calm", "annoyed": "Annoyed", "furious": "Furious"}}
    }
  }'
```

With an image, on a server started with `--vision --mmproj <projector>`:

```bash
curl -s http://127.0.0.1:8080/api/alpha/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model": "jev-gemma4-26b-a4b",
       "images": ["data:image/png;base64,iVBORw0KGgo..."],
       "state": {"hint": "Which of these is a mammal?", "image": "attached"},
       "questions": {"answer": {"type": "choice", "instructions": "Pick the mammal.",
                                "criteria": {"a": "shark", "b": "dolphin", "c": "trout"}}}}'
```
