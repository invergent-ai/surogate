# Contributing to NInfer

High-quality contributions are welcome. A pull request is expected to arrive as a reasoned,
validated implementation, not as a request for the maintainer to discover its basic correctness
problems, redesign its core approach, or finish its verification.

## Governing engineering rules

All applicable engineering requirements in [`AGENTS.md`](AGENTS.md) are mandatory for every
contribution, regardless of whether the code was written by a human or generated with AI.

In particular:

- choose the technically strongest coherent solution for the requested outcome;
- respect the documented architecture and ownership boundaries;
- do not preserve superseded project-owned compatibility paths, aliases, or fallbacks;
- do not add speculative generality, generic frameworks, or abstractions for unsupported products;
- validate numerical work against an independent mathematical oracle with criteria appropriate to
  the actual semantic contract and supported shapes;
- support performance claims with measurements at the level being claimed;
- update affected implementations, tests, tools, and active documentation together; and
- run focused verification that is sufficient to establish the changed behavior and its material
  claims.

The summary above does not replace `AGENTS.md`. Contributors must read and follow the applicable
requirements there before submitting a change.

## Contributor responsibility

Before opening a pull request, inspect the complete changed path, including boundary conditions,
error classification, state propagation, resource ownership, and externally observable behavior.
Do not use broad exception handling, blind retry loops, or heuristic fallback as substitutes for a
resource calculation, explicit validation, or a correctly defined recovery boundary.

The contributor must be able to explain the implementation and why it is correct. Maintainer review
is not a substitute for basic validation or debugging by the contributor.

Discuss a change before implementation when it would revise an established architecture, ownership,
artifact, numerical, or external protocol contract.

## AI-assisted and AI-generated contributions

AI-assisted contributions are welcome, but the contributor remains fully responsible for every
line submitted.

If an implementation is generated entirely by AI, it will only be considered when produced by a
current state-of-the-art coding model. The exact model and version must be disclosed in the pull
request. Whether a model satisfies this requirement is determined by the maintainer at the time of
submission.

Use of a state-of-the-art model does not replace understanding the generated code, reviewing the
complete diff, reasoning about edge cases, or performing the required correctness, numerical, and
performance validation. Generated code that has not received these basic checks is not ready for
submission.

## Evidence and verification

Select evidence according to the claim being made:

- exact transformations and formats require exact comparison;
- floating-point operators require an independent FP32 or FP64 mathematical oracle and a justified
  numerical criterion;
- stateful behavior requires verification of the complete affected state transition;
- CUDA changes require relevant supported shapes and execution routes;
- operator-level performance claims require direct operator measurements;
- end-to-end performance claims require end-to-end measurements; and
- CLI or serving changes require the affected observable behavior, schema tests, and documentation
  to remain consistent.

Use the existing project workflows instead of inventing parallel verification paths:

- [`tests/README.md`](tests/README.md) for test organization and commands;
- [`bench/README.md`](bench/README.md) for product and operator benchmarks; and
- [`docs/maintainer/op-development.md`](docs/maintainer/op-development.md) for numerical Op
  development and qualification.

Run the focused checks relevant to the change. State exactly what was run and summarize the result.
If a relevant check could not be run, state that limitation and why. Do not claim validation that
was not performed.

## Pull request requirements

A pull request description must include:

- the concrete problem being solved;
- the design and why it is the appropriate solution;
- the affected behavior or contract;
- the exact verification commands and summarized results;
- the workload, hardware, and toolchain for any performance claim;
- any relevant check that was not run and the resulting limitation; and
- for an entirely AI-generated implementation, the exact model and version used.

Keep the change coherent and exclude unrelated cleanup. A public CLI, serving, artifact, or
documentation contract must be updated consistently across its implementation, tests, and active
documentation.

## Review policy

Review is a limited maintainer resource and is not guaranteed. A pull request may be closed without
a line-by-line review when it:

- violates an applicable requirement in `AGENTS.md`;
- contains evident correctness, boundary, lifetime, state, or error-handling problems;
- lacks relevant verification or makes unsupported numerical or performance claims;
- is generated without adequate contributor review;
- changes an established contract without prior agreement; or
- would require the maintainer to redesign or rewrite its core approach.

Closing such a pull request is a determination that it does not meet the project's review threshold.
It does not create an obligation for the maintainer to debug, repair, or complete the contribution.
