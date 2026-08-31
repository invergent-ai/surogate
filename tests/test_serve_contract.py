"""The serving engine and the training DSL must agree about the model.

An architecture is described three times in this repo — the DSL declaration that
training compiles, the serve target's `config.h`, and the converter's artifact
inventory. Nothing used to notice when they drifted apart, and the failure mode is
expensive: a converter and a binder that disagree about a fused row count produce
a hundred-gigabyte artifact that fails at load, and one that disagrees about a
layer index produces an artifact that loads and is quietly wrong.

These tests run the two checkers on every target that has migrated, so drift is a
red test rather than an incident. They need no GPU, no weights and no built
extension — only a checkpoint's `config.json`.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GENERATE = REPO / "surogate/serve/tools/generate"
TARGETS = REPO / "csrc/src/serve/targets"

if str(GENERATE) not in sys.path:
    sys.path.insert(0, str(GENERATE))


@pytest.fixture(scope="module")
def emitters():
    check_contract = pytest.importorskip("check_contract")
    check_roundtrip = pytest.importorskip("check_roundtrip")
    from_dsl = pytest.importorskip("from_dsl")
    emit_config = pytest.importorskip("emit_config")
    return check_contract, check_roundtrip, from_dsl, emit_config


#: Targets whose `config.h` is emitted from the declaration and must match byte
#: for byte. Anything here has fully migrated.
GENERATED_TARGETS = ("qwen3_5_0_8b", "qwen3_5_4b")

#: Hand-written targets, checked value-by-value instead: forcing a generator to
#: reproduce prose that records *why* a constant holds would relocate the
#: duplication rather than remove it.
CHECKED_TARGETS = (("qwen4exp", "models/Qwen3.8-Flash-Next-frontend"),)


@pytest.mark.parametrize("target", GENERATED_TARGETS)
def test_generated_target_reproduces_from_declaration(emitters, target):
    check_contract, check_roundtrip, from_dsl, emit_config = emitters
    config_path = check_roundtrip.resolve_config(target)
    if config_path is None or not config_path.exists():
        pytest.skip(f"no checkpoint config for {target} (set {check_roundtrip.TARGETS[target][0]})")

    hf_config = json.loads(config_path.read_text())
    architecture = (hf_config.get("architectures") or [hf_config.get("model_type")])[0]
    spec = from_dsl.from_dsl(architecture, hf_config, name=target)

    committed = (TARGETS / target / "impl" / "config.h").read_text()
    assert emit_config.emit_config_h(spec) == committed, (
        f"{target}/impl/config.h no longer matches what the {architecture} declaration "
        f"emits; the declaration and the serve target disagree about the model"
    )


@pytest.mark.parametrize("target", GENERATED_TARGETS)
def test_generated_target_carries_adapter_slices(emitters, target):
    """The contract must reach serving with its LoRA slices intact — that is what
    lets an adapter trained on one logical projection land on the right row range
    of a fused serve tensor."""

    check_contract, check_roundtrip, from_dsl, _ = emitters
    config_path = check_roundtrip.resolve_config(target)
    if config_path is None or not config_path.exists():
        pytest.skip(f"no checkpoint config for {target}")

    hf_config = json.loads(config_path.read_text())
    architecture = (hf_config.get("architectures") or [hf_config.get("model_type")])[0]
    spec = from_dsl.from_dsl(architecture, hf_config, name=target)

    assert spec.params, "the contract carries no parameters"
    adapters = [p for p in spec.params if p.is_lora_target]
    assert adapters, "the contract carries no adapter-addressable parameters"
    for param in adapters:
        for slice_ in param.lora:
            assert slice_.size > 0, f"{param.dsl_name}: empty adapter slice {slice_.name}"
            assert slice_.offset >= 0, f"{param.dsl_name}: negative offset {slice_.name}"
    fused = [p for p in adapters if len(p.lora) > 1]
    assert fused, "no fused projection carries per-slice offsets; serving LoRA would guess"


@pytest.mark.parametrize("target,model_dir", CHECKED_TARGETS)
def test_hand_written_target_agrees_with_declaration(emitters, target, model_dir):
    check_contract, _, _, _ = emitters
    path = REPO / model_dir
    if not (path / "config.json").exists():
        pytest.skip(f"no checkpoint config at {model_dir}")
    assert check_contract.check(target, str(path), TARGETS) == 0, (
        f"{target} disagrees with its declaration; see the DISAGREE lines above"
    )


def test_text_struct_scoping_is_not_fooled_by_sibling_structs(emitters):
    """`DFlashConfig` reuses the name `layers`. Parsing the whole header once
    substituted the draft head's value for the text stack's, so the scoping this
    depends on gets its own test."""

    check_contract, _, _, _ = emitters
    header = """
    struct TextConfig { static constexpr int layers = 48; };
    struct DFlashConfig { static constexpr int layers = 1; };
    """
    assert check_contract.parse_constants(header)["layers"] == 48
