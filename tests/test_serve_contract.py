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

#: Every target whose converter inventory is derivable from the declaration, with
#: where its checkpoint config lives. `qwen3_6_27b` and `qwen3_8_27b` are absent
#: only because their configs are not on this machine — nothing about them is
#: known to be undeclarable.
#: (target, config source, what the target's C++ binder consumes). The engine
#: refuses to load an artifact holding an object no binder consumes, so an
#: artifact carries a capability's objects only when its target implements it —
#: the three Qwen3.5 targets and qwen4exp are text-only in C++ today.
INVENTORY_TARGETS = (
    ("qwen4exp", "dir:models/Qwen3.8-Flash-Next-frontend", {"text"}),
    ("qwen3_5_0_8b", "hub:models--Qwen--Qwen3.5-0.8B", {"text"}),
    ("qwen3_5_2b", "hub:models--Qwen--Qwen3.5-2B", {"text"}),
    ("qwen3_5_4b", "hub:models--Qwen--Qwen3.5-4B", {"text"}),
    ("qwen3_6_35b_a3b", "hub:models--Qwen--Qwen3.6-35B-A3B", {"text", "vision", "dflash"}),
)


def _resolve_source(source: str) -> pathlib.Path | None:
    kind, value = source.split(":", 1)
    if kind == "dir":
        path = REPO / value / "config.json"
        return path if path.exists() else None
    import glob as _glob
    hits = sorted(_glob.glob(
        str(pathlib.Path.home() / ".cache/huggingface/hub" / value / "snapshots/*/config.json")))
    return pathlib.Path(hits[0]) if hits else None


@pytest.mark.parametrize("target,source,capabilities", INVENTORY_TARGETS)
def test_artifact_inventory_derives_from_the_declaration(emitters, target, source, capabilities):
    """The whole artifact — text stack, MTP head, vision tower, DFlash scorer —
    must be what the declaration implies, name, shape and numeric width."""

    import importlib

    emit_inventory = pytest.importorskip("emit_inventory")
    config_path = _resolve_source(source)
    if config_path is None:
        pytest.skip(f"no checkpoint config for {target}")

    hf_config = json.loads(config_path.read_text())
    architecture = (hf_config.get("architectures") or [hf_config.get("model_type")])[0]
    derived = emit_inventory.inventory_for(architecture, hf_config, capabilities=capabilities)

    inventory = importlib.import_module(f"surogate.serve.tools.convert.{target}.inventory")
    committed = {s.name: (tuple(s.shape), s.format) for s in inventory.TENSOR_SPECS}
    emitted = {o["name"]: (o["shape"], o["format"]) for o in derived}

    assert set(emitted) == set(committed), (
        f"{target}: only-declaration={sorted(set(emitted) - set(committed))[:4]}, "
        f"only-converter={sorted(set(committed) - set(emitted))[:4]}"
    )
    bad = {
        n: (committed[n], emitted[n]) for n in committed
        if committed[n][0] != emitted[n][0]
        or not emit_inventory.formats_agree(emitted[n][1], committed[n][1], inventory)
    }
    assert not bad, f"{target}: shape/width disagreements: {list(bad.items())[:4]}"


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


def test_declaration_describes_more_than_any_target_exports():
    """The declaration is the model, not the artifact. Every Qwen3.5 checkpoint
    has a vision tower; the text-only targets do not export it because their
    binders reject it. Losing that distinction would mean the declaration had
    quietly become a description of one target's export instead."""

    emit_inventory = pytest.importorskip("emit_inventory")
    config_path = _resolve_source("hub:models--Qwen--Qwen3.5-2B")
    if config_path is None:
        pytest.skip("no Qwen3.5-2B config")

    hf_config = json.loads(config_path.read_text())
    architecture = (hf_config.get("architectures") or [hf_config.get("model_type")])[0]
    everything = emit_inventory.inventory_for(architecture, hf_config)
    text_only = emit_inventory.inventory_for(architecture, hf_config, capabilities={"text"})

    tower = [o for o in everything if o["name"].startswith("vision/")]
    assert tower, "the declaration should describe this model's vision tower"
    assert not [o for o in text_only if o["name"].startswith("vision/")]
    assert len(everything) > len(text_only)


@pytest.mark.parametrize("target,model_dir", CHECKED_TARGETS)
def test_fused_serve_objects_name_their_components(emitters, target, model_dir):
    """Every fused artifact object must say which declared parameters compose it,
    in row order — that mapping is what will place a LoRA adapter on the right
    rows of a fused serve tensor. Objects for subsystems the training graph does
    not yet cover are allowed to have none."""

    emit_inventory = pytest.importorskip("emit_inventory")
    from surogate.dsl.ir_builder import load_hf_config, resolve_architecture

    path = REPO / model_dir
    if not (path / "config.json").exists():
        pytest.skip(f"no checkpoint config at {model_dir}")

    hf_config = load_hf_config(str(path))
    derived = emit_inventory.inventory_for(resolve_architecture(hf_config), hf_config)

    # Subsystems a serving artifact carries but the training graph does not yet
    # compute: their objects are declared so the artifact is fully described, and
    # they name no source parameters because there are none to name.
    deferred = ("/indexer/", "/ple/", "text/ple/", "vision/", "mtp/", "dflash/",
                "draft_head")
    uncomposed = [
        obj["name"] for obj in derived
        if not obj["components"] and not any(mark in obj["name"] for mark in deferred)
    ]
    assert not uncomposed, f"artifact objects with no declared source: {uncomposed[:6]}"

    fused = [obj for obj in derived if len(obj["components"]) > 1]
    assert fused, "no fused objects declared; the composition map would be untested"


#: Converters that carry a conversion recipe alongside their inventory. The recipe
#: says where each artifact object comes from in the checkpoint, and it must cover
#: the inventory exactly — an inventory that grows without its recipe produces a
#: converter that cannot build the artifact it promises.
RECIPE_TARGETS = (
    "qwen3_5_0_8b",
    "qwen3_5_2b",
    "qwen3_5_4b",
    "qwen3_6_27b",
    "qwen3_6_35b_a3b",
)


@pytest.mark.parametrize("target", RECIPE_TARGETS)
def test_conversion_recipe_covers_its_inventory(target):
    """Importing a recipe runs its own coverage validation.

    This test exists because the inventory checks above did not: vision objects
    were added to three inventories whose recipes still returned `()`, and the
    inventory tests passed while conversion was broken. The two halves of a
    converter have to be checked together.
    """

    import importlib

    importlib.import_module(f"surogate.serve.tools.convert.{target}.recipe")


@pytest.mark.parametrize("target", RECIPE_TARGETS)
def test_converter_preflight_accepts_its_own_inventory(target):
    """The third half. `convert.py` guards conversion with `preflight_inventory`,
    which compares hardcoded section counts against the inventory — a fourth
    restatement of the same facts, and the one that actually blocks a conversion
    run. Growing an inventory without updating it fails at convert time, long
    after the tests have gone green, so it is checked here too."""

    import importlib

    module = importlib.import_module(f"surogate.serve.tools.convert.{target}.convert")
    preflight = getattr(module, "preflight_inventory", None)
    if preflight is None:
        pytest.skip(f"{target} has no preflight_inventory")
    preflight()
