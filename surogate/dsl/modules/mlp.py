"""MLP modules for the Python DSL.

``_hf_mapping_defaults_`` is a class-level attribute so
``surogate.dsl.hf`` can read it without instantiating.
"""

from __future__ import annotations

from typing import Any

from ..activations import activation_from_name
from ..dim import B, Dim, T
from ..hf import fuse
from ..mlp import MLPConfig
from ..nn import Module, Proxy, Tracer
from ..specs import LoRATarget


class GenericMLP(Module):
    """Config-driven MLP.

    Covers SwiGLU (gated, fused gate+up), gated MLP (separate gate/up),
    and plain up+down MLP.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        config: Any = None,
    ) -> None:
        super().__init__()
        self.config: MLPConfig = config if config is not None else MLPConfig()
        self.d_model = d_model
        self.d_ff = d_ff
        self.C = Dim("C")
        self.M = Dim("M")
        self.MUp = 2 * self.M  # used when gate+up are fused

        cfg = self.config
        if cfg.gated and cfg.fuse_gate_up:
            self._hf_mapping_defaults_ = {
                "up_weight": fuse(
                    "{prefix}.up_proj.weight",
                    "{prefix}.gate_proj.weight",
                    dim=0,
                ),
                "down_weight": "{prefix}.down_proj.weight",
            }
        elif cfg.gated:
            self._hf_mapping_defaults_ = {
                "gate_weight": "{prefix}.gate_proj.weight",
                "up_weight": "{prefix}.up_proj.weight",
                "down_weight": "{prefix}.down_proj.weight",
            }
        else:
            self._hf_mapping_defaults_ = {
                "up_weight": "{prefix}.up_proj.weight",
                "down_weight": "{prefix}.down_proj.weight",
            }

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args
        cfg = self.config

        _ff = self.d_ff
        _hidden = self.d_model

        # -- params -----------------------------------------------------
        # The fused SwiGLU weight concatenates up then gate along dim-0
        # (up_proj first, gate_proj second — matches the HF fuse() mapping
        # above). Each logical projection is a separately-addressable
        # LoRA target on the fused weight.
        if cfg.gated and cfg.fuse_gate_up:
            fused_targets = [
                LoRATarget(name="up", offset=0, size=_ff),
                LoRATarget(name="gate", offset=_ff, size=_ff),
            ]
            up_w = tracer.register_param("up_weight", ("MUp", "C"), lora_targets=fused_targets)
            down_w = tracer.register_param(
                "down_weight",
                ("C", "M"),
                lora_targets=[LoRATarget(name="down", size=_hidden)],
            )
            gate_w = None
        elif cfg.gated:
            gate_w = tracer.register_param(
                "gate_weight",
                (self.d_ff, "C"),
                lora_targets=[LoRATarget(name="gate", size=_ff)],
            )
            up_w = tracer.register_param(
                "up_weight",
                (self.d_ff, "C"),
                lora_targets=[LoRATarget(name="up", size=_ff)],
            )
            down_w = tracer.register_param(
                "down_weight",
                ("C", self.d_ff),
                lora_targets=[LoRATarget(name="down", size=_hidden)],
            )
        else:
            up_w = tracer.register_param(
                "up_weight",
                ("M", "C"),
                lora_targets=[LoRATarget(name="up", size=_ff)],
            )
            down_w = tracer.register_param(
                "down_weight",
                ("C", "M"),
                lora_targets=[LoRATarget(name="down", size=_hidden)],
            )
            gate_w = None

        # -- activation slots -------------------------------------------
        up_slot = act_slot = None
        if cfg.gated and cfg.fuse_gate_up:
            up_slot = tracer.register_activation(
                "up",
                ("B", "T", "MUp"),
                aliases=["up_flat"],
                share_policy="when_recomputed",
            )
            act_slot = tracer.register_activation(
                "act",
                ("B", "T", "M"),
                aliases=["act_flat"],
                share_policy="when_recomputed",
                description="Fused gated MLP activation output",
            )
            # Fused path: the 2D output of the down matmul is aliased
            # back to the 3D slot so existing tracing code / name remaps
            # (``mlp_down_flat``) keep working.
            down_aliases = ["down_flat"]
        else:
            down_aliases = None
        down_slot = tracer.register_activation(
            "down",
            ("B", "T", "C"),
            aliases=down_aliases,
            share_policy="when_recomputed",
            description="MLP down projection output",
        )

        # -- graph -------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        act_table = {
            "silu": g.silu,
            "gelu": g.gelu,
            "relu2": g.relu2,
        }

        if cfg.gated and cfg.fuse_gate_up:
            up_flat = g.matmul(
                x_flat,
                up_w,
                transpose="NT",
                out_name=tracer.prefixed("up_flat"),
            )
            up = g.view(
                up_flat,
                shape=[B, T, self.MUp],
                out_name=up_slot,
            )
            gated_op = cfg.activation.gated_cpp_op or "swiglu"
            if gated_op == "swiglu":
                act = g.swiglu(up, out_name=act_slot)
            elif gated_op == "gpt_oss_moe_act":
                act = g.gpt_oss_moe_act(
                    up,
                    out_name=act_slot,
                    **dict(cfg.activation.attrs),
                )
            else:
                raise ValueError(f"Unsupported fused gated activation '{gated_op}' in GenericMLP")
            act_flat = g.view(
                act,
                shape=[B * T, self.M],
                out_name=tracer.prefixed("act_flat"),
            )
        elif cfg.gated:
            gate_flat = g.matmul(x_flat, gate_w, transpose="NT")
            up_flat = g.matmul(x_flat, up_w, transpose="NT")
            act_fn = act_table.get(cfg.activation.cpp_op)
            if act_fn is None:
                raise ValueError(f"Unsupported activation '{cfg.activation.cpp_op}' for non-fused gated MLP")
            gate_act = act_fn(gate_flat, out_name=tracer.prefixed("gate_act"))
            act_flat = g.mul(gate_act, up_flat)
        else:
            up_flat = g.matmul(x_flat, up_w, transpose="NT")
            act_fn = act_table.get(cfg.activation.cpp_op)
            if act_fn is None:
                raise ValueError(f"Unsupported activation '{cfg.activation.cpp_op}' for simple MLP")
            act_flat = act_fn(up_flat, out_name=tracer.prefixed("act_flat"))

        out_flat = g.matmul(
            act_flat,
            down_w,
            transpose="NT",
            out_name=tracer.prefixed("down_flat"),
        )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=down_slot,
        )

        return Proxy(down_slot, out)


class SimpleMLP(Module):
    """Simple MLP with configurable activation (relu2/silu/gelu)."""

    _hf_mapping_defaults_ = {
        "up_weight": "{prefix}.up_proj.weight",
        "up_bias": "{prefix}.up_proj.bias",
        "down_weight": "{prefix}.down_proj.weight",
        "down_bias": "{prefix}.down_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu2",
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.use_bias = use_bias
        self.C = Dim("C")
        self.M = Dim("M")

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args

        _ff = self.d_ff
        _hidden = self.d_model

        # -- params ----------------------------------------------------------
        up_w = tracer.register_param(
            "up_weight",
            ("M", "C"),
            lora_targets=[LoRATarget(name="up", size=_ff)],
        )
        up_b = tracer.register_param("up_bias", ("M",), when="use_bias")
        down_w = tracer.register_param(
            "down_weight",
            ("C", "M"),
            lora_targets=[LoRATarget(name="down", size=_hidden)],
        )
        down_b = tracer.register_param("down_bias", ("C",), when="use_bias")

        # -- activation slots ------------------------------------------------
        up_slot = tracer.register_activation(
            "up",
            ("B", "T", "M"),
            aliases=["up_flat"],
            save=True,
            share_policy="when_recomputed",
        )
        act_slot = tracer.register_activation(
            "act",
            ("B", "T", "M"),
            aliases=["act_flat"],
            save=True,
            share_policy="when_recomputed",
            description="MLP activation output",
        )
        down_slot = tracer.register_activation(
            "down",
            ("B", "T", "C"),
            aliases=["down_flat"],
            share_policy="when_recomputed",
            description="MLP down projection output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        # Up projection
        if self.use_bias:
            up_flat = g.matmul_bias(
                x_flat,
                up_w,
                up_b,
                transpose="NT",
                out_name=tracer.prefixed("up_flat"),
            )
        else:
            up_flat = g.matmul(
                x_flat,
                up_w,
                transpose="NT",
                out_name=tracer.prefixed("up_flat"),
            )
        up = g.view(
            up_flat,
            shape=[B, T, self.M],
            out_name=up_slot,
        )

        # Activation
        act_map = {"relu2": g.relu2, "silu": g.silu, "gelu": g.gelu}
        act_fn = act_map.get(self.activation, g.relu2)
        act = act_fn(up, out_name=act_slot)

        # Down projection
        act_flat = g.view(
            act,
            shape=[B * T, self.M],
            out_name=tracer.prefixed("act_flat"),
        )
        if self.use_bias:
            out_flat = g.matmul_bias(
                act_flat,
                down_w,
                down_b,
                transpose="NT",
                out_name=tracer.prefixed("down_flat"),
            )
        else:
            out_flat = g.matmul(
                act_flat,
                down_w,
                transpose="NT",
                out_name=tracer.prefixed("down_flat"),
            )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=down_slot,
        )

        return Proxy(down_slot, out)


# Class-level default mapping so ``hf.py`` can read
# ``GenericMLP._hf_mapping_defaults_`` without instantiating. Instances
# override via ``self._hf_mapping_defaults_`` in __init__ when the
# config changes gate/up fusing.
GenericMLP._hf_mapping_defaults_ = {
    "up_weight": fuse(
        "{prefix}.up_proj.weight",
        "{prefix}.gate_proj.weight",
        dim=0,
    ),
    "down_weight": "{prefix}.down_proj.weight",
}

# ----------------------------------------------------------------------------
# Backwards-compat aliases
# ----------------------------------------------------------------------------


class SwiGLUMLP(GenericMLP):
    """Default SwiGLU MLP (fused gate+up, SiLU-style gating)."""

    _hf_mapping_defaults_ = {
        "up_weight": fuse(
            "{prefix}.up_proj.weight",
            "{prefix}.gate_proj.weight",
            dim=0,
        ),
        "down_weight": "{prefix}.down_proj.weight",
    }


class GatedMLP(GenericMLP):
    """Separate gate/up gated MLP (e.g. Gemma4's GELU gated MLP)."""

    _hf_mapping_defaults_ = {
        "gate_weight": "{prefix}.gate_proj.weight",
        "up_weight": "{prefix}.up_proj.weight",
        "down_weight": "{prefix}.down_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",
    ) -> None:
        super().__init__(
            d_model,
            d_ff,
            config=MLPConfig(
                activation=activation_from_name(activation),
                gated=True,
                fuse_gate_up=False,
            ),
        )


# ============================================================================
# Registry population
# ============================================================================
#
# The ``MLP.*`` namespace constants declared in ``surogate.dsl.mlp`` use
# ``factory=object`` as a placeholder because the concrete ``GenericMLP``
# class lives here, not there. Overwrite them now with real specs.

from ..activations import Activation  # noqa: E402
from ..mlp import MLP, MLPSpec, _register as _register_mlp  # noqa: E402

MLP.SWIGLU = _register_mlp(
    MLPSpec(
        name="swiglu",
        factory=GenericMLP,
        config=MLPConfig(activation=Activation.SILU, gated=True, fuse_gate_up=True),
    )
)
MLP.GELU_MLP = _register_mlp(
    MLPSpec(
        name="gelu_mlp",
        factory=GenericMLP,
        config=MLPConfig(activation=Activation.GELU, gated=False, fuse_gate_up=False),
    )
)
MLP.SIMPLE = _register_mlp(
    MLPSpec(
        name="simple",
        factory=GenericMLP,
        config=MLPConfig(activation=Activation.SILU, gated=False, fuse_gate_up=False),
    )
)
