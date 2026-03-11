"""Runtime GPTQModel support for MiniCPM-SALA.

GPTQModel AutoCompat infers a single module tree from layer 0, which is not
sound for MiniCPM-SALA because decoder layers mix sparse MiniCPM attention and
lightning attention families. This module registers an explicit model
definition at preprocess time so GPTQModel uses a stable, architecture-aware
layout for quantize/save flows.
"""

from __future__ import annotations

from gptqmodel.models.base import BaseQModel


MODEL_TYPE = "minicpm_sala"


class MiniCPMSALAGPTQ(BaseQModel):
    require_trust_remote_code = True
    layer_modules_strict = False
    pre_lm_head_norm_module = "model.norm"
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # MiniCPM-SALA mixes sparse MiniCPM attention layers with lightning layers.
    # We keep optional gating/norm helper modules in the tree with :! so GPTQ's
    # traversal understands the structure, but they remain native weights.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:0:!",
                "k_norm:0:!",
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_norm:1:!",
                "z_proj:1:!",
                "o_gate:1:!",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]


def register_minicpm_sala_gptq_model() -> None:
    from gptqmodel.models import auto as auto_module

    auto_module.MODEL_MAP[MODEL_TYPE] = MiniCPMSALAGPTQ

    supported_models = getattr(auto_module, "SUPPORTED_MODELS", None)
    if supported_models is None:
        return

    if isinstance(supported_models, set):
        supported_models.add(MODEL_TYPE)
        return

    if isinstance(supported_models, list):
        if MODEL_TYPE not in supported_models:
            supported_models.append(MODEL_TYPE)
        return

    if isinstance(supported_models, tuple) and MODEL_TYPE not in supported_models:
        auto_module.SUPPORTED_MODELS = (*supported_models, MODEL_TYPE)