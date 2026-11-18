# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


_MULTIMODAL_UNWRAP = {
    "Mistral3ForConditionalGeneration": "language_model",
}


def unwrap_multimodal(vllm_model):
    """Unwrap multimodal wrapper models to their underlying language model."""
    model_name = type(vllm_model).__name__
    attr = _MULTIMODAL_UNWRAP.get(model_name)
    if attr is not None:
        return getattr(vllm_model, attr)
    return vllm_model


def infer_model_from_vllm(
    vllm_model, blender, enable_sparse: bool = False, algorithm: str = "cacheblend"
):
    vllm_model = unwrap_multimodal(vllm_model)
    model_name = type(vllm_model).__name__

    if algorithm == "epic":
        # First Party
        from lmcache.v1.compute.models.epic_llama import (
            LMCEPICGlm4Model,
            LMCEPICLlamaModel,
            LMCEPICQwen3Model,
        )

        if model_name == "LlamaForCausalLM":
            return LMCEPICLlamaModel(vllm_model, blender, enable_sparse)
        elif model_name == "Qwen3ForCausalLM":
            return LMCEPICQwen3Model(vllm_model, blender, enable_sparse)
        elif model_name == "Glm4ForCausalLM":
            return LMCEPICGlm4Model(vllm_model, blender, enable_sparse)
        else:
            raise NotImplementedError(
                f"EPIC not implemented for model type {model_name}."
            )
    else:
        # CacheBlend (default)
        if model_name == "LlamaForCausalLM":
            # First Party
            from lmcache.v1.compute.models.llama import LMCLlamaModel

            return LMCLlamaModel(vllm_model, blender, enable_sparse)
        elif model_name == "Qwen3ForCausalLM":
            # First Party
            from lmcache.v1.compute.models.qwen3 import LMCQwen3Model

            return LMCQwen3Model(vllm_model, blender, enable_sparse)
        elif model_name == "Glm4ForCausalLM":
            # First Party
            from lmcache.v1.compute.models.glm4 import LMCGlm4Model

            return LMCGlm4Model(vllm_model, blender, enable_sparse)
        else:
            raise NotImplementedError(
                f"Model type {model_name} is not supported in LMCache."
            )


class VLLMModelTracker:
    _vllm_models: Dict[str, nn.Module] = {}

    @classmethod
    def register_model(
        cls,
        instance_id: str,
        vllm_model: nn.Module,
    ):
        """
        Register a vllm model by instance_id.
        """
        logger.info(f"Registering vllm model for {instance_id}")
        if instance_id not in cls._vllm_models:
            cls._vllm_models[instance_id] = vllm_model
        else:
            logger.warning(
                f"vllm model for {instance_id} already registered, doing nothing."
            )

    @classmethod
    def get_model(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the vllm model by instance_id.
        """
        if instance_id not in cls._vllm_models:
            raise ValueError(f"vllm model for {instance_id} not found.")
        return cls._vllm_models[instance_id]
