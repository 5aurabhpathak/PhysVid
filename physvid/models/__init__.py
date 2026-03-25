from typing import List

import numpy as np
import torch
from transformers.models.qwen2.modular_qwen2 import Qwen2DecoderLayer
from transformers.models.t5.modeling_t5 import T5Block

from .wan.wan_base.modules.model import WanAttentionBlock
from .wan.wan_wrapper import (WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper,
                              LocalConditioningWanDiffusionWrapper)

DIFFUSION_NAME_TO_CLASS = {
    "wan": WanDiffusionWrapper,
    "local_conditioning_wan": LocalConditioningWanDiffusionWrapper,
}


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


TEXTENCODER_NAME_TO_CLASS = {
    "wan": WanTextEncoder,
    "local_conditioning_wan": WanTextEncoder
}


class PhysicsTextEncoder:
    def __init__(self, config):
        self.config = config
        self.num_blocks = self.config.image_or_video_shape[1] // self.config.num_frame_per_block
        self.num_frame_per_block = np.ceil(self.config.video_data_shape[0] / self.num_blocks).astype(int)
        self.text_encoder = None
        self.physics_observer = None

    def set_encoder(self, encoder):
        self.text_encoder = encoder

    def set_physics_observer(self, physics_observer):
        assert not self.config.decoupled_vlm_mode, "Decoupled VLM does mode not require setting physics observer."
        self.physics_observer = physics_observer

    def __call__(self, text_prompts: List[str]) -> dict:
        local_pos_prompt_embeds = []
        for prompt in text_prompts:
            positive_prompts = prompt

            # randomly replace some of the positive prompts with empty strings for regularization
            if self.config.empty_prompt_prob > 0.:
                rnd = np.random.rand(len(positive_prompts))
                positive_prompts = [p if rand > self.config.empty_prompt_prob else ''
                                    for p, rand in zip(positive_prompts, rnd)]

            batched_prompts = positive_prompts
            batched_embeds = self.text_encoder(batched_prompts)
            batched_embeds = batched_embeds["prompt_embeds"]
            positive_embeds = batched_embeds.view(self.num_blocks * batched_embeds.shape[1],
                                                                    batched_embeds.shape[2])

            local_pos_prompt_embeds.append(positive_embeds)

        local_pos_prompt_embeds = torch.stack(local_pos_prompt_embeds, dim=0)
        return {
            "local_pos_prompt_embeds": local_pos_prompt_embeds,
        }


LOCAL_TEXTENCODER_NAME_TO_CLASS = {
    "conditioning": PhysicsTextEncoder,
}


def get_text_encoder_wrapper(model_name):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


def get_local_text_encoder_wrapper(model_name):
    if 'conditioning' in model_name:
        return LOCAL_TEXTENCODER_NAME_TO_CLASS['conditioning']
    else:
        raise ValueError(f"No local text encoder for model {model_name}")


VAE_NAME_TO_CLASS = {
    "wan": WanVAEWrapper,
    "local_conditioning_wan": WanVAEWrapper
}


def get_vae_wrapper(model_name):
    return VAE_NAME_TO_CLASS[model_name]


BLOCK_NAME_TO_BLOCK_CLASS = {
    "T5Block": T5Block,
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "WanAttentionBlock": WanAttentionBlock
}


def get_block_class(model_name):
    return BLOCK_NAME_TO_BLOCK_CLASS[model_name]
