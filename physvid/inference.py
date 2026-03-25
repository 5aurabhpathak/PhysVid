from typing import List

import torch
from torch import distributed as dist
from tqdm import tqdm

from physvid.models import get_diffusion_wrapper, get_text_encoder_wrapper, get_vae_wrapper
from physvid.models.physics_observer import LocalPromptGeneratorFromGlobalCaption


class InferencePipeline(torch.nn.Module):
    def __init__(self, args, device, dtype):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)().eval()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
        self.guidance_scale = args.guidance_scale
        self.device = device
        self.negative_prompt = args.negative_prompt
        self.dtype = dtype
        self.unconditional_dict = None
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    @torch.no_grad()
    def inference(self, noise: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        conditional_dict, unconditional_dict = self.get_text_conditioning(text_prompts)
        return self.denoise(noise, conditional_dict, unconditional_dict)

    def denoise(self, noise: torch.Tensor, conditional_dict: dict, unconditional_dict: dict) -> torch.Tensor:
        # -----cyclic inference-----
        t = torch.ones(noise.shape[:2], device=noise.device, dtype=torch.int64)
        for index, current_timestep in enumerate(tqdm(self.scheduler.timesteps, desc='denoising...',
                                                      dynamic_ncols=True,
                                                      disable=not self.is_main_process)):

            timestep = t * current_timestep

            x0_pred_cond = self.generator(
                noisy_image_or_video=noise,
                conditional_dict=conditional_dict,
                timestep=timestep)

            if self.guidance_scale > 0.:
                x0_pred_uncond = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=unconditional_dict,
                    timestep=timestep)

                x0_pred = x0_pred_uncond + self.guidance_scale * (
                        x0_pred_cond - x0_pred_uncond
                )
            else:
                x0_pred = x0_pred_cond

            flow_pred = self.generator._convert_x0_to_flow_pred(
                x0_pred=x0_pred.flatten(0, 1),
                xt=noise.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
                scheduler=self.scheduler
            ).unflatten(0, x0_pred.shape[:2])

            noise = self.scheduler.step(
                flow_pred.flatten(0, 1),
                self.scheduler.timesteps[index] * t.flatten(0, 1),
                noise.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2]).to(self.dtype)

        pred_video = (self.vae.decode_to_pixel(noise, micro_batch_size=1) * 0.5 + 0.5).clamp(0, 1)
        return pred_video

    def get_text_conditioning(self, text_prompts: List[str]):
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        if self.unconditional_dict is None:
            self.unconditional_dict = self.text_encoder(
                text_prompts=[self.negative_prompt] * len(text_prompts)
            )
        unconditional_dict = self.unconditional_dict
        return conditional_dict, unconditional_dict


class LocalConditioningInferencePipeline(InferencePipeline):
    def __init__(self, args, device, dtype):
        super().__init__(args, device, dtype)
        self.local_prompt_provider = LocalPromptGeneratorFromGlobalCaption()
        print('Using Local Prompt Generator to enable local prompt generation.')

        self.num_blocks = args.num_blocks
        self.empty_embeds = self.text_encoder([""] * self.num_blocks)["prompt_embeds"]
        self.empty_embeds = self.empty_embeds.view(self.num_blocks * self.empty_embeds.shape[1],
                                                   self.empty_embeds.shape[2]).to(device, dtype)
        self.generator.model.num_frame_per_block = args.num_frame_per_block

    def get_text_conditioning(self, text_prompts: List[str]):
        conditional_dict, unconditional_dict = super().get_text_conditioning(text_prompts)
        pos_embeds, neg_embeds = self.create_local_embeds(text_prompts)
        conditional_dict['local_prompt_embeds'] = pos_embeds
        unconditional_dict['local_prompt_embeds'] = neg_embeds
        return conditional_dict, unconditional_dict

    def call_physicist(self, caption):
        if caption.strip() == '':
            return [''] * self.num_blocks, [''] * self.num_blocks

        responses_list = self.local_prompt_provider.create_local_prompts(caption)

        negative_responses_list = []
        for response in responses_list:
            if response.strip() == '':
                negative_responses_list.append('')
                continue

            negative_response = self.local_prompt_provider.create_local_prompts(response, negative=True)
            negative_sentences = set(negative_response.strip('.').split('. '))
            positive_sentences = set(response.strip('.').split('. '))
            negative_sentences -= positive_sentences
            negative_response = '. '.join(negative_sentences)

            if negative_response.strip() != '':
                negative_response = negative_response + '.'
            # negative_response = ''

            if self.is_main_process:
                tqdm.write(f'[rank 0] Positive: {response}\nNegative: {negative_response}')

            negative_responses_list.append(negative_response)

        return responses_list, negative_responses_list

    def create_local_embeds(self, text_prompts: List[str]):
        def get_embeds(prompts):
            batched_embeds = self.text_encoder(prompts)
            batched_embeds = batched_embeds["prompt_embeds"]
            embeds = batched_embeds.view(self.num_blocks * batched_embeds.shape[1],
                                         batched_embeds.shape[2])
            return embeds

        batch_local_pos_prompts, batch_local_neg_prompts = [], []
        for prompt in text_prompts:
            positive_prompts, negative_prompts = self.call_physicist(prompt)
            pos_embeds = get_embeds(positive_prompts)
            neg_embeds = get_embeds(negative_prompts)
            # neg_embeds = self.empty_embeds
            batch_local_pos_prompts.append(pos_embeds)
            batch_local_neg_prompts.append(neg_embeds)

        local_pos_embeds = torch.stack(batch_local_pos_prompts, dim=0)
        local_neg_embeds = torch.stack(batch_local_neg_prompts, dim=0)
        return local_pos_embeds, local_neg_embeds
