import torch
from torch import nn

from physvid.loss import get_denoising_loss
from physvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper,
    get_local_text_encoder_wrapper
)
from physvid.models.wan.wan_wrapper import WanDiffusionWrapper


class Finetune(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)

        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.generator.set_module_grad(
            module_grad=dict(model=True)
        )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.text_encoder.requires_grad_(False)

        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        if 'conditioning' in args.generator_name:
            self.generator.to(device=device, dtype=self.dtype)
            self.generator.load_base_state_dict()
            self.local_text_encoder = get_local_text_encoder_wrapper(model_name=self.generator_model_name)(config=args)

        self.vae = get_vae_wrapper(model_name=args.model_name)()
        self.vae.requires_grad_(False)
        self.vae.model.eval()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)

        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        self.args = args
        self.device = device
        self.scheduler = self.generator.get_scheduler()
        self.denoising_loss_func = get_denoising_loss(
            args.denoising_loss_type)()

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
                device)
        else:
            self.scheduler.alphas_cumprod = None

    def _get_noisy_input_all_steps(self, image_or_video_shape, clean_latent):
        b, f = image_or_video_shape[:2]
        timestep = torch.randint(
            0,
            self.num_train_timestep,
            (b, 1),
            device=self.device,
            dtype=torch.long
        ).expand(b, f)

        if self.timestep_shift > 1:
            timestep = self.timestep_shift * \
                              (timestep / 1000) / (
                                          1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000

        timestep = timestep.clamp(self.min_step, self.max_step).to(torch.long)
        noise = torch.randn_like(clean_latent)

        noisy_input = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])
        return noisy_input, noise, timestep

    def _run_generator(self, image_or_video_shape, conditional_dict, clean_latent):
        noisy_input, noise, timestep = self._get_noisy_input_all_steps(
            image_or_video_shape=image_or_video_shape,
            clean_latent=clean_latent
        )

        pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        pred_image_or_video = pred_image_or_video.type_as(noisy_input)

        return pred_image_or_video, noise, noisy_input, timestep

    def finetune_loss(self, image_or_video_shape, conditional_dict: dict, clean_latent: torch.Tensor):
        # Step 1: Run generator on noisy input
        pred_image, noise, noisy_input, timestep = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            clean_latent=clean_latent
        )

        # Step 2: Compute the flow matching loss
        loss, log_dict = self.compute_loss(
            image_or_video_shape=image_or_video_shape,
            pred_image=pred_image,
            noisy_input=noisy_input,
            timestep=timestep,
            clean_image=clean_latent,
            pure_noise=noise
        )

        return loss, log_dict

    def compute_loss(self, image_or_video_shape, pred_image: torch.Tensor, noisy_input: torch.Tensor,
                     timestep: torch.Tensor, clean_image, pure_noise):
        if self.args.denoising_loss_type == "flow":
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_image.flatten(0, 1),
                xt=noisy_input.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        elif self.args.denoising_loss_type == "noise":
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_image.flatten(0, 1),
                xt=noisy_input.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])
        elif self.args.denoising_loss_type == "v":
            flow_pred = None
            pred_fake_noise = None
        else:
            raise NotImplementedError(
                f"Denoising loss type {self.args.denoising_loss_type} not implemented."
            )

        denoising_loss = self.denoising_loss_func(
            x=clean_image.flatten(0, 1),
            x_pred=pred_image.flatten(0, 1),
            noise=pure_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        log_dict = dict(finetunetrain_noisy_latent=noisy_input.detach(),
                        finetunetrain_pred_image=pred_image.detach())

        return denoising_loss, log_dict
