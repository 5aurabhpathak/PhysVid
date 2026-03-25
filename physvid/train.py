import argparse
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from physvid.data import LMDBMappedTextDataset, collate_prompts_batch_first
from physvid.finetune import Finetune
from physvid.models import get_block_class
from physvid.models.physics_observer import PhysicsObserver
from physvid.util import (
    launch_distributed_job,
    prepare_for_saving as _prepare_for_saving,
    set_model_seed, set_worker_seed, init_logging_folder,
    fsdp_wrap, cycle,
    fsdp_load_or_save,
    barrier, get_latest_checkpoint,
    keep_last_n_checkpoints
)


class Trainer:
    def __init__(self, config):
        self.config = config

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        assert dist.is_initialized()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0

        # use a random seed for the training
        if config.seed == 0:
            if global_rank == 0:
                # Generate on current device to avoid CPU backend issues
                random_seed = torch.randint(0, 10_000_000, (1,), device=self.device, dtype=torch.int64)
            else:
                random_seed = torch.empty(1, device=self.device, dtype=torch.int64)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_model_seed(config.seed)

        if self.is_main_process:
            self.output_path, self.wandb_folder = init_logging_folder(config)

        # Step 2: Initialize the model and optimizer
        self.model = Finetune(config, device=self.device)

        if 'conditioning' in config.generator_name and config.train_new_layers_only:
            self.model.generator.train_new_layers_only()

        if self.is_main_process:
            tqdm.write(f'Model: {self.model.generator.__class__.__name__}')
            tqdm.write(f'Number of parameters: {sum(p.numel() for p in self.model.generator.parameters())}')
            tqdm.write(f'Number of trainable parameters: {sum(p.numel() for p in self.model.generator.parameters()
                                                          if p.requires_grad)}')

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.generator_fsdp_transformer_module),
                                ) if config.generator_fsdp_wrap_strategy == "transformer" else None,
            use_orig_params=config.fsdp_use_orig_params
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.text_encoder_fsdp_transformer_module),
                                ) if config.text_encoder_fsdp_wrap_strategy == "transformer" else None
        )

        if 'conditioning' in config.generator_name:
            physics_observer = PhysicsObserver()
            physics_observer.model = fsdp_wrap(
                physics_observer.model,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.physics_encoder_fsdp_wrap_strategy,
                transformer_module=(get_block_class(config.physics_encoder_fsdp_transformer_module),
                                    ) if config.physics_encoder_fsdp_wrap_strategy == "transformer" else None,
                use_orig_params=config.fsdp_use_orig_params
            )
            self.model.local_text_encoder.set_physics_observer(physics_observer)

            self.model.local_text_encoder.set_encoder(self.model.text_encoder)

        if not config.no_visualize:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.generator_lr,
            betas=(config.beta1, config.beta2)
        )

        # Step 3: Initialize the dataloader
        dataset = LMDBMappedTextDataset(config.data_path, config.latents_map, config.image_or_video_shape[1:],
                                        return_key=False, dtype=np.float16,
                                        clip_len_s=5.1 if 'wan' in config.generator_name else 6.1)

        self.sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True, seed=config.seed)
        self.dataset = dataset

        dl_generator = torch.Generator()
        dl_generator.manual_seed(config.seed + global_rank)
        set_worker_seed_ = partial(set_worker_seed, seed=config.seed, rank=global_rank)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=self.sampler,
            worker_init_fn=set_worker_seed_,
            generator=dl_generator,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=collate_prompts_batch_first,
            num_workers=max(1, config.batch_size // 8))

        self.dataloader = dataloader
        self.steps_per_epoch = len(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None
        self.best_generator_loss = float('inf')

    def save(self):
        barrier()
        tqdm.write("Start gathering distributed model states...")
        generator_state_dict, generator_optimizer_state_dict = fsdp_load_or_save(
            self.model.generator, self.generator_optimizer)

        # Save random states for reproducibility
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        torch_cuda_state = torch.cuda.get_rng_state_all()

        state_dict = dict(
            generator=generator_state_dict,
            generator_optimizer=generator_optimizer_state_dict,
            step=self.step,
            epoch=self.sampler.epoch,
            random_state=random_state,
            numpy_state=numpy_state,
            torch_state=torch_state,
            torch_cuda_state=torch_cuda_state,
            seed=self.config.seed
        )

        barrier()
        if self.is_main_process:
            save_dirname = f"checkpoint_model_{int(min(self.best_generator_loss, 100) * 1e4):06d}_{self.step:06d}"
            os.makedirs(os.path.join(self.output_path, save_dirname), exist_ok=True)
            save_abspath = os.path.join(self.output_path, save_dirname, "model.pt")
            torch.save(state_dict, save_abspath)

            keep_last_n_checkpoints(self.output_path, n=3)
            # keep_best_n_checkpoints(self.output_path, n=3)
            tqdm.write(f'Saved checkpoint to {save_abspath}')

        barrier()
        torch.cuda.empty_cache()

    def add_visualization(self, generator_log_dict, wandb_loss_dict, clean_latent):

        def _decode_to_cpu(x):
            y = self.model.vae.decode_to_pixel(x, micro_batch_size=1).squeeze(1)
            return y.detach().cpu()

        input_latent = _decode_to_cpu(clean_latent)
        prepare_for_saving = partial(_prepare_for_saving, fps=16)
        wandb_loss_dict.update(dict(input_latent=prepare_for_saving(input_latent)))

        (finetunetrain_noisy_latent, finetunetrain_pred_image) = map(
            _decode_to_cpu,
            [generator_log_dict['finetunetrain_noisy_latent'], generator_log_dict['finetunetrain_pred_image']]
        )

        wandb_loss_dict.update(
            dict(finetunetrain_noisy_latent=prepare_for_saving(finetunetrain_noisy_latent),
                 finetunetrain_pred_image=prepare_for_saving(finetunetrain_pred_image))
        )

    def load(self):
        restart_ckpt = get_latest_checkpoint(self.config.output_path)
        if restart_ckpt:
            restart_ckpt = os.path.join(get_latest_checkpoint(self.config.output_path), "model.pt")
            self.config.from_ckpt = restart_ckpt
            self.config.load_optimizer = True
        elif not os.path.isfile(self.config.from_ckpt):
            tqdm.write('Training from scratch.')
            self.dataloader = cycle(self.dataloader, self.sampler)
            return

        checkpoint = torch.load(self.config.from_ckpt, map_location='cpu',
                                weights_only=False)

        # Load random states for reproducibility
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_state'])
        torch.set_rng_state(checkpoint['torch_state'])

        torch_cuda_states = checkpoint['torch_cuda_state']
        num_devices = torch.cuda.device_count()
        saved_len = len(torch_cuda_states)
        if saved_len < num_devices:
            torch_cuda_states = list(torch_cuda_states) + [torch.cuda.get_rng_state(device=i)
                                                           for i in range(saved_len, num_devices)]
        elif saved_len > num_devices:
            torch_cuda_states = torch_cuda_states[:num_devices]
        torch.cuda.set_rng_state_all(torch_cuda_states)

        self.config.seed = checkpoint['seed']

        fsdp_load_or_save(self.model.generator,
                          self.generator_optimizer if self.config.load_optimizer else None,
                          operation="load",
                          model_state_dict=checkpoint['generator'],
                          optim_state_dict=checkpoint.get('generator_optimizer'))

        self.step = checkpoint['step']
        self.dataloader = cycle(self.dataloader, self.sampler, start_epoch=checkpoint['epoch'])

        tqdm.write(f"Loaded checkpoint states from {self.config.from_ckpt}")

    def train_one_step_finetune(self):
        # Step 1: Get the next batch of text prompts
        _text_prompts, clean_latent = next(self.dataloader)
        clean_latent = clean_latent.to(device=self.device, dtype=self.dtype, non_blocking=True)

        if isinstance(_text_prompts, dict):
            text_prompts = _text_prompts['prompt']
            positive_prompts = _text_prompts['positive_prompts']
        else:
            assert not 'conditioning' in self.config.generator_name, 'conditioning requires positive prompts.'
            text_prompts = _text_prompts

        # randomly replace some of the text prompts with empty strings for regularization
        if self.config.empty_prompt_prob > 0.:
            rnd = np.random.rand(len(text_prompts))
            text_prompts = [p if rand > self.config.empty_prompt_prob else ''
                                for p, rand in zip(text_prompts, rnd)]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

        if 'conditioning' in self.config.generator_name:
            local_conditional_dict = self.model.local_text_encoder(text_prompts=positive_prompts)
            conditional_dict.update(dict(local_prompt_embeds=local_conditional_dict['local_pos_prompt_embeds']))

        # Step 3: Train the generator
        finetune_loss, finetune_log_dict = self.model.finetune_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            clean_latent=clean_latent
        )

        self.generator_optimizer.zero_grad()
        finetune_loss.backward()
        finetune_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        if finetune_loss.item() < self.best_generator_loss:
            self.best_generator_loss = finetune_loss.item()

        # Step 4: Logging
        if self.is_main_process:
            wandb_loss_dict = dict(finetune_loss=finetune_loss.item(),
                                   finetune_grad_norm=finetune_grad_norm.item())

            tqdm.write(f'Generator loss: {finetune_loss.item():.4f}')

            if self.step % self.config.log_iters == 0 and not self.config.no_visualize:
                self.add_visualization(finetune_log_dict, wandb_loss_dict, clean_latent)
                torch.cuda.empty_cache()

            wandb.log(wandb_loss_dict, step=self.step)

    def train(self):

        if getattr(self.config, 'from_ckpt', False):
            self.load()
            # steps_to_skip = self.step % self.steps_per_epoch
            # for _ in tqdm(range(steps_to_skip), dynamic_ncols=True, disable=not self.is_main_process,
            #               desc='Skipping data'):
            #     next(self.dataloader)
        else:
            self.dataloader = cycle(self.dataloader, self.sampler)

        total_epochs = (self.config.total_iters + self.steps_per_epoch - 1) // self.steps_per_epoch

        if self.is_main_process:
            pbar = tqdm(total=self.config.total_iters, initial=self.step, dynamic_ncols=True,
                        desc=f"epoch: {self.sampler.epoch} of {total_epochs}")

        step_func = self.train_one_step_finetune

        while self.step < self.config.total_iters:
            self.model.eval()

            if (self.step + 1) % 100 == 0:
                tqdm.write('Emptying cache...')
                torch.cuda.empty_cache()

            step_func()

            if not self.config.no_save and (self.step + 1) % self.config.log_iters == 0:
                self.save()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time -
                              self.previous_time}, step=self.step)
                    wandb.log({"epoch": self.sampler.epoch}, step=self.step)
                    self.previous_time = current_time

                pbar.set_description(f"epoch: {self.sampler.epoch} of {total_epochs}")
                pbar.update(1)
            self.step += 1

        if self.is_main_process:
            pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = Trainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
