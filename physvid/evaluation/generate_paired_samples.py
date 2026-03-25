import argparse
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from tqdm import tqdm

from physvid.data import LMDBMappedTextDataset, collate_prompts_batch_first
from physvid.inference import (InferencePipeline, LocalConditioningInferencePipeline)
from physvid.util import launch_distributed_job, set_model_seed, set_worker_seed, get_sha256_key, cycle


class SyntheticDatasetGenerator:
    def __init__(self, config):
        self.config = config

        launch_distributed_job()
        assert dist.is_initialized()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16
        self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.is_main_process = global_rank == 0

        # use a random seed
        if config.seed == 0:
            if global_rank == 0:
                random_seed = torch.randint(0, 10_000_000, (1,), device=self.device, dtype=torch.int64)
            else:
                random_seed = torch.empty(1, device=self.device, dtype=torch.int64)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        if self.is_main_process:
            tqdm.write(f'using seed: {config.seed}')

        set_model_seed(config.seed)

        if config.model_name == 'bidirectional_conditioning_wan':
            pipe = LocalConditioningInferencePipeline(config, device=self.device, dtype=self.dtype)
        else:
            pipe = InferencePipeline(config, device=self.device, dtype=self.dtype)

        if config.get('generator_checkpoint') is not None and self.is_main_process:
            state_dict = torch.load(config.generator_checkpoint, map_location='cpu', weights_only=False)['generator']
            pipe.generator.load_state_dict(state_dict, strict=True)

        self.pipe = pipe.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for p in self.pipe.generator.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.pipe.generator.buffers():
                dist.broadcast(b.data, src=0)
            dist.barrier()

        dataset = LMDBMappedTextDataset(config.data_path, config.latents_map, config.image_or_video_shape[1:],
                                        return_key=False, dtype=np.float16,
                                        clip_len_s=5.1 if 'wan' in config.model_name else 6.2)

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

        self.dataloader = cycle(dataloader, self.sampler)
        tqdm.write(f'Loaded dataset from {config.data_path}.')

        os.makedirs(config.generated_data_path, exist_ok=True)
        self.fps = 16

    @torch.no_grad()
    def run_one_step(self):
        # Step 1: Get the next batch of text prompts
        text_prompts, clean_latent = next(self.dataloader)
        clean_latent = clean_latent.to(device=self.device, dtype=self.dtype, non_blocking=True)
        input_videos = self._decode_to_cpu(clean_latent)
        text_prompts = text_prompts['prompt']

        keys = [get_sha256_key(prompt) for prompt in text_prompts]

        videos = self.pipe.inference(
            noise=torch.randn(tuple(self.config.image_or_video_shape),
                              dtype=self.dtype,
                              device=self.device),
            text_prompts=text_prompts
        ).permute(0, 1, 3, 4, 2).cpu().float().numpy()

        for key, video in zip(keys, videos):
            save_location = os.path.join(self.config.generated_data_path, f"{key}.mp4")
            export_to_video(video, save_location, fps=self.fps)

        for key, video in zip(keys, input_videos):
            save_location = os.path.join(self.config.generated_data_path, f"{key}_input.mp4")
            export_to_video(video, save_location, fps=self.fps)

    def _decode_to_cpu(self, x):
        y = (self.pipe.vae.decode_to_pixel(x, micro_batch_size=1) * 0.5 + 0.5).clamp(0, 1).permute(0, 1, 3, 4, 2)
        return y.cpu().float().numpy()

    @torch.no_grad()
    def run(self):
        generated_dir = Path(self.config.generated_data_path)
        if generated_dir.exists():
            files = [p.name for p in generated_dir.iterdir() if p.is_file() and p.name.endswith('.mp4')]
        else:
            files = []
        keys_main = set()
        keys_input = set()
        for name in files:
            if name.endswith('_input.mp4'):
                keys_input.add(name[:-10])  # remove "_input.mp4"
            else:
                keys_main.add(name[:-4])  # remove ".mp4"
        completed = keys_main & keys_input
        completed_count = len(completed)
        if completed_count > 0:
            tqdm.write(f'Found {completed_count} complete pairs in {self.config.generated_data_path}, skipping them')
        remaining_samples = max(0, self.config.num_samples - completed_count)

        self.config.num_samples = remaining_samples
        total_iters = int(np.ceil(self.config.num_samples / (self.config.batch_size * dist.get_world_size())))

        if self.is_main_process:
            pbar = tqdm(total=total_iters, dynamic_ncols=True)

        step = 0
        while step < total_iters:
            try:
                self.run_one_step()
            except json.decoder.JSONDecodeError as e:
                tqdm.write(f'Caught JSON decode error: {e}, skipping this batch.')
                continue

            if (step + 1) % 1 == 0:
                tqdm.write('Emptying cache...')
                torch.cuda.empty_cache()

            if self.is_main_process:
                pbar.update(1)
            step += 1

        if self.is_main_process:
            pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument('--generated_data_path', type=str, default=None,
                        help='Path to save the generated synthetic dataset. If not provided, '
                             'it will use the path in config file.')
    parser.add_argument('--model_name', type=str, default=None,
                        choices=[None, 'wan'],
                        help='If provided, override the model name in config file.')
    parser.add_argument('--from_ckpt', type=str, default=None,
                        help='Path to the generator checkpoint. If provided, override the path in config file.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='If provided, override the batch size in config file.')

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.generated_data_path is not None:
        config.generated_data_path = args.generated_data_path
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.from_ckpt is not None:
        config.generator_checkpoint = args.from_ckpt
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    generator = SyntheticDatasetGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
    dist.destroy_process_group()
