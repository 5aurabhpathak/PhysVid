import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from tqdm import tqdm

from physvid.data import VideoPhyDataset
from physvid.inference import (InferencePipeline,
                               LocalConditioningInferencePipeline)
from physvid.util import launch_distributed_job, set_model_seed, set_worker_seed, get_sha256_key


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

        if 'conditioning' in config.model_name:
            pipe = LocalConditioningInferencePipeline(config, device=self.device, dtype=self.dtype)
        else:
            pipe = InferencePipeline(config, device=self.device, dtype=self.dtype)

        if config.get('generator_checkpoint') is not None and self.is_main_process:
            state_dict = torch.load(config.generator_checkpoint, map_location='cpu', weights_only=False)['generator']
            pipe.generator.model.load_state_dict(state_dict, strict=True)

        self.pipe = pipe.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for p in self.pipe.generator.model.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.pipe.generator.model.buffers():
                dist.broadcast(b.data, src=0)
            dist.barrier()

        dataset = self.get_dataset()
        if self.is_main_process:
            tqdm.write(f'Loaded {config.hf_dataset_name} dataset with {len(dataset)} unique captions.')

        dl_generator = torch.Generator()
        dl_generator.manual_seed(config.seed + global_rank)
        set_worker_seed_ = partial(set_worker_seed, seed=config.seed, rank=global_rank)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False,
                                                                       seed=config.seed)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                                      drop_last=False,
                                                      sampler=sampler,
                                                      worker_init_fn=set_worker_seed_, generator=dl_generator,
                                                      collate_fn=self.pad_collate,
                                                      num_workers=max(1, config.batch_size // 8))

        os.makedirs(config.generated_data_path, exist_ok=True)

    def pad_collate(self, batch):
        valid_count = len(batch)
        if valid_count == self.config.batch_size:
            return batch, valid_count
        padded = list(batch) + [''] * (self.config.batch_size - valid_count)
        return padded, valid_count

    def get_dataset(self):
        dataset_name = getattr(self.config, 'hf_dataset_name', '')
        if 'videophy' in dataset_name:
            return VideoPhyDataset(caption_col=self.config.caption_col, version=dataset_name,
                                   generated_data_path=self.config.generated_data_path)
        else:
            raise ValueError(f"Unknown dataset name {dataset_name}")

    @torch.no_grad()
    def run(self):
        step = 0
        fps = 16
        for prompts, valid_count in tqdm(self.dataloader, dynamic_ncols=True, disable=not self.is_main_process):
            keys = [get_sha256_key(prompt) for prompt in prompts]

            videos = self.pipe.inference(
                noise=torch.randn(self.config.batch_size, *self.config.image_or_video_shape,
                                  dtype=self.dtype,
                                  device=self.device),
                text_prompts=prompts
            ).permute(0, 1, 3, 4, 2).cpu().float().numpy()

            for key, prompt, video in zip(keys, prompts, videos):
                save_location = os.path.join(self.config.generated_data_path, f"{key}.mp4")
                export_to_video(video, save_location, fps=fps)

            if (step + 1) % 10 == 0:
                torch.cuda.empty_cache()

            step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument('--generated_data_path', type=str, default=None,
                        help='Path to save the generated synthetic dataset. If not provided, '
                             'it will use the path in config file.')
    parser.add_argument('--caption_col', type=str, default=None,
                        choices=[None, 'caption', 'upsampled_caption'],
                        help='If provided, generate video using this caption style.')
    parser.add_argument('--hf_dataset_name', type=str, default=None,
                        choices=[None, 'videophy', 'videophy2'],
                        help='If provided, override the dataset name in config file.')
    parser.add_argument('--model_name', type=str, default=None,
                        choices=[None, 'wan', 'local_conditioning_wan'],
                        help='If provided, override the model name in config file.')
    parser.add_argument('--from_ckpt', type=str, default=None,
                        help='Path to the generator checkpoint. If provided, override the path in config file.')
    parser.add_argument('--task_type', type=str, default=None,
                        choices=[None, 'fast_denoise', 'slow_denoise'],
                        help='If provided, override the task type in config file.')
    parser.add_argument('--denoising_step_list', type=int, nargs='+', default=None,
                        help='If provided, override the denoising step list in config file.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='If provided, override the batch size in config file.')

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.generated_data_path is not None:
        config.generated_data_path = args.generated_data_path
    if args.caption_col is not None:
        config.caption_col = args.caption_col
    if args.hf_dataset_name is not None:
        config.hf_dataset_name = args.hf_dataset_name
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.from_ckpt is not None:
        config.generator_checkpoint = args.from_ckpt
    if args.task_type is not None:
        config.task_type = args.task_type
    if args.denoising_step_list is not None:
        config.denoising_step_list = args.denoising_step_list
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    generator = SyntheticDatasetGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
    dist.destroy_process_group()
