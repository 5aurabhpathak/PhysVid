import argparse
from functools import partial

import lmdb
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from physvid.data import VideoDataset, RemainingLMDBDatasetView
from physvid.models import WanVAEWrapper
from physvid.util import launch_distributed_job, set_model_seed, set_worker_seed, get_sha256_key


def store_arrays_to_lmdb(env, arrays_dict, start_index=0):
    """
    Store rows of multiple numpy arrays in a single LMDB.
    Each row is stored separately with a naming convention.
    """
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            for i, row in enumerate(array):
                # Convert row to bytes
                if isinstance(row, str):
                    row_bytes = row.encode()
                else:
                    row_bytes = row.tobytes()
                data_key = f'{array_name}_{start_index + i}_data'.encode()
                txn.put(data_key, row_bytes)


class VAELatentCreator:
    def __init__(self, config):
        self.config = config

        launch_distributed_job()
        assert dist.is_initialized()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16
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
        if config.vae == 'wan':
            tqdm.write('Using WanVAE')
            self.vae = WanVAEWrapper().eval().to(device=torch.cuda.current_device(), dtype=self.dtype)
        else:
            raise ValueError(f"Unknown VAE type {config.vae}")

        # figure out the maximum map size needed
        total_array_size = (1024 ** 4) // 2  # adapt to your need, set to 0.5x2=1TB by default, multiplication by 2 below
        self.write_env = lmdb.open(config.output_map, map_size=total_array_size * 2) if self.is_main_process else None
        dist.barrier()

        # Step 3: Initialize the dataloader
        dataset = RemainingLMDBDatasetView(VideoDataset(config.data_path, return_key=True, resolution=config.vae),
                                           config.output_map)

        self.sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False)

        set_worker_seed_ = partial(set_worker_seed, seed=config.seed, rank=global_rank)
        if config.batch_size >= 32:
            num_workers = config.batch_size // 8
        else:
            num_workers = 4

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                                      sampler=self.sampler,
                                                      worker_init_fn=set_worker_seed_,
                                                      num_workers=num_workers,
                                                      pin_memory=True,
                                                      prefetch_factor=4
                                                      )
        self.step = 0

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        device, dtype = videos[0].device, videos[0].dtype
        scale = [self.vae.mean.to(device=device, dtype=dtype),
                 1.0 / self.vae.std.to(device=device, dtype=dtype)]
        output = [
            self.vae.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in videos
        ]

        output = torch.stack(output, dim=0)
        return output

    @torch.no_grad()
    def run(self):
        for keys, start_ss, video_tensors in tqdm(self.dataloader, disable=not self.is_main_process, dynamic_ncols=True):
            video_tensors = video_tensors.float() / 255.0
            video_tensors = video_tensors * 2. - 1.
            video_tensors = video_tensors.transpose(2, 1).to(dtype=self.dtype, device=self.device, non_blocking=True)
            latents = self.encode(video_tensors).transpose(2, 1)

            arrays_dict = {}
            for key, start_s, latent in zip(keys, start_ss, latents):
                key = get_sha256_key(f'{key}_{start_s:.1f}')
                arrays_dict[key] = latent.half().cpu().numpy()[np.newaxis, ...]

            if self.is_main_process:
                gathered = [None] * dist.get_world_size()
                dist.gather_object(arrays_dict, gathered, dst=0)
                merged = {}
                for d in gathered:
                    merged.update(d)
                store_arrays_to_lmdb(self.write_env, merged)
            else:
                dist.gather_object(arrays_dict, None, dst=0)

            if (self.step + 1) % 100 == 0:
                torch.cuda.empty_cache()
            self.step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    annotator = VAELatentCreator(config)
    annotator.run()
    dist.barrier()

    if annotator.is_main_process:
        annotator.write_env.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
