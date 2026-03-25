import argparse
import json
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from physvid.data import DepletedDatasetView, VideoDataset
from physvid.models.physics_observer import PhysicsObserver, VideoCaptioner
from physvid.util import (
    launch_distributed_job,
    set_model_seed, set_worker_seed
)


class Annotator:
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

        # Step 2: Initialize the model
        self.physics_observer = PhysicsObserver()
        self.physics_observer.set_video_prompt(config.positive_vlm_instruction_path)
        self.physics_observer.set_no_video_prompt(config.negative_vlm_instruction_path)

        # reuse the same model for captioning
        self.captioner = VideoCaptioner(model=self.physics_observer.model, processor=self.physics_observer.processor)

        # Step 3: Initialize the dataloader
        self.num_frame_per_block = np.ceil(config.image_or_video_shape[0] / config.num_blocks).astype(int)
        dataset = DepletedDatasetView(VideoDataset(config.data_path, return_key=True))

        dl_generator = torch.Generator()
        dl_generator.manual_seed(config.seed + global_rank)
        set_worker_seed_ = partial(set_worker_seed, seed=config.seed, rank=global_rank)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False,
                                                                       seed=config.seed)

        self.dataloader = torch.utils.data.DataLoader(dataset, worker_init_fn=set_worker_seed_,
                                                      sampler=sampler,
                                                 generator=dl_generator,
                                                 pin_memory=True,
                                                 prefetch_factor=4,
                                                 num_workers=2)
        self.step = 0

        with open(config.data_path) as f:
            self.video_info = json.load(f)

    def call_physicist(self, video, caption):
        responses_list = self.physics_observer.create_local_prompts(video, caption, max_fps=.5,
                                                                    duration=self.num_frame_per_block)
        return responses_list

    def process_one_batch(self, batch):
        for key, start_s, video in zip(*batch):
            video = video.to(self.device, non_blocking=True)
            prompt = self.captioner.create_global_prompt(video).strip()
            assert prompt != '', "Generated empty prompt!"

            positive_prompts = self.call_physicist(video, prompt)

            data_dict = self.video_info[key]
            data_dict[str(start_s.item())] = dict(prompt=prompt, positive_prompts=positive_prompts)

    def run(self):
        for video_tensors in tqdm(self.dataloader, dynamic_ncols=True, disable=not self.is_main_process):
            self.process_one_batch(video_tensors)

            if self.step % (self.config.log_iters * 10) == 0:
                torch.cuda.empty_cache()

            if self.step % self.config.log_iters == 0:
                self.consolidate()

            self.step += 1

        # merge saved json files
        self.consolidate()
        dist.barrier()
        if self.is_main_process:
            self.merge_json_files()

    def merge_json_files(self):
        path_wo_ext = self.config.data_path.rsplit('.', 1)[0]
        merged_video_info = self.video_info
        for rank in range(1, dist.get_world_size()):
            load_path = f'{path_wo_ext}_rank{rank}.json'
            with open(load_path, 'r') as f:
                video_info_rank = json.load(f)
                for k, v in video_info_rank.items():
                    merged_video_info[k].update(v)

        save_path = f'{path_wo_ext}_merged_v3.json'
        with open(save_path, 'w') as f:
            json.dump(merged_video_info, f)
        print(f'[rank0] Merged json saved to {save_path}')


    def consolidate(self):
        rank = dist.get_rank()
        path_wo_ext = self.config.data_path.rsplit('.', 1)[0]
        save_path = f'{path_wo_ext}_rank{rank}.json'
        with open(save_path, 'w') as f:
            json.dump(self.video_info, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    annotator = Annotator(config)
    annotator.run()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
