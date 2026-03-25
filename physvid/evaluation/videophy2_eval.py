import json
import os
from functools import partial

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from physvid.data import VideoPhyDataset
from physvid.evaluation.videophy2.mplug_owl_video import (MplugOwlImageProcessor,
                                                          MplugOwlProcessor,
                                                          MplugOwlForConditionalGeneration)
from physvid.evaluation.videophy2.template import PROMPT_PHYSICS, PROMPT_RULE, PROMPT_SA
from physvid.util import launch_distributed_job, set_model_seed, set_worker_seed, get_sha256_key

generate_kwargs = {
    'do_sample': False,
    'top_k': 1,
    'temperature': 0.001,
    'max_new_tokens': 256,
}


class VideoPhy2Evaluator:
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
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_model_seed(config.seed)

        config.checkpoint = 'weights/videophy2'

        # Processors
        self.tokenizer = LlamaTokenizer.from_pretrained(config.checkpoint)
        image_processor = MplugOwlImageProcessor.from_pretrained(config.checkpoint)
        self.processor = MplugOwlProcessor(image_processor, self.tokenizer)

        # Instantiate evaluator model
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            config.checkpoint,
            torch_dtype=self.dtype,
            device_map={'': 'cpu'}
        ).eval().to(device=self.device)

        dataset = self.get_dataset()

        if self.is_main_process:
            tqdm.write(f'Loaded VideoPhy2 dataset with {len(dataset)} unique captions.')

        self.sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, seed=config.seed,
                                                                       drop_last=False)

        dl_generator = torch.Generator()
        dl_generator.manual_seed(config.seed + global_rank)
        set_worker_seed_ = partial(set_worker_seed, seed=config.seed, rank=global_rank)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                                      sampler=self.sampler, worker_init_fn=set_worker_seed_,
                                                      generator=dl_generator,
                                                      drop_last=False,
                                                      num_workers=max(config.batch_size // 8, 1))

        self.num_map = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "0": 0, "1": 1, "2": 2, "3": 3,
                        "4": 4, "5": 5}
        self.out_dict = dict()

    def get_dataset(self):
        dataset_name = getattr(self.config, 'hf_dataset_name', '')
        if 'videophy' in dataset_name:
            return VideoPhyDataset(caption_col=self.config.caption_col, version=dataset_name)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    @staticmethod
    def build_prompts(prompts, task, rule='*'):
        if task == 'sa':
            prompts = [PROMPT_SA.format(caption=prompt.strip()) for prompt in prompts]
        elif task == 'pc':
            if rule == '*':
                prompts = [PROMPT_PHYSICS for _ in prompts]
            else:
                prompts = [PROMPT_RULE.format(rule=rule) for _ in prompts]
        return prompts

    def post_process_outputs(self, outputs):
        scores = []
        for output in outputs:
            output = self.tokenizer.decode(output, skip_special_tokens=True)
            output_lower = output.lower().strip()
            # tqdm.write(f'[rank{dist.get_rank()}] Model output: {output_lower}')
            score = None
            for key, val in self.num_map.items():
                if key in output_lower:
                    score = val
                    break

            if score is None:
                # Optionally, try to extract a digit with a simple filter.
                digits = ''.join([c for c in output_lower if c.isdigit()])
                score = int(digits) if digits and int(digits) in self.num_map.values() else 0
                tqdm.write(f"[rank{dist.get_rank()}] Warning: Could not parse output '{output}'. "
                           f"Defaulting to {score}.")

            scores.append(score)

        return scores

    @torch.no_grad()
    def run_once(self, videopaths, prompts, task):
        # tqdm.write(f'[rank{dist.get_rank()}] Running evaluation for task: {task} ')
        prompts = self.build_prompts(prompts, task)
        inputs = self.processor(text=prompts, videos=videopaths, num_frames=self.config.num_frames, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **generate_kwargs)
        return self.post_process_outputs(outputs)

    def run(self):
        step = 0
        for prompts in tqdm(self.dataloader, disable=not self.is_main_process, dynamic_ncols=True):
            keys = [get_sha256_key(prompt) for prompt in prompts]

            # create_paths
            videopaths = [os.path.join(self.config.generated_data_path, f'{key}.mp4') for key in keys]

            scores_sa = self.run_once(videopaths, prompts, task='sa')
            # tqdm.write(f'[rank{dist.get_rank()}] SA scores: {scores_sa}')
            scores_pc = self.run_once(videopaths, prompts, task='pc')

            for key, prompt, score_sa, score_pc in zip(keys, prompts, scores_sa, scores_pc):
                self.out_dict[key] = dict(caption=prompt, sa=score_sa, pc=score_pc)

            if (step + 1) % self.config.save_iters == 0:
                torch.cuda.empty_cache()
                self.save_output()

            step += 1

        self.save_output()

    def save_output(self):
        dist.barrier()

        # all gather out_dict from all ranks
        local_dict = self.out_dict
        all_dicts = [None] * dist.get_world_size() if self.is_main_process else None
        dist.gather_object(local_dict, all_dicts, dst=0)

        if self.is_main_process:
            merged = {}
            for d in all_dicts:
                merged.update(d)

            all_values = merged.values()
            sa_score = sum(int(v['sa'] > 3) for v in all_values) / len(all_values)
            pc_score = sum(int(v['pc'] > 3) for v in all_values) / len(all_values)
            jp_score = sum(int(v['sa'] > 3 and v['pc'] > 3) for v in all_values) / len(all_values)

            tqdm.write(f'[rank{dist.get_rank()}] SA: {sa_score:.2f} PC: {pc_score:.2f} '
                       f'Joint Performance: {jp_score:.2f}')
            with open(self.config.eval_result_file, 'w') as f:
                json.dump(merged, f)

            tqdm.write(f'Wrote {len(merged)} entries to {self.config.eval_result_file}')

        dist.barrier()
