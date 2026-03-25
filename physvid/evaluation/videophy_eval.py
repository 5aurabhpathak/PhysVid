import json
import os
import re
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from physvid.data import VideoPhyDataset
from physvid.evaluation.videophy.constants import PROMPT_PHYSICS, PROMPT_VTA
from physvid.evaluation.videophy.mplug_owl_video import (MplugOwlImageProcessor,
                                                         MplugOwlProcessor,
                                                         MplugOwlForConditionalGeneration)
from physvid.evaluation.videophy.utils import batchify
from physvid.util import launch_distributed_job, set_model_seed, set_worker_seed, get_sha256_key


class VideoPhyEvaluator:
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

        config.checkpoint = 'weights/videophy'

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
            tqdm.write(f'Loaded VideoPhy dataset with {len(dataset)} unique captions.')

        self.sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, seed=config.seed,
                                                                       drop_last=False)
        dl_generator = torch.Generator()
        dl_generator.manual_seed(config.seed + global_rank)
        set_worker_seed_ = partial(set_worker_seed, seed=config.seed, rank=global_rank)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                                      sampler=self.sampler, worker_init_fn=set_worker_seed_,
                                                      drop_last=False,
                                                      generator=dl_generator,
                                                      num_workers=max(config.batch_size // 8, 1))
        self.out_dict = dict()

        media_tokens = ['<image>', '<|video|>']
        self.media_tokens = {k: -int(i+1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {'<image>': 1+64,'<|video|>': 1+64}
        self.max_length = 256

    def get_dataset(self):
        dataset_name = getattr(self.config, 'hf_dataset_name', '')
        if 'videophy' in dataset_name:
            return VideoPhyDataset(caption_col=self.config.caption_col, version=dataset_name)
        else:
            raise NotImplementedError(f'Unknown dataset {dataset_name}')

    def get_entail(self, logits, input_ids):
        logits = nn.Softmax(dim=2)(logits)
        token_id_yes = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        token_id_no = self.tokenizer.encode('No', add_special_tokens=False)[0]
        entailment = []
        for j in range(len(logits)):
            for i in range(len(input_ids[j])):
                if input_ids[j][i] == self.tokenizer.pad_token_id:  # pad token if the answer is not present
                    i = i - 1
                    break
                elif i == len(input_ids[j]) - 1:
                    break
            score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
            entailment.append(score)
        entailment = torch.stack(entailment)
        return entailment

    def _extract_text_token_from_conversation(self, data, max_length):
        # output enc_chunk
        enc_chunk = []

        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        # conversation = data["completion"]
        conversation = data

        # For Text only data
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            pattern = '|'.join(map(re.escape, ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            prompt_length = -1
            stop_flag = False
            for idx, chunk_str in enumerate(chunk_strs):
                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_strs[idx-1] == 'AI: ':
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length:
                            curr_chunk = curr_chunk[:max_length-enc_length]
                            stop_flag = True
                        curr_chunk += [self.tokenizer.eos_token_id]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [1] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length + 1:
                            curr_chunk = curr_chunk[:max_length+1-enc_length]
                            stop_flag = True
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    if stop_flag:
                        break

        # For Image-Text Data
        else:
            enc_length = 0
            prompt_length = -2
            pattern = '|'.join(
                map(re.escape, list(self.media_tokens.keys()) + ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break

                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_str in self.media_tokens:
                        # [CLS] + 256 + [EOS]
                        if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                            break
                        else:
                            enc_chunk += [self.media_tokens[chunk_str]
                                          ] * self.media_lengths[chunk_str]
                            enc_length += self.media_lengths[chunk_str]
                            label_chunk += [0] * self.media_lengths[chunk_str]
                    else:

                        if chunk_strs[idx-1] == 'AI: ':
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length:
                                curr_chunk = curr_chunk[:max_length-enc_length]
                            curr_chunk += [self.tokenizer.eos_token_id]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [1] * len(curr_chunk)
                        else:
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length + 1:
                                curr_chunk = curr_chunk[:max_length +
                                                        1-enc_length]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [0] * len(curr_chunk)

        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * \
                (max_length + 1 - enc_length)
            padding_length = len(padding_chunk)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + 1, (prompt_length, enc_length,
                                                               padding_length, max_length + 1)
        assert len(label_chunk) == max_length + 1, (len(label_chunk), max_length + 1)
        non_padding_mask = [1 if i < enc_length -
                            1 else 0 for i in range(max_length)]

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        prompt_length = torch.tensor([prompt_length]).long()

        # Create loss mask
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = enc_chunk.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = torch.tensor(tmp_enc_chunk).long()
            non_media_mask = non_media_mask[1:].long()
        return {'input_ids': enc_chunk, "prompt_length": prompt_length, 'seq_length': enc_length,
                "non_padding_mask": non_padding_mask, 'non_media_mask': non_media_mask, 'prompt_mask': prompt_mask}

    @staticmethod
    def build_prompts(prompts, task):
        if task == 'sa':
            prompts = [PROMPT_VTA.format(caption=prompt.strip()) for prompt in prompts]
        elif task == 'pc':
            prompts = [PROMPT_PHYSICS for _ in prompts]
        return prompts

    @torch.no_grad()
    def run_once(self, videopaths, prompts, task):
        prompts = self.build_prompts(prompts, task)
        video_inputs = self.processor(videos=videopaths, num_frames=self.config.num_frames, return_tensors='pt')
        batch = []
        for caption, video_input in zip(prompts, video_inputs):
            text_input = self._extract_text_token_from_conversation(caption, self.max_length)
            batch.append({'video': video_input.unsqueeze(0), 'text': text_input})

        inputs = batchify(batch)
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if v.dtype == torch.float:
                    inputs[k] = v.bfloat16()
                inputs[k] = inputs[k].to(self.model.device)

        outputs = self.model(pixel_values=inputs['pixel_values'], video_pixel_values=inputs['video_pixel_values'],
                        labels=None,
                        num_images=inputs['num_images'], num_videos=inputs['num_videos'],
                        input_ids=inputs['input_ids'], non_padding_mask=inputs['non_padding_mask'],
                        non_media_mask=inputs['non_media_mask'], prompt_mask=inputs['prompt_mask'])
        logits = outputs['logits']
        entail_scores = self.get_entail(logits, inputs['input_ids'])
        return entail_scores.cpu().float().numpy().tolist()

    def run(self):
        step = 0
        for prompts in tqdm(self.dataloader, disable=not self.is_main_process, dynamic_ncols=True):
            keys = [get_sha256_key(prompt) for prompt in prompts]

            # create_paths
            videopaths = [os.path.join(self.config.generated_data_path, f'{key}.mp4') for key in keys]

            scores_sa = self.run_once(videopaths, prompts, task='sa')
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
            sa_score = sum(int(v['sa'] >= .5) for v in all_values) / len(all_values)
            pc_score = sum(int(v['pc'] >= .5) for v in all_values) / len(all_values)
            jp_score = sum(int(v['sa'] >= .5 and v['pc'] >= .5) for v in all_values) / len(all_values)

            tqdm.write(f'[rank{dist.get_rank()}] SA: {sa_score:.2f} PC: {pc_score:.2f} '
                       f'Joint Performance: {jp_score:.2f}')
            with open(self.config.eval_result_file, 'w') as f:
                json.dump(merged, f)

            tqdm.write(f'Wrote {len(merged)} entries to {self.config.eval_result_file}')

        dist.barrier()
