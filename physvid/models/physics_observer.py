import json
from typing import List

import numpy as np
import outlines
import torch
import torch.distributed as dist
from pydantic import BaseModel, Field, constr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from physvid.util import load_video


class VideoAnalysis(BaseModel):
    """A Pydantic JSON schema to describe events in a video."""
    visible_elements: List[constr(max_length=50)] = Field(..., min_length=1, max_length=10,
                                                          description="A list of visible elements in the video.")
    thinking: str = Field(..., max_length=1000, min_length=300,
                         description="A detailed reasoning process analyzing the physical interactions and phenomena in the video.")
    physics: str = Field(..., max_length=500, min_length=100,
                         description="A JSON object mapping each visible element to its physics observations.")


class VideoCaptioning(BaseModel):
    """A Pydantic JSON schema to describe events in a video."""
    visible_elements: List[constr(max_length=50)] = Field(..., min_length=1, max_length=10,
                                                          description="A list of visible elements in the video.")
    thinking: str = Field(..., max_length=1000, min_length=300,
                         description="A detailed reasoning process analyzing the video.")
    caption: str = Field(..., max_length=500, min_length=100,
                        description="A detailed caption describing the video content.")


class LocalPromptsFromCaption(BaseModel):
    """A Pydantic JSON schema to describe events in a video."""
    thinking: str = Field(..., max_length=1000, min_length=300,
                         description="A detailed reasoning process analyzing the physical interactions and phenomena in"
                                     " the scene described by the caption.")
    visible_elements: List[constr(max_length=50)] = Field(..., min_length=1, max_length=10,
                                                          description="A list of visible elements in the scene "
                                                                      "described by the caption.")
    physics: List[constr(min_length=100, max_length=500)] = Field(..., min_length=7, max_length=7,
                         description="A JSON object mapping each time segment to its physics observations.")


class PhysicsObserver:
    def __init__(self, max_duration=None):
        cache_kwargs = dict()
        model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=None,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            **cache_kwargs
        )
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.model.to(self.device).eval().requires_grad_(False)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True,
                                                       **cache_kwargs
                                                       )
        self.system_prompt = "You are an experienced physicist. Follow the instructions carefully."

        model = outlines.Generator(outlines.from_transformers(self.model, self.processor), VideoAnalysis)
        self.logits_processor = model.logits_processor
        self.max_duration = max_duration
        self.prompt_with_video = None
        self.prompt_without_video = None

    def set_video_prompt(self, prompt_path):
        with open(prompt_path) as f:
            self.prompt_with_video = f.read()

    def set_no_video_prompt(self, prompt_path):
        with open(prompt_path) as f:
            self.prompt_without_video = f.read()

    @torch.no_grad()
    def generate(self, x, caption):
        self.logits_processor.reset()
        if x is None:
            prompt = self.prompt_without_video + caption
        else:
            prompt = self.prompt_with_video + caption

        content = [{"type": "text", "text": prompt}]
        if x is not None:
            content.insert(0, {"type": "video", "num_frames": len(x)})

        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        inputs = self.processor(conversation=conversation, images=x, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # print(dist.get_rank(), '==>', caption, flush=True)
        output_ids = self.model.generate(
            **inputs,
        logits_processor=[self.logits_processor],
        max_new_tokens = 1024,
        do_sample = True,
        temperature = .3,
        # synced_gpus = True
        )

        # ignore model output for empty caption. Model forward pass can not be avoided with distributed setup.
        if caption.strip() == '':
            return ''

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def _create_local_prompts(self, video_path, caption, duration=None, max_fps=5, sliding_window=False):
        if isinstance(video_path, str):
            frames, timestamps = load_video(video_path, end_time=self.max_duration)
        elif isinstance(video_path, (list, np.ndarray, torch.Tensor)):
            frames, timestamps = video_path, np.arange(len(video_path))
        elif video_path is None:
            response = self.generate(None, caption)
            return {'<no duration>': response}
        else:
            raise ValueError(f"Unsupported video_path type: {type(video_path)}")

        prompts = {}
        max_time = timestamps[-1]
        start_ts = timestamps[0]

        if duration is None:
            duration = timestamps[-1]
        else:
            duration = max(duration, timestamps[1] - timestamps[0])

        start_time = start_ts
        while start_time < max_time:
            end_time = min(start_time + duration, max_time)

            cur_frames = self.select_max_num_frames(frames, timestamps, start_time, end_time, max_fps=max_fps)
            response = self.generate(cur_frames, caption)

            key = f"[{start_time:.2f}, {end_time:.2f}]"
            prompts[key] = response

            if sliding_window:
                start_time += duration / 2.
            else:
                start_time += duration

        return prompts

    def create_local_prompts(self, *args, **kwargs):
        responses_dict = self._create_local_prompts(*args, **kwargs)
        return [self.post_process_prompt(response) for response in responses_dict.values()]

    @staticmethod
    def post_process_prompt(response):
        if not dist.is_initialized() or dist.get_rank() == 0:
            tqdm.write(response)

        parsed_json = ''
        try:
            parsed_json = json.loads(response)
            parsed_json = parsed_json['physics']
        except json.decoder.JSONDecodeError:
            pass
        return parsed_json

    def create_prompt_without_video(self, *args, **kwargs):
        response = self.generate(None, *args, **kwargs)
        return self.post_process_prompt(response)

    @staticmethod
    def select_max_num_frames(frames, timestamps, start_time, end_time, max_fps):
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        frames = frames[mask]
        max_frames = int(np.ceil(max_fps * (end_time - start_time)))
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = frames[indices]

        frames = [frame for frame in frames]
        return frames


class VideoCaptioner:
    def __init__(self, model=None, processor=None):
        cache_kwargs = dict()
        model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=None,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                **cache_kwargs
            )
        else:
            self.model = model

        self.device = torch.device('cuda', torch.cuda.current_device())
        self.model.to(self.device).eval().requires_grad_(False)

        if processor is None:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True,
                                                           **cache_kwargs
                                                           )
        else:
            self.processor = processor

        self.system_prompt = "You are a helpful assistant. Follow the instructions carefully."

        model = outlines.Generator(outlines.from_transformers(self.model, self.processor), VideoCaptioning)
        self.logits_processor = model.logits_processor

        with open('physvid/models/prompt_nocaption.txt') as f:
            self.prompt = f.read()

    @torch.no_grad()
    def generate(self, x):
        self.logits_processor.reset()
        prompt = self.prompt

        content = [{"type": "video", "num_frames": len(x)}, {"type": "text", "text": prompt}]
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        inputs = self.processor(conversation=conversation, images=x, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # print(dist.get_rank(), '==>', caption, flush=True)
        output_ids = self.model.generate(
            **inputs,
        logits_processor=[self.logits_processor],
        max_new_tokens = 1024,
        do_sample = True,
        temperature = .3,
        # synced_gpus = True
        )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def create_global_prompt(self, video):
        response = self.generate(video)
        return self.post_process_prompt(response)

    @staticmethod
    def post_process_prompt(response):
        if not dist.is_initialized() or dist.get_rank() == 0:
            tqdm.write(response)
        try:
            parsed_json = json.loads(response)
            parsed_json = parsed_json['caption']
        except json.decoder.JSONDecodeError:
            parsed_json = ''
        return parsed_json


class LocalPromptGeneratorFromGlobalCaption:
    def __init__(self, model=None, processor=None):
        cache_kwargs = dict()
        model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=None,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                **cache_kwargs
            )
        else:
            self.model = model

        self.device = torch.device('cuda', torch.cuda.current_device())
        self.model.to(self.device).eval().requires_grad_(False)

        if processor is None:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True,
                                                           **cache_kwargs
                                                           )
        else:
            self.processor = processor

        self.system_prompt = "You are an experienced physicist. Follow the instructions carefully."

        model = outlines.Generator(outlines.from_transformers(self.model, self.processor), LocalPromptsFromCaption)
        self.pos_logits_processor = model.logits_processor

        model = outlines.Generator(outlines.from_transformers(self.model, self.processor), VideoAnalysis)
        self.neg_logits_processor = model.logits_processor

        with open('physvid/models/prompt_novideo.txt') as f:
            self.prompt1 = f.read()

        with open('physvid/models/negative_prompt_v2.txt') as f:
            self.prompt2 = f.read()

    def set_positive_prompt(self, prompt_path):
        with open(prompt_path) as f:
            self.prompt1 = f.read()

    def set_negative_prompt(self, prompt_path):
        with open(prompt_path) as f:
            self.prompt2 = f.read()

    @torch.no_grad()
    def generate(self, caption, negative=False):
        if negative:
            prompt = self.prompt2
            logits_processor = self.neg_logits_processor
        else:
            prompt = self.prompt1
            logits_processor = self.pos_logits_processor

        logits_processor.reset()
        prompt = prompt + caption

        content = [{"type": "text", "text": prompt}]
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        inputs = self.processor(conversation=conversation, images=None, return_tensors="pt").to(self.device)

        # print(dist.get_rank(), '==>', caption, flush=True)
        output_ids = self.model.generate(
            **inputs,
            logits_processor=[logits_processor],
            max_new_tokens=1024,
            do_sample=True,
            temperature=.3,
            # synced_gpus = True
        )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def create_local_prompts(self, caption, negative=False):
        response = self.generate(caption, negative=negative)
        return self.post_process_prompt(response)

    @staticmethod
    def post_process_prompt(response):
        if not dist.is_initialized() or dist.get_rank() == 0:
            tqdm.write(response)

        parsed_json = None
        try:
            parsed_json = json.loads(response)
            parsed_json = parsed_json['physics']
        except json.decoder.JSONDecodeError:
            pass
        return parsed_json

def test():
    po = LocalPromptGeneratorFromGlobalCaption()

    prompt = 'A refrigerator door closing after getting a soda can from inside.'
    local_prompts = po.create_local_prompts(prompt)
    print(local_prompts)


if __name__ == "__main__":
    test()
