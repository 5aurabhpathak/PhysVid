import json
import os
from pathlib import Path

import ffmpeg
import lmdb
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm

from physvid.util import get_sha256_key, is_correct_aspect_ratio


def filter_already_generated(keys, generated_data_path: str):
    seen_keys = set()
    if os.path.exists(generated_data_path):
        for filename in os.listdir(generated_data_path):
            key = filename[:-4]  # remove .mp4 extension
            seen_keys.add(key)

    sha256_keys = [get_sha256_key(prompt) for prompt in keys]
    filtered_captions = [caption for caption, key in zip(keys, sha256_keys) if key not in seen_keys]
    return filtered_captions


def collate_prompts_batch_first(batch):
    # batch is a list of (prompt, video) or (key, video); adapt as needed
    prompts, videos = zip(*batch)  # len == batch_size

    # prompts is a list of dicts; keep lists per sample (no transpose)
    batched_prompts = {k: [p[k] for p in prompts] for k in prompts[0].keys()}
    # videos can be stacked as usual
    batched_videos = default_collate(videos)
    return batched_prompts, batched_videos


def get_key_duration_index(video_info: dict[str, dict], clip_len_s: float) -> list[tuple[str, float]]:
    # Build deterministic global index of (video_path, start_time_s)
    _keys = list(video_info.keys())
    keys: list[tuple[str, float]] = []
    for key in _keys:
        duration = video_info[key]['duration']
        num_clips = int((duration - clip_len_s) // clip_len_s + 1)
        for i in range(num_clips):
            start = i * clip_len_s
            keys.append((key, start))

    tqdm.write(f"Total {len(_keys)} videos in dataset. Total 5s clips: {len(keys)}.")
    return keys


class VideoDataset(Dataset):
    def __init__(self, data_path, return_key=False, resolution='wan'):
        self.clip_len_s = 5.1 if 'wan' in resolution else 6.2
        self.return_key = return_key
        self.max_frames = 81 if 'wan' in resolution else 49
        self.fps = 16 if 'wan' in resolution else 8
        self.width = 832 if 'wan' in resolution else 720
        self.height = 480
        tqdm.write(f'Using settings for {resolution}: max_frames={self.max_frames}, fps={self.fps}, width={self.width},'
              f'height={self.height}, clip_len_s={self.clip_len_s}')

        path = Path(data_path)
        cache_file_path = path.parent / f'{resolution}_preprocessed' / f"{path.stem}_filtered.json"
        if path.is_file() or cache_file_path.is_file():
            with open(path) as f:
                keys = json.load(f)
        else:
            keys = dict()
            curlen = 0
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            duration_limit = self.clip_len_s
            # target_ratio = self.width / self.height
            target_ratio = 16 / 9

            for input_path in tqdm(list(path.iterdir()), desc='Scanning directories', unit='dir', dynamic_ncols=True):
                if not input_path.is_dir():
                    continue

                for input_file in tqdm(list(input_path.iterdir()), desc=f'Processing {input_path.name}',
                                       unit='file', dynamic_ncols=True):
                    if not input_file.is_file() or input_file.suffix.lower() not in video_extensions:
                        continue

                    probe = ffmpeg.probe(input_file)
                    duration = float(probe['format']['duration'])
                    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                    w, h = int(video_stream['width']), int(video_stream['height'])

                    if duration < duration_limit or not is_correct_aspect_ratio(w, h, target_ratio=target_ratio):
                        continue

                    keys[str(input_file)] = dict(duration=duration)

                tqdm.write(f"Found {len(keys)-curlen} valid videos in {input_path}")
                curlen = len(keys)

            with open(cache_file_path, 'w') as f:
                json.dump(keys, f)

        self.video_info = keys

        # Build deterministic global index of (video_path, start_time_s)
        self.keys = get_key_duration_index(self.video_info, self.clip_len_s)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key, start_s = self.keys[idx]
        input_dict = self.video_info[key]
        fps = self.fps
        width = self.width
        height = self.height

        stream = ffmpeg.input(key, ss=start_s, t=self.clip_len_s)
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
        stream = ffmpeg.filter(stream, 'scale', width, height)
        stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")
        out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=True)
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3]).transpose([0, 3, 1, 2])

        if len(frames) < self.max_frames:
            tqdm.write(f"Warning: Video {key} has only {len(frames)} frames < {self.max_frames}")
            # pad frames by repeating last frame
            pad_len = self.max_frames - len(frames)
            frames = np.concatenate([frames, np.repeat(frames[-1][None, ...], pad_len, axis=0)], axis=0)

        frames = torch.from_numpy(frames[:self.max_frames])
        prompt = input_dict.get(start_s, '')

        if self.return_key:
            return key, start_s, frames
        return prompt, frames


class LMDBMappedTextDataset(Dataset):
    def __init__(self, data_path, video_map, image_or_video_shape, return_key=True, dtype=np.uint8,
                 clip_len_s=None):
        with open(data_path) as f:
            self.video_info = json.load(f)

        self.keys = list(self.video_info.keys())

        if isinstance(self.video_info[self.keys[0]], dict):
            # Build deterministic global index of (video_path, start_time_s)
            self.keys = get_key_duration_index(self.video_info, clip_len_s=clip_len_s)

        self.video_map = video_map
        self.env = None
        self.txn = None
        self.image_or_video_shape = image_or_video_shape
        self.return_key = return_key
        self.dtype = dtype

    def _require_env(self):
        if self.txn is None:
            self.env = lmdb.open(self.video_map, readonly=True, lock=False, readahead=False, meminit=False)
            self.txn = self.env.begin(write=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._require_env()
        val = self.keys[idx]
        if not isinstance(val, tuple):
            key = val
            prompt = self.video_info[key]
            _key = get_sha256_key(prompt.strip())
        else:
            key, start_s = self.keys[idx]
            start_s = f'{start_s:.1f}'
            prompt = self.video_info[key][start_s]
            _key = get_sha256_key(f'{key}_{start_s}')

        video = self.retrieve_row_from_lmdb(_key, 0)
        video_tensor = torch.from_numpy(video)
        if self.return_key:
            key_or_prompt = _key
        else:
            key_or_prompt = prompt
        return key_or_prompt, video_tensor

    def retrieve_row_from_lmdb(self, array_name, row_index):
        """
        Retrieve a specific row from a specific array in the LMDB.
        """
        data_key = f'{array_name}_{row_index}_data'.encode()

        row_bytes = self.txn.get(data_key)

        if self.dtype == str:
            array = row_bytes.decode()
        else:
            array = np.frombuffer(row_bytes, dtype=self.dtype)

        if self.image_or_video_shape is not None and len(self.image_or_video_shape) > 0:
            array = array.reshape(self.image_or_video_shape)
        return array

    def reverse_match(self, shakey):
        """
        Given a SHA-256 key, return the corresponding global prompt
        Args:
            shakey: SHA-256 key
        Returns: global prompt or None
        """
        for key in self.keys:
            if not isinstance(key, tuple):
                prompt = self.video_info[key]
                key_hashed = get_sha256_key(prompt.strip())
            else:
                key_path, start_s = key
                start_s = f'{start_s:.1f}'
                prompt = self.video_info[key_path][start_s]['prompt']
                key_hashed = get_sha256_key(prompt.strip())

            if key_hashed == shakey:
                return prompt

    def __del__(self):
        if self.env is not None:
            self.env.close()


class DepletedDatasetView(Dataset):
    """
    A generic wrapper that filters out items already present in a FileStore.
    Works out-of-the-box for datasets exposing:
      - `keys` and `video_info` (e.g., JSON/LMDB mapped datasets), or
      - `captions` (e.g., VideoPhyDataset).
    Otherwise, pass `idx_to_hash` to define how to compute the hash for each item.
    """
    def __init__(
        self,
        base_dataset: Dataset
    ):
        self.base = base_dataset

        remaining_keys = []
        for key in base_dataset.keys:
            if base_dataset.video_info.get(key) is None:
                input_dict = base_dataset.video_info[key[0]]
                if input_dict.get(str(key[1])) is None:
                    remaining_keys.append(key)
                    continue

        base_dataset.keys = remaining_keys
        tqdm.write(f"{base_dataset.__class__.__name__}: {len(base_dataset.keys)} entries to process. ")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[int(idx)]


class RemainingLMDBDatasetView(Dataset):
    """
    A generic wrapper that filters out items already present in a LMDB filestore.
    Works out-of-the-box for datasets exposing:
      - `keys` and `video_info` (e.g., JSON/LMDB mapped datasets), or
      - `captions` (e.g., VideoPhyDataset).
    Otherwise, pass `idx_to_hash` to define how to compute the hash for each item.
    """
    def __init__(
        self,
        base_dataset: Dataset,
        lmdb_path: str,
    ):
        self.base = base_dataset
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

        # Pre-filter keys so we are not processing already done entries
        remaining_keys = []
        stored_keys = set()
        self.cursor = self.txn.cursor()
        for key in self.cursor.iternext(values=False):
            stored_keys.add(key.decode().split('_')[0])  # extract base key without index

        tqdm.write(f'LMDB filestore has {len(stored_keys)} entries.')

        for key in base_dataset.keys:
            key_hashed = get_sha256_key(f'{key[0]}_{key[1]:.1f}')
            if key_hashed not in stored_keys:
                remaining_keys.append(key)

        base_dataset.keys = remaining_keys
        tqdm.write(f"{base_dataset.__class__.__name__}: {len(base_dataset.keys)} entries to process.")
        self.cleanup()

    def cleanup(self):
        self.cursor.close()
        self.txn.abort()
        self.env.close()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[int(idx)]

    def __del__(self):
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error during RemainingLMDBDatasetView cleanup: {e}")


class VideoPhyDataset(Dataset):
    def __init__(self, split='test', caption_col='caption', generated_data_path=None, version='videophy'):
        if version == 'videophy':
            dataset = f'videophysics/videophy_{split}_public'
        elif version == 'videophy2':
            dataset = f'videophysics/videophy2_{split}'
        else:
            raise ValueError(f"Unknown version: {version}")

        ds = load_dataset(dataset, split=split)
        to_remove = [c for c in ds.column_names if c != caption_col]
        if to_remove:
            ds = ds.remove_columns(to_remove)

        # keep only entries with non-empty and non-duplicate captions
        unique = {}
        for caption in ds[caption_col]:
            if not caption:
                continue
            caption = caption.strip().strip('"\'').strip()
            if caption and caption not in unique:
                unique[caption] = None
        self.keys = list(unique)

        if generated_data_path is not None:
            self.keys = filter_already_generated(self.keys, generated_data_path)

        tqdm.write(f"{self.__class__.__name__}: {len(self.keys)} entries to process.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.keys[int(idx)]
