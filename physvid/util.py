import hashlib
import os
import random
import shutil
import time
from datetime import timedelta, datetime
from functools import partial

import ffmpeg
import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP, FullOptimStateDictConfig
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torchvision.utils import make_grid


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    print('Rank:', rank, 'Local Rank:', local_rank, 'World Size:', world_size, 'Host:', host, 'Port:', port)

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"

    _set_cuda_device_with_retry(local_rank)
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))


def _set_cuda_device_with_retry(local_rank, retries=5, backoff=1.0):
    """
    Attempt to set the CUDA device with exponential backoff.
    Only masks transient init races; persistent failures are re-raised.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check driver / nvidia-smi.")
    dev_count = torch.cuda.device_count()
    if local_rank >= dev_count:
        raise RuntimeError(f"local_rank {local_rank} >= device_count {dev_count}.")

    for attempt in range(retries):
        try:
            torch.cuda.set_device(local_rank)
            _ = torch.cuda.current_device()  # force context
            print('set_device succeeded on attempt', attempt, 'for local rank', local_rank)
            return
        except RuntimeError as e:
            print('set_device failed for local_rank', local_rank, ', retrying:', e)
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 1.5
            else:
                raise


def set_model_seed(seed: int, deterministic: bool = False):
    """Set a single seed for model initialization across all ranks."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def set_worker_seed(worker_id, seed=None, rank=None):
    """Set a different seed for each dataloader worker."""
    worker_seed = seed + rank * 1000 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_logging_folder(args):
    date = str(datetime.now()).replace(" ", "-").replace(":", "-")
    output_path = os.path.join(
        args.output_path,
        f"{date}_seed{args.seed}"
    )
    os.makedirs(output_path, exist_ok=False)

    os.makedirs(args.output_path, exist_ok=True)
    wandb.login(host=args.wandb_host, key=args.wandb_key)
    run = wandb.init(config=OmegaConf.to_container(args, resolve=True), dir=args.output_path, **
                     {"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project,
                      "group": getattr(args, "wandb_group", None)})
    wandb.run.log_code(".")
    wandb.run.name = args.wandb_name
    print(f"run dir: {run.dir}")
    wandb_folder = run.dir
    os.makedirs(wandb_folder, exist_ok=True)

    return output_path, wandb_folder


def is_distributed():
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7),
              transformer_module=None, use_orig_params=False, sync_module_states=False):

    if not is_distributed() or sharding_strategy == 'no_shard':
        # override settings if not sharded or not distributed
        sharding_strategy = 'no_shard'
        wrap_strategy = 'none'

    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        if transformer_module is None:
            raise ValueError("transformer_module must be provided when wrap_strategy is 'transformer'")

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    elif wrap_strategy == 'none':
        auto_wrap_policy = None
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states
    )
    return module


def cycle(dl, sampler, start_epoch=0):
    epoch = start_epoch
    while True:
        print('Starting epoch', epoch)
        sampler.set_epoch(epoch)
        try:
            for data in dl:
                yield data
        except Exception as e:
            print(f"DataLoader exception: {e}. Restarting epoch {epoch}.")
        epoch += 1


def fsdp_load_or_save(model, optimizer, operation="save", model_state_dict=None, optim_state_dict=None):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    optim_fullstate_save_policy = FullOptimStateDictConfig(rank0_only=True)

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy, optim_fullstate_save_policy
    ):
        if operation == "save":
            model_state_dict = model.state_dict()
            optim_state_dict = FSDP.optim_state_dict(model, optimizer)
        elif operation == "load":
            if model_state_dict is None:
                raise ValueError("model_state_dict must be provided for loading.")

            model.load_state_dict(model_state_dict, strict=True)

            if optimizer is None or optim_state_dict is None:
                print('Warning: fsdp loader received None optimizer or optim_state_dict, only loading model weights.')
                return None, None

            # Cache current per-group learning rates
            saved_lrs = [pg.get('lr', None) for pg in optimizer.param_groups]
            sharded_optim_state_dict = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=optim_state_dict
            )
            optimizer.load_state_dict(sharded_optim_state_dict)

            # Restore cached learning rates
            for pg, lr in zip(optimizer.param_groups, saved_lrs):
                if lr is not None:
                    pg['lr'] = lr

    if operation == "save":
        return model_state_dict, optim_state_dict
    return None, None


def barrier():
    if dist.is_initialized():
        dist.barrier()


def prepare_for_saving(tensor, fps=16, caption=None):
    # Convert range [-1, 1] to [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)

    if tensor.ndim == 4:
        # Assuming it's an image and has shape [batch_size, 3, height, width]
        tensor = make_grid(tensor, 4, padding=0, normalize=False)
        return wandb.Image((tensor * 255).float().numpy().astype(np.uint8), caption=caption)
    elif tensor.ndim == 5:
        # Assuming it's a video and has shape [batch_size, num_frames, 3, height, width]
        return wandb.Video((tensor * 255).float().numpy().astype(np.uint8), fps=fps, format="webm",
                           caption=caption)
    else:
        raise ValueError("Unsupported tensor shape for saving. Expected 4D (image) or 5D (video) tensor.")


def get_sha256_key(prompt: str) -> str:
    """
    Generates a fixed-length SHA-256 hash for a given string prompt.

    Args:
        prompt: The input string to be hashed.

    Returns:
        A 64-character hexadecimal string representing the SHA-256 hash.
    """
    # Encode the string into bytes using UTF-8
    prompt_bytes = prompt.encode('utf-8')
    sha256_hash = hashlib.sha256(prompt_bytes)
    hex_key = sha256_hash.hexdigest()
    return hex_key


def keep_last_n_checkpoints(checkpoint_dir, n=3):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_model')]
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split('_')[-1]))
    for ckpt in ckpts_sorted[:-n]:
        shutil.rmtree(os.path.join(checkpoint_dir, ckpt))


def keep_best_n_checkpoints(checkpoint_dir, n=3):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_model')]
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split('_')[-2]))
    for ckpt in ckpts_sorted[n:]:
        shutil.rmtree(os.path.join(checkpoint_dir, ckpt))


def _get_latest_wandb_run(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    # List all nonempty subdirectories in the checkpoint directory except 'wandb'
    runs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))
            and any(os.scandir(os.path.join(checkpoint_dir, d))) and d != 'wandb']

    if not runs:
        return None
    runs_sorted = sorted(runs, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    latest_run = runs_sorted[-1]
    return os.path.join(checkpoint_dir, latest_run)


def get_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = _get_latest_wandb_run(checkpoint_dir)
    if checkpoint_dir is None:
        return None
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_model')]
    if not ckpts:
        return None
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split('_')[-1]))
    latest_ckpt = ckpts_sorted[-1]
    return os.path.join(checkpoint_dir, latest_ckpt)


def load_video(
        video_path: str,
        start_time: float = None,
        end_time: float = None,
        fps: float = None,
        size: int = None,
        size_divisible: int = 1,
        precise_time: bool = False,
        verbose: bool = False,
):
    """
    Load and process a video file and return the frames and the timestamps of each frame.

    Args:
        video_path (str): Path to the video file.
        start_time (float, optional): Start time in seconds. Defaults to None.
        end_time (float, optional): End time in seconds. Defaults to None.
        fps (float, optional): Frames per second. Defaults to None.
        size (int, optional): Size of the shortest side. Defaults to None.
        size_divisible (int, optional): Size divisible by this number. Defaults to 1.
        precise_time (bool, optional): Whether to use precise time. Defaults to False.
        verbose (bool, optional): Print ffmpeg output. Defaults to False.

    Returns:
        frames (List[PIL.Image]): List of frames.
        timestamps (List[float]): List of timestamps.
    """
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    w, h = int(video_stream['width']), int(video_stream['height'])

    kwargs, input_kwargs, output_kwargs = {}, {}, {}
    do_trim = start_time is not None or end_time is not None
    if start_time is not None:
        new_start_time = max(float(video_stream['start_time']), start_time)
        duration -= new_start_time - start_time
        start_time = new_start_time
    else:
        start_time = float(video_stream['start_time'])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    else:
        duration = duration
    if do_trim:
        kwargs = {'ss': start_time, 't': duration}
    if precise_time:
        output_kwargs.update(kwargs)
    else:
        input_kwargs.update(kwargs)

    if size is not None:
        scale_factor = size / min(w, h)
        new_w, new_h = round(w * scale_factor), round(h * scale_factor)
    else:
        new_w, new_h = w, h
    new_w = new_w // size_divisible * size_divisible
    new_h = new_h // size_divisible * size_divisible

    # NOTE: It may result in unexpected number of frames in ffmpeg
    # if calculate the fps directly according to max_frames
    # if max_frames is not None and (fps is None or duration * fps > 2 * max_frames):
    #     fps = round(max_frames / duration * 2)

    stream = ffmpeg.input(video_path, **input_kwargs)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    if new_w != w or new_h != h:
        stream = ffmpeg.filter(stream, 'scale', new_w, new_h)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

    frames = np.frombuffer(out, np.uint8).reshape([-1, new_h, new_w, 3]).transpose([0, 3, 1, 2])

    if fps is not None:
        timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[:len(frames)]
    else:
        timestamps = np.linspace(start_time, start_time + duration, len(frames))
    return frames, timestamps


def is_correct_aspect_ratio(width: int, height: int, target_ratio: float, tolerance: float = 0.1) -> bool:
    actual_ratio = width / height
    return abs(actual_ratio - target_ratio) <= (target_ratio * tolerance)
