import torch
from physvid.util import set_model_seed
from diffusers.utils import export_to_video

from physvid.data import VideoPhyDataset
from physvid.inference import (InferencePipeline,
                               LocalConditioningInferencePipeline)


class Config:
    seed = 100
    model_name = "local_conditioning_wan"
    # generator_checkpoint = "data/wisa80k/wandb_runs/finetune-2e6-all-steps-local-from-ckpt/2025-10-24-07-32-29.356486_seed9060043/checkpoint_model_000722_002999/model.pt"
    num_frame_per_block = 3
    video_data_shape = (81, 3, 480, 832)  # (frames, channels, height, width) for Wan models
    image_or_video_shape = (1, 21, 16, 60, 104)  # (batch, frames, channels, height, width) for Wan models
    decoupled_vlm_mode = False
    hf_dataset_name = "videophy"
    caption_col = "caption"
    num_inference_steps = 50
    guidance_scale = 6.
    negative_prompt = ('Unrealistic colors, overexposed, static, blurred details, subtitles, style, artwork, painting, '
                       'image, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, '
                       'mutilated, extra fingers, deformed, disfigured, malformed limbs, fused fingers, still image, '
                       'cluttered background, three legs, walking backwards, antigravity,frictionless, lowres, blurry, '
                       'fuzzy, out of focus, extra limbs, cloned face, missing arms, '
                       'missing legs, extra arms, extra legs, too many fingers, bad physics, '
                       'reversed gravity, reversed time')
    num_blocks = 7
    positive_vlm_instruction_path =  'physvid/models/prompt.txt'
    negative_vlm_instruction_path = 'physvid/models/negative_prompt_v2.txt'
    use_physics_observer = False


def main():
    device = torch.cuda.current_device()
    dtype = torch.bfloat16
    config = Config()

    if config.seed == 0:
        random_seed = torch.randint(0, 10000000, (1,), device=device)
        config.seed = random_seed.item()

    print(f'using seed: {config.seed}')
    set_model_seed(config.seed)

    if 'conditioning' in config.model_name:
        pipe = LocalConditioningInferencePipeline(config, device=device, dtype=dtype)
    else:
        pipe = InferencePipeline(config, device=device, dtype=dtype)

    if hasattr(config, 'generator_checkpoint'):
        state_dict = torch.load(config.generator_checkpoint, map_location="cpu", weights_only=False)['generator']
        pipe.generator.model.load_state_dict(state_dict, strict=True)

    pipe = pipe.to(device=device, dtype=dtype)

    dataset = VideoPhyDataset(caption_col=config.caption_col)
    # text_prompt = 'A car driving in the snowy weather.'
    text_prompt = dataset[10]
    print(f"Text prompt: {text_prompt}")

    noise = torch.randn(config.image_or_video_shape,
                              dtype=dtype,
                              device=device)

    with torch.no_grad():
        video = pipe.inference(
            noise=noise,
            text_prompts=[text_prompt]
        ).permute(0, 1, 3, 4, 2).cpu().float().numpy()[0]

    save_location = f"{config.model_name}_output.mp4"
    export_to_video(video, save_location, fps=16)


if __name__ == '__main__':
    main()
