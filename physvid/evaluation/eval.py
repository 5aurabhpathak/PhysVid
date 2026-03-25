import argparse

from omegaconf import OmegaConf
from torch import distributed as dist

from physvid.evaluation.videophy2_eval import VideoPhy2Evaluator
from physvid.evaluation.videophy_eval import VideoPhyEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument('--evaluator_name', type=str, required=True,
                        choices=['videophy2', 'videophy'],
                        help='The name of the evaluator to use.')
    parser.add_argument('--generated_data_path', type=str, default=None,
                        help='Path to load the generated synthetic dataset from. If not provided, '
                             'it will use the path in config file.')
    parser.add_argument('--eval_result_file', type=str, default=None,
                        help='Path to save the evaluation results. If not provided, '
                        'it will use the path in config file.')
    parser.add_argument('--caption_col', type=str, default=None,
                        choices=[None, 'caption', 'upsampled_caption'],
                        help='If provided, generate video using this caption style.')
    parser.add_argument('--hf_dataset_name', type=str, default=None,
                        choices=[None, 'videophy', 'videophy2'],
                        help='If provided, override the dataset name in config file.')


    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.generated_data_path is not None:
        config.generated_data_path = args.generated_data_path
    if args.eval_result_file is not None:
        config.eval_result_file = args.eval_result_file
    if args.caption_col is not None:
        config.caption_col = args.caption_col
    if args.hf_dataset_name is not None:
        config.hf_dataset_name = args.hf_dataset_name

    if args.evaluator_name == 'videophy2':
        evaluator = VideoPhy2Evaluator(config)
    elif args.evaluator_name == 'videophy':
        evaluator = VideoPhyEvaluator(config)
    else:
        raise ValueError(f"Unknown evaluator name: {args.evaluator_name}")

    evaluator.run()


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
