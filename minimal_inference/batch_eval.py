# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
import yaml

# import liveavatar.models.wan.wan_2_2 as wan
from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from liveavatar.models.wan.wan_2_2.distributed.util import init_distributed_group
from liveavatar.models.wan.wan_2_2.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from liveavatar.models.wan.wan_2_2.utils.utils import merge_video_audio, save_video, str2bool
from liveavatar.utils.args_config import parse_args_for_training_config as training_config_parser
from liveavatar.utils.router.synthesize_audio import merge_multiple_audio_files

EXAMPLE_PROMPT = {
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def load_sample_list(file_path):
    """
    Load sample list from JSON or YAML file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        dict: Dictionary of samples with required fields (prompt, image, audio, num_clip)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample list file not found: {file_path}")
    
    # Determine file format by extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    elif file_ext in ['.yaml', '.yml']:
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .json, .yaml, .yml are supported.")
    
    # Validate samples
    if not isinstance(samples, dict):
        raise ValueError("Sample list must be a dictionary with sample names as keys")
    
    required_fields = ['prompt', 'image', 'audio', 'num_clip']
    for sample_name, sample_data in samples.items():
        if not isinstance(sample_data, dict):
            raise ValueError(f"Sample '{sample_name}' must be a dictionary")
        
        missing_fields = [field for field in required_fields if field not in sample_data]
        if missing_fields:
            raise ValueError(f"Sample '{sample_name}' is missing required fields: {missing_fields}")
    
    return samples


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The directory to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='euler',
        choices=['euler', 'unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    parser.add_argument(
        "--load_lora",
        action="store_true",
        default=False,
        help="Whether to load the LoRA weights.")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="The path to the LoRA weights.")
    parser.add_argument(
        "--lora_path_dmd",
        type=str,
        default=None,
        help="The path to the LoRA weights for DMD.")
    parser.add_argument(
        "--training_config",
        type=str,
        default=None,
        help="The path to the training config file.")
    parser.add_argument(
        "--use_dataset",
        action="store_true",
        default=False,
        help="Whether to use the dataset for inference.")
    parser.add_argument(
        "--dataset_sample_idx",
        type=int,
        default=0,
        help="The index of the sample to use for inference.")
    parser.add_argument(
        "--drop_motion_noisy",
        action="store_true",
        default=False,
        help="Whether to drop the motion noisy.")
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        default=False,
        help="Whether to use a single GPU.")
    parser.add_argument(
        "--using_merged_ckpt",
        action="store_true",
        default=False,
        help="Whether to use the merged ckpt.")
    parser.add_argument(
        "--num_gpus_dit",
        type=int,
        default=4,
        help="The number of GPUs to use for DiT.")
    parser.add_argument(
        "--enable_vae_parallel",
        action="store_true",
        default=False,
        help="Whether to enable VAE parallel decoding on a separate GPU.")
    parser.add_argument(
        "--sample_list",
        type=str,
        default=None,
        required=True,
        help="Path to JSON or YAML file containing batch sample configurations.")

    parser.add_argument(
    "--offload_kv_cache",
    action="store_true",
    default=False,
    help="Whether to offload the KV cache to CPU.")
    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate_single_sample(wan_s2v, args, cfg, sample_idx, sample_name, sample_data, rank, world_size):
    """
    Generate video for a single sample.
    
    Args:
        wan_s2v: The WanS2V pipeline instance
        args: Command line arguments
        cfg: Model configuration
        sample_idx: Index of the sample (for naming)
        sample_name: Name of the sample from config file
        sample_data: Sample configuration dict containing prompt, image, audio, num_clip
        rank: Distributed training rank
        world_size: Number of processes
    """
    logging.info(f"Generating video for sample {sample_idx}: {sample_name}")
    logging.info(f"  Prompt: {sample_data['prompt']}")
    logging.info(f"  Image: {sample_data['image']}")
    logging.info(f"  Audio: {sample_data['audio']}")
    logging.info(f"  Num_clip: {sample_data['num_clip']}")
    
    video, dataset_info = wan_s2v.generate(
        input_prompt=sample_data['prompt'],
        ref_image_path=sample_data['image'],
        audio_path=sample_data['audio'],
        enable_tts=args.enable_tts,
        tts_prompt_audio=args.tts_prompt_audio,
        tts_prompt_text=args.tts_prompt_text,
        tts_text=args.tts_text,
        num_repeat=sample_data['num_clip'],
        pose_video=args.pose_video,
        generate_size=args.size,
        max_area=MAX_AREA_CONFIGS[args.size],
        infer_frames=args.infer_frames,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        init_first_frame=args.start_from_ref,
        use_dataset=args.use_dataset,
        dataset_sample_idx=args.dataset_sample_idx,
        drop_motion_noisy=args.drop_motion_noisy,
        num_gpus_dit=args.num_gpus_dit,
        enable_vae_parallel=args.enable_vae_parallel,
        input_video_for_sam2=None,
    )

    logging.info(f"Denoising video done for sample {sample_idx}")
    print(f"rank: {rank}")
    
    if args.enable_vae_parallel:
        save_rank = args.num_gpus_dit 
    else:
        save_rank = 0 if world_size == 1 else args.num_gpus_dit - 1 
    
    if rank == save_rank:
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = sample_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
        suffix = '.mp4'
        
        # New naming format with sample index prefix
        save_file = f"sample_{sample_idx:04d}_{formatted_time}_{args.sample_steps}step_{formatted_prompt}"
        
        if args.lora_path_dmd is not None:
            # Only add lora suffix for .pt files (local paths with sufficient depth)
            path_parts = args.lora_path_dmd.split("/")
            if args.lora_path_dmd.endswith(".pt") and len(path_parts) >= 3:
                lora_suffix = path_parts[-3] + "_" + path_parts[-1].split(".")[0]
                save_file = save_file + "_" + lora_suffix
        
        if args.save_dir is None:
            if args.lora_path_dmd is not None and args.lora_path_dmd.endswith(".pt") and len(args.lora_path_dmd.split("/")) >= 3:
                path_parts = args.lora_path_dmd.split("/")
                lora_suffix = path_parts[-3] + "_" + path_parts[-1].split(".")[0]
                args.save_dir = "./output/batch_eval/" + lora_suffix + "/"
            else:
                args.save_dir = "./output/batch_eval/"
        
        os.makedirs(args.save_dir, exist_ok=True)
        save_file = args.save_dir + save_file + suffix
        
        logging.info(f"Saving generated video to {save_file}")
        save_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        
        if "s2v" in args.task:
            if args.use_dataset:
                from tests.help_functions.clip_with_audio import clip_by_frames_with_wav
                clip_by_frames_with_wav(
                    input_video_path=dataset_info['video_path'],
                    start_frame_inclusive=dataset_info['start'],
                    end_frame_inclusive=dataset_info['end'],
                    output_video_path=f"./tmp/video/{os.path.basename(dataset_info['video_path'])}",
                    output_wav_path=f"./tmp/audio/{os.path.basename(dataset_info['video_path']).split('.mp4')[0]}.wav",
                )
                resample_file = save_file.replace('.mp4', '_resample.mp4')
                command = f"ffmpeg -i '{save_file}' -vf fps=25 -r 25 '{resample_file}'"
                os.system(command)
                merge_video_audio(video_path=resample_file, audio_path=f"./tmp/audio/{os.path.basename(dataset_info['video_path']).split('.mp4')[0]}.wav")
            elif args.enable_tts is False:
                merge_video_audio(video_path=save_file, audio_path=sample_data['audio'])
            else:
                merge_video_audio(video_path=save_file, audio_path="tts.wav")
    
    del video
    torch.cuda.empty_cache()
    
    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()


def generate(args, training_settings):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print(f"world_size: {world_size}")
    if world_size == 1:
        rank = 0
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size)
    if world_size > 1:
        assert world_size >= 5, "At least 5 GPUs are supported for distributed inference."
        assert args.num_gpus_dit == 4, "Only 4 GPUs are supported for distributed inference."
        assert args.enable_vae_parallel is True, "VAE parallel is required for distributed inference."
        args.single_gpu = False
        from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
        print(f"Using TPP distributed inference.")
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."
        args.enable_vae_parallel = False
        args.num_gpus_dit = 1
        args.single_gpu = True
        from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
        print(f"Using single GPU inference with offload mode: {args.offload_model}")

    if args.ulysses_size > 1:
        assert False, "Sequence parallel is not supported."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            sp_size=args.ulysses_size,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            single_gpu=args.single_gpu,
            offload_kv_cache=args.offload_kv_cache,
        )
        if args.load_lora and args.lora_path_dmd is not None:
            print(f'Use LoRA: lora path: {args.lora_path_dmd}, lora rank:', training_settings['lora_rank'] ,", lora alpha: ",training_settings['lora_alpha'])
            # for version <1.5.0, using ckpt not merged 还需要把ckpt path改了


            if args.lora_path_dmd is not None:
                wan_s2v.noise_model = wan_s2v.add_lora_to_model(
                    wan_s2v.noise_model,
                    lora_rank=training_settings['lora_rank'],
                    lora_alpha=training_settings['lora_alpha'],
                    lora_target_modules=training_settings['lora_target_modules'],
                    init_lora_weights=training_settings['init_lora_weights'],
                    pretrained_lora_path=args.lora_path_dmd,
                    load_lora_weight_only=False,
                )

    else:
        assert False, "Only s2v is supported for now."


    # Load sample list
    logging.info(f"Loading sample list from {args.sample_list}")
    samples = load_sample_list(args.sample_list)
    logging.info(f"Loaded {len(samples)} samples")
    
    # Process each sample
    for idx, (sample_name, sample_data) in enumerate(samples.items(), start=1):
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing sample {idx}/{len(samples)}: {sample_name}")
        logging.info(f"{'='*80}")
        
        try:
            generate_single_sample(
                wan_s2v=wan_s2v,
                args=args,
                cfg=cfg,
                sample_idx=idx,
                sample_name=sample_name,
                sample_data=sample_data,
                rank=rank,
                world_size=world_size
            )
            logging.info(f"Successfully completed sample {idx}: {sample_name}")
        except Exception as e:
            logging.error(f"Error processing sample {idx} ({sample_name}): {str(e)}")
            raise  # Stop on error as requested

    if dist.is_initialized():
        print(f"rank {rank} done, waiting for other ranks to finish...")
        # dist.barrier()
        print(f"rank {rank} start destroying process group...")
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    training_settings = training_config_parser(args.training_config)
    generate(args, training_settings)
