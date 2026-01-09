# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import json
import time
import subprocess
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from decord import VideoReader
from PIL import Image
import torch.nn.functional as F
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import subprocess
from diffusers import FlowMatchEulerDiscreteScheduler
from .wan_2_2.distributed.fsdp import shard_model
from .wan_2_2.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .wan_2_2.distributed.util import get_world_size
from .causal_audio_encoder import AudioEncoder
from .causal_audio_encoder import AudioEncoder_Training
from .causal_model_s2v import CausalWanModel_S2V, sp_attn_forward_s2v
from .wan_2_2.modules.t5 import T5EncoderModel
from .wan_2_2.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .wan_2_2.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ...utils.load_weight_utils import load_state_dict
from liveavatar.utils.router.utils import process_masks_to_routing_logits
from .inference_utils import STREAMING_VAE
if STREAMING_VAE:
    from .wan_2_2.modules.vae_streaming import WanVAE as Wan2_1_VAE
else:
    from .wan_2_2.modules.vae2_1 import Wan2_1_VAE



def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


class WanS2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        sp_size=None,
        t5_cpu=False,
        init_on_cpu=True,
        drop_part_motion_frames=0.3,
        convert_model_dtype=False,
        is_training=False,
        single_gpu=False,
        offload_kv_cache=False
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.is_training = is_training
        self.num_train_timesteps = config.num_train_timesteps # 1000
        self.num_frames_per_block = config.num_frames_per_block
        self.param_dtype = config.param_dtype
        self.checkpoint_dir = checkpoint_dir
        self.drop_part_motion_frames = drop_part_motion_frames
        self.single_gpu = single_gpu
        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,dtype=self.param_dtype)

        if self.is_training:
            from liveavatar.models.wan.flow_match import FlowMatchScheduler_Omni
            self.scheduler = FlowMatchScheduler_Omni(shift=5, sigma_min=0.0, extra_one_step=True)
            self.scheduler.set_timesteps(1000, training=True)
        else:
            if config.sample_solver == 'euler':
                self.sample_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=3)
            elif config.sample_solver == 'unipc':#default
                self.sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            elif config.sample_solver == 'dpm++':
                self.sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            else:
                raise NotImplementedError("Unsupported solver.")

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        if not dit_fsdp:
            self.noise_model = CausalWanModel_S2V.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.param_dtype,
                device_map=self.device)
        else:
            self.noise_model = CausalWanModel_S2V.from_pretrained(
                checkpoint_dir, torch_dtype=self.param_dtype)
        
        self.noise_model.freqs.to(device=self.device)

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            sp_size=sp_size,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        self.noise_model.num_frame_per_block = self.num_frames_per_block

        self.audio_encoder = AudioEncoder(
            device=self.device,
            model_id=os.path.join(checkpoint_dir,
                                "wav2vec2-large-xlsr-53-english"))

        if use_sp:
            self.sp_size = sp_size if sp_size is not None else get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        self.drop_first_motion = config.drop_first_motion
        self.fps = config.sample_fps
        self.audio_sample_m = 0
        self.tgt_gpu_id = 0
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None, load_only=False, load_lora_weight_only=False):
        if not load_only:
            self.lora_alpha = lora_alpha
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=True if init_lora_weights == "kaiming" else init_lora_weights,
                target_modules=lora_target_modules.split(","),
            )
            model = get_peft_model(model, lora_config)
                
        if pretrained_lora_path is not None:
            ori_pretrained_lora_path = pretrained_lora_path
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            
            # get_peft_model adds "base_model.model." prefix to the keys
            first_key = next(iter(state_dict.keys()))
            if not first_key.startswith("base_model.model."):
                state_dict = {f"base_model.model.{k}": v for k, v in state_dict.items()}

            if load_lora_weight_only:
                state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {ori_pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
            
            # Merge weights and return base model
            print(f"Merging LoRA weights from {ori_pretrained_lora_path}...")
            model = model.merge_and_unload()
            print(f"LoRA merged successfully.")
            
        return model
    
    def set_all_model_to_dtype_device(self, dtype, device):
        models = [
            self.noise_model,
            self.text_encoder.model,
            self.vae.model,
            self.audio_encoder.model
        ]
        
        for model in models:
            for param in model.parameters():
                param.data = param.data.to(device=device, dtype=dtype)
                if param.grad is not None:
                    param.grad = param.grad.to(device=device, dtype=dtype)
            
    
    def set_requires_grad(self, requires_grad=True):
        self.requires_grad = requires_grad
        self.noise_model.requires_grad_(requires_grad)
        self.text_encoder.model.requires_grad_(requires_grad)
        self.vae.model.requires_grad_(requires_grad)
        self.audio_encoder.model.requires_grad_(requires_grad)
    
    def set_eval(self):
        self.noise_model.eval()
        self.text_encoder.model.eval()
        self.vae.model.eval()
        self.audio_encoder.model.eval()
    
    def set_train(self):
        self.noise_model.train()
        self.text_encoder.model.train()
        self.vae.model.train()
        self.audio_encoder.model.train()
    
    def set_device_dtype(self, device, dtype):
        self.noise_model.to(device, dtype)
        self.text_encoder.model.to(device, dtype)
        self.vae.model.to(device, dtype)
        self.audio_encoder.model.to(device, dtype)
    
    def _configure_model(self, model, use_sp, sp_size, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            sp_size (`int`):
                Sequence parallel size to use instead of world_size.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)
        if use_sp:
            # Store sp_size in model for use in forward pass
            model.sp_size = sp_size
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_s2v, block.self_attn)
                # Store sp_size in each block for access in forward pass
                block.sp_size = sp_size
            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _process_timestep(self, timestep: torch.Tensor, type: str="causal_video") -> torch.Tensor:
        """
        copy from liveavatar/dmd.py:
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.
            - type: a string indicating the type of the current model (image, bidirectional_video, or causal_video).
        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if type == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frames_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def get_size_less_than_area(self,
                                height,
                                width,
                                target_area=1024 * 704,
                                divisor=64):
        if height * width <= target_area:
            # If the original image area is already less than or equal to the target,
            # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            # Resize to fit within the target area and then pad to multiples of `divisor`
            max_upper_area = target_area  # Maximum allowed total pixel count after padding
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area

            # Calculate scale boundaries using quadratic equation
            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (
                2 * a)  # Scale when maximum padding is applied
            max_scale = math.sqrt(max_upper_area /
                                  (height * width))  # Scale without any padding

        # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
        # Use binary search-like iteration to find this scale
        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)

            # Pad to make dimensions divisible by 64
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            padded_height, padded_width = new_height + pad_height, new_width + pad_width

            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width
        else:
            # Fallback: calculate target dimensions based on aspect ratio and divisor alignment
            aspect_ratio = width / height
            target_width = int(
                (target_area * aspect_ratio)**0.5 // divisor * divisor)
            target_height = int(
                (target_area / aspect_ratio)**0.5 // divisor * divisor)

            # Ensure the result is not larger than the original resolution
            if target_width >= width or target_height >= height:
                target_width = int(width // divisor * divisor)
                target_height = int(height // divisor * divisor)

            return target_height, target_width

    def prepare_default_cond_input(self,
                                   map_shape=[3, 12, 64, 64],
                                   motion_frames=5,
                                   lat_motion_frames=2,
                                   enable_mano=False,
                                   enable_kp=False,
                                   enable_pose=False):
        default_value = [1.0, -1.0, -1.0]
        cond_enable = [enable_mano, enable_kp, enable_pose]
        cond = []
        for d, c in zip(default_value, cond_enable):
            if c:
                map_value = torch.ones(
                    map_shape, dtype=self.param_dtype, device=self.device) * d
                cond_lat = torch.cat([
                    map_value[:, :, 0:1].repeat(1, 1, motion_frames, 1, 1),
                    map_value
                ],
                                     dim=2)
                cond_lat = torch.stack(
                    self.vae.encode(cond_lat.to(
                        self.param_dtype)))[:, :, lat_motion_frames:].to(
                            self.param_dtype)

                cond.append(cond_lat)
        if len(cond) >= 1:
            cond = torch.cat(cond, dim=1)
        else:
            cond = None
        return cond

    def encode_audio(self, audio_path, infer_frames):
        assert self.is_training is False
        z = self.audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat
    
    def encode_audio_from_array(self, audio_array, infer_frames):
        """
        从 numpy array 编码音频（用于流式输入）
        
        Args:
            audio_array: numpy array, shape [samples], 16kHz
            infer_frames: 推理帧数
        
        Returns:
            audio_embed_bucket: 编码后的音频特征
            num_repeat: 重复次数
        """
        assert self.is_training is False
        z = self.audio_encoder.extract_audio_feat_from_array(
            audio_array, 
            return_all_layers=True,
            dtype=self.param_dtype
        )
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device, self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def _streaming_encode_next_audio_block_or_random(self, block_frames: int):
        chunk = self.get_audio_callback()
        audio_embed, _ = self.encode_audio_from_array(chunk, infer_frames=block_frames)
        return audio_embed[..., :block_frames].contiguous() #torch.Size([1, 25, 1024, 12])
    
    def encode_audio_training(self, audio_tensor, infer_frames,fps,audio_sample_m = 0):
        assert self.is_training is True
        z = self.audio_encoder.extract_audio_feat_training(
            audio_tensor, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=fps, batch_frames=infer_frames, m=audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def read_last_n_frames(self,
                           video_path,
                           n_frames,
                           target_fps=16,
                           reverse=False):
        """
        Read the last `n_frames` from a video at the specified frame rate.

        Parameters:
            video_path (str): Path to the video file.
            n_frames (int): Number of frames to read.
            target_fps (int, optional): Target sampling frame rate. Defaults to 16.
            reverse (bool, optional): Whether to read frames in reverse order. 
                                    If True, reads the first `n_frames` instead of the last ones.

        Returns:
            np.ndarray: A NumPy array of shape [n_frames, H, W, 3], representing the sampled video frames.
        """
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        interval = max(1, round(original_fps / target_fps))

        required_span = (n_frames - 1) * interval

        start_frame = max(0, total_frames - required_span -
                          1) if not reverse else 0

        sampled_indices = []
        for i in range(n_frames):
            indice = start_frame + i * interval
            if indice >= total_frames:
                break
            else:
                sampled_indices.append(indice)

        return vr.get_batch(sampled_indices).asnumpy()

    def encode_prompt(self, input_prompt, n_prompt=None, offload_model=True):
        """
        编码文本提示词
        三种情况：只有正向，cfg但是提供空 neg，提供 neg
        
        Args:
            input_prompt (str): 正面文本提示词
            n_prompt (str): 负面文本提示词，默认为空字符串
            offload_model (bool): 是否在编码后将模型卸载到CPU，默认为True
            
        Returns:
            tuple: (context, context_null) - 编码后的正面和负面文本上下文
        """
        context_null = None
        if n_prompt == "": #case 3
            n_prompt = self.sample_neg_prompt
            
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if n_prompt is not None:
                context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            if n_prompt is not None:
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context_null = [t.to(self.device) for t in context_null]
                
        return context, context_null

    def load_pose_cond(self, pose_video, num_repeat, infer_frames, size):
        HEIGHT, WIDTH = size
        if not pose_video is None:
            pose_seq = self.read_last_n_frames(
                pose_video,
                n_frames=infer_frames * num_repeat,
                target_fps=self.fps,
                reverse=True)

            resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
            crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
            tensor_trans = transforms.ToTensor()

            cond_tensor = torch.from_numpy(pose_seq)
            cond_tensor = cond_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop_opreat(resize_opreat(cond_tensor)).permute(
                1, 0, 2, 3).unsqueeze(0)

            padding_frame_num = num_repeat * infer_frames - cond_tensor.shape[2]
            cond_tensor = torch.cat([
                cond_tensor,
                - torch.ones([1, 3, padding_frame_num, HEIGHT, WIDTH])
            ],
                                    dim=2)

            cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-torch.ones([1, 3, infer_frames, HEIGHT, WIDTH])]

        COND = []
        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond],
                             dim=2)
            cond_lat = torch.stack(
                self.vae.encode(
                    cond.to(dtype=self.param_dtype,
                            device=self.device)))[:, :,
                                                  1:].cpu()  # for mem save
            COND.append(cond_lat)
        return COND

    def get_gen_size(self, size, max_area, ref_image_path, pre_video_path):
        if not size is None:
            HEIGHT, WIDTH = size
        else:
            if pre_video_path:
                ref_image = self.read_last_n_frames(
                    pre_video_path, n_frames=1)[0]
            else:
                ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
            HEIGHT, WIDTH = ref_image.shape[:2]
        HEIGHT, WIDTH = self.get_size_less_than_area(
            HEIGHT, WIDTH, target_area=max_area)
        return (HEIGHT, WIDTH)

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size=13500):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        gpu_id : "1","2","3","4"
        """
        kv_cache1 = []

        for _ in range(self.noise_model.num_layers):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
                "cond_k": torch.zeros([batch_size, 3000, 40, 128], dtype=dtype, device=device), #hard-coded for 720*400 resolution
                "cond_v": torch.zeros([batch_size, 3000, 40, 128], dtype=dtype, device=device), #hard-coded for 720*400 resolution
                "cond_end": torch.tensor([0], dtype=torch.long, device=device) #dynamically updated 2640
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache
    
    def _move_kv_cache_to_working_gpu(self,moved_id, gpu_id=0):
        """
        Move the KV cache to the working GPU.
        move_id : "1","2","3","4"
        """
        if gpu_id == 0: #move to working device
            tgt_device =f"cuda:{gpu_id}"
        else: #offload
            tgt_device = "cpu" if self.single_gpu else f"cuda:{gpu_id}"
            
        kv_cache1 = self.kv_cache1[str(moved_id)]
        for layer in kv_cache1:
            layer["k"] = layer["k"].to(tgt_device)
            layer["v"] = layer["v"].to(tgt_device)
            layer["cond_k"] = layer["cond_k"].to(tgt_device)
            layer["cond_v"] = layer["cond_v"].to(tgt_device)
        self.kv_cache1[str(moved_id)] = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.noise_model.num_layers):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache
    
    def _initialize_comm_group(self, num_gpus_dit=4, enable_vae_parallel=False):
        local_gpu_id = torch.distributed.get_rank()

        self.tgt_gpu = (local_gpu_id + 1) % (num_gpus_dit+int(enable_vae_parallel)) if (local_gpu_id!=num_gpus_dit - 1 + int(enable_vae_parallel)) else None
        self.src_gpu = (local_gpu_id - 1) % (num_gpus_dit+int(enable_vae_parallel)) if (local_gpu_id!=0) else None
        self.audio_tgt_gpu = (local_gpu_id + 1) % (num_gpus_dit+int(enable_vae_parallel)) if (local_gpu_id!=num_gpus_dit - 1) else None
        self.audio_src_gpu = (local_gpu_id - 1) % (num_gpus_dit+int(enable_vae_parallel)) if (local_gpu_id!=num_gpus_dit - 1 + int(enable_vae_parallel)) else None
        
    def generate(
        self,
        input_prompt=None,
        ref_image_path=None,
        audio_path=None,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=1,
        pose_video=None,
        generate_size=None,
        max_area=720 * 1280,
        infer_frames=80,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        init_first_frame=False,
        use_dataset=False,
        dataset_sample_idx=0,
        drop_motion_noisy=False,
        num_gpus_dit=4,
        max_repeat=100000000,
        enable_vae_parallel=False,
        mask=None,
        input_video_for_sam2=None,
        enable_online_decode=False,
    ):
        # ------------------------------------Step 1: prepare conditional inputs--------------------------------------
        
        size = self.get_gen_size(
            size=None,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None)
        HEIGHT, WIDTH = size
        channel = 3
        resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
        crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()

        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))

        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()
        audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(
            0) * 2 - 1.0  # b c 1 h w
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.vae.dtype, device=self.vae.device)
        ref_pixel_values = ref_pixel_values.repeat(1, 1, 5, 1, 1)
        ref_latents = torch.stack(self.vae.encode(ref_pixel_values))[:,:,1:]


        # drop_first_motion = self.drop_first_motion
        drop_first_motion = False
        motion_latents = ref_pixel_values.repeat(1, 1, self.motion_frames, 1, 1)
        videos_last_frames = motion_latents.detach()
        motion_latents = torch.stack(self.vae.encode(motion_latents))
        
        if drop_motion_noisy:
            zero_motion_latents = torch.zeros_like(motion_latents)

        # get pose cond input if need
        COND = self.load_pose_cond(
            pose_video=pose_video,
            num_repeat=num_repeat,
            infer_frames=infer_frames,
            size=size) # list(1):[1,16,12,48,32]当num_repeat=1

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # process prompt
        context, context_null = self.encode_prompt(input_prompt, n_prompt, offload_model) #list(1):[len,4096]
        dataset_info = {}

        print("complete prepare conditional inputs")
        if sample_solver == 'euler':#default
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=3)
        else:
            raise NotImplementedError("Unsupported solver.")
        self._initialize_comm_group(num_gpus_dit=num_gpus_dit, enable_vae_parallel=enable_vae_parallel)
        in_dit_device = dist.get_rank() < num_gpus_dit
        dist.barrier() # wait all ranks to finish initialization


        #--------------------------------------Step 2: generate--------------------------------------
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
        ):
            out = []
            self.kv_cache1 = None
            active_nr = max_repeat
            for r in range(active_nr):
            #-------------------------------------------rollout loop------------------------------------------------------
                #----------------------------------------------Step 2.1: clip-level init------------------------------------------------------ 

                if r==0 or in_dit_device:
                    seed_g = torch.Generator(device=self.device)
                    seed_g.manual_seed(seed + r)

                    lat_target_frames = (infer_frames + 3 + self.motion_frames
                                        ) // 4 - lat_motion_frames
                    target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                    frame_seq_length = HEIGHT // 8 * WIDTH // 8 // 2 // 2
                    clip_noise = [
                        torch.randn(
                            16,
                            target_shape[0],
                            target_shape[1],
                            target_shape[2],
                            dtype=self.param_dtype,
                            device=self.device,
                            generator=seed_g)
                    ]
                    clip_output = torch.zeros_like(clip_noise[0]) #[16,f,h,w]
                    max_seq_len = np.prod(target_shape) // 4
                    if self.kv_cache1 is None:
                        local_rank = torch.distributed.get_rank()
                        if local_rank < num_gpus_dit:
                            self._initialize_kv_cache(
                                    batch_size=1,
                                    dtype=self.param_dtype,
                                    device=f"cuda:{local_rank}",
                                    kv_cache_size=max_seq_len
                                )

                        self._initialize_crossattn_cache(
                            batch_size=1,
                            dtype=self.param_dtype,
                            device=self.device
                        )

                    clip_latents = deepcopy(clip_noise)
                    with torch.no_grad():
                        left_idx = r * infer_frames
                        right_idx = r * infer_frames + infer_frames
                        cond_latents = COND[r] if pose_video else COND[0] * 0
                        cond_latents = cond_latents.to(
                            dtype=self.param_dtype, device=self.device)
                        audio_input = audio_emb[..., left_idx:right_idx]
                    input_motion_latents = motion_latents.clone()

                    # if offload_model or self.init_on_cpu:
                    #     self.noise_model.to(self.device)
                    #     torch.cuda.empty_cache()

                #-----------------------------------------------Temporal denoising loop in single clip---------------------------------
                # 2.2.0 prefill cond caching
                if (r==0 or r==1) and (dist.get_rank() != num_gpus_dit-1+int(enable_vae_parallel)): #考虑要不要r==1的时候替换一下ref cond，如果要的话clip r=0的时候还不能并行，要让每卡都有clean的latent0
                    if r==1:
                        ref_latents = torch.empty_like(ref_latents).type_as(clip_latents[0])
                        dist.broadcast(ref_latents, src=num_gpus_dit-1+int(enable_vae_parallel))
                    block_index = 0
                    block_latents = clip_latents[0][:, block_index *
                                    self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block] #[16,f,h,w]
                    left_idx = block_index * (self.num_frames_per_block * 4)
                    right_idx = (block_index+1) * (self.num_frames_per_block * 4)
                    block_arg_c = {
                        'context': context[0:1], #list(1) torch.Size([19, 4096])
                        'seq_len': None,
                        'cond_states': cond_latents[:,:,block_index * 
                                        self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block],
                        "motion_latents": input_motion_latents,
                        'ref_latents': ref_latents,
                        "audio_input": audio_input[..., left_idx:right_idx],
                        "motion_frames": [self.motion_frames, lat_motion_frames],
                        "drop_motion_frames": drop_first_motion and r == 0,
                        "sink_flag": True,
                    }
                    timestep = torch.ones(
                        [1, self.num_frames_per_block], device=self.device, dtype=self.param_dtype) * 0
                    self.noise_model( #update clean kv cache
                        [block_latents], t=timestep*0, **block_arg_c, 
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frames_per_block * frame_seq_length,
                        current_end=(block_index + 1) * self.num_frames_per_block * frame_seq_length)
                        
                num_blocks = target_shape[0] // self.num_frames_per_block
                for block_index in range(num_blocks):
                    if enable_vae_parallel and (not in_dit_device) and r >= 2:
                        if getattr(self, "audio_template", None) is None:
                            # Use file-based audio embedding to build a template shape once.
                            left0 = 0
                            right0 = self.num_frames_per_block * 4
                            self.audio_template = audio_emb[..., left0:right0].contiguous()
                        audio_block = self._streaming_encode_next_audio_block_or_random(
                            block_frames=self.num_frames_per_block * 4
                        )
                        if audio_block is not None and self.audio_tgt_gpu is not None:
                            dist.send(audio_block.contiguous(), self.audio_tgt_gpu)

                    # 2.2.1 prepare block-level cond
                    if getattr(self, '_sampler_timesteps', None) is None:
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=self.device)
                        self._sampler_timesteps = sample_scheduler.timesteps
                        self._sampler_sigmas = sample_scheduler.sigmas

                    timesteps = self._sampler_timesteps
                    sample_scheduler.timesteps = timesteps
                    sample_scheduler.sigmas = self._sampler_sigmas
                    sample_scheduler._step_index = dist.get_rank() 
                    sample_scheduler._begin_index = 0

                    block_latents = clip_latents[0][:, block_index *
                                self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block] #[16,f,h,w]
                    if r==0 or in_dit_device:
                        left_idx = block_index * (self.num_frames_per_block * 4)
                        right_idx = (block_index+1) * (self.num_frames_per_block * 4)
                        block_arg_c = {
                            'context': context[0:1], #list(1) torch.Size([19, 4096])
                            'seq_len': None,
                            'cond_states': cond_latents[:,:,0 * 
                                            self.num_frames_per_block:(0 + 1) * self.num_frames_per_block],
                            "motion_latents": input_motion_latents,
                            'ref_latents': ref_latents,
                            # "audio_input": audio_input[..., left_idx:right_idx],
                            "motion_frames": [self.motion_frames, lat_motion_frames],
                            "drop_motion_frames": drop_first_motion and r == 0,
                        }

                    for i, t in enumerate(tqdm(timesteps)):
                        if i != dist.get_rank():
                            continue
                        if self.src_gpu is None:
                            latent_model_input = block_latents #[16,num_frames_per_block,h,w]
                        else:  
                            latent_model_input = torch.empty_like(block_latents)  # 创建空tensor接收
                            dist.recv(latent_model_input, self.src_gpu)
                        if getattr(self, 'audio_template', None) is None:
                            self.audio_template = audio_input[..., 0: (self.num_frames_per_block * 4)].contiguous()
                        if dist.get_rank() == 0:
                            if r <=2:
                                audio_input_block = torch.randn_like(self.audio_template)
                            else:
                                audio_input_block = torch.empty_like(self.audio_template)
                                dist.recv(audio_input_block, self.audio_src_gpu)
                        else:
                            audio_input_block = torch.empty_like(self.audio_template) #torch.Size([1, 25, 1024, 12]
                            dist.recv(audio_input_block, self.audio_src_gpu)
                        block_arg_c["audio_input"] = audio_input_block

                        timestep = [t] * self.num_frames_per_block
                        timestep = torch.tensor(timestep).to(self.device).unsqueeze(0)
                        

                        noise_pred_cond = self.noise_model(
                            [latent_model_input], t=timestep, **block_arg_c, 
                            kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                            current_start=block_index * self.num_frames_per_block * frame_seq_length + r * num_blocks * self.num_frames_per_block * frame_seq_length,
                            current_end=(block_index + 1) * self.num_frames_per_block * frame_seq_length + r * num_blocks *self.num_frames_per_block * frame_seq_length,
                            mask=mask)

                        noise_pred = [torch.cat(noise_pred_cond, dim=0)]

                        temp_x0 = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0),# [16,f,h,w]
                            t,
                            latent_model_input.unsqueeze(0), #[1,16,f,h,w]
                            return_dict=False,
                            generator=seed_g)[0]
                        block_latents = temp_x0.squeeze(0) #[16,num_frames_per_block,h,w]
                        if self.tgt_gpu is None:
                            pass
                        else:
                            dist.send(block_latents.contiguous(), self.tgt_gpu)
                        if self.audio_tgt_gpu is None:
                            pass
                        else:
                            dist.send(audio_input_block.contiguous(), self.audio_tgt_gpu)
                        yield None
 
                    if enable_vae_parallel and (not in_dit_device):
                        block_latents = torch.empty_like(block_latents)

                        dist.recv(block_latents, self.src_gpu)

                        if r == 0 and active_nr != 1:
                            if block_index == 0: #cache new ref
                                ref_latents = block_latents.unsqueeze(0)[:,:,0:1] # 更新attention sink anchor到generated image，broadcast到所有rank
                            elif block_index == num_blocks-1: #broadcast ref to all ranks
                                dist.broadcast(ref_latents.contiguous(), src=num_gpus_dit-1+int(enable_vae_parallel))
                            else:
                                pass

                        # decode to rgb
                        if r == 0 and block_index == 0:
                            decode_latents = motion_latents[:,:,:7]
                            self.vae.stream_decode(decode_latents)
                        decode_latents = block_latents.unsqueeze(0)

                        torch.cuda.synchronize()
                        vae_wait_start = time.time()
                        image = torch.stack(self.vae.stream_decode(decode_latents))
                        torch.cuda.synchronize()
                        vae_wait_time = time.time() - vae_wait_start
                        print(f"[VAE] decoding for data from GPU {self.src_gpu}: {vae_wait_time:.4f}s")
                        
                        image = image[:, :, -(infer_frames)//num_blocks:] # 3
                        
                        if r == 0 and block_index == 0:
                            image = image[:, :, 3:]#第一个clip第一个block保留0帧，后面3
                 
                        yield image.cpu()

    

    def tts(self, tts_prompt_audio, tts_prompt_text, tts_text):
        if not hasattr(self, 'cosyvoice'):
            self.load_tts()
        speech_list = []
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        prompt_speech_16k = load_wav(tts_prompt_audio, 16000)
        if tts_prompt_text is not None:
            for i in self.cosyvoice.inference_zero_shot(tts_text, tts_prompt_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        else:
            for i in self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        torchaudio.save('tts.wav', torch.concat(speech_list, dim=1), self.cosyvoice.sample_rate)
        return 'tts.wav'

    def load_tts(self):
        if not os.path.exists('CosyVoice'):
            from wan.utils.utils import download_cosyvoice_repo
            download_cosyvoice_repo('CosyVoice')
        if not os.path.exists('CosyVoice2-0.5B'):
            from wan.utils.utils import download_cosyvoice_model
            download_cosyvoice_model('CosyVoice2-0.5B', 'CosyVoice2-0.5B')
        sys.path.append('CosyVoice')
        sys.path.append('CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.cosyvoice = CosyVoice2('CosyVoice2-0.5B')