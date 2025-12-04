#!/bin/bash
# Single GPU Gradio Launch Script


CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /primus_xpfs_workspace_T04/huangshijie/miniconda3/envs/wan_s2v/bin/torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    minimal_inference/gradio_app.py \
    --ulysses_size 1 \
    --task s2v-14B \
    --size "704*384" \
    --base_seed 420 \
    --training_config liveavatar/configs/s2v_causal_sft.yaml \
    --offload_model True \
    --convert_model_dtype \
    --infer_frames 48 \
    --load_lora \
    --lora_path_dmd "/primus_xpfs_workspace_T04/huangyubo/LongLive/1027_s2v_selfforcing_1_13_2/ckptcopy/step_2500_generator_lora.pt" \
    --sample_steps 4 \
    --sample_guide_scale 0 \
    --num_clip 100 \
    --num_gpus_dit 1 \
    --sample_solver euler \
    --single_gpu \
    --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
    --server_port 7860 \
    --server_name "0.0.0.0"

