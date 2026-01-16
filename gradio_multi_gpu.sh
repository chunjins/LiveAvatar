#!/bin/bash
# Multi-GPU Gradio Launch Script (TPP mode)


export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400

CUDA_VISIBLE_DEVICES=0,1,2,3,4
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "Starting Gradio Web UI in Multi-GPU mode"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=5 \
    --master_port=29502 \
    minimal_inference/gradio_app.py \
    --ulysses_size 1 \
    --task s2v-14B \
    --size "720*400" \
    --base_seed 420 \
    --training_config liveavatar/configs/s2v_causal_sft.yaml \
    --offload_model False \
    --convert_model_dtype \
    --infer_frames 48 \
    --load_lora \
    --lora_path_dmd "Quark-Vision/Live-Avatar" \
    --sample_steps 4 \
    --sample_guide_scale 0 \
    --num_clip 100 \
    --num_gpus_dit 4 \
    --sample_solver euler \
    --enable_vae_parallel \
    --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
    --server_port 7860 \
    --server_name "0.0.0.0" \
    --fp8
