# --- LOAD MODULES ---
module load bsc/1.0
module load miniforge/25.3.0-3
module load nvidia-hpc-sdk/23.9
module load gcc/13.2.0
module load cuda/12.8
module load nccl/2.24.3-1
module load ucx/1.19.0
module load git/2.43.0
module load ffmpeg

# --- ACTIVATE VIRTUAL ENVIRONMENT ---
# This ensures 'torchrun' uses your installed packages
source .venv/bin/activate

# --- ENVIRONMENT & CACHE FIXES ---
# Prevents the "df: No such file" error on the cluster
mkdir -p $HOME/.triton/autotune
export TRITON_CACHE_DIR="$HOME/.triton"
export HF_HUB_OFFLINE=1       # Prevent hanging on download checks
export PYTHONUNBUFFERED=1     # Ensure you see logs immediately


# --- MULTI-GPU CONFIG ---
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE=true

torchrun --nproc_per_node=1 --master_port=29101  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "704*384" \
     --base_seed 420 \
     --training_config liveavatar/configs/s2v_causal_sft.yaml \
     --offload_model True \
     --convert_model_dtype \
     --prompt "In a casino, a dealer is dealing with cards, sitting next to a table"  \
     --image "../DiffSynth/data/live_dealer/test/live_dealer_5_1_2026_1.jpg" \
     --audio "../DiffSynth/data/live_dealer/test/audio_2025-06-15_19-12-29_trimmed_0-10.mp3" \
     --pose_video "../DiffSynth/data/live_dealer/test/2025-06-15_19-12-29_trimmed_0-10_yolo_cards.mp4" \
     --infer_frames 48 \
     --load_lora \
     --lora_path_dmd "Quark-Vision/Live-Avatar" \
     --sample_steps 4 \
     --sample_guide_scale 0 \
     --num_clip 10000 \
     --num_gpus_dit 1 \
     --sample_solver euler \
     --single_gpu \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
     --fp8
     
