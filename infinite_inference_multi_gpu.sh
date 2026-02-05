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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE=true

# Set timeout to 2 hours (in seconds)
export NCCL_TIMEOUT=7200
export NCCL_ASYNC_ERROR_HANDLING=1

# Fix NCCL network interface issues on HPC clusters
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# Use UCX for better InfiniBand support
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring


torchrun --nproc_per_node=4 --master_port=29102  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "720*400" \
     --base_seed 420 \
     --training_config liveavatar/configs/s2v_causal_sft.yaml \
     --offload_model False \
     --convert_model_dtype \
     --prompt "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge."  \
     --image "examples/dwarven_blacksmith.jpg" \
     --audio "examples/dwarven_blacksmith.wav" \
     --infer_frames 48 \
     --load_lora \
     --lora_path_dmd "Quark-Vision/Live-Avatar" \
     --sample_steps 4 \
     --sample_guide_scale 0 \
     --num_clip 10000 \
     --num_gpus_dit 4 \
     --sample_solver euler \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
     --fp8
