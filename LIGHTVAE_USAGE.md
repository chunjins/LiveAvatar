# Using Lightweight VAE (LightVAE) in LiveAvatar

This guide explains how to use the lightweight VAE model (`lightvaew2_1.safetensors`) in the LiveAvatar project, based on the implementation from [LightX2V](https://github.com/ModelTC/LightX2V).

## Overview

The lightweight VAE offers:
- **Faster decoding**: Accelerated video generation
- **Lower memory usage**: Reduced VRAM requirements
- **75% model pruning**: Maintains quality with fewer parameters

## Download the Model

Download `lightvaew2_1.safetensors` from HuggingFace:
```bash
cd /gpfs/projects/ehpc552/chunjin/LiveAvatar/ckpt
huggingface-cli download lightx2v/Autoencoders lightvaew2_1.safetensors --local-dir ./vae/
```

Or place it in your checkpoint directory:
```
ckpt/Wan2.2-S2V-14B/lightvaew2_1.safetensors
```

## Configuration

Edit `liveavatar/models/wan/wan_2_2/configs/wan_s2v_14B_modified.py`:

### Option 1: Using Standard VAE (Default)
```python
# vae
s2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
s2v_14B.use_lightvae = False
```

### Option 2: Using Lightweight VAE
```python
# vae
s2v_14B.vae_checkpoint = 'lightvaew2_1.safetensors'
s2v_14B.use_lightvae = True  # Enable 75% pruning for lightweight VAE
```

## Implementation Details

### What Changed

1. **Added `use_lightvae` parameter** to VAE wrapper classes:
   - `Wan2_1_VAE` in `vae2_1.py`
   - `WanVAE` in `vae_streaming.py`

2. **Added `pruning_rate` parameter** to core VAE components:
   - `_video_vae()` function - passes pruning_rate to model construction
   - `WanVAE_` - stores pruning_rate and passes to Encoder3d/Decoder3d
   - `Encoder3d` - applies pruning to all dimension channels: `dims = [int(d * (1 - pruning_rate)) for d in dims]`
   - `Decoder3d` - applies pruning to all dimension channels: `dims = [int(d * (1 - pruning_rate)) for d in dims]`

3. **Pruning mechanism**:
   - `use_lightvae=True` → `pruning_rate=0.75` (lightweight)
   - `use_lightvae=False` → `pruning_rate=0.0` (standard)
   - With 75% pruning, dimensions are reduced: `96 * (1-0.75) = 24`, `192 * (1-0.75) = 48`, etc.
   - This matches the checkpoint dimensions in `lightvaew2_1.safetensors`

4. **Safetensors support**: Added `.safetensors` file loading in addition to `.pth`

5. **Pipeline integration**: Updated `causal_s2v_pipeline_tpp.py` to pass `use_lightvae` parameter

### Code Architecture

```python
# Configuration (wan_s2v_14B_modified.py)
s2v_14B.vae_checkpoint = 'lightvaew2_1.safetensors'
s2v_14B.use_lightvae = True

# Pipeline instantiation (causal_s2v_pipeline_tpp.py)
self.vae = Wan2_1_VAE(
    vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
    device=self.device,
    dtype=self.param_dtype,
    use_lightvae=getattr(config, 'use_lightvae', False)
)

# VAE initialization (vae2_1.py / vae_streaming.py)
pruning_rate = 0.75 if use_lightvae else 0.0
self.model = _video_vae(
    pretrained_path=vae_pth,
    z_dim=z_dim,
    pruning_rate=pruning_rate,
)
```

## Performance Benefits

Based on LightX2V benchmarks:
- **Decoding Speed**: ~1.5-2x faster than standard VAE
- **VRAM Usage**: ~25% reduction in VAE memory footprint
- **Quality**: Minimal quality degradation with proper pruning

## Compatibility

- ✅ Supports both `.pth` and `.safetensors` formats
- ✅ Works with streaming and non-streaming VAE
- ✅ Compatible with multi-GPU pipeline
- ✅ Same mean/std normalization as standard VAE

## Reference

Implementation based on:
- [LightX2V VAE implementation](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/video_encoders/hf/wan/vae.py)
- [LightVAE model weights](https://huggingface.co/lightx2v/Autoencoders)

## Troubleshooting

### Model not found
Ensure the checkpoint path is correct:
```python
# Check file exists
import os
checkpoint_path = os.path.join('ckpt/Wan2.2-S2V-14B', 'lightvaew2_1.safetensors')
assert os.path.exists(checkpoint_path), f"File not found: {checkpoint_path}"
```

### Safetensors import error
Install safetensors if not already installed:
```bash
pip install safetensors
```

### Performance not improved
Verify `use_lightvae=True` is set in config and the lightweight checkpoint is loaded:
```python
# In pipeline code, add logging
logging.info(f"Using LightVAE: {self.vae.use_lightvae}")
logging.info(f"VAE checkpoint: {config.vae_checkpoint}")
```
