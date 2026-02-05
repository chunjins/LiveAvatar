## Installation

1. Install virtual environment

```
pip install uv
uv venv --python=3.10
source .venv/bin/activate
```

2. Install pytorch. Note the version depends on hardware and cuda version.

```
uv pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

3. Install flash_attn_3.

```
uv pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 --extra-index-url https://download.pytorch.org/whl/cu128
```

or

```
pip install flash-attn==2.8.3 --no-build-isolation

```
4. Install requirements.

```
uv pip install -r requirements.txt
```

5. Download models

```
uv pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./ckpt/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir ./ckpt/LiveAvatar
```

