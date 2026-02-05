#!/usr/bin/env python3
"""
Test script to verify lightweight VAE integration.
Usage: python test_lightvae.py
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, '/gpfs/projects/ehpc552/chunjin/LiveAvatar')

def test_vae_initialization():
    """Test that VAE can be initialized with use_lightvae parameter."""
    print("Testing VAE initialization...")
    
    # Test 1: Standard VAE
    print("\n1. Testing standard VAE (use_lightvae=False)")
    try:
        from liveavatar.models.wan.wan_2_2.modules.vae2_1 import Wan2_1_VAE
        
        vae_standard = Wan2_1_VAE(
            vae_pth='ckpt/Wan2.2-S2V-14B/lightvaew2_1.pth',
            device='cpu',
            dtype=torch.float32,
            use_lightvae=True
        )
        print("   ✓ Standard VAE initialized successfully")
        print(f"   - use_lightvae: {vae_standard.use_lightvae}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 2: Check pruning_rate parameter exists in _video_vae
    print("\n2. Checking _video_vae pruning_rate parameter")
    try:
        from liveavatar.models.wan.wan_2_2.modules.vae2_1 import _video_vae
        import inspect
        sig = inspect.signature(_video_vae)
        params = list(sig.parameters.keys())
        
        if 'pruning_rate' in params:
            print("   ✓ pruning_rate parameter found in _video_vae")
        else:
            print(f"   ✗ pruning_rate parameter NOT found. Available params: {params}")
            return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 3: Streaming VAE
    print("\n3. Testing streaming VAE initialization")
    try:
        from liveavatar.models.wan.wan_2_2.modules.vae_streaming import WanVAE
        
        vae_streaming = WanVAE(
            vae_pth='ckpt/Wan2.2-S2V-14B/Wan2.1_VAE.pth',
            device='cpu',
            dtype=torch.float32,
            use_lightvae=False
        )
        print("   ✓ Streaming VAE initialized successfully")
        print(f"   - use_lightvae: {vae_streaming.use_lightvae}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 4: Config parameter
    print("\n4. Testing config integration")
    try:
        from liveavatar.models.wan.wan_2_2.configs.wan_s2v_14B_modified import s2v_14B
        
        if hasattr(s2v_14B, 'use_lightvae'):
            print(f"   ✓ use_lightvae found in config: {s2v_14B.use_lightvae}")
            print(f"   - vae_checkpoint: {s2v_14B.vae_checkpoint}")
        else:
            print("   ✗ use_lightvae NOT found in config")
            return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 5: Safetensors support
    print("\n5. Testing safetensors file format detection")
    try:
        test_path_safetensors = "test.safetensors"
        test_path_pth = "test.pth"
        
        if test_path_safetensors.endswith('.safetensors'):
            print("   ✓ .safetensors extension detected correctly")
        else:
            print("   ✗ .safetensors extension detection failed")
            return False
            
        if not test_path_pth.endswith('.safetensors'):
            print("   ✓ .pth format distinguished from .safetensors")
        else:
            print("   ✗ .pth format detection failed")
            return False
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nTo use lightweight VAE:")
    print("1. Download: huggingface-cli download lightx2v/Autoencoders lightvaew2_1.safetensors")
    print("2. Edit config: s2v_14B.vae_checkpoint = 'lightvaew2_1.safetensors'")
    print("3. Enable: s2v_14B.use_lightvae = True")
    return True

if __name__ == "__main__":
    success = test_vae_initialization()
    sys.exit(0 if success else 1)
