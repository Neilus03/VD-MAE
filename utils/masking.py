import torch 
import torch.nn as nn
import numpy as np

def tubemasking(mask_ratio, rgb_patches, depth_patches, mask_type='random'):
    """
    Apply tube masking to the input patches for the whole batch (frame sequence)
    
    mask_ratio: float -> ~85%
    rgb_patches: (B, C, H, W) -> (B, 3, H, W)
    depth_patches: (B, C, H, W) -> (B, 1, H, W)
    mask_type: str -> 'random' or 'block'
    """
    
def random_temporal_masking(mask_ratio, rgb_patches, depth_patches, mask_type='random'):
    """
    Apply random temporal masking to the input patches for the whole batch (frame sequence)
    
    mask_ratio: float -> ~85%
    rgb_patches: (B, C, H, W) -> (B, 3, H, W)
    depth_patches: (B, C, H, W) -> (B, 1, H, W)
    """

if __name__ == "__main__":
    # Test
    
    rgb_patches = torch.randn(4, 3, 224, 224) #mejor imagenes de vd
    depth_patches = torch.randn(4, 1, 224, 224) #mejor imagenes de vd
    mask_ratio = 0.85
    mask_type = 'random'
    tubemasking(mask_ratio, rgb_patches, depth_patches, mask_type)
    random_temporal_masking(mask_ratio, rgb_patches, depth_patches, mask_type)