import torch 
import torch.nn as nn
import numpy as np
import random

def tubemasking(mask_ratio, frames, patch_size, mask_type='random'):
    """
    Apply tube masking to the input patches for the whole batch (frame sequence)
    
    mask_ratio: float -> ~85%
    frames: (B, C, H, W)
    patch_size: int -> 16
    mask_type: str -> 'random' or 'block'
    """

    (B, C, H, W) = frames.size()

    # Calculate the number of patches along height and width
    H_patches = H // patch_size
    W_patches = W // patch_size
    
    # Create the mask shape (only for the spatial dimensions H_patches, W_patches)
    mask_shape = (H_patches, W_patches)  # mask will be the same for all B and C dimensions
    
    # Calculate the number of patches to be masked based on the masking ratio
    total_patches = H_patches * W_patches
    num_masked_patches = int(mask_ratio * total_patches)

    if mask_type == "random":
        # Create a random mask for spatial dimensions (H_patches, W_patches)
        mask = torch.ones(mask_shape, dtype=torch.float32)
        masked_indices = torch.randperm(total_patches)[:num_masked_patches]
        
        # Convert flat indices to 2D (H_patches, W_patches) indices
        mask_view = mask.view(-1)
        mask_view[masked_indices] = 0
        mask = mask.view(mask_shape)
    
    elif mask_type == "block":
        raise NotImplementedError("Block masking is not implemented yet")
    
    else:
        raise ValueError("mask_type must be either 'random' or 'block'")
    
    # Expand mask along the batch dimension (same mask for all B samples)
    mask = mask.expand(B, C, H_patches, W_patches)
    
    return mask
    

def random_temporal_masking(mask_ratio, frames, patch_size):
    """
    Apply random temporal masking to the input patches for the whole batch (frame sequence)
    
    mask_ratio: float -> ~85%
    frames: (B, C, H, W)
    patch_size: int -> 16
    """

    (B, C, H, W) = frames.size()

    # Calculate the number of patches along height and width
    H_patches = H // patch_size
    W_patches = W // patch_size
    
    # Create the mask shape (B, C, H_patches, W_patches)
    mask_shape = (B, C, H_patches, W_patches)
    
    # Calculate the number of patches to mask along the B dimension
    num_masked_patches = int(mask_ratio * B)

    # Initialize the mask with ones (all unmasked initially)
    mask = torch.ones(mask_shape, dtype=torch.float32)

    # For each patch position (h, w), mask random indices along the B dimension
    for h in range(H_patches):
        for w in range(W_patches):
            # For each (h, w) position, randomly mask `num_masked_patches` along B
            masked_indices = torch.randperm(B)[:num_masked_patches]
            
            # Set the selected B indices to 0 (masked) for ALL C channels at this (h, w) location
            mask[masked_indices, :, h, w] = 0

    return mask


if __name__ == "__main__":
    # Test
    rgb_frame_sequence = torch.randn(4, 3, 224, 224)
    mask_ratio = 0.95
    random_tube = tubemasking(mask_ratio, rgb_frame_sequence, patch_size=16, mask_type='random')
    random_block = tubemasking(mask_ratio, rgb_frame_sequence, patch_size=16, mask_type='random')
    random_temporal = random_temporal_masking(mask_ratio, rgb_frame_sequence, patch_size=16)
    
    #save random tube figure as image
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(random_tube[i, 0].detach().numpy(), cmap='gray')
    plt.savefig('random_tube.png')
    plt.close(fig)
    
    #save random temporal figure as image
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(random_temporal[i, 0].detach().numpy(), cmap='gray')
    plt.savefig('random_temporal.png')
    plt.close(fig)
    
    print('Random Tube Masking:', random_tube.shape)
    print(random_tube)
    print('Random Temporal Masking:', random_temporal.shape)
    print(random_temporal)