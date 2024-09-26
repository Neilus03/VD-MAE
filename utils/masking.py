import torch 
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

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


def plot_mask(mask, title, save_name):
    """
    Plot and save the mask as an image. Each batch (B) is visualized separately.
    
    mask: The mask tensor (B, C, H_patches, W_patches)
    title: The title of the plot
    save_name: File name for saving the plot
    """
    
    # Extract batch size (B) and height, width patches
    B, C, H_patches, W_patches = mask.size()

    plt.figure(figsize=(10, 10))

    # Plot each batch (B) feature map, taking the first channel (C=0)
    for b in range(B):
        mask_np = mask[b, 0].cpu().numpy()  # Extract the first channel for each batch sample
        
        plt.subplot(1, B, b + 1)
        plt.imshow(mask_np, cmap='gray', interpolation='nearest')
        plt.title(f"{title} - Batch {b+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


if __name__ == "__main__":
    # Test
    rgb_frame_sequence = torch.randn(4, 3, 64, 64)
    mask_ratio = 0.75
    random_tube = tubemasking(mask_ratio, rgb_frame_sequence, patch_size=16, mask_type='random')
    random_temporal = random_temporal_masking(mask_ratio, rgb_frame_sequence, patch_size=16)
    print('Random Tube Masking:', random_tube.shape)
    print(random_tube)
    print('Random Temporal Masking:', random_temporal.shape)
    print(random_temporal)
    plot_mask(random_tube, 'Random Tube Masking', 'random_tube_mask.png')
    plot_mask(random_temporal, 'Random Temporal Masking', 'random_temporal_mask.png')