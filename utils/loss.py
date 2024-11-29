
import torch
import torch.distributed as dist
#from utils.visualization_utils_old_utils_mireia import new_log_visualizations
from visualization_utils import new_log_visualizations
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# def reshape_reconstructions(recon, H, W, patch_size, num_channels):
#     B, T, num_patches, _ = recon.shape
#     H_patches, W_patches = H // patch_size, W // patch_size

#     # Reshape to (B, T, H_patches, W_patches, patch_size, patch_size, num_channels)
#     recon = recon.view(B, T, H_patches, W_patches, patch_size, patch_size, num_channels)

#     # Permute to bring spatial dimensions together: (B, T, num_channels, H_patches, patch_size, W_patches, patch_size)
#     recon = recon.permute(0, 1, 6, 2, 4, 3, 5)

#     # Reshape to merge patches into full resolution: (B, num_channels, T, H, W)
#     frames = recon.reshape(B, num_channels, T, H, W)

#     return frames

def reshape_reconstructions(recon, H, W, patch_size, num_channels):
    B, T, num_patches, _ = recon.shape
    H_patches, W_patches = H // patch_size, W // patch_size

    # Reshape to (B, T, H_patches, W_patches, patch_size, patch_size, num_channels)
    recon = recon.view(B, T, H_patches, W_patches, patch_size, patch_size, num_channels)
    print("After reshaping to patches:", recon.shape)

    # Visualize a slice before permute
    slice_idx = 0  # Example slice for visualization
    plt.figure(figsize=(5, 5))
    plt.title("Before Permute - Recon Slice")
    plt.imshow(recon[slice_idx, 0, :, :, :, :, 0].detach().cpu().numpy().reshape(H, W), cmap='viridis')
    plt.show()

    # Permute to bring spatial dimensions together: (B, T, num_channels, H_patches, patch_size, W_patches, patch_size)
    recon = recon.permute(0, 1, 6, 2, 4, 3, 5)
    print("After permute:", recon.shape)

    # Reshape to merge patches into full resolution: (B, num_channels, T, H, W)
    frames = recon.reshape(B, num_channels, T, H, W)
    print("Final reshaped frames:", frames.shape)

    # Visualize a single frame after reconstruction
    plt.figure(figsize=(5, 5))
    plt.title("Reconstructed Frame - Recon Slice")
    plt.imshow(frames[slice_idx, 0, 0, :, :].detach().cpu().numpy(), cmap='viridis')
    plt.show()

    return frames


# TODO new version of masks reshape, check this and run the code

# def reshape_masks_to_frames(
#     masks, batch_size, channels, time, frame_height, frame_width,
#     num_patches_in_H, num_patches_in_W,
#     num_pixels_in_patch_H, num_pixels_in_patch_W
# ):
#     """
#     Reshapes the mask tensor from [B, N] to [B, C, T, H, W].
    
#     Args:
#         masks (torch.Tensor): The input mask tensor of shape [B, N].
#         batch_size (int): Batch size (B).
#         channels (int): Number of channels (C).
#         time (int): Number of frames per batch (T).
#         frame_height (int): Height of the frame in pixels (H).
#         frame_width (int): Width of the frame in pixels (W).
#         num_patches_in_H (int): Number of patches along height.
#         num_patches_in_W (int): Number of patches along width.
#         num_pixels_in_patch_H (int): Number of pixels per patch along height.
#         num_pixels_in_patch_W (int): Number of pixels per patch along width.
    
#     Returns:
#         torch.Tensor: The reshaped mask tensor of shape [B, C, T, H, W].
#     """
#     # Step 1: Divide N into subsets for each frame's patches
#     num_patches_per_frame = num_patches_in_H * num_patches_in_W
#     assert masks.shape[1] % num_patches_per_frame == 0, "N must be divisible by num_patches_per_frame."
    
#     # Calculate tubelet size (number of frames per tubelet)
#     tubelet_size = masks.shape[1] // (num_patches_per_frame * time)
#     assert frame_height == num_patches_in_H * num_pixels_in_patch_H, "Frame height mismatch."
#     assert frame_width == num_patches_in_W * num_pixels_in_patch_W, "Frame width mismatch."
    
#     # Reshape N into [B, T, num_patches_per_frame]
#     masks = masks.view(batch_size, time, num_patches_per_frame)
    
#     # Step 2: Repeat along the tubelet size dimension
#     masks = masks.unsqueeze(2).repeat(1, 1, tubelet_size, 1)  # [B, T, tubelet_size, num_patches_per_frame]
#     masks = masks.view(batch_size, time * tubelet_size, num_patches_per_frame)
    
#     # Step 3: Repeat along the channel dimension
#     masks = masks.unsqueeze(1).repeat(1, channels, 1, 1)  # [B, C, T * tubelet_size, num_patches_per_frame]
    
#     # Step 4: Reshape the patches into 2D grids
#     masks = masks.view(batch_size, channels, time * tubelet_size, num_patches_in_H, num_patches_in_W)
    
#     # Step 5: Restructure patch-wise masks into pixel-wise masks
#     masks = masks.repeat_interleave(num_pixels_in_patch_H, dim=3)  # Expand patches along height
#     masks = masks.repeat_interleave(num_pixels_in_patch_W, dim=4)  # Expand patches along width
    
#     return masks

def reshape_masks_to_frames(
    masks, batch_size, channels, time, frame_height, frame_width,
    num_patches_in_H, num_patches_in_W,
    num_pixels_in_patch_H, num_pixels_in_patch_W
):
    num_patches_per_frame = num_patches_in_H * num_patches_in_W
    tubelet_size = masks.shape[1] // (num_patches_per_frame * time)

    print("Initial masks shape:", masks.shape)

    # Reshape N into [B, T, num_patches_per_frame]
    masks = masks.view(batch_size, time, num_patches_per_frame)
    print("After reshaping to frame patches:", masks.shape)

    # Step 2: Repeat along the tubelet size dimension
    masks = masks.unsqueeze(2).repeat(1, 1, tubelet_size, 1)
    masks = masks.view(batch_size, time * tubelet_size, num_patches_per_frame)
    print("After repeating along tubelet size:", masks.shape)

    # Step 3: Repeat along the channel dimension
    masks = masks.unsqueeze(1).repeat(1, channels, 1, 1)
    print("After repeating along channels:", masks.shape)

    # Step 4: Reshape the patches into 2D grids
    masks = masks.view(batch_size, channels, time * tubelet_size, num_patches_in_H, num_patches_in_W)
    print("After reshaping to patch grids:", masks.shape)

    # Step 5: Restructure patch-wise masks into pixel-wise masks
    masks = masks.repeat_interleave(num_pixels_in_patch_H, dim=3)
    masks = masks.repeat_interleave(num_pixels_in_patch_W, dim=4)
    print("After converting to pixel-wise masks:", masks.shape)

    # Visualize one frame of the mask
    slice_idx = 0  # Example slice
    plt.figure(figsize=(5, 5))
    plt.title("Reshaped Mask Frame")
    plt.imshow(masks[slice_idx, 0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    plt.show()

    return masks

def reshape_masks(rgb_masks, depth_masks, H, W, T, patch_size, tubelet_size):
    # Reshape masks for visualization

    print('rgb_masks shape:', rgb_masks.shape)
    B, num_tubelets = rgb_masks.shape
    H_patches = H // patch_size
    W_patches = W // patch_size
    num_patches_per_frame = H_patches * W_patches
    num_temporal_positions = T // tubelet_size

    # Reshape masks to [B, num_temporal_positions, H_patches, W_patches]
    rgb_masks_reshaped = rgb_masks.view(B, num_temporal_positions, H_patches, W_patches)

    # Expand masks along temporal dimension (tubelet_size)
    rgb_masks_expanded = rgb_masks_reshaped.unsqueeze(2).expand(-1, -1, tubelet_size, -1, -1)

    # Merge temporal positions and tubelet_size to get masks per frame
    rgb_masks_expanded = rgb_masks_expanded.contiguous().view(B, T, H_patches, W_patches)

    # Reshape masks for visualization
    B, num_tubelets = depth_masks.shape
    H_patches = H // patch_size
    W_patches = W // patch_size
    num_patches_per_frame = H_patches * W_patches
    num_temporal_positions = T // tubelet_size

    # Reshape masks to [B, num_temporal_positions, H_patches, W_patches]
    depth_masks_reshaped = depth_masks.view(B, num_temporal_positions, H_patches, W_patches)

    # Expand masks along temporal dimension (tubelet_size)
    depth_masks_expanded = depth_masks_reshaped.unsqueeze(2).expand(-1, -1, tubelet_size, -1, -1)

    # Merge temporal positions and tubelet_size to get masks per frame
    depth_masks_expanded = depth_masks_expanded.contiguous().view(B, T, H_patches, W_patches)

    # Determine upscaling factor
    upscale_factor = patch_size

    # TODO shape mismatch in interpolate

    # Adjust Depth Mask
    depth_masks = depth_masks_expanded.unsqueeze(1)  # Add channel dimension (B, 1, T, H_patches, W_patches)
    depth_masks = F.interpolate(
        depth_masks.float(),  # Ensure input is float for interpolation
        size=(H, W),  # Resize to match pixel dimensions
        mode='nearest'  # Nearest neighbor to maintain binary values
    ).to(depth_masks_expanded.dtype)  # Cast back to original dtype if needed

    # Adjust RGB Mask
    rgb_masks = rgb_masks_expanded.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # Triplicate along the channel dimension (B, 3, T, H_patches, W_patches)
    rgb_masks = F.interpolate(
        rgb_masks.float(),  # Ensure input is float for interpolation
        size=(H, W),  # Resize to match pixel dimensions
        mode='nearest'  # Nearest neighbor to maintain binary values
    ).to(rgb_masks_expanded.dtype)  # Cast back to original dtype if needed

    return rgb_masks, depth_masks

def compute_loss(rgb_frames, depth_maps, rgb_recon, depth_recon, rgb_masks, depth_masks, epoch):
    '''
    Compute the reconstruction loss with separate masks for RGB and Depth.

    Parameters:
        - rgb_frames: Original RGB frames, shape [B, 3, T, H, W]
        - depth_maps: Original Depth maps, shape [B, 1, T, H, W]
        - rgb_recon: Reconstructed RGB patches, shape [B, T, num_patches_per_frame, patch_size^2 * 3]
        - depth_recon: Reconstructed Depth patches, shape [B, T, num_patches_per_frame, patch_size^2 * 1]
        - rgb_masks: Boolean Tensor indicating masked positions for RGB, shape [B, N]
        - depth_masks: Boolean Tensor indicating masked positions for Depth, shape [B, N]

    Returns:
        - rgb_loss: RGB reconstruction loss
        - depth_loss: Depth reconstruction loss
        - total_loss: The total loss computed as a weighted sum of RGB and Depth losses
    '''

    B, _, T, H, W = rgb_frames.shape
    patch_size = 16  # TODO hardcoded for now

    # Reshape reconstructions to frame dimensions
    rgb_recon = reshape_reconstructions(rgb_recon, H, W, patch_size, 3)  # Shape: [B, 3, T, H, W]
    depth_recon = reshape_reconstructions(depth_recon, H, W, patch_size, 1)  # Shape: [B, 1, T, H, W]

    # Reshape masks
    rgb_masks = reshape_masks_to_frames(rgb_masks, B, 3, T, H, W, H // patch_size, W // patch_size, patch_size, patch_size)
    depth_masks = reshape_masks_to_frames(depth_masks, B, 1, T, H, W, H // patch_size, W // patch_size, patch_size, patch_size)

    # Flatten tensors for loss computation
    rgb_frames_flat = rgb_frames.view(B, 3, -1)  # Shape: [B, 3, T * H * W]
    depth_maps_flat = depth_maps.view(B, 1, -1)  # Shape: [B, 1, T * H * W]
    rgb_recon_flat = rgb_recon.view(B, 3, -1)  # Shape: [B, 3, T * H * W]
    depth_recon_flat = depth_recon.view(B, 1, -1)  # Shape: [B, 1, T * H * W]

    rgb_masks_flat = rgb_masks.view(B, 3, -1)  # Shape: [B, 3, T * H * W]
    depth_masks_flat = depth_masks.view(B, 1, -1)  # Shape: [B, 1, T * H * W]

    # Select masked positions for RGB and Depth separately
    rgb_patches_masked = rgb_frames_flat[rgb_masks_flat]
    depth_patches_masked = depth_maps_flat[depth_masks_flat]
    rgb_recon_masked = rgb_recon_flat[rgb_masks_flat]
    depth_recon_masked = depth_recon_flat[depth_masks_flat]

    # Compute the loss
    loss_fn = nn.MSELoss()
    rgb_loss = loss_fn(rgb_recon_masked, rgb_patches_masked)
    depth_loss = loss_fn(depth_recon_masked, depth_patches_masked)

    alpha = 0.5
    beta = 0.5
    total_loss = alpha * rgb_loss + beta * depth_loss

    # Visualization: Heatmaps for Flattened RGB Original vs Reconstructed
    save_visual_heatmap(rgb_frames_flat[0], rgb_recon_flat[0], epoch, prefix='RGB')
    save_visual_heatmap(depth_maps_flat[0], depth_recon_flat[0], epoch, prefix='Depth')

    return rgb_loss, depth_loss, total_loss

def save_visual_heatmap(original_flat, reconstructed_flat, epoch, prefix='RGB'):
    '''
    Saves heatmap comparison of flattened original and reconstructed images.

    Parameters:
        - original_flat: Flattened original tensor, shape [C, H*W]
        - reconstructed_flat: Flattened reconstructed tensor, shape [C, H*W]
        - epoch: Current training epoch
        - prefix: Prefix for file naming ('RGB' or 'Depth')
    '''
    original_flat_np = original_flat.detach().cpu().numpy()
    reconstructed_flat_np = reconstructed_flat.detach().cpu().numpy()

    # Create figure with side-by-side heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_flat_np, aspect='auto', cmap='viridis')
    axs[0].set_title(f"{prefix} Original Flattened")
    axs[0].axis('off')

    axs[1].imshow(reconstructed_flat_np, aspect='auto', cmap='viridis')
    axs[1].set_title(f"{prefix} Reconstructed Flattened")
    axs[1].axis('off')

    # Save figure
    output_dir = './visualizations'
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'{prefix}_flattened_epoch_{epoch}.png')
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Saved heatmap comparison to {fig_path}")
