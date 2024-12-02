import torch
import numpy as np

def create_tubelet_mask(batch_size, num_channels, num_frames, height, width, patch_size, tubelet_size, mask_ratio):
    # Calculate the patch dimensions in pixels
    patch_dim = int(np.sqrt(patch_size))
    patches_per_row = height // patch_dim
    patches_per_col = width // patch_dim
    num_tubelets = num_frames // tubelet_size
    
    # Initialize the mask to ones (no masking initially) in the shape of the frames
    mask = np.ones((batch_size, num_channels, num_frames, height, width), dtype=np.int32)
    
    # Calculate the number of patches to mask in each tubelet based on the mask_ratio
    num_patches_to_mask = int(mask_ratio * patches_per_row * patches_per_col)

    for b in range(batch_size):
        for t in range(num_tubelets):
            # Select the start frame for the current tubelet
            start_frame = t * tubelet_size
            # Randomly select patches to mask within this tubelet
            masked_indices = np.random.choice(
                patches_per_row * patches_per_col, num_patches_to_mask, replace=False
            )
            for idx in masked_indices:
                row = idx // patches_per_col
                col = idx % patches_per_col
                
                # Set mask values to 0 for the entire patch across all frames in the tubelet
                mask[b, :, start_frame:start_frame + tubelet_size,
                     row * patch_dim:(row + 1) * patch_dim,
                     col * patch_dim:(col + 1) * patch_dim] = 0

    return mask


def flatten_mask(mask, patch_size):
    """
    Convert a tubelet-wise mask to match the shape of the flattened patch representation.

    Parameters:
        - mask: Binary mask with shape [B, C, T, H, W] as a numpy array or tensor
        - patch_size: Size of each patch (int)

    Returns:
        - flattened_mask: Mask reshaped to [B * N, patch_size^2 * C]
    """
    # If mask is a numpy array, convert it to a PyTorch tensor
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    B, C, T, H, W = mask.shape

    # Step 1: Use unfold to extract patches from the mask, as we did for the frames
    mask_patches = mask.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    # mask_patches shape: [B, C, T, num_patches_H, num_patches_W, patch_size, patch_size]

    # Step 2: Permute to move channel to the last position, matching `extract_patches`
    mask_patches = mask_patches.permute(0, 2, 3, 4, 1, 5, 6)  # Shape: [B, T, num_patches_H, num_patches_W, C, patch_size, patch_size]

    # Step 3: Flatten the patch and spatial dimensions
    num_patches_per_frame = mask_patches.shape[2] * mask_patches.shape[3]
    mask_patches_flat = mask_patches.contiguous().view(B, T, num_patches_per_frame, patch_size * patch_size * C)
    # Shape: [B, T, num_patches_per_frame, patch_size^2 * C]

    # Step 4: Reshape to match the flattened patch structure [B * num_patches, patch_size^2 * C]
    flattened_mask = mask_patches_flat.view(-1, patch_size * patch_size * C)  # Shape: [B * N, patch_size^2 * C]

    return flattened_mask
