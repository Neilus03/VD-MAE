def extract_patches(self, frames, patch_size):
    """
    Extract non-overlapping patches from frames.

    Parameters:
        - frames: Tensor of shape [B, C, T, H, W]
        - patch_size: Patch size (int)

    Returns:
        - patches: Tensor of shape [B, T, num_patches_per_frame, patch_size^2 * C]
    """
    B, C, T, H, W = frames.shape

    # Use unfold to extract patches
    patches = frames.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    # patches shape: [B, C, T, num_patches_H, num_patches_W, patch_size, patch_size]

    # Rearrange dimensions to [B, T, num_patches_H, num_patches_W, C, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6)  # [B, T, num_patches_H, num_patches_W, C, patch_size, patch_size]

    # Flatten the spatial dimensions of the patches (num_patches_H * num_patches_W) and the patch pixels (patch_size * patch_size)
    num_patches_per_frame = patches.shape[2] * patches.shape[3]  # num_patches_H * num_patches_W
    patches = patches.contiguous().view(B, T, num_patches_per_frame, patch_size * patch_size * C)  # Shape: [B, T, num_patches_per_frame, patch_size^2 * C]

    return patches