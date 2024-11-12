
import torch
import torch.distributed as dist
from visualization_utils import log_visualizations
from torch import nn

def compute_loss(self, rgb_frames, depth_maps, rgb_recon, depth_recon, rgb_masks, depth_masks):
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
    B = 1
    T = 8  # TODO hardcoded for now

    # Change reconstruction shape: divide the last dimension into 3 for RGB
    rgb_recon_visual = rgb_recon.reshape(B, T, 196, 3, 256)  # TODO this 256 is hardcoded

    # Change reconstruction shape: divide the last dimension into 1 for Depth
    depth_recon_visual = depth_recon.reshape(B, T, 196, 1, 256)  # TODO this 256 is hardcoded
    # shape [1, 8, 196, 3, 256]

    # Permute channel indexation to idx 1:
    rgb_recon_visual = rgb_recon_visual.permute(0, 3, 1, 2, 4)
    depth_recon_visual = depth_recon_visual.permute(0, 3, 1, 2, 4)
    # shape [1, 3, 8, 196, 256]

    rgb_recon_visual = rgb_recon_visual.reshape(B, 3, T, 14, 14, 16, 16)
    depth_recon_visual = depth_recon_visual.reshape(B, 1, T, 14, 14, 16, 16)

    # Permute to get the desired shape [batch_size, num_channels, num_frames, frame_height, frame_width]
    rgb_recon_visual = rgb_recon_visual.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 3, T, 224, 224)
    depth_recon_visual = depth_recon_visual.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, T, 224, 224)

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        log_visualizations(rgb_frames, depth_maps,
                        rgb_recon_visual,
                        depth_recon_visual,
                        rgb_masks,
                        depth_masks,
                        0,
                        'Train')

    # Extract original patches
    rgb_patches = self.extract_patches(rgb_frames, self.rgb_tubelet_embed.patch_size[0])  # Shape: [B, N, patch_size^2 * 3]
    depth_patches = self.extract_patches(depth_maps, self.depth_tubelet_embed.patch_size[0])  # Shape: [B, N, patch_size^2 * 1]

    # Flatten the RGB and Depth masks to use as indices
    # TODO I think this is not the correct structure
    rgb_masks_flat = rgb_masks.view(-1)
    depth_masks_flat = depth_masks.view(-1)

    # Flatten the patches and reconstructions
    rgb_patches_flat = rgb_patches.view(-1, rgb_patches.size(-1))  # Shape [B * num_patches, patch_size^2 * 3]
    depth_patches_flat = depth_patches.view(-1, depth_patches.size(-1))
    rgb_recon_flat = rgb_recon.view(-1, rgb_recon.size(-1))  # Shape [B * N, patch_size^2 * 3]
    depth_recon_flat = depth_recon.view(-1, depth_recon.size(-1))

    # Select masked positions for RGB and Depth separately
    rgb_patches_masked = rgb_patches_flat[rgb_masks_flat]
    depth_patches_masked = depth_patches_flat[depth_masks_flat]
    rgb_recon_masked = rgb_recon_flat[rgb_masks_flat]
    depth_recon_masked = depth_recon_flat[depth_masks_flat]

    # Compute the loss
    loss_fn = nn.MSELoss()
    rgb_loss = loss_fn(rgb_recon_masked, rgb_patches_masked)
    depth_loss = loss_fn(depth_recon_masked, depth_patches_masked)

    # Total loss with weighting factors
    total_loss = self.alpha * rgb_loss + self.beta * depth_loss

    return rgb_loss, depth_loss, total_loss
