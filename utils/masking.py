import torch 
import numpy as np
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
    

class RandomMaskingGenerator(MaskingGenerator):

    def __call__(self):
        """
        Generate the random temporal mask for the entire batch.
        Returns:
            mask: A binary mask tensor of shape (frames, num_patches_H, num_patches_W)
                  where each patch (from the spatial dimension) is masked across time (frames)
                  in proportion to the mask_ratio.
        """

        mask = np.zeros((self.frames, self.num_patches_per_frame), dtype=np.float32)
        num_masks_per_patch = int(self.frames * (self.num_masks_per_frame / self.num_patches_per_frame))

        for patch_idx in range(self.num_patches_per_frame):
            frame_mask = np.hstack([
                np.zeros(self.frames - num_masks_per_patch),
                np.ones(num_masks_per_patch),
            ])
            np.random.shuffle(frame_mask)

            mask[:, patch_idx] = frame_mask
        mask = mask.reshape(self.frames, self.height, self.width)

    return mask


def plot_mask(mask, title, save_name):
    """
    Plot and save the mask as an image in a grid layout. Each frame is visualized separately.
    
    mask: The mask numpy array (frames, H_patches, W_patches)
    title: The title of the plot
    save_name: File name for saving the plot
    """

    frames, H_patches, W_patches = mask.shape

    grid_size = int(np.ceil(np.sqrt(frames)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for frame in range(frames):
        row, col = divmod(frame, grid_size)
        mask_frame = mask[frame]

        axes[row, col].imshow(mask_frame, cmap='gray', interpolation='nearest')
        axes[row, col].set_title(f"Frame {frame+1}")
        axes[row, col].axis('off')

    for idx in range(frames, grid_size * grid_size):
        fig.delaxes(axes.flatten()[idx])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_name)
    plt.close()


if __name__ == "__main__":
    # Test
    rgb_frame_sequence = torch.randn(4, 3, 224, 224)
    mask_ratio = 0.95
    random_tube = tubemasking(mask_ratio, rgb_frame_sequence, patch_size=16, mask_type='random')
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
    plot_mask(random_tube, 'Random Tube Masking', 'random_tube_mask.png')
    plot_mask(random_temporal, 'Random Temporal Masking', 'random_temporal_mask.png')