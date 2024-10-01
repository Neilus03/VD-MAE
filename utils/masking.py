import torch 
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

class MaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        """
        Initialize the Tube Masking Generator class
        Args:
            input_size: Tuple (num_frames, num_patches_H, num_patches_W)
            mask_ratio: float -> ~85%
        """

        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        """
        Return the string representation of the class
        """

        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str


class TubeMaskingGenerator(MaskingGenerator):
    
    def __call__(self):
        """
        Generate the tube mask for the entire batch
        Returns:
            mask: A binary mask tensor of shape (frames, num_patches_H, num_patches_W)
        """

        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask_per_frame = mask_per_frame.reshape(self.height, self.width)
        mask = np.tile(mask_per_frame, (self.frames, 1, 1))

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

    rgb_frame_sequence = torch.randn(16, 3, 224, 224)  # Simulating a 16-frame RGB sequence
    mask_ratio = 0.85
    patch_size = 16

    H_patches = 224 // patch_size
    W_patches = 224 // patch_size

    # Tube masking
    tube_mask_generator = TubeMaskingGenerator(input_size=(rgb_frame_sequence.shape[0], H_patches, W_patches), mask_ratio=mask_ratio)
    random_tube_mask = tube_mask_generator()

    total_masked_tube = int(np.sum(random_tube_mask))
    total_unmasked_tube = (rgb_frame_sequence.shape[0] * H_patches * W_patches) - total_masked_tube

    print(f'Tube Masking:')
    print(f'Total masked patches: {total_masked_tube}')
    print(f'Total unmasked patches: {total_unmasked_tube}')
    print('Random Tube Masking (3D):', random_tube_mask.shape)

    # Random masking
    random_mask_generator = RandomMaskingGenerator(input_size=(rgb_frame_sequence.shape[0], H_patches, W_patches), mask_ratio=mask_ratio)
    random_mask = random_mask_generator()

    total_masked_random = int(np.sum(random_mask))
    total_unmasked_random = (rgb_frame_sequence.shape[0] * H_patches * W_patches) - total_masked_random

    print(f'\nRandom Masking:')
    print(f'Total masked patches: {total_masked_random}')
    print(f'Total unmasked patches: {total_unmasked_random}')
    print('Random Masking (3D):', random_mask.shape)