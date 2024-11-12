from timm.models.layers import to_2tuple
import torch.nn as nn
import torch

class TubeletEmbed(nn.Module):
    """Video to Tubelet Embedding"""
    def __init__(self, img_size=224, patch_size=16, num_frames=16, tubelet_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        # Ensure img_size and patch_size are tuples
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Calculate the number of patches
        self.num_tubelets = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (num_frames // tubelet_size)
        print(f"Number of tubelets: {self.num_tubelets}")
        # Define the 3D convolutional layer
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]) # Stride = kernel size, ensures that the convolutional kernels do not overlap and that the tubelets are extracted without redundancy.
        )

    def forward(self, x):
        """
        x: Input tensor of shape [B, C, T, H, W]
        """
        print(f"Input shape: {x.shape}")
        x = self.proj(x)  # Shape: [B, embed_dim, T', H', W']
        print(f"Shape after 3D Conv: {x.shape}")
        x = x.flatten(2)  # Flatten the spatial and temporal dimensions
        print(f"Shape after flatten(2): {x.shape}")
        x = x.transpose(1, 2)  # Shape: [B, N, embed_dim], where N = T' * H' * W'
        print(f"Shape after transpose(1, 2): {x.shape}")
        return x
    

# Example usage
if __name__ == "__main__":
    model = TubeletEmbed()
    x = torch.randn(1, 3, 16, 224, 224)
    out = model(x)
    print(out.shape)
# Output: torch.Size([1, 784, 768])
