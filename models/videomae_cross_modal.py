# models/videomae_cross_modal.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers import *

from functools import partial
import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from tubeletembed import TubeletEmbed

# Import visualization libraries and set backend for clusters without display
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure the logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level to INFO

class CrossModalVideoMAE(nn.Module):
    ''' Cross-modal VideoMAE model for Masked Video Reconstruction '''
    def __init__(self, config):
        super(CrossModalVideoMAE, self).__init__()
        
        # Initialize RGB tubelet embedding layer
        self.rgb_tubelet_embed = TubeletEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            num_frames=config['num_frames'],
            tubelet_size=config['tubelet_size'],
            in_chans=3,
            embed_dim=config['embed_dim']
        )
        
        # Initialize the Depth tubelet embedding layer
        self.depth_tubelet_embed = TubeletEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            num_frames=config['num_frames'],
            tubelet_size=config['tubelet_size'],
            in_chans=1,
            embed_dim=config['embed_dim']
        )
        
        # Get the number of tubelets from the tubelet embedding layer
        num_tubelets = self.rgb_tubelet_embed.num_tubelets  # Same number of patches for RGB and Depth
        
        # Initialize positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tubelets, config['embed_dim']))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize the transformer encoder layers
        self.encoder = nn.ModuleList([
            Block(
                dim=config['embed_dim'],
                num_heads=config['encoder_num_heads'],
                mlp_ratio=config['encoder_mlp_ratio'],
                qkv_bias=True,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            ) for _ in range(config['num_layers_encoder'])
        ])

        # Normalize the encoder output
        self.encoder_norm = nn.LayerNorm(config['embed_dim'], eps=1e-6)
        
        # Initialize two separate decoders for RGB and Depth
        
        # RGB decoder
        self.rgb_decoder = nn.ModuleList([
            Block(
                dim=config['decoder_embed_dim'],
                num_heads=config['decoder_num_heads'],
                mlp_ratio=config['decoder_mlp_ratio'],
                qkv_bias=True,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            ) for _ in range(config['num_layers_decoder'])
        ])
        # Normalize the RGB decoder output
        self.rgb_decoder_norm = nn.LayerNorm(config['decoder_embed_dim'], eps=1e-6)
        # Output layer for RGB frames
        self.rgb_head = nn.Linear(config['decoder_embed_dim'], config['tubelet_size'] * config['patch_size'] ** 2 * 3)
        
        # Depth decoder
        self.depth_decoder = nn.ModuleList([
            Block(
                dim=config['decoder_embed_dim'],
                num_heads=config['decoder_num_heads'],
                mlp_ratio=config['decoder_mlp_ratio'],
                qkv_bias=True,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            )
            for _ in range(config['num_layers_decoder'])
        ])
        # Normalize the Depth decoder output
        self.depth_decoder_norm = nn.LayerNorm(config['decoder_embed_dim'], eps=1e-6)
        # Output layer for Depth frames
        self.depth_head = nn.Linear(config['decoder_embed_dim'], config['tubelet_size'] * config['patch_size'] ** 2 * 1)
        
        # Initialize the mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim'])) 
        nn.init.trunc_normal_(self.mask_token, std=0.02)
      
        # Loss weighting factors
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 1.0)

        # Get embed dimensions
        self.embed_dim = config['embed_dim']
        self.decoder_embed_dim = config['decoder_embed_dim']
        
        # Function to project encoder output to decoder input over the last dimension
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.decoder_embed_dim)  # [B, N, embed_dim] -> [B, N, decoder_embed_dim]
        
        # Store necessary config parameters as class attributes
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.num_frames = config['num_frames']
        self.tubelet_size = config['tubelet_size']
        self.mask_ratio = config.get('mask_ratio', 0.25)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, rgb_frames, depth_maps):
        '''
        Forward pass of the CrossModalVideoMAE model.

        Parameters:
            - rgb_frames: Tensor of shape [B, 3, T, H, W]
            - depth_maps: Tensor of shape [B, 1, T, H, W]

        Returns:
            - rgb_reconstruction: Reconstructed RGB patches.
            - depth_reconstruction: Reconstructed depth patches.
        '''
        B, C, T, H, W = rgb_frames.shape
        assert C == 3, "Input RGB tensor must have 3 channels"

        _, C_d, T_d, H_d, W_d = depth_maps.shape
        assert C_d == 1, "Input Depth tensor must have 1 channel"
        assert T == T_d and H == H_d and W == W_d, "RGB and Depth tensors must have the same dimensions"

        # Apply tubelet embedding to the RGB and Depth frames
        rgb_embed = self.rgb_tubelet_embed(rgb_frames)  # Shape: [B, N, embed_dim]
        depth_embed = self.depth_tubelet_embed(depth_maps)  # Shape: [B, N, embed_dim]

        # Generate masks inside the forward method
        N = rgb_embed.size(1)  # Total number of tubelets across all frames
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        num_patches_per_frame = H_patches * W_patches
        num_temporal_positions = T // self.tubelet_size
        num_tubelets = num_temporal_positions * num_patches_per_frame
        num_masks_per_temporal_position = int(num_patches_per_frame * self.mask_ratio)

        # Generate masks for RGB
        rgb_masks = torch.zeros(B, num_tubelets, dtype=torch.bool, device=rgb_frames.device)
        for b in range(B):
            for t_pos in range(num_temporal_positions):
                start_idx = t_pos * num_patches_per_frame
                end_idx = (t_pos + 1) * num_patches_per_frame
                indices = torch.randperm(num_patches_per_frame, device=rgb_frames.device)[:num_masks_per_temporal_position]
                rgb_masks[b, start_idx + indices] = True  # Apply the masks

        # Generate masks for Depth
        depth_masks = torch.zeros(B, num_tubelets, dtype=torch.bool, device=depth_maps.device)
        for b in range(B):
            for t_pos in range(num_temporal_positions):
                start_idx = t_pos * num_patches_per_frame
                end_idx = (t_pos + 1) * num_patches_per_frame
                indices = torch.randperm(num_patches_per_frame, device=depth_maps.device)[:num_masks_per_temporal_position]
                depth_masks[b, start_idx + indices] = True  # Apply the masks

        # For visualization, store the masks separately
        self.rgb_masks = rgb_masks
        self.depth_masks = depth_masks

        # Apply masking to RGB embeddings
        mask_tokens_rgb = self.mask_token.expand(B, N, -1)  # Shape: [B, N, embed_dim]
        rgb_embeddings = torch.where(rgb_masks.unsqueeze(-1), mask_tokens_rgb, rgb_embed)  # Shape: [B, N, embed_dim]

        # Apply masking to Depth embeddings
        mask_tokens_depth = self.mask_token.expand(B, N, -1)  # Shape: [B, N, embed_dim]
        depth_embeddings = torch.where(depth_masks.unsqueeze(-1), mask_tokens_depth, depth_embed)  # Shape: [B, N, embed_dim]

        # Add positional encoding
        rgb_embeddings += self.pos_embed
        depth_embeddings += self.pos_embed

        ''''REVISAR'''
        #only_visible_rgb_embeddings = rgb_embed[~rgb_masks.unsqueeze(-1)].view(B, -1, self.embed_dim)
        #only_visible_depth_embeddings = depth_embed[~depth_masks.unsqueeze(-1)].view(B, -1, self.embed_dim)
        '''REVISAR'''
        
        # Encode RGB and Depth separately
        for block in self.encoder:
            rgb_embeddings = block(rgb_embeddings)
            depth_embeddings = block(depth_embeddings)

        # Normalize encoder outputs
        rgb_embeddings = self.encoder_norm(rgb_embeddings)
        depth_embeddings = self.encoder_norm(depth_embeddings)
        
        '''NOW ADD THE MASKED TOKENS TO THEIR RESPECTIVE POSITION'''
        '''REVISAR'''
        #rgb_decoder_input = torch.cat((mask_tokens_rgb, rgb_embeddings), dim=1)  # Shape: [B, N+1, embed_dim]
        #depth_decoder_input = torch.cat((mask_tokens_depth, depth_embeddings), dim=1)  # Shape: [B, N+1, embed_dim]
        '''REVISAR'''
        
        # Prepare the sequence for decoders
        # RGB Decoder
        rgb_decoder_embeddings = self.encoder_to_decoder(rgb_embeddings) if self.embed_dim != self.decoder_embed_dim else rgb_embeddings  #REVISAR
        for block in self.rgb_decoder:
            rgb_decoder_embeddings = block(rgb_decoder_embeddings)

        rgb_decoder_embeddings = self.rgb_decoder_norm(rgb_decoder_embeddings)
        rgb_reconstruction = self.rgb_head(rgb_decoder_embeddings)  # Shape: [B, N, output_dim]

        # Depth Decoder
        depth_decoder_embeddings = self.encoder_to_decoder(depth_embeddings) if self.embed_dim != self.decoder_embed_dim else depth_embeddings 
        for block in self.depth_decoder:
            depth_decoder_embeddings = block(depth_decoder_embeddings)

        depth_decoder_embeddings = self.depth_decoder_norm(depth_decoder_embeddings)
        depth_reconstruction = self.depth_head(depth_decoder_embeddings)  # Shape: [B, N, output_dim]

        # Reshape RGB reconstruction to the desired shape
        rgb_reconstruction = rgb_reconstruction.view(B, T, num_patches_per_frame, -1)  # Adjust the last dimension accordingly

        # Reshape Depth reconstruction to the desired shape
        depth_reconstruction = depth_reconstruction.view(B, T, num_patches_per_frame, -1)  # Adjust the last dimension accordingly

        logger.info(f"RGB Reconstruction Shape: {rgb_reconstruction.shape}")
        logger.info(f"Depth Reconstruction Shape: {depth_reconstruction.shape}")

        return rgb_reconstruction, depth_reconstruction

# Visualization functions

def visualize_masks(mask_tensor, title, filename_prefix):
    """
    Visualize the mask tensor over time and save as PNG images.

    Args:
        mask_tensor (torch.Tensor): Tensor of shape [T, H_patches, W_patches]
        title (str): Title for the plots
        filename_prefix (str): Prefix for the saved image files
    """
    T, H_patches, W_patches = mask_tensor.shape
    for t in range(T):
        fig, ax = plt.subplots(figsize=(4, 4))
        mask = mask_tensor[t].cpu().numpy()
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        ax.set_title(f'{title} - Frame {t}')
        ax.axis('off')
        filename = f"{filename_prefix}_frame_{t}.png"
        plt.savefig(filename)
        plt.close(fig)
        # Log the number of masked patches in the frame
        num_masked_patches = mask.sum()
        logger.info(f"Frame {t}: Number of masked patches: {int(num_masked_patches)}")
        logger.info(f"Saved mask visualization to {filename}")

def animate_masks(mask_tensor, title, filename):
    """
    Create an animation of the mask tensor over time and save as GIF.

    Args:
        mask_tensor (torch.Tensor): Tensor of shape [T, H_patches, W_patches]
        title (str): Title for the animation
        filename (str): Filename for the saved animation (should end with .gif)
    """
    T, H_patches, W_patches = mask_tensor.shape
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'{title} - Frame 0')
    im = ax.imshow(mask_tensor[0].cpu().numpy(), cmap='gray', interpolation='nearest')
    ax.axis('off')

    def update(t):
        im.set_array(mask_tensor[t].cpu().numpy())
        ax.set_title(f'{title} - Frame {t}')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, blit=True)
    ani.save(filename, writer='pillow', fps=2)
    plt.close(fig)
    logger.info(f"Saved animation to {filename}")




#################################################
#     Test the model with dummy input data      #       
#################################################


if __name__ == "__main__":
    import argparse
    import yaml

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration settings from config.yaml
    with open('../config/config.yaml', 'r') as f:
        full_config = yaml.safe_load(f)

    # Combine model and training configs
    config = full_config['model']
    training_config = full_config['training']
    data_config = full_config['data']

    # Add training parameters to config
    config['mask_ratio'] = training_config.get('mask_ratio', 0.25)
    config['alpha'] = training_config.get('alpha', 1.0) 
    config['beta'] = training_config.get('beta', 1.0)

    # Instantiate the model
    model = CrossModalVideoMAE(config)
    logger.info("Model instantiated successfully.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Create dummy input data
    B = 2  # Batch size (set to 1 for visualization purposes)
    C_rgb = 3
    C_depth = 1
    T = config['num_frames']
    H = config['img_size']
    W = config['img_size']

    rgb_frames = torch.randn(B, C_rgb, T, H, W).to(device)
    depth_maps = torch.randn(B, C_depth, T, H, W).to(device)
    logger.info(f"Dummy RGB frames shape: {rgb_frames.shape}")
    logger.info(f"Dummy Depth maps shape: {depth_maps.shape}")

    # Forward pass
    with torch.no_grad():
        rgb_reconstruction, depth_reconstruction = model(rgb_frames, depth_maps)
        logger.info("Forward pass completed.")

    # Access the masks from the model
    rgb_masks = model.rgb_masks  # Shape: [B, num_tubelets]
    depth_masks = model.depth_masks  # Shape: [B, num_tubelets]

    # Reshape masks for visualization
    B, num_tubelets = rgb_masks.shape
    H_patches = H // config['patch_size']
    W_patches = W // config['patch_size']
    num_patches_per_frame = H_patches * W_patches
    num_temporal_positions = T // config['tubelet_size']

    # Reshape masks to [B, num_temporal_positions, H_patches, W_patches]
    rgb_masks_reshaped = rgb_masks.view(B, num_temporal_positions, H_patches, W_patches)

    # Expand masks along temporal dimension (tubelet_size)
    rgb_masks_expanded = rgb_masks_reshaped.unsqueeze(2).expand(-1, -1, config['tubelet_size'], -1, -1)

    # Merge temporal positions and tubelet_size to get masks per frame
    rgb_masks_expanded = rgb_masks_expanded.contiguous().view(B, T, H_patches, W_patches)

    # Reshape masks for visualization
    B, num_tubelets = depth_masks.shape
    H_patches = H // config['patch_size']
    W_patches = W // config['patch_size']
    num_patches_per_frame = H_patches * W_patches
    num_temporal_positions = T // config['tubelet_size']

    # Reshape masks to [B, num_temporal_positions, H_patches, W_patches]
    depth_masks_reshaped = depth_masks.view(B, num_temporal_positions, H_patches, W_patches)

    # Expand masks along temporal dimension (tubelet_size)
    depth_masks_expanded = depth_masks_reshaped.unsqueeze(2).expand(-1, -1, config['tubelet_size'], -1, -1)

    # Merge temporal positions and tubelet_size to get masks per frame
    depth_masks_expanded = depth_masks_expanded.contiguous().view(B, T, H_patches, W_patches)

    # Visualize masks for a sample
    batch_index = 0  # Since B=1, index is 0

    # Visualize RGB Masks
    #visualize_masks(rgb_masks_expanded[batch_index], title='RGB Masks', filename_prefix='rgb_mask')

    # Visualize Depth Masks
    #visualize_masks(depth_masks_expanded[batch_index], title='Depth Masks', filename_prefix='depth_mask')

    # Animate Masks
    animate_masks(rgb_masks_expanded[batch_index], title='Masks Animation', filename='masks_animation_rgb.gif')
    animate_masks(depth_masks_expanded[batch_index], title='Masks Animation', filename='masks_animation_depth.gif')
    # Print output shapes
    logger.info(f"RGB Reconstruction Shape: {rgb_reconstruction.shape}")      # Expected: [B, T, num_patches_per_frame, -1]
    logger.info(f"Depth Reconstruction Shape: {depth_reconstruction.shape}")  # Expected: [B, T, num_patches_per_frame, -1] 

    logger.info("All assessments passed successfully.")
