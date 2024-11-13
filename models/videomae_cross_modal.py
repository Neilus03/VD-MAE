# models/videomae_cross_modal.py

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import PatchEmbed, Block
#from timm.vision_transformer import PatchEmbed, Block
from functools import partial
import os
import torch.distributed as dist
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from tubeletembed import TubeletEmbed
#from visualization_utils import log_visualizations

# Configure the logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level to INFO


class CrossModalVideoMAE(nn.Module):
    ''' Cross-modal VideoMAE model for Masked Video Reconstruction '''
    def __init__(self, config):
        super(CrossModalVideoMAE, self).__init__()
        '''
        Initialize the CrossModalVideoMAE model's components:
            - RGB and Depth patch embedding layers
            - Positional encoding
            - Transformer encoder layers
            - Transformer decoder layers for RGB and Depth
            - Mask token
        '''
        
        #Initialize RGB tubelet embedding layer
        self.rgb_tubelet_embed = TubeletEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            num_frames=config['num_frames'],
            tubelet_size=config['tubelet_size'],
            in_chans=3,
            embed_dim=config['embed_dim']
        )
        
        
        #Initialize the Depth tubelet embedding layer
        self.depth_tubelet_embed = TubeletEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            num_frames=config['num_frames'],
            tubelet_size=config['tubelet_size'],
            in_chans=1,
            embed_dim=config['embed_dim']
        )
        
        #Get the number of patches from the tubelet embedding layer
        num_tubelets = self.rgb_tubelet_embed.num_tubelets #The saem number of patches for RGB and Depth
        
        #Initialize positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tubelets, config['embed_dim']))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        
        #Initialize the transformer encoder layers which will process the RGB and Depth embeddings
        self.encoder = nn.ModuleList([
            Block(
                dim=config['embed_dim'],  # Corrected to use 'embed_dim'
                num_heads=config['encoder_num_heads'],
                mlp_ratio=config['encoder_mlp_ratio'],
                qkv_bias=True,

                attn_drop=0.,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            ) for _ in range(config['num_layers_encoder'])
        ])
 
        #Normalize the encoder output
        self.encoder_norm = nn.LayerNorm(config['embed_dim'], eps=1e-6)
        
        
        #Initialize two separate decoders for RGB and Depth (as in MultiMAE)
        
        #RGB decoder: a lightweight transformer decoder that takes the RGB encoder output and reconstructs the RGB frames
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
        #Normalize the RGB decoder output
        self.rgb_decoder_norm = nn.LayerNorm(config['embed_dim'], eps=1e-6)
        #Output layer for RGB frames to a shape of (patch_size, patch_size, 3)
        self.rgb_head = nn.Linear(config['embed_dim'], config['tubelet_size']*config['patch_size'] ** 2 * 3) #CHECK THIS LINE'S OUTPUT DIMENSION (SUSPECTED TO BE PATCH_SIZE ** 2 * NUM_PATCHES**2 * 3)

        
        #Depth decoder: a lightweight transformer decoder that takes the Depth encoder output and reconstructs the Depth frames
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
        #Normalize the Depth decoder output
        self.depth_decoder_norm = nn.LayerNorm(config['decoder_embed_dim'], eps=1e-6)
        #Output layer for Depth frames to a shape of (patch_size, patch_size, 1) 
        self.depth_head = nn.Linear(config['decoder_embed_dim'], config['tubelet_size']*config['patch_size']**2 * 1) #CHECK THIS LINE'S OUTPUT DIMENSION (SUSPECTED TO BE PATCH_SIZE ** 2 * NUM_PATCHES**2 * 1)


        #Initialize the mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim'])) 
        nn.init.trunc_normal_(self.mask_token, std=0.02)
      
        # Loss weighting factors
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 1.0)

        # Get embed dimensions
        self.embed_dim = config['embed_dim']
        self.decoder_embed_dim = config['decoder_embed_dim']
        
        # Function to project encoder output to decoder input over the last dimension
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.decoder_embed_dim) # [B, N, embed_dim] -> [B, N, decoder_embed_dim]
        
    def forward(self, rgb_frames, depth_maps, rgb_masks, depth_masks):
        '''
        Forward pass of the CrossModalVideoMAE model.

        Parameters:
            - rgb_frames: Tensor of shape [B, 3, T, H, W]
            - depth_maps: Tensor of shape [B, 1, T, H, W]
            - rgb_masks: Boolean Tensor of shape [B, N], where N is the number of TUBELETS for RGB frames.
            - depth_masks: Boolean Tensor of shape [B, N], where N is the number of TUBELETS for depth maps.

        Returns:
            - rgb_reconstruction: Reconstructed RGB patches.
            - depth_reconstruction: Reconstructed depth patches.
        '''
        B, C, T, H, W = rgb_frames.shape
        assert C == 3, "Input RGB tensor must have 3 channels"

        _, C_d, T_d, H_d, W_d = depth_maps.shape
        assert C_d == 1, "Input Depth tensor must have 1 channel"
        assert T == T_d and H == H_d and W == W_d, "RGB and Depth tensors must have the same dimensions"

        # TODO check fused embeddings; compare with previous version

        # Apply tubelet embedding to the RGB and Depth frames
        rgb_embed = self.rgb_tubelet_embed(rgb_frames)  # Shape: [B, N, embed_dim]
        depth_embed = self.depth_tubelet_embed(depth_maps)  # Shape: [B, N, embed_dim]

        # Ensure that rgb_masks and depth_masks have the correct shape
        N = rgb_embed.size(1)  # Total NUMBER OF TUBELETS across all frames (for both RGB and depth)
        assert rgb_masks.shape == (B, N), "RGB mask tensor must have shape [B, N]"
        assert depth_masks.shape == (B, N), "Depth mask tensor must have shape [B, N]"

        # Apply masking to RGB embeddings
        mask_tokens_rgb = self.mask_token.expand(B, N, -1)  # Shape: [B, N, embed_dim]
        rgb_embeddings = torch.where(rgb_masks.unsqueeze(-1), mask_tokens_rgb, rgb_embed)  # Shape: [B, N, embed_dim]

        # Apply masking to Depth embeddings
        mask_tokens_depth = self.mask_token.expand(B, N, -1)  # Shape: [B, N, embed_dim]
        depth_embeddings = torch.where(depth_masks.unsqueeze(-1), mask_tokens_depth, depth_embed)  # Shape: [B, N, embed_dim]

        # Add positional encoding
        rgb_embeddings += self.pos_embed
        depth_embeddings += self.pos_embed

        # Encode RGB and Depth separately
        for block in self.encoder:
            rgb_embeddings = block(rgb_embeddings)
            depth_embeddings = block(depth_embeddings)

        # Normalize encoder outputs
        rgb_embeddings = self.encoder_norm(rgb_embeddings)
        depth_embeddings = self.encoder_norm(depth_embeddings)

        # Prepare the sequence for decoders
        # RGB Decoder
        rgb_decoder_embeddings = self.encoder_to_decoder(self.embed_dim, self.decoder_embed_dim) if self.embed_dim != self.decoder_embed_dim else rgb_embeddings 
        for block in self.rgb_decoder:
            rgb_decoder_embeddings = block(rgb_decoder_embeddings)

        rgb_decoder_embeddings = self.rgb_decoder_norm(rgb_decoder_embeddings)
        rgb_reconstruction = self.rgb_head(rgb_decoder_embeddings)  # Shape: [B, N, 3 * num_patches]

        # Depth Decoder
        depth_decoder_embeddings = self.encoder_to_decoder(self.embed_dim, self.decoder_embed_dim) if self.embed_dim != self.decoder_embed_dim else depth_embeddings 
        for block in self.depth_decoder:
            depth_decoder_embeddings = block(depth_decoder_embeddings)

        depth_decoder_embeddings = self.depth_decoder_norm(depth_decoder_embeddings)
        depth_reconstruction = self.depth_head(depth_decoder_embeddings)  # Shape: [B, N, 1 * num_patches]

        # Get the number of patches per frame by dividing the image size by the patch size
        num_patches_per_frame = (H // config['patch_size']) * (W // config['patch_size']) # 14 * 14 = 196 for 224x224 images and 16x16 patches
    

        # Reshape RGB reconstruction to the desired shape
        rgb_reconstruction = rgb_reconstruction.view(B, T, num_patches_per_frame, -1)  # Last dim will be 768

        # Reshape Depth reconstruction to the desired shape
        depth_reconstruction = depth_reconstruction.view(B, T, num_patches_per_frame, -1)  # Last dim will be 256

        logger.info(f"RGB Reconstruction Shape: {rgb_reconstruction.shape}")
        logger.info(f"Depth Reconstruction Shape: {depth_reconstruction.shape}")

        return rgb_reconstruction, depth_reconstruction



if __name__ == "__main__":
    import argparse
    import yaml

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Sample configuration dictionary
    config = {
        'img_size': 224,           # Image size (height and width)
        'patch_size': 16,          # Patch size
        'num_frames': 32,           # Number of frames in the video
        'tubelet_size': 4,         # Number of frames per tubelet
        'embed_dim': 768,          # Embedding dimension for encoder
        'decoder_embed_dim': 768,  # Embedding dimension for decoder
        'encoder_num_heads': 12,   # Number of attention heads in encoder
        'decoder_num_heads': 12,   # Number of attention heads in decoder
        'encoder_mlp_ratio': 4.0,  # MLP ratio in encoder
        'decoder_mlp_ratio': 4.0,  # MLP ratio in decoder
        'num_layers_encoder': 12,  # Number of encoder layers
        'num_layers_decoder': 8,   # Number of decoder layers
        'alpha': 1.0,              # Loss weighting factor for RGB
        'beta': 1.0,               # Loss weighting factor for Depth
    }

    # Instantiate the model
    model = CrossModalVideoMAE(config)
    logger.info("Model instantiated successfully.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Create dummy input data
    B = 16  # Batch size
    C_rgb = 3
    C_depth = 1
    T = config['num_frames']
    H = config['img_size']
    W = config['img_size']

    rgb_frames = torch.randn(B, C_rgb, T, H, W).to(device)
    depth_maps = torch.randn(B, C_depth, T, H, W).to(device)
    logger.info(f"Dummy RGB frames shape: {rgb_frames.shape}")
    logger.info(f"Dummy Depth maps shape: {depth_maps.shape}")

    # Calculate number of tubelets
    num_patches = (H // config['patch_size']) * (W // config['patch_size'])
    num_tubelets = (T // config['tubelet_size']) * num_patches
    logger.info(f"Number of tubelets: {num_tubelets}")

    # Create dummy masks (randomly mask 25% of the tubelets)
    rgb_masks = torch.rand(B, num_tubelets) < 0.25
    depth_masks = torch.rand(B, num_tubelets) < 0.25
    rgb_masks = rgb_masks.to(device)
    depth_masks = depth_masks.to(device)
    logger.info(f"Dummy RGB masks shape: {rgb_masks.shape}")
    logger.info(f"Dummy Depth masks shape: {depth_masks.shape}")

    # Forward pass
    with torch.no_grad():
        rgb_reconstruction, depth_reconstruction = model(rgb_frames, depth_maps, rgb_masks, depth_masks)
        logger.info("Forward pass completed.")

    # Print output shapes
    logger.info(f"RGB Reconstruction Shape: {rgb_reconstruction.shape}")      # Expected: [B, T, num_patches_per_frame, embed_dim]
    logger.info(f"Depth Reconstruction Shape: {depth_reconstruction.shape}")  # Expected: [B, T, num_patches_per_frame, embed_dim]

 

    logger.info("All assessments passed successfully.")