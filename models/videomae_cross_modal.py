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

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from tubeletembed import TubeletEmbed
from visualization_utils import log_visualizations


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
        # Count number of patches
        #num_patches = (config['img_size'] // config['patch_size']) ** 2
        #self.rgb_head = nn.Linear(config['embed_dim'], config['patch_size'] ** 2 * num_patches * 3)
        
        
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
        #self.depth_head = nn.Linear(config['decoder_embed_dim'], config['patch_size'] ** 2 * num_patches * 1)

        #Initialize the mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim']))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
      
        # Loss weighting factors
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 1.0)

        # Get embed dimensions
        self.embed_dim = config['embed_dim']
        self.decoder_embed_dim = config['decoder_embed_dim']
        
    def forward(self, rgb_frames, depth_maps, rgb_masks, depth_masks):
        '''
        Forward pass of the CrossModalVideoMAE model.

        Parameters:
            - rgb_frames: Tensor of shape [B, 3, T, H, W]
            - depth_maps: Tensor of shape [B, 1, T, H, W]
            - rgb_masks: Boolean Tensor of shape [B, N], where N is the number of patches for RGB frames.
            - depth_masks: Boolean Tensor of shape [B, N], where N is the number of patches for depth maps.

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
        N = rgb_embed.size(1)  # Total patches across all frames (for both RGB and depth)
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
        rgb_decoder_embeddings = self.encoder_to_decoder(rgb_embeddings) if self.embed_dim != self.decoder_embed_dim else rgb_embeddings
        for block in self.rgb_decoder:
            rgb_decoder_embeddings = block(rgb_decoder_embeddings)

        rgb_decoder_embeddings = self.rgb_decoder_norm(rgb_decoder_embeddings)
        rgb_reconstruction = self.rgb_head(rgb_decoder_embeddings)  # Shape: [B, N, 3 * num_patches]

        # Depth Decoder
        depth_decoder_embeddings = self.encoder_to_decoder(depth_embeddings) if self.embed_dim != self.decoder_embed_dim else depth_embeddings
        for block in self.depth_decoder:
            depth_decoder_embeddings = block(depth_decoder_embeddings)

        depth_decoder_embeddings = self.depth_decoder_norm(depth_decoder_embeddings)
        depth_reconstruction = self.depth_head(depth_decoder_embeddings)  # Shape: [B, N, 1 * num_patches]

        # Get the number of patches per frame
        num_patches_per_frame = 14 * 14  # TODO hardcoded for now

        # Reshape RGB reconstruction to the desired shape
        rgb_reconstruction = rgb_reconstruction.view(B, T, num_patches_per_frame, -1)  # Last dim will be 768

        # Reshape Depth reconstruction to the desired shape
        depth_reconstruction = depth_reconstruction.view(B, T, num_patches_per_frame, -1)  # Last dim will be 256

        return rgb_reconstruction, depth_reconstruction

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


if __name__ == '__main__':
    import yaml
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = config['model']
    
    # Create an instance of the model
    model = CrossModalVideoMAE(config)

    # Create dummy data matching your data loader's output
    batch_size = 4
    B = batch_size
    num_frames = config['num_frames']
    img_size = config['img_size']
    C_rgb = 3
    C_depth = 1

    # Generate dummy RGB frames and depth maps
    rgb_frames = torch.randn(B, C_rgb, num_frames, img_size, img_size)  # Shape: [B, 3, T, H, W]
    depth_maps = torch.randn(B, C_depth, num_frames, img_size, img_size)  # Shape: [B, 1, T, H, W]
    
    #Print shapes for debugging
    print(f"RGB Frames Shape: {rgb_frames.shape}")
    print(f"Depth Maps Shape: {depth_maps.shape}")

    # Generate random masks
    num_tubelets = model.rgb_tubelet_embed.num_tubelets
    masks = torch.rand(B, num_tubelets) < 0.75  # 75% of patches are masked

    print(f'num_tubelets: {num_tubelets}')
    print(f"Mask Shape: {masks.shape}")
    

    # Forward pass
    rgb_recon, depth_recon = model(rgb_frames, depth_maps, masks)

    # Print the output shapes
    print(f"RGB Reconstruction Shape: {rgb_recon.shape}")  # Expected: [B, N, patch_size^2 * 3]
    print(f"Depth Reconstruction Shape: {depth_recon.shape}")  # Expected: [B, N, patch_size^2 * 1]")
