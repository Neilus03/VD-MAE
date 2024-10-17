# models/videomae_cross_modal.py

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from tubeletembed import TubeletEmbed


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
        self.rgb_head = nn.Linear(config['embed_dim'], config['patch_size'] ** 2 * 3) #CHECK THIS LINE'S OUTPUT DIMENSION (SUSPECTED TO BE PATCH_SIZE ** 2 * NUM_PATCHES**2 * 3)
        
        
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
        self.depth_head = nn.Linear(config['decoder_embed_dim'], config['patch_size']**2 * 1) #CHECK THIS LINE'S OUTPUT DIMENSION (SUSPECTED TO BE PATCH_SIZE ** 2 * NUM_PATCHES**2 * 1)


        #Initialize the mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim']))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
      
        # Loss weighting factors
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 1.0)
        
    def forward(self, rgb_frames, depth_maps, masks):
        '''
        Forward pass of the CrossModalVideoMAE model.

        Parameters:
            - rgb_frames: Tensor of shape [B, 3, T, H, W]
            - depth_maps: Tensor of shape [B, 1, T, H, W]
            - masks: Boolean Tensor of shape [B, N], where N is the number of patches.

        Returns:
            - rgb_reconstruction: Reconstructed RGB patches.
            - depth_reconstruction: Reconstructed depth patches.
        '''
        # Inputs are already in shape [B, 3, T, H, W] and [B, 1, T, H, W]

        B, C, T, H, W = rgb_frames.shape
        assert C == 3, "Input RGB tensor must have 3 channels"

        _, C_d, T_d, H_d, W_d = depth_maps.shape
        assert C_d == 1, "Input Depth tensor must have 1 channel"
        assert T == T_d and H == H_d and W == W_d, "RGB and Depth tensors must have the same dimensions"

        # Apply tubelet embedding to the RGB and Depth frames
        rgb_embed = self.rgb_tubelet_embed(rgb_frames)  # Shape: [B, N, embed_dim]
        depth_embed = self.depth_tubelet_embed(depth_maps)  # Shape: [B, N, embed_dim]

        # Fuse the embeddings via element-wise addition
        fused_embed = rgb_embed + depth_embed  # Shape: [B, N, embed_dim]

        # Assign fused embeddings to 'embeddings' variable
        embeddings = fused_embed

        # Ensure masks tensor shape is as expected and convert to boolean tensor if not already
        assert masks.shape == (B, embeddings.size(1)), "Mask tensor must have shape [B, N]"
        masks = masks.bool()  # Convert to boolean tensoR

        # Apply masking, replace the masked embeddings with the mask token
        # Expand the mask token to the same shape as the embeddings
        mask_tokens = self.mask_token.expand(B, embeddings.size(1), -1)  # Shape: [B, N, embed_dim]
        # Replace the masked embeddings with the mask token
        embeddings = torch.where(masks.unsqueeze(-1), mask_tokens, embeddings)  # Shape: [B, N, embed_dim]

        # Add positional encoding to the embeddings
        embeddings += self.pos_embed  # Shape: [B, N, embed_dim]

        # Pass the embeddings through the transformer encoder layers
        for block in self.encoder:
            embeddings = block(embeddings)

        # Normalize the encoder output
        embeddings = self.encoder_norm(embeddings)

         #AT THIS POINT WE HAVE THE EMBEDDINGS OF THE BOTTLENECK
        
        
        # Prepare the full sequence for the decoders
        # Map encoder output to decoder input dimension if they are different
        if config['embed_dim'] != config['decoder_embed_dim']:
            decoder_embeddings = nn.Linear(config['embed_dim'], config['decoder_embed_dim'], bias=False)
        else:
            decoder_embeddings = embeddings 

        # RGB Decoder
        rgb_embeddings = decoder_embeddings  # Start with the prepared embeddings
        for block in self.rgb_decoder:
            rgb_embeddings = block(rgb_embeddings)
        # Normalize the RGB decoder output
        rgb_embeddings = self.rgb_decoder_norm(rgb_embeddings)
        # Reconstruct the RGB patches
        rgb_reconstruction = self.rgb_head(rgb_embeddings)  # Shape: [B, N, 3 * patch_size ** 2]

        # Depth Decoder
        depth_embeddings = decoder_embeddings
        for block in self.depth_decoder:
            depth_embeddings = block(depth_embeddings)
        # Normalize the Depth decoder output
        depth_embeddings = self.depth_decoder_norm(depth_embeddings)
        # Reconstruct the Depth patches
        depth_reconstruction = self.depth_head(depth_embeddings)  # Shape: [B, N, 1 * patch_size ** 2]

        return rgb_reconstruction, depth_reconstruction

            
    def extract_patches(self, frames, patch_size):
        """
        Extract non-overlapping patches from frames.

        Parameters:
            - frames: Tensor of shape [B, C, T, H, W]
            - patch_size: Patch size (int)

        Returns:
            - patches: Tensor of shape [B, N, patch_size^2 * C], where N is the number of patches
        """
        B, C, T, H, W = frames.shape
        # Rearrange frames to [B * T, C, H, W]
        frames = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # Shape: [B*T, C, H, W]
        # Use unfold to extract patches
        patches = frames.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches shape: [B*T, C, num_tubelets_H, num_tubelets_W, patch_size, patch_size]
        # Rearrange dimensions
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # Shape: [B*T, num_tubelets_H, num_tubelets_W, C, patch_size, patch_size]
        # Flatten spatial dimensions
        patches = patches.contiguous().view(B, -1, patch_size * patch_size * C)  # Shape: [B, N, patch_size^2 * C]
        return patches

        
    def compute_loss(self, rgb_frames, depth_maps, rgb_recon, depth_recon, masks):
        '''
        Compute the reconstruction loss.

        Parameters:
            - rgb_frames: Original RGB frames, shape [B, 3, T, H, W]
            - depth_maps: Original Depth maps, shape [B, 1, T, H, W]
            - rgb_recon: Reconstructed RGB patches, shape [B, N, patch_size^2 * 3]
            - depth_recon: Reconstructed Depth patches, shape [B, N, patch_size^2 * 1]
            - masks: Boolean Tensor indicating masked positions, shape [B, N]

        Returns:
            - rgb_loss: RGB reconstruction loss
            - depth_loss: Depth reconstruction loss
            - total_loss: The total loss computed as a weighted sum of RGB and Depth losses
        '''
        # Extract original patches
        rgb_patches = self.extract_patches(rgb_frames, self.rgb_tubelet_embed.patch_size[0])  # Shape: [B, N, patch_size^2 * 3]
        depth_patches = self.extract_patches(depth_maps, self.depth_tubelet_embed.patch_size[0])  # Shape: [B, N, patch_size^2 * 1]

        # Only compute loss on masked positions
        # Flatten the masks to use as indices
        masks_flat = masks.view(-1)
        
        # Flatten the patches and reconstructions
        rgb_patches_flat = rgb_patches.view(-1, rgb_patches.size(-1))
        depth_patches_flat = depth_patches.view(-1, depth_patches.size(-1))
        rgb_recon_flat = rgb_recon.view(-1, rgb_recon.size(-1))
        depth_recon_flat = depth_recon.view(-1, depth_recon.size(-1))

        # Select masked positions
        rgb_patches_masked = rgb_patches_flat[masks_flat]
        depth_patches_masked = depth_patches_flat[masks_flat]
        rgb_recon_masked = rgb_recon_flat[masks_flat]
        depth_recon_masked = depth_recon_flat[masks_flat]

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
