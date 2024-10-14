# imports necessary libraries and modules
import math  # Provides mathematical functions
import torch  # PyTorch library for tensor computations
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional interface for neural network operations
import torch.utils.checkpoint as checkpoint  # Enables gradient checkpointing for memory efficiency
from functools import partial  # Allows creating partial functions

# imports modules from other files within the project or from external libraries
from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table  # imports related to fine-tuning
from timm.models.registry import register_model  # imports model registration from timm library
from timm.models.layers import trunc_normal_ as __call_trunc_normal_  # imports truncated normal initialization


# Defines a function for truncated normal initialization
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std) # applies the truncated normal initialization from TIMM


# A list of model names that are exported from this file
__all__ = [
    'pretrain_videomae_small_patch16_224',  # small variant for 224x224 input with 16x16 patches
    'pretrain_videomae_base_patch16_224',  # base variant for 224x224 input with 16x16 patches
    'pretrain_videomae_large_patch16_224',  # large variant for 224x224 input with 16x16 patches
    'pretrain_videomae_huge_patch16_224',  # huge variant for 224x224 input with 16x16 patches
]


class PretrainVisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder for VideoMAE pre-training.

    This encoder processes masked video patches and outputs features for visible patches.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int | tuple): Patch size. Default: 16.
        in_chans (int): Number of input channels. Default: 3.
        num_classes (int): Number of classes for classification head (not used in pre-training). Default: 0.
        embed_dim (int): Embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: False.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        init_values (float): Init value for LayerScale. Default: None.
        tubelet_size (int): Tubelet size. Default: 2.
        use_checkpoint (bool): Whether to use checkpointing. Default: False.
        use_learnable_pos_emb (bool): Whether to use learnable positional embeddings. Default: False.
    """

    # the init method initializes the encoder part of the VideoMAE model
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        # calls the init method of the parent class (nn.Module)
        super().__init__()
        # initializes class variables
        self.num_classes = num_classes  # Number of classes for the classification head (unused in pretraining).
        self.num_features = self.embed_dim = embed_dim  # Embedding dimension.
        # creates a patch embedding layer to convert image to patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            tubelet_size=tubelet_size)  # img_size: image size, patch_size: patch size, tubelet_size: number of frames to be considered as tubelet
        num_patches = self.patch_embed.num_patches  # Total number of patches.
        self.use_checkpoint = use_checkpoint #  flag for gradient checkpointing

        # TODO: Add the cls token

        # creates learnable or fixed sinusoidal positional embeddings
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1,
                                                      embed_dim))  # Learnable positional embeddings. Shape: (1, num_patches + 1, embed_dim), +1 accounts for unused cls token
        else:
            # uses fixed sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches,
                                                         embed_dim)  # Fixed sine-cosine positional embeddings. Shape: (1, num_patches, embed_dim)

        # creates a list of stochastic depth decay rates (drop_path_rate), linearly increasing from 0 to drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # creates the transformer encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values) # creates each transformer encoder block. depth: number of encoder blocks
            for i in range(depth)]) # iterates to create multiple encoder blocks
        self.norm = norm_layer(embed_dim)  # Normalization layer.
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # creates a classification head for linear projection

        # initializes the learnable positional embedding with truncated normal distribution
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # initializes the weights of the model
        self.apply(self._init_weights)  # applies the weight initialization function to all submodules


    # initializes weights of linear and layer normalization layers
    def _init_weights(self, m):
        # initializes weights for linear layers with xavier uniform initialization
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # initializes linear layer weights
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # initializes linear layer biases to 0
        # initializes the layer normalization layers with bias and weight values
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # initializes layer norm bias to 0
            nn.init.constant_(m.weight, 1.0)  # initializes layer norm weight to 1

    # returns the number of encoder blocks (transformer layers)
    def get_num_layers(self):
        return len(self.blocks)  # Returns the number of transformer blocks.

    # specifies parameter names that should not be subjected to weight decay during training
    @torch.jit.ignore  # Tells TorchScript to ignore this method.
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}  # Returns a set of parameters to exclude from weight decay.

    # returns the classification head of the encoder
    def get_classifier(self):
        return self.head  # Returns the classification head.

    # allows resetting the classifier (linear head) with a new number of classes. Useful for transfer learning
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes  # sets the new number of classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity() # defines new linear classifier if num_classes>0 else Identity()

    # This method defines the forward pass for the feature extraction part of the encoder. It takes masked video input, performs patch embedding, adds positional embedding and then feeds the visible patches through the transformer blocks.
    def forward_features(self, x, rgb_mask, depth_mask):
        # input x: [B, C, T, H, W] - mask: [B, T*H*W]
        _, _, T, _, _ = x.shape # extracts the temporal dimension (T) from the input video 'x'
        rgb_x = self.patch_embed(x[:, :3, :, :, :])  # Applies patch embedding. [B, C, T, H, W] -> [B, N, D] - N: number of patches per video (= T*H*W/P*P*Tp), D: embedding dimension
        depth_x = self.patch_embed(x[:, 3:, :, :, :])  # Applies patch embedding. [B, C, T, H, W] -> [B, N, D] - N: number of patches per video (= T*H*W/P*P*Tp), D: embedding dimension

        rgb_x = rgb_x + self.pos_embed.type_as(rgb_x).to(rgb_x.device).clone().detach()  # Adds positional embeddings. [B, N, D]
        depth_x = depth_x + self.pos_embed.type_as(depth_x).to(depth_x.device).clone().detach()  # Adds positional embeddings. [B, N, D]

        # TODO Is this conctaenation correct?
        x = torch.cat([rgb_x, depth_x], dim=1)  # Concatenates rgb and depth embeddings. [B, 2*N, D]

        # TODO I am also concatenating the masks, this will work depending on how the rgb_x and depth_x are concatenated
        mask = torch.cat([rgb_mask, depth_mask], dim=1)  # Concatenates rgb and depth masks. [B, 2*T*H*W]

        B, _, C = x.shape # extracts the batch size (B) and embedding dimension (C)
        x_vis = x[~mask].reshape(B, -1, C)  # Selects visible patches and reshapes. [B, N_vis, D]  N_vis: number of visible patches per video,

        if self.use_checkpoint: # checks if gradient checkpointing is enabled
            for blk in self.blocks: # iterates over transformer blocks
                x_vis = checkpoint.checkpoint(blk, x_vis)  # Applies a transformer block with gradient checkpointing. [B, N_vis, D]
        else: # if gradient checkpointing is disabled
            for blk in self.blocks: # iterates over transformer blocks
                x_vis = blk(x_vis)  # Applies a transformer block. [B, N_vis, D]

        x_vis = self.norm(x_vis)  # Applies normalization. [B, N_vis, D]
        return x_vis  # Returns the processed features.

    # The forward method performs a forward pass of the encoder
    def forward(self, rgb_x, depth_x, rgb_mask, depth_mask):
        # input x: [B, C, T, H, W] - mask: [B, T*H*W]
        x = self.forward_features(rgb_x, depth_x, rgb_mask, depth_mask)  # Extracts features from the input. [B, N_vis, D]
        x = self.head(x)  # Applies the classification head. output [B, N_vis, num_classes] if num_classes>0 else [B, N_vis, D] (not useful in pre-training since num_classes==0)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    # The init method initializes the decoder part of the VideoMAE model.
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False
                 ):
        # calls the init method of the parent class (nn.Module)
        super().__init__()
        self.num_classes = num_classes  # Number of classes (output channels of the decoder).
        # TODO changed the assertion from 3 classes to 4 classes
        assert num_classes == 4 * tubelet_size * patch_size ** 2  # asserts output dimension, num_classes should be number of pixels (3 channles) * number of frames in tubelet
        self.num_features = self.embed_dim = embed_dim  # Embedding dimension.
        self.patch_size = patch_size # Patch size
        self.use_checkpoint = use_checkpoint  # Whether to use gradient checkpointing.


        # creates a list of stochastic depth decay rates (drop_path_rate), linearly increasing from 0 to drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # creates the transformer decoder blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)  # Creates each transformer decoder block. depth: number of decoder blocks
            for i in range(depth)])  # iterates to create multiple decoder blocks
        self.norm = norm_layer(embed_dim)  # Normalization layer.
        # TODO not sure about this...?
        # Linear Projection Heads (Modified)
        self.rgbd_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # creates a linear projection head for combined RGBD

        # initializes weights and biases
        self.apply(self._init_weights)  # applies the weight initialization function to all submodules


    # initializes weights of linear and layer normalization layers
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) # initializes weights for linear layers
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) # initializes bias for linear layers
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0) # initializes bias for layer normalization layers
            nn.init.constant_(m.weight, 1.0) # initializes weights for layer normalization layers


    # returns the number of transformer blocks (layers) in the decoder
    def get_num_layers(self):
        return len(self.blocks)  # returns the number of decoder blocks (transformer layers)

    # specifies parameter names that should not be subjected to weight decay
    @torch.jit.ignore  # Tells TorchScript to ignore this method.
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # returns the classifier head of the decoder
    def get_classifier(self):
        return self.head  # returns the classification head

    # reset classifier
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes  # sets new num of classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity() # initializes new linear head


    # The forward method performs a forward pass of the decoder. It takes the full set of tokens (visible + mask tokens), passes them through the transformer blocks, normalizes, and projects to pixel values using a linear head.
    def forward(self, x, return_token_num):
        # TODO rgb & depth outputs should be handled here. I separated the output into two heads, not sure if this is correct
        # x: [B, N, C_d] - return_token_num: int
        if self.use_checkpoint: # checks if gradient checkpointing is enabled
            for blk in self.blocks: # loops over each decoder block
                x = checkpoint.checkpoint(blk, x)  # Applies a transformer block with gradient checkpointing. [B, N, C_d]
        else:   # if checkpointing is not enabled
            for blk in self.blocks: # loops over each decoder block
                x = blk(x)  # Applies a transformer block. [B, N, C_d]

        # Applies normalization and the projection head to the masked tokens to predict pixel values.
        if return_token_num > 0: # checks if reconstruction is to be done on a subset of tokens
            # x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels [B, N_mask, num_classes] where N_mask = return_token_num
            rgb_x = self.rgb_head(self.norm(x[:, -return_token_num:]))
            depth_x = self.depth_head(self.norm(x[:, -return_token_num:]))
        else:  # otherwise computes output for all tokens
            # x = self.head(self.norm(x)) # [B, N, num_classes]
            rgb_x = self.rgb_head(self.norm(x))
            depth_x = self.depth_head(self.norm(x))

        return rgb_x, depth_x  # returns the predicted pixel values for rgb and depth


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, # image size
                 patch_size=16, # patch size
                 encoder_in_chans=3, # input channels
                 encoder_num_classes=0, # encoder classifier output dim (unused)
                 encoder_embed_dim=768, # embedding dimension of encoder
                 encoder_depth=12, # number of transformer blocks in encoder
                 encoder_num_heads=12, # number of heads for multi head attention in encoder
                 decoder_num_classes=1536,  #  decoder classifier output dim (number of pixels x channels x tubelet size)
                 decoder_embed_dim=512, # decoder embed dim
                 decoder_depth=8, #  number of transformer blocks in decoder
                 decoder_num_heads=8, #  number of heads for multi head attention in decoder
                 mlp_ratio=4., # mlp expansion ratio
                 qkv_bias=False, # bias for qkv projection
                 qk_scale=None, # qk scale
                 drop_rate=0., # droput rate
                 attn_drop_rate=0., # attention dropout rate
                 drop_path_rate=0.,  # stochastic depth drop rate
                 norm_layer=nn.LayerNorm,  # normalization layer
                 init_values=0., # initialization values for layer scale
                 use_learnable_pos_emb=False,  # learnable positional embeddings
                 use_checkpoint=False,  # gradient checkpointing flag
                 tubelet_size=2,  # tubelet size in frames
                 num_classes=0,  # number of classes for final classification (unused)
                 in_chans=0,  # number of input channels (unused)
                 ):
        # call parent class init
        super().__init__()
        # Encoder
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,  # image size
            patch_size=patch_size,  # patch size
            in_chans=encoder_in_chans,  # number of input channels
            num_classes=encoder_num_classes,  # number of classes for encoder (unused)
            embed_dim=encoder_embed_dim,  # embedding dimension
            depth=encoder_depth,  # encoder depth
            num_heads=encoder_num_heads, # num of heads in encoder multi-head attn
            mlp_ratio=mlp_ratio, # mlp ratio
            qkv_bias=qkv_bias,  # bias for qkv
            qk_scale=qk_scale,  # qk scale
            drop_rate=drop_rate,  # dropout rate
            attn_drop_rate=attn_drop_rate, # attention dropout rate
            drop_path_rate=drop_path_rate,  # stochastic depth drop rate
            norm_layer=norm_layer,  # normalization layer
            init_values=init_values, # init values for layer scale
            tubelet_size=tubelet_size,  # tubelet size
            use_checkpoint=use_checkpoint,  # checkpointing flag
            use_learnable_pos_emb=use_learnable_pos_emb)  # learnable pos emb flag

        # Decoder
        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,  # patch size
            num_patches=self.encoder.patch_embed.num_patches, # number of patches
            num_classes=decoder_num_classes,  # decoder output dim (pixels x channels x tubelet_size)
            embed_dim=decoder_embed_dim, # decoder embed dim
            depth=decoder_depth, # decoder depth
            num_heads=decoder_num_heads,  # decoder num heads
            mlp_ratio=mlp_ratio, # mlp ratio
            qkv_bias=qkv_bias, # qkv bias
            qk_scale=qk_scale, # qk scale
            drop_rate=drop_rate, # dropout
            attn_drop_rate=attn_drop_rate, # attn dropout
            drop_path_rate=drop_path_rate,  # stochastic depth
            norm_layer=norm_layer,  # normalization
            init_values=init_values, # init values for layer scale
            tubelet_size=tubelet_size,  # tubelet size
            use_checkpoint=use_checkpoint) # checkpointing flag

        # Linear projection from encoder output to decoder input
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim,
                                            bias=False)  # projects encoder output to decoder input dimension

        # Mask token for masked patches
        # TODO not sure about this
        self.rgb_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # mask token learnable parameter [1, 1, decoder_embed_dim]
        self.depth_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # mask token learnable parameter [1, 1, decoder_embed_dim]

        # Positional embeddings for decoder
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches,
                                                     decoder_embed_dim)  # pos emb for decoder [1, num_patches, decoder_embed_dim]

        # Initialize mask token
        trunc_normal_(self.mask_token, std=.02)  # initializes the mask token

    # Weight initialization
    def _init_weights(self, m):
        # initializes weights of linear layers with xavier uniform and biases with constant 0
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization for linear layer weights.
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) # constant initialization for bias
        # initializes weights and biases of layer normalization layers with constant values
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0) # constant initialization of bias
            nn.init.constant_(m.weight, 1.0)  # constant initialization of weights

    # Get number of layers
    def get_num_layers(self):
        return len(self.blocks) # number of blocks

    # Parameters to exclude from weight decay
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    # Forward pass
    def forward(self, x, mask):
        # x: [B, C, T, H, W] - mask: [B, T*H*W]
        _, _, T, _, _ = x.shape # extracts temporal dim (T)
        x_vis = self.encoder(x, mask)  # Encodes visible patches. [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # Projects encoder output to decoder dimension. [B, N_vis, C_d]
        B, N, C = x_vis.shape  # Extracts batch size (B), number of visible tokens(N), and channel dimension (C).

        # Expands positional embeddings for the batch and adds to the visible tokens
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(
            x.device).clone().detach()  # expands pos emb [B, N, C_d]
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C) # selects pos emb related to visible tokens and reshapes. [B, N_vis, C_d]
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C) # selects pos emb related to masked tokens and reshapes. [B, N_mask, C_d]
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask],
                           dim=1)  # Concatenates visible and masked tokens. [B, N, C_d]

        rgb_x, depth_x = self.decoder(x_full, pos_emd_mask.shape[1]) # The single output x has RGB and Depth info

        return rgb_x, depth_x

# register model small
@register_model # timm register model
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224, # input image size
        patch_size=16,  # patch size
        encoder_embed_dim=384,  # encoder embedding dimension
        encoder_depth=12, # encoder depth
        encoder_num_heads=6, # encoder heads
        encoder_num_classes=0, # unused
        decoder_num_classes=1536, # decoder output channels
        decoder_embed_dim=192, # decoder embed dim
        decoder_num_heads=3, # decoder heads
        mlp_ratio=4, # mlp ratio
        qkv_bias=True,  # qkv bias is used
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # layer norm with specified epsilon
        **kwargs) # other parameters
    model.default_cfg = _cfg() # assigns default configuration
    if pretrained: # for loading pretrained weights if available. Not used in pre-training
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
# register model base
@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224, # image size
        patch_size=16,  # patch size
        encoder_embed_dim=768, # encoder embed dim
        encoder_depth=12,  # encoder depth
        encoder_num_heads=12, # encoder heads
        encoder_num_classes=0,  # unused
        decoder_num_classes=1536,  # decoder output dim (pixels x channels x tubelet_size)
        decoder_embed_dim=384, # decoder embed dim
        decoder_num_heads=6,  # decoder heads
        mlp_ratio=4, # mlp ratio
        qkv_bias=True, # qkv bias
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # layer norm
        **kwargs)
    model.default_cfg = _cfg() # default cfg
    if pretrained: # load pretrained weights if available. Not used in pre-training
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

# register model large
@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224, # image size
        patch_size=16, # patch size
        encoder_embed_dim=1024, # encoder embed dim
        encoder_depth=24, # encoder depth
        encoder_num_heads=16,  # encoder num heads
        encoder_num_classes=0, # unused
        decoder_num_classes=1536, # decoder output channels
        decoder_embed_dim=512, # decoder embed dim
        decoder_num_heads=8, # decoder num heads
        mlp_ratio=4,  # mlp ratio
        qkv_bias=True, # qkv bias
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # layer norm
        **kwargs)
    model.default_cfg = _cfg() # default cfg
    if pretrained: #  load pretrained weights if available. Not used during pre-training
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

# register model huge
@register_model
def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224, # image size
        patch_size=16,  # patch size
        encoder_embed_dim=1280,  # encoder embedding dimension
        encoder_depth=32,  # encoder depth
        encoder_num_heads=16,  # encoder num heads
        encoder_num_classes=0,  # unused
        decoder_num_classes=1536, # decoder output dim (pixels x channels x tubelet_size)
        decoder_embed_dim=640, # decoder embed dim
        decoder_num_heads=8, # decoder num heads
        mlp_ratio=4, # mlp ratio
        qkv_bias=True, # qkv bias
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # layer norm
        **kwargs)
    model.default_cfg = _cfg() # default cfg
    if pretrained:  # load pretrained weights if available. Not used during pre-training
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
