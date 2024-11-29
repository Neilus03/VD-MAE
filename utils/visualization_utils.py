# Description: Utility functions for visualizing images and depth maps during training and validation.

import numpy as np
import sys, os, yaml
import matplotlib.pyplot as plt
import wandb
import io
import torch
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist

#sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

#from data.dataloader import denormalize_RGB, denormalize_depth


# Load configuration
config_path = '../config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def denormalize_RGB(tensor):
    '''
    Denormalizes the RGB channels of a tensor containing an RGB or RGB-D image
    '''
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    
    # Check if the input is a 4D tensor (batch, channels, height, width)
    if tensor.dim() == 4:
        # Only denormalize the first 3 channels (RGB)
        denormalized_rgb = tensor[:, :3] * std + mean
        
        # If there's a depth channel, keep it unchanged
        if tensor.shape[1] == 4:
            denormalized_tensor = torch.cat([denormalized_rgb, tensor[:, 3:]], dim=1)
        else:
            denormalized_tensor = denormalized_rgb
    else:
        # For 3D tensors (single image)
        if tensor.shape[0] >= 3:
            denormalized_rgb = tensor[:3] * std + mean
            if tensor.shape[0] == 4:
                denormalized_tensor = torch.cat([denormalized_rgb, tensor[3:]], dim=0)
            else:
                denormalized_tensor = denormalized_rgb
        else:
            denormalized_tensor = tensor * std + mean

    return denormalized_tensor

def denormalize_depth(tensor, mean, std):
    '''
    Denormalizes the depth channel of a tensor containing an RGB-D image
    '''
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    denormalized_depth = tensor * std + mean
    return denormalized_depth

def unpatchify(patches, img_size=(224, 224), patch_size=16):
    '''
    Reconstructs the image from patches.

    Args:
        patches (torch.Tensor): The output patches from the decoder
                                (shape: [batch_size, num_patches, patch_size*patch_size*num_channels]).
        img_size (tuple): The original image size (height, width).
        patch_size (int): The size of each patch (assuming square patches).

    Returns:
        tuple: (reconstructed_rgb, reconstructed_depth)
            reconstructed_rgb: [batch_size, 3, height, width]
            reconstructed_depth: [batch_size, 1, height, width]
    '''
    batch_size, num_patches, patch_elements = patches.shape
    num_channels = patch_elements // (patch_size * patch_size)
    assert num_channels == 4, "Expected 4 channels (RGB + Depth)"
    assert patch_elements == patch_size * patch_size * num_channels, "Incorrect patch elements"

    # Calculate the number of patches per dimension
    num_patches_per_dim = img_size[0] // patch_size
    assert num_patches_per_dim ** 2 == num_patches, "Number of patches does not match image dimensions"

    # Reshape patches into the image grid
    patches = patches.view(batch_size, num_patches_per_dim, num_patches_per_dim, patch_size, patch_size, num_channels)
    # Rearrange dimensions to match image shape
    patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    # Combine patches into full images
    images = patches.view(batch_size, num_channels, img_size[0], img_size[1])

    # Split the channels into RGB and depth
    reconstructed_rgb = images[:, :3, :, :]
    reconstructed_depth = images[:, 3:, :, :]

    print(f"Unpatchify - Reconstructed RGB image shape: {reconstructed_rgb.shape}")
    print(f"Unpatchify - Reconstructed Depth image shape: {reconstructed_depth.shape}")

    return reconstructed_rgb, reconstructed_depth


def extract_patches(image, patch_size):
    '''
    Extract patches from an image tensor.
    Args:
        image: Tensor of shape (C, H, W)
        patch_size: size of the patches
    Returns:
        patches: Tensor of shape (num_patches, C, patch_size, patch_size)
    '''
    # Unfold the image to get patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Rearrange dimensions to get patches in (num_patches, channels, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, image.shape[0], patch_size, patch_size)
    return patches

def assemble_patches_with_gaps(patches, gap_size, num_patches_per_row, patch_size, num_channels=3, depth=False):
    '''
    Assembles patches into an image with gaps between patches for cooler visualization.
    Args:
        patches: numpy array of shape (num_patches, num_channels, patch_size, patch_size)
        gap_size: size of the gap between patches (in pixels)
        num_patches_per_row: number of patches per row/column
        patch_size: size of each patch (assuming square patches)
        num_channels: number of channels in the image
        depth: whether it's a depth map (True) or RGB image (False)
    Returns:
        image_with_gaps: numpy array of shape (grid_size, grid_size, num_channels) or (grid_size, grid_size)
    '''
    #define the grid size as the size of the image with gaps
    grid_size = num_patches_per_row * patch_size + (num_patches_per_row - 1) * gap_size # in pixels
    if depth:
        #for depth use a single channel
        image_with_gaps = np.ones((grid_size, grid_size))
    else:
        #for RGB use 3 channels
        image_with_gaps = np.ones((grid_size, grid_size, num_channels))
    idx = 0
    #iterate over the patches and place them in the image with gaps
    for row in range(num_patches_per_row):
        for col in range(num_patches_per_row):
            #calculate the start position of the patch
            y_start = row * (patch_size + gap_size)
            x_start = col * (patch_size + gap_size)
            if depth:
                #for depth maps we only have one channel
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size] = patches[idx]
            else:
                #for RGB images we have 3 channels
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = patches[idx].transpose(1, 2, 0)
            idx += 1
    return image_with_gaps

def log_visualizations(rgb_frames, depth_maps, reconstructed_image, reconstructed_depth, mask, epoch, batch_idx, prefix='Train'):
    '''
    Logs visualizations to WandB.
    Args:
        rgb_frames: input images (batch_size, 3, T, H, W)
        depth_maps: input depth maps (batch_size, 1, T, H, W)
        reconstructed_image: reconstructed RGB images (batch_size, 3, T, H, W)
        reconstructed_depth: reconstructed depth maps (batch_size, 1, T, H, W)
        mask: mask used during training (batch_size, num_patches)
        epoch: current epoch
        batch_idx: current batch index
        prefix: 'Train' or 'Validation'
    '''
    rank = dist.get_rank()
    if rank != 0:
        return
    
    depth_mean = config['data']['depth_stats']['mean']
    depth_std = config['data']['depth_stats']['std']
    img_size = config['model']['img_size']
    patch_size = config['model']['patch_size']
    print(f"1-Original RGB frames shape: {rgb_frames.shape}")
    print(f"1-Original Depth maps shape: {depth_maps.shape}")
    # Select the first sample in the batch
    original_image = rgb_frames[0].detach().cpu()  # Shape: (3, T, H, W)
    original_depth = depth_maps[0].detach().cpu()  # Shape: (1, T, H, W)
    print(f'2-First image in batch shape: {original_image.shape}')
    print(f'2-First depth map in batch shape: {original_depth.shape}')

    # Select the first frame in the sequence
    original_image = original_image[:, 0]  # Shape: (3, H, W)
    original_depth = original_depth[:, 0]  # Shape: (1, H, W)
    print(f"3-Original RGB image shape: {original_image.shape}")
    print(f"3-Original Depth map shape: {original_depth.shape}")
    # Same for reconstructed image
    reconstructed_image = reconstructed_image.detach().cpu()
    reconstructed_depth = reconstructed_depth.detach().cpu()
    print(f"4-Reconstructed RGB image shape: {reconstructed_image.shape}")
    print(f"4-Reconstructed Depth map shape: {reconstructed_depth.shape}")
    # Same for mask
    mask = mask[:, 0]
    print('MASKS SHAPE:', mask.shape)

    # Select the first frame in the sequence
    reconstructed_image = reconstructed_image[:, :, 0]  # Shape: (B, 3, H, W)
    reconstructed_depth = reconstructed_depth[:, :, 0]  # Shape: (B, 1, H, W)
    print(f"5-Reconstructed RGB image shape: {reconstructed_image.shape}")
    print(f"5-Reconstructed Depth map shape: {reconstructed_depth.shape}")    

    # Denormalize original RGB image
    #original_rgb_image = denormalize_RGB(original_image).permute(1, 2, 0).clamp(0, 1).numpy()
    original_rgb_image = original_image
    # Denormalize original depth map
    original_depth_map = original_depth.numpy()
    original_depth_map = original_depth_map * depth_std + depth_mean

    # Normalize depth map for visualization
    original_depth_map_viz = (original_depth_map - original_depth_map.min()) / (original_depth_map.max() - original_depth_map.min() + 1e-8)

    # Extract patches from the original image
    original_patches_rgb = extract_patches(original_image, patch_size)  # Shape: (num_patches, 3, patch_size, patch_size)
    original_patches_depth = extract_patches(original_depth, patch_size)  # Shape: (num_patches, 1, patch_size, patch_size)
    original_patches = torch.cat([original_patches_rgb, original_patches_depth], dim=1)  # Shape: (num_patches, 4, patch_size, patch_size)

    # Denormalize patches
    #original_patches_rgb_denorm = denormalize_RGB(original_patches_rgb).clamp(0, 1).numpy()  # Shape: (num_patches, 3, patch_size, patch_size)
    original_patches_rgb_denorm = original_patches_rgb.numpy()
    original_patches_depth_denorm = original_patches_depth.numpy()
    original_patches_depth_denorm = original_patches_depth_denorm * depth_std + depth_mean  # Shape: (num_patches, 1, patch_size, patch_size)
    original_patches_depth_denorm = original_patches_depth_denorm.squeeze(1)  # Shape: (num_patches, patch_size, patch_size)

    # Normalize depth patches for visualization
    depth_min = original_patches_depth_denorm.min()
    depth_max = original_patches_depth_denorm.max()
    original_patches_depth_viz = (original_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)

    # Assemble original patches with gaps
    gap_size = 2  # pixels
    num_patches_per_row = img_size // patch_size
    assembled_original_rgb = assemble_patches_with_gaps(original_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_original_depth = assemble_patches_with_gaps(original_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)
    print(f"6-Assembled original RGB image shape: {assembled_original_rgb.shape}")
    print(f"6-Assembled original Depth map shape: {assembled_original_depth.shape}")
    # Masked patches
    # masked_patches = original_patches.clone()
    # print(f'masked_patches shape: {masked_patches.shape}')
    # masked_indices = (mask[0] == 1).nonzero(as_tuple=False).squeeze()
    # masked_patches[masked_indices] = 0  # Set masked patches to zero

    # TODO the shapes mismatch

    # Clone the original patches for masking
    masked_patches = original_patches.clone()
    print(f'masked_patches shape: {masked_patches.shape}')

    # Ensure mask is reshaped to match patch dimensions
    batch_size, num_patches = mask.shape[0], masked_patches.shape[0]
    print(f"Batch size: {batch_size}, Number of patches: {num_patches}")

    # Flatten mask to match patch dimensions
    mask = mask.view(batch_size, -1)  # Flatten spatial dimensions into patches
    print(f"Flattened mask shape: {mask.shape}")

    # Get masked indices
    masked_indices = (mask[0] == 1).nonzero(as_tuple=False).squeeze()  # Adjust for the first batch element
    print(f"Masked indices: {masked_indices}")

    # Ensure masked indices are within valid range
    assert masked_indices.max() < masked_patches.shape[0], "Masked indices exceed the number of patches!"
    masked_patches[masked_indices] = 0  # Set masked patches to zero

    # Extract masked patches for rgb and depth separately
    masked_patches_rgb = masked_patches[:, :3, :, :] # Shape: (num_patches, 3, patch_size, patch_size)
    masked_patches_depth = masked_patches[:, 3:, :, :] # Shape: (num_patches, 1, patch_size, patch_size)

    # Denormalize masked patches
    masked_patches_rgb_denorm = denormalize_RGB(masked_patches_rgb).clamp(0, 1).numpy()
    masked_patches_depth_denorm = masked_patches_depth.numpy()
    # Denormalize masked depth patches
    masked_patches_depth_denorm = denormalize_depth(masked_patches_depth, depth_mean, depth_std)
    masked_patches_depth_denorm = masked_patches_depth_denorm.squeeze(1)

    # Normalize masked depth patches for visualization
    depth_min = masked_patches_depth_denorm.min()
    depth_max = masked_patches_depth_denorm.max()
    masked_patches_depth_viz = (masked_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)

    # Assemble masked patches with gaps
    assembled_masked_rgb = assemble_patches_with_gaps(masked_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_masked_depth = assemble_patches_with_gaps(masked_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)

    # Reconstructed images
    #reconstructed_rgb_denorm = denormalize_RGB(reconstructed_image[0].detach().cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
    reconstructed_rgb_denorm = reconstructed_image[0].detach().cpu()
    reconstructed_depth_map = reconstructed_depth[0, 0].detach().cpu().numpy()
    reconstructed_depth_map = reconstructed_depth_map * depth_std + depth_mean
    print(f"7-Reconstructed RGB image shape: {reconstructed_rgb_denorm.shape}")
    print(f"7-Reconstructed Depth map shape: {reconstructed_depth_map.shape}")
    # Normalize reconstructed depth map for visualization
    recon_depth_min = reconstructed_depth_map.min()
    recon_depth_max = reconstructed_depth_map.max()
    reconstructed_depth_map_viz = (reconstructed_depth_map - recon_depth_min) / (recon_depth_max - recon_depth_min + 1e-8)

    # Create depth images using matplotlib and save them to buffers
    depth_images = {}
    # Original Depth Map
    fig1 = plt.figure()
    plt.imshow(np.squeeze(original_depth_map_viz), cmap='viridis')
    plt.axis('off')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    buf1.seek(0)
    depth_images[f'{prefix} Original Depth Map'] = wandb.Image(Image.open(buf1), caption='Original Depth Map')

    # Assembled Original Depth Patches
    fig2 = plt.figure()
    plt.imshow(np.squeeze(assembled_original_depth), cmap='viridis')
    plt.axis('off')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    buf2.seek(0)
    depth_images[f'{prefix} Assembled Original Depth Patches'] = wandb.Image(Image.open(buf2), caption='Original Depth Patches')

    # Masked Depth Map
    fig3 = plt.figure()
    plt.imshow(assembled_masked_depth, cmap='viridis')
    plt.axis('off')
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig3)
    buf3.seek(0)
    depth_images[f'{prefix} Masked Depth Map'] = wandb.Image(Image.open(buf3), caption='Masked Depth Map')

    # Reconstructed Depth Map
    fig4 = plt.figure()
    plt.imshow(reconstructed_depth_map_viz, cmap='viridis')
    plt.axis('off')
    buf4 = io.BytesIO()
    plt.savefig(buf4, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig4)
    buf4.seek(0)
    depth_images[f'{prefix} Reconstructed Depth Map'] = wandb.Image(Image.open(buf4), caption='Reconstructed Depth Map')

    # Log images to WandB
    wandb.log({
        f'{prefix} Original RGB Image': wandb.Image(original_rgb_image),
        f'{prefix} Assembled Original RGB Patches': wandb.Image(assembled_original_rgb),
        #f'{prefix} Masked RGB Image': wandb.Image(assembled_masked_rgb),
        f'{prefix} Reconstructed RGB Image': wandb.Image(reconstructed_rgb_denorm),
        **depth_images,
        'Epoch': epoch,
        'Batch': batch_idx
    })


def new_log_visualizations(rgb_frames, depth_maps, reconstructed_image, reconstructed_depth, rgb_masks, depth_masks, epoch, batch_idx, prefix='Train'):
    '''
    Logs visualizations to WandB.
    Args:
        rgb_frames: input images (batch_size, 3, T, H, W)
        depth_maps: input depth maps (batch_size, 1, T, H, W)
        reconstructed_image: reconstructed RGB images (batch_size, 3, T, H, W)
        reconstructed_depth: reconstructed depth maps (batch_size, 1, T, H, W)
        rgb_masks: masks used during training for rgb frames (batch_size, 3, T, H, W)
        depth_masks: masks used during training for depth maps (batch_size, 1, T, H, W)
        epoch: current epoch
        batch_idx: current batch index
        prefix: 'Train' or 'Validation'
    '''
    rank = dist.get_rank()
    if rank != 0:
        return
    
    depth_mean = config['data']['depth_stats']['mean']
    depth_std = config['data']['depth_stats']['std']
    img_size = config['model']['img_size']
    patch_size = config['model']['patch_size']
    
    print(f"1-Original RGB frames shape: {rgb_frames.shape}")
    print(f"1-Original Depth maps shape: {depth_maps.shape}")
    print(f"1-RGB masks shape: {rgb_masks.shape}")
    print(f"1-Depth masks shape: {depth_masks.shape}")
    
    # Select the first sample in the batch
    original_image = rgb_frames[0].detach().cpu()  # Shape: (3, T, H, W)
    original_depth = depth_maps[0].detach().cpu()  # Shape: (1, T, H, W)
    rgb_mask = rgb_masks[0].detach().cpu()         # Shape: (3, T, H, W)
    depth_mask = depth_masks[0].detach().cpu()     # Shape: (1, T, H, W)
    
    print(f"2-First image in batch shape: {original_image.shape}")
    print(f"2-First depth map in batch shape: {original_depth.shape}")
    print(f"2-First RGB mask shape: {rgb_mask.shape}")
    print(f"2-First Depth mask shape: {depth_mask.shape}")
    
    # Select the first frame in the sequence
    original_image = original_image[:, 0]  # Shape: (3, H, W)
    original_depth = original_depth[:, 0]  # Shape: (1, H, W)
    rgb_mask = rgb_mask[:, 0]              # Shape: (3, H, W)
    depth_mask = depth_mask[:, 0]          # Shape: (1, H, W)
    
    print(f"3-Original RGB image shape: {original_image.shape}")
    print(f"3-Original Depth map shape: {original_depth.shape}")
    print(f"3-RGB mask shape: {rgb_mask.shape}")
    print(f"3-Depth mask shape: {depth_mask.shape}")
    
    # Same for reconstructed image
    reconstructed_image = reconstructed_image[0].detach().cpu()[:, 0]  # Shape: (3, H, W)
    reconstructed_depth = reconstructed_depth[0].detach().cpu()[0, 0]  # Shape: (H, W)
    
    print(f"4-Reconstructed RGB image shape: {reconstructed_image.shape}")
    print(f"4-Reconstructed Depth map shape: {reconstructed_depth.shape}")
    
    # Denormalize and normalize original RGB image
    original_rgb_image = original_image  # Shape: (3, H, W)
    
    # Denormalize depth map
    original_depth_map = original_depth.numpy() * depth_std + depth_mean
    original_depth_map_viz = (original_depth_map - original_depth_map.min()) / (original_depth_map.max() - original_depth_map.min() + 1e-8)
    
    # Extract patches
    original_patches_rgb = extract_patches(original_image, patch_size)  # Shape: (num_patches, 3, patch_size, patch_size)
    original_patches_depth = extract_patches(original_depth, patch_size)  # Shape: (num_patches, 1, patch_size, patch_size)
    original_patches = torch.cat([original_patches_rgb, original_patches_depth], dim=1)  # Shape: (num_patches, 4, patch_size, patch_size)
    
    # Denormalize patches
    original_patches_rgb_denorm = original_patches_rgb.numpy()
    original_patches_depth_denorm = original_patches_depth.numpy() * depth_std + depth_mean
    original_patches_depth_denorm = original_patches_depth_denorm.squeeze(1)  # Shape: (num_patches, patch_size, patch_size)
    
    # Normalize depth patches for visualization
    depth_min = original_patches_depth_denorm.min()
    depth_max = original_patches_depth_denorm.max()
    original_patches_depth_viz = (original_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)
    
    # Assemble patches with gaps
    gap_size = 2
    num_patches_per_row = img_size // patch_size
    assembled_original_rgb = assemble_patches_with_gaps(original_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_original_depth = assemble_patches_with_gaps(original_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)
    
    print(f"6-Assembled original RGB image shape: {assembled_original_rgb.shape}")
    print(f"6-Assembled original Depth map shape: {assembled_original_depth.shape}")
    
    # Visualize masks
    rgb_mask_viz = rgb_mask.permute(1, 2, 0).numpy()  # Shape: (H, W, 3)
    depth_mask_viz = depth_mask.squeeze(0).numpy()    # Shape: (H, W)
    
    # Normalize reconstructed depth map for visualization
    reconstructed_depth_map_viz = (reconstructed_depth - reconstructed_depth.min()) / (reconstructed_depth.max() - reconstructed_depth.min() + 1e-8)
    
    # Create depth images using matplotlib and save them to buffers
    depth_images = {}
    # Original Depth Map
    fig1 = plt.figure()
    plt.imshow(np.squeeze(original_depth_map_viz), cmap='viridis')
    plt.axis('off')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    buf1.seek(0)
    depth_images[f'{prefix} Original Depth Map'] = wandb.Image(Image.open(buf1), caption='Original Depth Map')
    
    # Assembled Original Depth Patches
    fig2 = plt.figure()
    plt.imshow(np.squeeze(assembled_original_depth), cmap='viridis')
    plt.axis('off')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    buf2.seek(0)
    depth_images[f'{prefix} Assembled Original Depth Patches'] = wandb.Image(Image.open(buf2), caption='Original Depth Patches')
    
    # Reconstructed Depth Map
    fig3 = plt.figure()
    plt.imshow(reconstructed_depth_map_viz, cmap='viridis')
    plt.axis('off')
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig3)
    buf3.seek(0)
    depth_images[f'{prefix} Reconstructed Depth Map'] = wandb.Image(Image.open(buf3), caption='Reconstructed Depth Map')
    
    # Log images to WandB
    wandb.log({
        f'{prefix} Original RGB Image': wandb.Image(original_rgb_image),
        f'{prefix} Assembled Original RGB Patches': wandb.Image(assembled_original_rgb),
        f'{prefix} RGB Mask': wandb.Image(rgb_mask_viz),
        f'{prefix} Depth Mask': wandb.Image(depth_mask_viz),
        f'{prefix} Reconstructed RGB Image': wandb.Image(reconstructed_image),
        **depth_images,
        'Epoch': epoch,
        'Batch': batch_idx
    })
