#pretraining/pretraining.py

'''Import Libraries'''
import os
import torch
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import torch.optim as optim
import wandb
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from tubeletembed import TubeletEmbed

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from videomae_cross_modal import CrossModalVideoMAE

sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))
from dataloader import VideoFrameDataset


def main():
    '''Set Up Configuration'''
    # Load configuration
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    wandb_config = config['wandb']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = CrossModalVideoMAE(model_config)
    
    #Send the model to the device
    model = model.to(device)

    '''Setup DepthAnythingV2 Model'''
    sys.path.append(os.path.join(os.path.dirname(__file__), '../depth_anything_v2'))
    from depth_anything_v2.dpt import DepthAnythingV2
    

    model_configs_depth = {
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        }
    }

    depth_model = DepthAnythingV2(**model_configs_depth['vitl'])

    #Set it into evaluation mode
    with torch.no_grad():
        depth_model.load_state_dict(torch.load(data_config['depth_model_checkpoint'], map_location=device))
    depth_model.eval()
    

    '''Set Up Dataset and Dataloader'''

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((model_config['img_size'], model_config['img_size'])),
        transforms.ToTensor()
    ])

    # Initialize the dataset
    video_folder = data_config['finevideo_path'] if os.path.exists(data_config['finevideo_path']) else '/home/ndelafuente/VD-MAE/sports_videos'

    dataset = VideoFrameDataset(
            video_folder = video_folder,
            transform = transform,
            depth_model=depth_model,
            num_frames = model_config['num_frames'],
            frame_interval = model_config['frame_interval']
        )


    # Initialize the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        pin_memory=True 
    )
    

    '''Set Up Optimizer and Scheduler'''
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay']
    )

    #Define lr scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config['num_epochs'])

    '''Set Up Wandb'''
    wandb.init(entity=wandb_config['entity'], project=wandb_config['project'], name=wandb_config['name'])


    '''Training Loop'''
    num_epochs = training_config['num_epochs']
    mask_ratio = training_config['mask_ratio']

    scaler = torch.cuda.amp.GradScaler('cuda') #Mixed precision training for optimization

    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()
        # Set the total losses to 0
        total_loss = 0.0
        total_rgb_loss = 0.0
        total_depth_loss = 0.0
        
        # Set the progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        #Iterate over the dataloader
        for batch_idx, batch in progress_bar:
            frames, _, depth_maps, _, _ = batch
            #Send data to device
            frames = frames.to(device)
            depth_maps = depth_maps.to(device)
            
            # Batch size
            batch_size = frames.size(0) 
            assert batch_size == frames.size(0) == depth_maps.size(0), "Batch size mismatch"
            
            #Get number of patches
            num_tubelets = model.rgb_tubelet_embed.num_tubelets
            
            #Generate the masks
            masks = torch.rand(training_config['batch_size'], num_tubelets).to(device) < mask_ratio
            
            #Zero the gradients
            optimizer.zero_grad()
            
            #Forward pass
            with torch.cuda.amp.autocast():
                #Get the reconstructions for the frames and depth maps
                rgb_recon, depth_recon = model(frames, depth_maps, masks)
                #Calculate the losses
                rgb_loss, depth_loss, loss = model.compute_loss(frames, depth_maps, rgb_recon, depth_recon, masks)
                
            #Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #Update the total losses
            total_loss += loss.item()
            total_rgb_loss += rgb_loss.item()
            total_depth_loss += depth_loss.item()
            
            #Log metrics to wandb every N batches
            if batch_idx % wandb_config['log_interval'] == 0:
                wandb.log({
                    'loss': total_loss / (batch_idx + 1),
                    'rgb_loss': total_rgb_loss / (batch_idx + 1),
                    'depth_loss': total_depth_loss / (batch_idx + 1)
                })
                
        #Step the scheduler
        scheduler.step()
        
        #Average the total losses
        avg_loss = total_loss / len(dataloader)
        avg_rgb_loss = total_rgb_loss / len(dataloader)
        avg_depth_loss = total_depth_loss / len(dataloader)
        
        #Log epoch metrics to wandb and console
        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'rgb_loss': avg_rgb_loss,
            'depth_loss': avg_depth_loss
        })
        
        #Save the model checkpoint if the loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, os.path.join(config['checkpoint_dir'], 'model_at_epoch_{}.pth'))
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f} RGB Loss: {avg_rgb_loss:.4f} Depth Loss: {avg_depth_loss:.4f}")
            

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
