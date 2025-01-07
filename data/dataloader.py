import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random

# Import necessary modules and libraries:
# - os: For interacting with the operating system (e.g., file paths).
# - cv2: OpenCV library for image and video processing.
# - torch: PyTorch library for tensor computations and neural networks.
# - torch.utils.data: Provides tools for dataset handling and data loading.
# - torchvision.transforms: Common image transformations.
# - matplotlib.pyplot: For plotting and visualization.
# - numpy: Numerical computations.
# - warnings: To manage warning messages.

# Adjust the Python path to include the 'depth_anything_v2' directory.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../depth_anything_v2'))

# Import the Depth Anything  V2 model from the specified module.
from depth_anything_v2.dpt import DepthAnythingV2

# Load configuration settings from a YAML file (assuming you have a 'config.yaml' file).
import yaml
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configure the device to be used for computation.
# Use GPU ('cuda') if available; otherwise, fall back to CPU ('cpu').
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define model configurations for the Depth Anything V2 model.
# The 'vitl' configuration specifies the model architecture and parameters.
#model_configs = {
#    'vitl': {
#        'encoder': 'vitl',               # Use the 'vitl' encoder architecture.
#        'features': 256,                 # Number of feature maps.
#        'out_channels': [256, 512, 1024, 1024]  # Output channels at different layers.
#    }
#}

# Initialize the Depth Anything V2 model with the specified configuration.
#depth_model = DepthAnythingV2(**model_configs['vitl'])

# Suppress future warnings during model loading.
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore", category=FutureWarning)
    # Load the pre-trained model weights from the checkpoint specified in the configuration.
#    depth_model.load_state_dict(torch.load(config['data']['depth_model_checkpoint'], map_location=device))

# Move the model to the configured device (GPU or CPU) and set it to evaluation mode.
#depth_model = depth_model.to(device).eval()

# Define a custom dataset class that inherits from PyTorch's Dataset class.
class VideoFrameDataset(Dataset):
    def __init__(self, video_folder, transform=None, depth_model=None, num_frames=32, frame_interval=1, single_video_path=None):
        """
        Initialize the dataset with the specified parameters.

        Parameters:
        - video_folder (str): Path to the folder containing video files.
        - transform (callable, optional): Transformations to apply to each frame.
        - depth_model (nn.Module, optional): Pre-loaded depth estimation model.
        - num_frames (int): Number of frames to sample per sequence.
        - frame_interval (int): Interval at which frames are sampled (e.g., every nth frame).
        """
        self.video_folder = video_folder          # Store the video folder path.
        self.transform = transform                # Store the transformation function.
        self.depth_model = depth_model            # Store the depth estimation model.
        self.num_frames = num_frames              # Number of frames per sequence.
        self.frame_interval = frame_interval      # Interval at which frames are sampled.

        # Initialize data structures for video paths and frame counts.
        self.video_paths = []                     # List to store video file paths.
        self.video_frame_counts = {}              # Dictionary mapping video paths to frame counts.
        self.video_sequence_indices = []          # List to store (video_path, sequence_start_frame_idx) tuples.
        self.video_files = []                     # List to store video files
        
        # Initialize single_video_path to whatever is passed in
        self.single_video_path = single_video_path
        
        # Build the list of video paths and frame counts.
        self._build_video_list()

        # Build the list of sequences for each video.
        self._build_sequence_indices()

    def _build_video_list(self):
        """
        Build a list of video file paths and their corresponding frame counts.
        """
        #If single_video_path is provided, use that instead of the video_folder
        if self.single_video_path:
            if os.path.isfile(self.single_video_path):
                self.video_files = [self.single_video_path]
        
        else:
            # Create a list of full paths to video files in the video folder.
            self.video_files = [
                os.path.join(self.video_folder, f) for f in os.listdir(self.video_folder)
                if f.endswith(('.mp4', '.avi', '.mov'))  # Include only specified video file extensions.
            ]

        # Iterate over each video file to process its frames.
        for video_path in self.video_files:
            cap = cv2.VideoCapture(video_path)  # Open the video file using OpenCV.

            if not cap.isOpened():
                # If the video cannot be opened, print a warning and skip to the next video.
                print(f"Warning: Cannot open video {video_path}")
                continue

            # Retrieve the total number of frames in the video.
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Release the video capture object.
            cap.release()

            if frame_count == 0:
                # If the video has zero frames, skip it.
                continue

            # Store the video path and frame count.
            self.video_paths.append(video_path)
            self.video_frame_counts[video_path] = frame_count

        # Debugging print statements.
        print(f"Total number of videos: {len(self.video_paths)}")

    def _build_sequence_indices(self):
        """
        Build a list of (video_path, sequence_start_frame_idx) tuples for sampling sequences.
        Sequences are non-overlapping and start from the beginning of the video.
        """
        for video_path in self.video_paths:
            total_frames = self.video_frame_counts[video_path]
            # Adjust total frames based on frame_interval.
            effective_total_frames = total_frames // self.frame_interval

            # Calculate the number of sequences that can be formed.
            num_sequences = effective_total_frames // self.num_frames

            # Generate start indices for each sequence.
            for seq_idx in range(num_sequences):
                start_frame_idx = seq_idx * self.num_frames * self.frame_interval
                self.video_sequence_indices.append((video_path, start_frame_idx))

        # Debugging print statements.
        print(f"Total number of sequences: {len(self.video_sequence_indices)}")

    def __len__(self):
        """
        Return the total number of sequences in the dataset.
        """
        return len(self.video_sequence_indices)

    def __getitem__(self, idx):
        """
        Retrieve the data for a single sequence given its index.

        Parameters:
        - idx (int): Index of the sequence to retrieve.

        Returns:
        - frames (Tensor): Tensor of frames of shape (CHANNELS, TIME, HEIGHT, WIDTH).
        - frame_patches (Tensor): Tensor of image patches of shape (TIME, NUM_PATCHES, CHANNELS, PATCH_SIZE, PATCH_SIZE).
        - depth_maps (Tensor): Tensor of depth maps of shape (TIME, 1, HEIGHT, WIDTH).
        - depth_patches (Tensor): Tensor of depth map patches of shape (TIME, NUM_PATCHES, 1, PATCH_SIZE, PATCH_SIZE).
        - torch.tensor(idx) (Tensor): Tensor containing the sequence index.
        """
        # Retrieve the video path and start frame index from the list.
        video_path, start_frame_idx = self.video_sequence_indices[idx]

        # Debugging prints.
        print(f"Processing sequence idx: {idx}")
        print(f"Video path: {video_path}")
        print(f"Start frame index: {start_frame_idx}")

        # Generate the list of frame indices for the sequence.
        frame_indices = [start_frame_idx + i * self.frame_interval for i in range(self.num_frames)]

        # Initialize lists to store frames, depth maps, and patches.
        frames_list = []
        depth_maps_list = []
        frame_patches_list = []
        depth_patches_list = []

        # Read and process each sampled frame.
        for frame_idx in frame_indices:
            # Read the frame.
            frame = self._read_frame(video_path, frame_idx)

            # Create a copy of the original frame for depth map computation.
            original_frame = frame.copy()

            if self.transform:
                # Apply the specified transformations to the frame (e.g., resizing, converting to tensor).
                frame = self.transform(frame)  # Resulting tensor shape: (3, H, W)

            # Compute the depth map for the original frame.
            depth_map = self._get_depth_map(original_frame)

            # Resize the depth map to match the frame size (e.g., 224x224).
            depth_map = cv2.resize(depth_map, (frame.shape[2], frame.shape[1]))

            # Convert the depth map to a tensor and add a channel dimension.
            depth_map = torch.from_numpy(depth_map).unsqueeze(0)  # Shape: (1, H, W)

            # Extract patches from the transformed frame.
            frame_patches = self._get_patches(frame)  # Shape: (NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE)

            # Extract patches from the depth map.
            depth_patches = self._get_patches(depth_map)  # Shape: (NUM_PATCHES, 1, PATCH_SIZE, PATCH_SIZE)

            # Append to the lists.
            frames_list.append(frame)
            depth_maps_list.append(depth_map)
            frame_patches_list.append(frame_patches)
            depth_patches_list.append(depth_patches)

        # Stack frames and patches along the TIME dimension.
        frames = torch.stack(frames_list, dim=1)  # Shape: (3, TIME, H, W)
        depth_maps = torch.stack(depth_maps_list, dim=1)  # Shape: (1, TIME, H, W)
        frame_patches = torch.stack(frame_patches_list, dim=0)  # Shape: (TIME, NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE)
        depth_patches = torch.stack(depth_patches_list, dim=0)  # Shape: (TIME, NUM_PATCHES, 1, PATCH_SIZE, PATCH_SIZE)

        # Debugging prints.
        print(f"Frames tensor shape: {frames.shape}")
        print(f"Depth maps tensor shape: {depth_maps.shape}")
        print(f"Frame patches tensor shape: {frame_patches.shape}")
        print(f"Depth patches tensor shape: {depth_patches.shape}")

        # Return the processed data along with the sequence index.
        return frames, frame_patches, depth_maps, depth_patches, torch.tensor(idx)

    def _read_frame(self, video_path, frame_idx):
        """
        Read a specific frame from a video file.

        Parameters:
        - video_path (str): Path to the video file.
        - frame_idx (int): Index of the frame to read.

        Returns:
        - frame (ndarray): The read frame in RGB format.
        """
        # Open the video file using OpenCV.
        cap = cv2.VideoCapture(video_path)

        # Set the position of the video to the desired frame index.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame at the current position.
        ret, frame = cap.read()

        # Release the video capture object.
        cap.release()

        if not ret or frame is None:
            # If the frame cannot be read, raise an error.
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        # Convert the frame from BGR (OpenCV default) to RGB color space.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Return the RGB frame.
        return frame

    def _get_depth_map(self, image):
        """
        Compute the depth map for a given image using the depth estimation model.

        Parameters:
        - image (ndarray): Input image in RGB format.

        Returns:
        - depth (ndarray): Normalized depth map of shape (H, W) with values in [0, 1].
        """
        # Convert the image to BGR format as expected by the depth model.
        raw_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Disable gradient computation for efficiency.
        with torch.no_grad():
            # Infer the depth map using the depth estimation model.
            depth = self.depth_model.infer_image(raw_img)

        # Normalize the depth map to the range [0, 1].
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Convert the depth map to a 32-bit floating-point numpy array.
        depth = depth.astype(np.float32)

        # Return the normalized depth map.
        return depth  # Shape: (H, W)

    def _get_patches(self, tensor):
        """
        Split a tensor into non-overlapping patches of size 16x16.

        Parameters:
        - tensor (Tensor): Input tensor of shape (C, H, W).

        Returns:
        - patches (Tensor): Tensor containing patches of shape (NUM_PATCHES, C, 16, 16).
        """
        # Use the unfold method to extract patches along the height and width dimensions.
        patches = tensor.unfold(1, 16, 16).unfold(2, 16, 16)
        # The shape of patches is now (C, NUM_PATCHES_H, NUM_PATCHES_W, PATCH_SIZE_H, PATCH_SIZE_W).

        # Rearrange the dimensions to bring the patch indices to the first two dimensions.
        patches = patches.permute(1, 2, 0, 3, 4)
        # The shape of patches is now (NUM_PATCHES_H, NUM_PATCHES_W, C, 16, 16).

        # Flatten the patch grid into a single dimension.
        patches = patches.contiguous().view(-1, tensor.shape[0], 16, 16)
        # The final shape of patches is (NUM_PATCHES, C, 16, 16).

        # Return the tensor containing the patches.
        return patches



if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt

    # Import the Depth Anything  V2 model from the specified module.
    from depth_anything_v2.dpt import DepthAnythingV2

    # Load configuration settings from a YAML file (assuming you have a 'config.yaml' file).
    import yaml
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Configure the device to be used for computation.
    # Use GPU ('cuda') if available; otherwise, fall back to CPU ('cpu').
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define model configurations for the Depth Anything V2 model.
    # The 'vitl' configuration specifies the model architecture and parameters.
    model_configs = {
        'vitl': {
            'encoder': 'vitl',               # Use the 'vitl' encoder architecture.
            'features': 256,                 # Number of feature maps.
            'out_channels': [256, 512, 1024, 1024]  # Output channels at different layers.
        }
    }

    # Initialize the Depth Anything V2 model with the specified configuration.
    depth_model = DepthAnythingV2(**model_configs['vitl'])

    # Suppress future warnings during model loading.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # Load the pre-trained model weights from the checkpoint specified in the configuration.
        depth_model.load_state_dict(torch.load(config['data']['depth_model_checkpoint'], map_location=device))

    # Move the model to the configured device (GPU or CPU) and set it to evaluation mode.
    depth_model = depth_model.to(device).eval()

        
    # Set the random seed for reproducibility.
    random.seed(24)
    torch.manual_seed(24)

    # Define the transformation pipeline to apply to each frame.
    transform = transforms.Compose([
        transforms.ToPILImage(),         # Convert the frame to a PIL Image.
        transforms.Resize((224, 224)),   # Resize the image to 224x224 pixels.
        transforms.ToTensor(),           # Convert the image to a PyTorch tensor with shape (3, 224, 224).
    ])

    # Define the number of frames to sample per sequence.
    NUM_FRAMES = 16  # T = 32

    # Define the frame interval (e.g., take every nth frame).
    FRAME_INTERVAL = 1  # Take one frame every 4 frames.

    # Specify the path to the folder containing video files.
    video_folder = config['data']['finevideo_path'] + '/sports_videos'

    # Check if the specified video folder exists.
    if not os.path.exists(video_folder):
        video_folder = '/data/datasets/finevideo/sports_videos'
        single_video_path = '/data/datasets/finevideo/sports_videos/circle_animation.mp4'
        if not os.path.exists(video_folder):
            # If the folder does not exist, raise a FileNotFoundError.
            raise FileNotFoundError(f"The specified video folder does not exist: {video_folder}")


    # Create an instance of the VideoFrameDataset with the given parameters.
    dataset = VideoFrameDataset(
        video_folder=video_folder,       # Path to the video folder.
        transform=transform,             # Transformation to apply to each frame.
        depth_model=depth_model,         # Depth estimation model.
        num_frames=NUM_FRAMES,           # Number of frames per sequence.
        frame_interval=FRAME_INTERVAL,    # Frame interval for sampling.,
        single_video_path=single_video_path
    )

    # Print dataset statistics to provide insight into the dataset composition.
    print("Dataset Statistics:")
    print(f"Total number of sequences: {len(dataset)}")

    # Set the batch size, which is the number of sequences per batch.
    batch_size = 4  # Adjust this value based on your memory constraints and requirements.

    # Create a DataLoader using the dataset.
    dataloader = DataLoader(
        dataset,                         # The dataset from which to load data.
        batch_size=batch_size,           # Number of sequences per batch.
        shuffle=True,                    # Shuffle the data at every epoch.
        num_workers=0,                   # Number of subprocesses to use for data loading.
    )

    # Iterate over the DataLoader to process batches of data.
    for batch_idx, batch in enumerate(dataloader):
        
        # Print information about the current batch.
        print(f"Batch {batch_idx}:")
        #Print shape of the batch
        print(f"Batch {batch_idx} has shapes: \nframes: {batch[0].shape},\nframe_patches: {batch[1].shape},\ndepth_maps: {batch[2].shape}, \ndepth_patches: {batch[3].shape}, \ntorch.tensor(idx): {batch[4].shape}")
        
        # Unpack the batch into individual components.
        frames, frame_patches, depth_maps, depth_patches, sequence_indices = batch
        
        print(f"Frames shape: {frames.shape}")             # Expected shape: (batch_size, 3, TIME, H, W)
        print(f"Frame patches shape: {frame_patches.shape}")  # Expected shape: (batch_size, TIME, NUM_PATCHES, 3, 16, 16)
        print(f"Depth maps shape: {depth_maps.shape}")     # Expected shape: (batch_size, TIME, 1, H, W)
        print(f"Depth patches shape: {depth_patches.shape}")  # Expected shape: (batch_size, TIME, NUM_PATCHES, 1, 16, 16)
        print(f"Sequence indices: {sequence_indices}")

        # Visualize the frames of the first sequence in the batch.
        first_sequence_frames = frames[0]  # Shape: (3, TIME, H, W)
        first_sequence_index = sequence_indices[0].item()

        # Permute dimensions for visualization.
        frames_np = first_sequence_frames.permute(1, 2, 3, 0).numpy()  # Shape: (TIME, H, W, 3)

        # Create a figure to display the frames.
        fig, axs = plt.subplots(4, 8, figsize=(16, 8))  # Adjust grid size based on NUM_FRAMES
        axs = axs.flatten()
        for idx in range(NUM_FRAMES):
            ax = axs[idx]
            ax.imshow(frames_np[idx])
            ax.axis('off')
            ax.set_title(f"Frame {idx}")
        plt.suptitle(f"Frames of Sequence {first_sequence_index} in Batch {batch_idx}")
        plt.tight_layout()
        plt.savefig(f"video_frames_batch_{batch_idx}.png")
        plt.close(fig)

        # Visualize the depth maps of the first sequence in the batch.
        first_sequence_depth_maps = depth_maps[0]  # Shape: (TIME, 1, H, W)
        depth_maps_np = first_sequence_depth_maps.squeeze(1).numpy()  # Shape: (TIME, H, W)

        fig, axs = plt.subplots(4, 8, figsize=(16, 8))
        axs = axs.flatten()
        for idx in range(NUM_FRAMES):
            ax = axs[idx]
            ax.imshow(depth_maps_np[idx], cmap='magma')
            ax.axis('off')
            ax.set_title(f"Depth {idx}")
        plt.suptitle(f"Depth Maps of Sequence {first_sequence_index} in Batch {batch_idx}")
        plt.tight_layout()
        plt.savefig(f"video_depth_maps_batch_{batch_idx}.png")
        plt.close(fig)

        # Visualize the patches of the first frame of the first sequence.
        first_frame_patches = frame_patches[0, 0]  # Shape: (NUM_PATCHES, 3, 16, 16)

        fig, axs = plt.subplots(14, 14, figsize=(20, 20))  # Assuming 196 patches (14x14).
        for i in range(14):
            for j in range(14):
                ax = axs[i, j]
                patch_idx = i * 14 + j
                patch = first_frame_patches[patch_idx].permute(1, 2, 0).numpy()
                ax.imshow(patch)
                ax.axis('off')
        plt.suptitle(f"Patches of First Frame of Sequence {first_sequence_index} in Batch {batch_idx}")
        plt.tight_layout()
        plt.savefig(f"video_frame_patches_batch_{batch_idx}.png")
        plt.close(fig)

        # For testing purposes, limit the processing to 1 batch.
        if batch_idx == 4:
            break
