import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings

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

# Import the Depth Anything V2 model from the specified module.
from depth_anything_v2.dpt import DepthAnythingV2

# Load configuration settings from a YAML file (assuming you have a 'config.yaml' file).
import yaml
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configure the device to be used for computation.
# Use GPU ('cuda') if available; otherwise, fall back to CPU ('cpu').
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
    depth_model.load_state_dict(torch.load(config['data']['depth_model_checkpoint'], map_location=DEVICE))

# Move the model to the configured device (GPU or CPU) and set it to evaluation mode.
depth_model = depth_model.to(DEVICE).eval()

# Define a custom dataset class that inherits from PyTorch's Dataset class.
class VideoFrameDataset(Dataset):
    def __init__(self, video_folder, transform=None, depth_model=None, shuffle=True):
        """
        Initialize the dataset with the specified parameters.

        Parameters:
        - video_folder (str): Path to the folder containing video files.
        - transform (callable, optional): Transformations to apply to each frame.
        - depth_model (nn.Module, optional): Pre-loaded depth estimation model.
        """
        self.video_folder = video_folder          # Store the video folder path.
        self.transform = transform                # Store the transformation function.
        self.depth_model = depth_model            # Store the depth estimation model.

        # Initialize statistics for dataset analysis.
        self.total_videos = 0                     # Total number of videos in the dataset.
        self.total_frames_before_sampling = 0     # Total number of frames before sampling.
        self.total_frames_after_sampling = 0      # Total number of frames after sampling.
        self.frames_per_video_before = {}         # Dictionary mapping video paths to frame counts before sampling.
        self.frames_per_video_after = {}          # Dictionary mapping video paths to frame counts after sampling.

        # Initialize data structures for frame indexing.
        self.frame_info = []                      # List to store tuples of (video_path, frame_idx) for each frame.
        self.video_frame_indices = {}             # Dictionary mapping video paths to lists of frame indices.

        # Build the index of frames across all videos in the dataset.
        self._build_frame_index()

    def _build_frame_index(self, shuffle=True):
        """
        Build an index of all frames from all videos in the specified folder.
        This method populates self.frame_info and self.video_frame_indices.
        """
        # Create a list of full paths to video files in the video folder.
        video_files = [
            os.path.join(self.video_folder, f) for f in os.listdir(self.video_folder)
            if f.endswith(('.mp4', '.avi', '.mov'))  # Include only specified video file extensions.
        ]
        idx = 0  # Initialize a global index counter for frames.

        # Update the total number of videos.
        self.total_videos = len(video_files)

        # Iterate over each video file to process its frames.
        for video_path in video_files:
            cap = cv2.VideoCapture(video_path)  # Open the video file using OpenCV.

            if not cap.isOpened():
                # If the video cannot be opened, print a warning and skip to the next video.
                print(f"Warning: Cannot open video {video_path}")
                continue

            # Retrieve the total number of frames in the video.
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Retrieve the frames per second (fps) of the video.
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Release the video capture object.
            cap.release()

            if frame_count == 0:
                # If the video has zero frames, skip it.
                continue

            # Update statistics for frames before sampling.
            self.total_frames_before_sampling += frame_count
            self.frames_per_video_before[video_path] = frame_count

            # Calculate the frame interval to sample frames at 1/5 of the original fps.
            if fps > 0:
                # If fps is available and valid, calculate the frame interval.
                frame_interval = max(1, int(round(fps / 5)))  # Scale down fps to 1/5.
            else:
                # If fps is not available, default to sampling every 5th frame.
                frame_interval = 5
                print(f"Warning: FPS not available for video {video_path}. Assuming frame interval of {frame_interval}.")

            # Initialize a list to store indices for frames in this video.
            indices = []

            # Generate a list of frame indices to sample, starting from 0 to frame_count, stepping by frame_interval.
            frame_indices = list(range(0, frame_count, frame_interval))
            

            # Iterate over the sampled frame indices.
            for frame_idx in frame_indices:
                # Append a tuple of (video_path, frame_idx) to the frame_info list.
                self.frame_info.append((video_path, frame_idx))

                # Append the global frame index to the indices list.
                indices.append(idx)

                # Increment the global index counter.
                idx += 1

            # Update statistics for frames after sampling.
            sampled_frame_count = len(frame_indices)
            self.total_frames_after_sampling += sampled_frame_count
            self.frames_per_video_after[video_path] = sampled_frame_count

            # Map the video path to its list of frame indices in the video_frame_indices dictionary.
            self.video_frame_indices[video_path] = indices
        
        # If specified, shuffle the video paths keeping the frames in the same video together.
        if shuffle:
            #print("Video order before shuffling: ", list(self.video_frame_indices.keys())[:10])
            # Shuffle the video paths
            video_paths = list(self.video_frame_indices.keys())
            random.shuffle(video_paths)
            self.video_frame_indices = {path: self.video_frame_indices[path] for path in video_paths}
            #print("Video order after shuffling: ", list(self.video_frame_indices.keys())[:10])

    def __len__(self):
        """
        Return the total number of frames in the dataset after sampling.
        """
        return len(self.frame_info)

    def __getitem__(self, idx):
        """
        Retrieve the data for a single frame given its index.

        Parameters:
        - idx (int): Index of the frame to retrieve.

        Returns:
        - frame (Tensor): Transformed frame image tensor of shape (3, 224, 224).
        - frame_patches (Tensor): Tensor of image patches of shape (196, 3, 16, 16).
        - depth_map (Tensor): Depth map tensor of shape (1, 224, 224).
        - depth_patches (Tensor): Tensor of depth map patches of shape (196, 1, 16, 16).
        - torch.tensor(frame_idx) (Tensor): Tensor containing the original frame index.
        """
        # Retrieve the video path and frame index from the frame_info list.
        video_path, frame_idx = self.frame_info[idx]

        # Read the specified frame from the video.
        frame = self._read_frame(video_path, frame_idx)

        # Create a copy of the original frame for depth map computation.
        original_frame = frame.copy()

        if self.transform:
            # Apply the specified transformations to the frame (e.g., resizing, converting to tensor).
            frame = self.transform(frame)  # Resulting tensor shape: (3, 224, 224)

        # Compute the depth map for the original frame.
        depth_map = self._get_depth_map(original_frame)

        # Resize the depth map to match the frame size (224x224).
        depth_map = cv2.resize(depth_map, (224, 224))

        # Convert the depth map to a tensor and add a channel dimension.
        depth_map = torch.from_numpy(depth_map).unsqueeze(0)  # Shape: (1, 224, 224)

        # Extract patches from the transformed frame.
        frame_patches = self._get_patches(frame)  # Shape: (196, 3, 16, 16)

        # Extract patches from the depth map.
        depth_patches = self._get_patches(depth_map)  # Shape: (196, 1, 16, 16)

        # Return the processed data along with the frame index.
        return frame, frame_patches, depth_map, depth_patches, torch.tensor(frame_idx)

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
        depth = (depth - depth.min()) / (depth.max() - depth.min())

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
        - patches (Tensor): Tensor containing patches of shape (num_patches, C, 16, 16).
        """
        # Use the unfold method to extract patches along the height and width dimensions.
        patches = tensor.unfold(1, 16, 16).unfold(2, 16, 16)
        # The shape of patches is now (C, num_patches_H, num_patches_W, patch_size_H, patch_size_W).

        # Rearrange the dimensions to bring the patch indices to the first two dimensions.
        patches = patches.permute(1, 2, 0, 3, 4)
        # The shape of patches is now (num_patches_H, num_patches_W, C, 16, 16).

        # Flatten the patch grid into a single dimension.
        patches = patches.contiguous().view(-1, tensor.shape[0], 16, 16)
        # The final shape of patches is (num_patches, C, 16, 16).

        # Return the tensor containing the patches.
        return patches

# Import necessary modules for custom sampling and randomization.
from torch.utils.data.sampler import Sampler
import random

# Define a custom batch sampler to ensure that each batch contains frames from the same video.
class VideoBatchSampler(Sampler):
    def __init__(self, video_frame_indices, batch_size):
        """
        Initialize the batch sampler with the specified parameters.

        Parameters:
        - video_frame_indices (dict): Mapping from video paths to lists of frame indices.
        - batch_size (int): Number of frames per batch.
        - shuffle (bool): Whether to shuffle the frames within each video.
        """
        self.video_frame_indices = video_frame_indices  # Store the mapping of video paths to frame indices.
        self.batch_size = batch_size                    # Store the batch size.

        # Create a list of video paths from the keys of the video_frame_indices dictionary.
        self.video_paths = list(video_frame_indices.keys())


    def __iter__(self):
        """
        Create an iterator that yields batches of frame indices.

        Yields:
        - batch (list): A list of frame indices representing a batch.
        """
        # Iterate over each video path.
        for video_path in self.video_paths:
            # Retrieve the list of frame indices for the current video.
            indices = self.video_frame_indices[video_path]
            
            # Generate batches of indices for the current video.
            for i in range(0, len(indices), self.batch_size):
                # Create a batch by slicing the indices list.
                batch = indices[i:i + self.batch_size]

                # Yield the batch of indices.
                yield batch

    def __len__(self):
        """
        Return the total number of batches across all videos.
        """
        total_batches = 0  # Initialize a counter for the total number of batches.

        # Iterate over the lists of indices for each video.
        for indices in self.video_frame_indices.values():
            # Calculate the number of batches for the current video.
            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size

            # Add the number of batches to the total count.
            total_batches += num_batches

        # Return the total number of batches.
        return total_batches

if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt

    # Set the random seed for reproducibility.
    random.seed(23)
    torch.manual_seed(23)

    # Define the transformation pipeline to apply to each frame.
    transform = transforms.Compose([
        transforms.ToPILImage(),         # Convert the frame to a PIL Image.
        transforms.Resize((224, 224)),   # Resize the image to 224x224 pixels.
        transforms.ToTensor(),           # Convert the image to a PyTorch tensor with shape (3, 224, 224).
    ])

    # Specify the path to the folder containing video files.
    video_folder = '../sports_videos'  # Replace with your actual folder path.

    # Check if the specified video folder exists.
    if not os.path.exists(video_folder):
        # If the folder does not exist, raise a FileNotFoundError.
        raise FileNotFoundError(f"The specified video folder does not exist: {video_folder}")

    # Create an instance of the VideoFrameDataset with the given parameters.
    dataset = VideoFrameDataset(
        video_folder=video_folder,       # Path to the video folder.
        transform=transform,             # Transformation to apply to each frame.
        depth_model=depth_model          # Depth estimation model.
    )

    # Print dataset statistics to provide insight into the dataset composition.
    #print("Dataset Statistics:")
    #print(f"Total number of videos: {dataset.total_videos}")
    #print(f"Total number of frames before sampling: {dataset.total_frames_before_sampling}")
    #print(f"Total number of frames after sampling: {dataset.total_frames_after_sampling}")
    #print("Frames per video before and after sampling:")

    # Iterate over the videos to print per-video frame counts.
    for i, video_path in enumerate(dataset.frames_per_video_before):
        video_name = os.path.basename(video_path)  # Extract the video file name.
        frames_before = dataset.frames_per_video_before[video_path]  # Frames before sampling.
        frames_after = dataset.frames_per_video_after[video_path]    # Frames after sampling.

        #if i % 100 == 0:
            # For large datasets, print statistics every 100 videos to avoid clutter.
            #print(f"- {video_name}: {frames_before} frames before, {frames_after} frames after sampling")

    # Set the batch size, which is the number of frames per batch from the same video.
    batch_size = 256  # Adjust this value based on your memory constraints and requirements.

    # Create an instance of the custom VideoBatchSampler.
    batch_sampler = VideoBatchSampler(
        dataset.video_frame_indices,     # Mapping from video paths to frame indices.
        batch_size=batch_size,           # Number of frames per batch.
    )

    # Create a DataLoader using the dataset and the custom batch sampler.
    dataloader = DataLoader(
        dataset,                         # The dataset from which to load data.
        batch_sampler=batch_sampler,     # The custom batch sampler for batching frames.
        num_workers=0,                   # Number of subprocesses to use for data loading (0 means data will be loaded in the main process).
        #drop_last=True                   # Drop the last incomplete batch if it is smaller than the specified batch size.
    )

    # Iterate over the DataLoader to process batches of data.
    for batch_idx, batch in enumerate(dataloader):
        # Unpack the batch into individual components.
        frames, frame_patches, depth_maps, depth_patches, frame_indices = batch
        
        # get the actual batch size for each batch as last batch may have less frames
        batch_size = frames.shape[0]
        

        # Print information about the current batch.
        print(f"Batch {batch_idx}:")
        print(f"Frames shape: {frames.shape}")             # Expected shape: (batch_size, 3, 224, 224)
        print(f"Frame patches shape: {frame_patches.shape}")  # Expected shape: (batch_size, 196, 3, 16, 16)
        print(f"Depth maps shape: {depth_maps.shape}")     # Expected shape: (batch_size, 1, 224, 224)
        print(f"Depth patches shape: {depth_patches.shape}")  # Expected shape: (batch_size, 196, 1, 16, 16)

        # Calculate the grid size (number of rows and columns)
        grid_size = int(np.ceil(np.sqrt(batch_size)))  # Round up to ensure all images fit

        # --- Visualize the Frames ---
        # Prepare frames for visualization by permuting dimensions and converting to NumPy arrays.
        frames_np = frames.permute(0, 2, 3, 1).numpy()  # Shape: (batch_size, 224, 224, 3)

        # Create a figure with a grid of subplots to display each frame in the batch.
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

        # Flatten the axs array for easy indexing.
        axs = axs.flatten()

        # Iterate over each frame in the batch to display it.
        for idx in range(batch_size):
            ax = axs[idx]                          # Get the appropriate subplot axis.
            ax.imshow(frames_np[idx])              # Display the frame image.
            ax.axis('off')                         # Hide axis ticks and labels.
            ax.set_title(f"Frame {frame_indices[idx].item()}")  # Set the subplot title.

        # Turn off any unused subplots.
        for idx in range(batch_size, grid_size * grid_size):
            axs[idx].axis('off')  # Hide axes without images.

        # Set the overall title for the figure.
        plt.suptitle(f"Frames from the same video in Batch {batch_idx}")

        # Adjust layout to prevent overlap.
        plt.tight_layout()

        # Display the figure containing the frames.
        plt.show()

        # --- Visualize the Depth Maps ---
        # Prepare depth maps for visualization by removing the channel dimension.
        depth_maps_np = depth_maps.squeeze(1).numpy()  # Shape: (batch_size, 224, 224)

        # Create a figure with a grid of subplots to display each depth map in the batch.
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

        # Flatten the axs array for easy indexing.
        axs = axs.flatten()

        # Iterate over each depth map in the batch to display it.
        for idx in range(batch_size):
            ax = axs[idx]                           # Get the appropriate subplot axis.
            ax.imshow(depth_maps_np[idx], cmap='magma')  # Display the depth map using the 'magma' colormap.
            ax.axis('off')                          # Hide axis ticks and labels.
            ax.set_title(f"Depth Map {frame_indices[idx].item()}")  # Set the subplot title.

        # Turn off any unused subplots.
        for idx in range(batch_size, grid_size * grid_size):
            axs[idx].axis('off')  # Hide axes without images.

        # Set the overall title for the figure.
        plt.suptitle(f"Depth Maps from the same video in Batch {batch_idx}")

        # Adjust layout to prevent overlap.
        plt.tight_layout()

        # Display the figure containing the depth maps.
        plt.show()

        # Visualize the patches of the first frame in the batch.
        fig, axs = plt.subplots(14, 14, figsize=(20, 20))  # Create a 14x14 grid of subplots.

        # Iterate over the patch grid to display each patch.
        for i in range(14):
            for j in range(14):
                ax = axs[i, j]
                # Calculate the patch index.
                patch_idx = i * 14 + j

                # Retrieve the patch and permute dimensions for visualization.
                patch = frame_patches[0, patch_idx].permute(1, 2, 0).numpy()

                # Display the patch image.
                ax.imshow(patch)

                # Hide axis ticks and labels.
                ax.axis('off')

        # Set the overall title for the figure.
        plt.suptitle(f"Frame Patches from the same video in Batch {batch_idx}")

        # Adjust layout to prevent overlap.
        plt.tight_layout()

        # Display the figure containing the patches.
        plt.show()

        # For testing purposes, limit the processing to 10 batches.
        if batch_idx == 10:
            break
