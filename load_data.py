import os
from datasets import load_dataset

# Create a folder to save the sports videos
save_directory = "sports_videos"
os.makedirs(save_directory, exist_ok=True)

# Load the FineVideo dataset
dataset = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)

# Filter only the sports category videos
sports_videos = (sample for sample in dataset if sample['json']['content_fine_category'] == 'Sports')

# Download and save the sports videos in the specified directory
for sample in sports_videos:
    video_filename = os.path.join(save_directory, f"sports_video_{sample['json']['original_video_filename']}.mp4")
    with open(video_filename, 'wb') as video_file:
        video_file.write(sample['mp4']) 
    print(f"Downloaded {video_filename}")
