import os
from tqdm import tqdm
from datasets import load_dataset

from datasets import load_dataset
import json

# Load the dataset in streaming mode
dataset = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)

# Define a filter function to select only Sports videos
def is_sports(sample):
    return sample['json']['content_parent_category'] == 'Sports'

# Apply the filter to the dataset
sports_dataset = filter(is_sports, dataset)

# Create directories to save videos and metadata
import os
os.makedirs("sports_videos", exist_ok=True)
os.makedirs("sports_metadata", exist_ok=True)

# Iterate over the filtered dataset and save the samples
for idx, sample in enumerate(sports_dataset):
    # Print some information
    print(f"Processing sample {idx}")

    # Save the video file
    video_filename = f"sports_videos/sample_{idx}.mp4"
    with open(video_filename, 'wb') as video_file:
        video_file.write(sample['mp4'])

    # Save the json metadata
    json_filename = f"sports_metadata/sample_{idx}.json"
    with open(json_filename, 'w') as json_file:
        json.dump(sample['json'], json_file)


