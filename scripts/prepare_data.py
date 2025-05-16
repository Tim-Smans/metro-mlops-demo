from datasets import load_dataset
import pandas as pd
import os
import boto3

# Label mapping
label_map = {
    0: "anger",
    1: "joy",
    2: "optimism",
    3: "sadness"
}

s3 = boto3.client('s3',
    # Use the external ip for minio here
    endpoint_url='http://localhost',
    # Minio username
    aws_access_key_id='minio',
    # Minio password
    aws_secret_access_key='minio123',
    # Region, You can keep this us-east-1
    region_name='us-east-1'
)

bucket_name = 'ml-data'

# Load the 'Emotion' subset
dataset = load_dataset("tweet_eval", "emotion")

# Convert to dataframes and write to CSV file
for split in ["train", "validation", "test"]:
    df = dataset[split].to_pandas()
    df['label'] = df['label'].map(label_map)
    df.to_csv(f"{split}.csv", index=False)

for split in ['train', 'validation', 'test']:
    s3.upload_file(f"{split}.csv", bucket_name, f"{split}.csv")
