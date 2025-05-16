import boto3
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dataset_train', type=str, required=True)
parser.add_argument('--output_dataset_validation', type=str, required=True)
parser.add_argument('--output_dataset_test', type=str, required=True)

args = parser.parse_args()

s3 = boto3.client('s3',
    # Use the internal ip for minio here
    endpoint_url='http://istio-ingressgateway.istio-system.svc.cluster.local',
    # Minio username
    aws_access_key_id='minio',
    # Minio password
    aws_secret_access_key='minio123',
    # Region, You can keep this us-east-1
    region_name='us-east-1'
)     

bucket_name = "ml-data"

files = ["train.csv", "validation.csv", "test.csv"]
# Download and load files


output_paths = {
    "train": args.output_dataset_train,
    "validation": args.output_dataset_validation,
    "test": args.output_dataset_test
}

for split in ["train", "validation", "test"]:
    local_path = f"/tmp/{split}.csv"
    s3.download_file(bucket_name, f"{split}.csv", local_path)
    df = pd.read_csv(local_path)
    df.to_csv(output_paths[split], index=False)
