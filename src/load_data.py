import boto3
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from io import StringIO

parser = argparse.ArgumentParser()
parser.add_argument('--output_dataset_train_X', type=str, required=True)
parser.add_argument('--output_dataset_test_X', type=str, required=True)
parser.add_argument('--output_dataset_train_y', type=str, required=True)
parser.add_argument('--output_dataset_test_y', type=str, required=True)

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

# Download dataset
obj = s3.get_object(Bucket="ml-data", Key="iris/iris.csv")
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save datasets to the provided paths
X_train.to_csv(args.output_dataset_train_X, index=False)
X_test.to_csv(args.output_dataset_test_X, index=False)
y_train.to_csv(args.output_dataset_train_y, index=False)
y_test.to_csv(args.output_dataset_test_y, index=False)