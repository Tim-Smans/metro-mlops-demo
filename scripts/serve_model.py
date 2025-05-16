import boto3
import mlflow
from mlflow.tracking import MlflowClient

# MinIO client configuration
s3 = boto3.client(
    "s3",
    endpoint_url="http://istio-ingressgateway.istio-system.svc.cluster.local",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)

# MLflow client
MLFLOW_TRACKING_URI = "http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow/"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Model name
model_name = "mood-machine-model"

# Get the latest model version
latest_version = max(
    [v.version for v in client.search_model_versions(f"name='{model_name}'")]
)

# Retrieve the download path and UUID
latest_path = client.get_model_version_download_uri(model_name, latest_version)
latest_uuid = latest_path.split("/")[3]
run_id = client.get_model_version(model_name, latest_version).run_id


print(f"Latest path: {latest_path}")
print(f"Latest UUID: {latest_uuid}")

# Prefixes
source_prefix = f"{latest_uuid}/artifacts/model/"
latest_prefix = "latest/"

print(f"Expected source prefix: ml-models/{source_prefix}")

# Ensure source_prefix ends with a '/'
if not source_prefix.endswith('/'):
    source_prefix += '/'

# Clean the old 'latest/' directory
objects = s3.list_objects_v2(Bucket="ml-models", Prefix=latest_prefix)
if "Contents" in objects:
    print(f"Cleaning existing 'latest/' prefix...")
    for obj in objects["Contents"]:
        s3.delete_object(Bucket="ml-models", Key=obj["Key"])
else:
    print(f"no existing 'latest/' to clean.")

# Copy all files from the latest model version to the 'latest/' directory
# Copy all files from the latest model version to the 'latest/' directory
response = s3.list_objects_v2(Bucket="ml-models", Prefix=source_prefix)

if "Contents" in response and len(response["Contents"]) > 0:
    print(f"Found {len(response['Contents'])} files under {source_prefix}. Copying...")
    for obj in response["Contents"]:
        src_key = obj["Key"]

        # Make sure we are not copying "empty" directory keys
        if src_key.endswith('/'):
            continue

        dest_key = src_key.replace(source_prefix, latest_prefix)

        print(f"→ Copying {src_key} → {dest_key}")
        
        s3.copy_object(
            Bucket="ml-models",
            CopySource={"Bucket": "ml-models", "Key": src_key},
            Key=dest_key
        )
    print(f"'latest/' now points to {latest_uuid}")
else:
    print(f"No files found under prefix '{source_prefix}'. Nothing copied.")