# src/save_iris_to_minio.py
from sklearn.datasets import load_iris
import pandas as pd
import boto3
from io import StringIO

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

# Load the data from sklearn
iris = load_iris(as_frame=True)
# This uses pandas to concatinate 2 columns from the dataset into one
df = pd.concat(
    [
        iris.data, 
        pd.DataFrame(iris.target, columns=["target"])
    ]
    , axis=1
)
# Using StringIO to create a fake csv file to write to minio.
csv_buffer = StringIO()
# Writing the concatinated table into the temporary fake CSV file.
df.to_csv(csv_buffer, index=False)

s3.put_object(Bucket="ml-data", Key="iris/iris.csv", Body=csv_buffer.getvalue())

print("Successfully uploaded data to MinIO")