import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_train_X', type=str, required=True)
parser.add_argument('--dataset_test_X', type=str, required=True)
parser.add_argument('--dataset_train_y', type=str, required=True)
parser.add_argument('--dataset_test_y', type=str, required=True)
args = parser.parse_args()

X_train = pd.read_csv(args.dataset_train_X)
X_test = pd.read_csv(args.dataset_test_X)
y_train = pd.read_csv(args.dataset_train_y)
y_test = pd.read_csv(args.dataset_test_y)

y_train = y_train.squeeze()
y_test = y_test.squeeze()

# MLflow
mlflow.set_tracking_uri("http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow") # bv. http://mlflow-service.kubeflow:5000
mlflow.set_experiment("iris-experiment")

# Configuring the environmental variables for MinIO
# Force S3 settings to avoid IncompleteBody errors
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT", "http://istio-ingressgateway.istio-system.svc.cluster.local")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("MINIO_ACCESS_KEY", "minio")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("MINIO_SECRET_KEY", "minio123")
os.environ['MLFLOW_S3_UPLOAD_EXTRA_ARGS'] = '{"ACL": "bucket-owner-full-control"}'


with mlflow.start_run():
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_metric("accuracy", acc)

    logged_model = mlflow.sklearn.log_model(model, "iris-model")

    mlflow.register_model(
        model_uri=logged_model.model_uri,
        name="iris-model"
    )