import boto3
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import joblib
import mlflow
import mlflow.sklearn
import os

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT", "http://istio-ingressgateway.istio-system.svc.cluster.local")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("MINIO_ACCESS_KEY", "minio")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("MINIO_SECRET_KEY", "minio123")
os.environ['MLFLOW_S3_UPLOAD_EXTRA_ARGS'] = '{"ACL": "bucket-owner-full-control"}'

s3 = boto3.client('s3',
    # Use the external ip for minio here
    endpoint_url='http://istio-ingressgateway.istio-system.svc.cluster.local',
    # Minio username
    aws_access_key_id='minio',
    # Minio password
    aws_secret_access_key='minio123',
    # Region, You can keep this us-east-1
    region_name='us-east-1'
)     


parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset_train', type=str, required=True)
parser.add_argument('--input_dataset_validation', type=str, required=True)
parser.add_argument('--input_dataset_test', type=str, required=True)
parser.add_argument('--output_model', type=str, required=True)
args = parser.parse_args()

# MLFlow setup:
MLFLOW_TRACKING_URI = "http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Creating an MLFlow experiment
experiment_name = "mood-machine-experiment"
try:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location="s3://ml-models/"
    )
except mlflow.exceptions.MlflowException:
    pass

print("MLflow Tracking URI:", mlflow.get_tracking_uri())


# Load our datasets from the arguments, taken from MinIO in previous pipeline step.
df_train = pd.read_csv(args.input_dataset_train)
df_val = pd.read_csv(args.input_dataset_validation)
df_test = pd.read_csv(args.input_dataset_test)

# Encoding our labels, ML models work with numeric labels.
# We transform our string labels to numeric labels using the LabelEncoder()
# joy -> 1
# sadness -> 3
le = LabelEncoder()
y_train = le.fit_transform(df_train['label'])
y_val = le.transform(df_val['label'])
y_test = le.transform(df_test['label'])

with mlflow.start_run(experiment_id="1") as run:
# Training pipeline
# We are using a TfidVectorizer to transorm our text to numeric values
# Text: "I feel sad" -> Vector: [0, 0.4, 0.1, 0, 0.9]
    pipeline = Pipeline([
      ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,1))),
      ('clf', LogisticRegression(max_iter=2000, penalty='l2', solver='liblinear', C=10))
    ])
    
    param_grid = {
        'tfidf__max_features': [3000, 5000, 8000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10],
        'clf__penalty': ['l2'],
        'clf__solver': ['liblinear', 'saga']
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', verbose=2, n_jobs=-1)

    pipeline.fit(df_train['text'], y_train)

    grid.fit(df_train['text'], y_train)

    pipeline = grid.best_estimator_
    print("Best params:", grid.best_params_)
    mlflow.log_params(grid.best_params_)
    
    # Evaluate our model on our test set
    y_pred = pipeline.predict(df_test["text"])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    print("label encoders:")
    print(dict(enumerate(le.classes_)))

    # Save combined model + label encoder
    model_bundle = {'model': pipeline, 'label_encoder': le}
    joblib.dump(model_bundle, args.output_model)

    # Forceer flush en controleer het bestand
    assert os.path.exists(args.output_model), "Model file not found!"
    assert os.path.getsize(args.output_model) > 0, "Model file is empty!"

    
    # Log the model as an MLflow model
    logged_model = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="mood-machine-model",
        input_example=[df_train["text"].iloc[0]]
    )
    
    mlflow.register_model(
        model_uri=logged_model.model_uri,
        name="mood-machine-model"
    )
    