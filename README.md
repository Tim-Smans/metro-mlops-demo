# Metro MLOps usage guide
This guide includes a very simple, practical example of how to use the Metro MLOps platform
It describes the different steps to take when setting up a workflow/pipeline. 

We will be using the Iris dataset as an example:
[Iris Github](https://github.com/venky14/Machine-Learning-with-Iris-Dataset)

---

## Table of contents

1. Preparing training/testing data
2. Setting up the training scripts and implementing MLFLow
3. Setting up the kubeflow pipeline
4. Using model serving API

---

## Preparing training/testing data

A `ml-data` bucket in MinIO was created to handle the training and testing data. How you structure this totally depends on the usecase. Some options are:

- Data pipeline into MinIO
- Create a training data pre-processing script in python
- Manually uploading the data into MinIO

In this guide i will be creating a data preprocessing script that will before training. 

### Viewing MinIO
Before actually writing code it might be a good idea to check the MinIO UI. 

The UI should be located at < external-ip>/ 

To check the external ip run:

`kubectl get svc -n istio-system` (Using Minikube? Make sure to use minikube tunnel!)

You should see something like this:
![Kubectl get service](https://i.imgur.com/PW4WvT0.png)
The external ip is on the 'istio-ingressgateway' row under external-ip (When using minikube probably local host)

When opening the UI it will prompt you to login. The first time the login should be:

Username: minio

Password: minio123

Now you should see the MinIO layout with some pre-created buckets:
![Minio UI](https://i.imgur.com/cwiXWt9.png)

### 1. Making a connection to MinIO
We will start of by making a connection to MinIO, to connect to our MinIO client we will be using the Boto3 package, start out by installing it (We strongly recommend installing all the packages inside of a virtual environment) using: 

`pip install boto3`

Import the library and connect to the client using following code:

```python
# src/save_iris_to_minio.py
import boto3

s3 = boto3.client('s3',
    # Use the external ip for minio here
    endpoint_url='http://external-ip>',
    # Minio username
    aws_access_key_id='minio',
    # Minio password
    aws_secret_access_key='minio123',
    # Region, You can keep this us-east-1
    region_name='us-east-1'
)
```

Because this script will be run outside of the kubeflow workflow we need to use the external ip when connecting to the client.
When using the default configuration the MinIO ip is just:
`<external-ip>/`
Without a suffix.

### 2. Loading the data
The Iris dataset is built into sklearn so downloading it is very simple. This step would be replaced by any other way of pre-processing the data.

I added following code to load the dataset and write it to csv

```python
# src/save_iris_to_minio.py

# Load the data from sklearn
iris = load_iris(as_frame=True)
# This uses pandas to concatenate 2 columns from the dataset into one
df = pd.concat(
    [
        iris.data, 
        pd.DataFrame(iris.target, columns=["target"])
    ]
    , axis=1
)
# Using StringIO to create an in-memory csv file to write to minio.
csv_buffer = StringIO()
# Writing the concatinated table into the temporary in-memory CSV file.
df.to_csv(csv_buffer, index=False)
```

Your training data doesn't necessarily need to be in CSV format. It all depends on your method of training. This step of the process completely depends on your usecase and data.

### 3. Uploading it to MinIO
Actually uploading the data to MinIO is very simple:

```python
# src/save_iris_to_minio.py

# Uploading the files
# Bucket -> The name of the bucket to use in MinIO
# Key -> The name of the directory/file in MinIO
# Body -> The object you want to upload to MinIO 
s3.put_object(Bucket="ml-data", Key="iris/iris.csv", Body=csv_buffer.getvalue())
```

If you want to upload local files that is also possible:
```python
# Uploading the files
# line 1 -> Where the file is located locally
# line 2 -> The name of the bucket to use in MinIO
# line 3 -> The name of the directory/file in MinIO
s3.upload_file(
    "./data/saves/trainset.npz", 
    "ml-data",
    "data/train.npz" 
)
```

Make sure to upload to an existing bucket in MinIO!

When running the `save_iris_to_minio.py` script it will now upload my data to MinIO:
![Saved data to minio](https://i.imgur.com/KaMAAxj.png)

---

## Creating a training script and implementing MLFLow

This step is mostly dependent on your usecase, i will create a very simple training script for the iris model but this will probably way more extensive in your case.

Make sure to install the mlflow python library using:

`pip install mlflow`

### 1. Connecting to MinIO and downloading data
Because we need to get the training data from MinIO we need to create another connection with the MinIO client. 

This is mostly the same as we did in the last step. But because we are using this script in the pipeline (inside of our kubernetes architecture) we are able to use our internal ip to connect with MinIO. The default internal ip for MinIO is:

`http://istio-ingressgateway.istio-system.svc.cluster.local`

So a connection would look like this:

```python
# src/train.py

import boto3

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
```

Next step is how to download the data from MinIO to start training with it. This is easily done using:

```python
# src/train.py

# Download dataset
obj = s3.get_object(Bucket="datasets", Key="iris/iris.csv")
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
``` 
The first line downloads the object from MinIO (important for your usecase) and the second line will put use pandas to put it in a dataframe, this line might be different in your case.

### 2. Training the model and using MLFLow
I won't explain this part in depth because this will be heavily depend on your training method. The training in this workflow can happen any way using all machine learning libraries supported by MLFLow.

This is my training code:
```python
# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow
mlflow.set_tracking_uri("http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow") # bv. http://mlflow-service.kubeflow:5000
mlflow.set_experiment("iris-experiment")

with mlflow.start_run():
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_metric("accuracy", acc)

    logged_model = mlflow.sklearn.log_model(model, "model")

    mlflow.register_model(
        model_uri=logged_model.model_uri,
        name="iris-model"
    )
```

The first 3 lines split the data into training/testing headers and train/test data.

Then we are configuring our MLFlow endpoint. It is very important to take the right tracking URL for MLFLow.
We can use the internal URL for this again. The default internal url for MLFlow in our platform is:

`http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow`

With mlflow.set_experiment you are able to declare an experiment name.

`mlflow.start_run()` is used to start a new training run. If you have a training cycle with multiple epochs the epochs are looped inside of here.

We can use `mlflow.log_param("param_name", "param_value")` to declare the parameters that were used for this run.

We can also use `mlflow.log_metric("metric_name", "metric_value")` to log different metrics like the accuracy. If we are using epochs and these metrics are logged in every epoch this is also displayed in a nice graph inside of the MLFLow UI. 

The usage of the MLFLow UI is explained in a different step.

The next step is to log the model to MLFLow This is done by doing `mlflow.<machinelearning_library>.log_model(trained_model, "name")` it is important to use the machine learning library you used to train your model when logging the model. That is why it's important to train your model with a library that is supported by MLFlow.
It will also save the logged model for registering

The final important step is to register your model to the MLFlow model registry, we do this by running `mlflow.register_model(..)`. Make to do this, otherwise the model serving will not be possible.

### 3. Preparing the training script as a docker image

In the root of your project create a requirements.txt, this needs to contain all the python packages that your training script needs to run (Our platform needs at least boto3 and mlflow).
Also create a Dockerfile looking something like this:
```Dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/
``` 

It is important to copy the training scripts you created. Go ahead and build this image and upload it to your dockerhub account:
```bash
docker build -t metro-mlops-demo .
docker tag metro-mlops-demo timsmans/metro-mlops-demo
docker push metro-mlops-demo
```

Now we can use this image to build our pipeline.
Currently my docker image exists of one script. You are also able to upload an entire directory of scripts in your image, this will allow you to use all of these scripts as pipeline steps.

## Setting up a kubeflow Pipeline

The next step is to create a pipeline in kubeflow to automize our machine learning workflow. I split up the script we created earlier into 2 different scripts. A script to load the data from MinIO, and the actual training script. This will give us an example to use arguments.

We can't just use the loaded data from the first step of the pipeline in the second step (this will become more clear soon). To send data from one step to the next you are able to use argument parser.

### 1. Spliting the script, and adding the argument parser

We start out by splitting up the script that i started with into a script that will load the data and the other script to train the model.

#### Data loading script
Start by importing the `argparse` library and define arguments like this:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--output_dataset_train_X', type=str, required=True)
parser.add_argument('--output_dataset_test_X', type=str, required=True)
parser.add_argument('--output_dataset_train_y', type=str, required=True)
parser.add_argument('--output_dataset_test_y', type=str, required=True)

args = parser.parse_args()
```

You can define an argument for everything you want to pass on in the pipeline. This will only declare it, you need to pass the values to it aswell.
This is done by doing:

```python
# Save datasets to the provided paths
X_train.to_csv(args.output_dataset_train_X, index=False)
X_test.to_csv(args.output_dataset_test_X, index=False)
y_train.to_csv(args.output_dataset_train_y, index=False)
y_test.to_csv(args.output_dataset_test_y, index=False)
```

#### Training script

Now all we have to do to use these datasets in our training script is declare the arguments once more.

```python
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
```

By adding this we will be able to use the datasets in our training script same as before.

You should take a look at the `load_data.py` and `train.py` to get a better overview of how the argument system works.


### 2. Creating the pipeline script

The pipeline is script is made up of 3 parts. The components, the pipeline declaration and the compiler declaration.
Combining these components makes for an easy way to build a pipeline

#### The components
Components is how you define the different steps of your pipeline. In this demo i will be using container components, when doing this each components get a docker image that it will run inside of that step. It will get clearer with the practical example:

```python
@dsl.container_component
def load_data(
    output_dataset_train: Output[Dataset],
    output_dataset_test: Output[Dataset],
    output_dataset_train_y: Output[Dataset],
    output_dataset_test_y: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='timsmans/metro-mlops-demo:latest', 
        command=['python', '/app/load_data.py'],
        args=[
            '--output_dataset_train_X', output_dataset_train_X.path,
            '--output_dataset_train_y', output_dataset_train_y.path,
            '--output_dataset_test_X', output_dataset_test_X.path,
            '--output_dataset_test_y', output_dataset_test_y.path,
        ]
    )
```

A container component uses the `@dsl.container_component` decorator. After that it's just a normal function definition that returns a dsl ContainerSpec object (This object is where you will clarify the structure and specifications of the container) and receives our arguments as parameters.

Inside of the `ContainerSpec` we clarify multiple properties. The most important one is the docker image we are using in that step. It is possible for multiple container components to use the same image (it's the same in the demo).
Command is the command that is being ran inside the container when that step is initalised. In our case we want to script (the is connected to that step) to be ran using python.
And last but not least we want to declare the arguments one more time. Make sure to not make typo's when declaring these.

We create these components for every step of our pipeline. Check the `pipeline.py` file if you want to take a look at the other component in this pipeline.

#### Declaring the pipeline

Inside of the pipeline declaration we will set up the flow of the pipeline, we decide which steps depend on eachother and we can even disable/enable step caching.

```python
@dsl.pipeline(
    name="demo-pipeline",
    description="End-to-end demo pipeline using the Iris dataset"
)
def demo_pipeline():
    # Data loading
    load_data_task = load_data()
    
    # Training
    train_task = train_model(
        dataset_train_X=load_data_task.outputs['output_dataset_train_X'],
        dataset_test_X=load_data_task.outputs['output_dataset_test_X'],
        dataset_train_y=load_data_task.outputs['output_dataset_train_y'],
        dataset_test_y=load_data_task.outputs['output_dataset_test_y']
    )
    
    train_task.set_caching_options(False)

    # Define execution order
    train_task.after(load_data_task)
```

At the top of this function we use the `@dsl.pipeline` decorator. Inside of this decorator we can supply a name and description for our pipeline.

Inside of the function we have to declare our container components as steps. We do this as seen above, by declaring the functions as variables.
After that we can also use these variables to define an execution order. Another usefull property we can configure is the `set_caching_options()` function.
This is forces a step to be re-run every time the pipeline is ran. This is especially important for training steps or steps that are critical to be ran every time. If this is not altered a step will be pulled from cache when running multiple times.

#### Declaring the compiler
```python
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=demo_pipeline,
        package_path="demo_pipeline.yaml"
    )
```

This step is basically always the same, it is important for the `pipeline_func` property to be defined correctly. The package path is just the path where this pipeline file will be saved when generated.

If you want a better overview of this file please do take a look at the `pipeline.py` script.

#### Generating and using the pipeline

To generate the pipeline after creating the script is very easy. Just run the pipeline.py script and this should generate a yaml file.

When you have this pipeline file go to the kubeflow UI at
`<external-ip>/pipeline/` 

**Do not forget the final / otherwise the UI won't load**

You should see this:
![Kubeflow startpage](https://i.imgur.com/BP16jqW.png)
*I do recommend looking at these tutorials as wel*

You can add our new pipeline by clicking the `Upload Pipeline` button at the top right.

![New pipeline page](https://i.imgur.com/AFzHuJM.png)

Go ahead and fill in the name and description, also make sure to select the right yaml file for your pipeline.
When you have done this, go ahead and click `Create`

If successful, this should display something like this:
![Added pipeline](https://i.imgur.com/OtCAcS4.png)

This is the overview of your pipeline, if you want to test if everything works like it should it's time to create a new run.
Doing this is very easy. Go ahead and click the `Create run` button.

You can asign a name and description to the run. Because we are using MLFlow for experiment tracking you can keep the experiment field as default.

Go ahead and click start, this will start a run:
![Active run display](https://i.imgur.com/4187YiV.png)

A step will display a red exclamation mark if it fails. You can debug this viewing the logs of a step by clicking that step and looking at the 'logs' tab.

If the step is successful it will display a green checkmark.

Wait and debug until all your steps are successful (This takes a while sometimes).

### 3. Checking the experiment in MLFlow

When you have a successful run your model should be trained, we can look at the training metrics inside of MLFlow.

Start of by opening the MLFlow UI at `<external-ip>/mlflow`

![MLFLow UI](https://i.imgur.com/BsGrstW.png)

At the left side you can open your experiment and you should be able to see the completed runs.
Go ahead and open a run to look at the metrics:

![MLFlow experiment details](https://i.imgur.com/4SRjCT1.png)

You can use these metrics to optimize your model training. 
We also registered the model after training, so if we go take a look at the `models` section (click models in navbar) we should see our model and it's different versions.

Another important place to look for our models is in MinIO. If we go to MinIO and take a look at the `ml-models` bucket we should see the `/1/` directory, go ahead and open that up. This is where we will find the files needed to serve our model (We will come back to this in the next step)

![](https://i.imgur.com/fiONSQz.png)

## Using model serving API
Now ofcourse we want to use our trained model. To do this will use the built in model serving API.

### 1. Preparing the model.

In a production setup we would want to create an easy way to do this, the platform is not at that point yet. So this part might be a bit confusing and sometimes difficult.
Start by going to MinIO into the `ml-models` bucket and into the `/1/` direcory. These are all the runs MLFlow has registered. Pick the one you want to serve (The run id's are visible on MLFlow)

For now i created this script to serve a model, feel free to use it:

```python
import boto3
import mlflow
from mlflow.tracking import MlflowClient

# MinIO config
s3 = boto3.client(
    "s3",
    endpoint_url="http://istio-ingressgateway.istio-system.svc.cluster.local",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123"
)

# Get the latest version of the mnist model in MLFlow model registry
MLFLOW_TRACKING_URI = "http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow/"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
model_name = "<CHANGE TO YOUR MODEL>"

latest_version = max(
    [v.version for v in client.search_model_versions(f"name='{model_name}'")]
)

latest_path = client.get_model_version_download_uri(model_name, latest_version)
latest_uuid = latest_path.split("/")[3] 

print(f"🔍 Latest path {latest_path}")

print(f"🔍 Latest uuid {latest_uuid}")

# MinIO Path for the newest version
source_prefix = f"{latest_uuid}/artifacts/mnist_model/"
latest_prefix = "latest/"

print(f"🔍 Expected Key in MinIO: ml-models/{latest_uuid}/artifacts/mnist_model/")


# Delete the old 'latest' directory
objects = s3.list_objects_v2(Bucket="ml-models", Prefix=latest_prefix)
if "Contents" in objects:
    for obj in objects["Contents"]:
        s3.delete_object(Bucket="ml-models", Key=obj["Key"])

# Copy all files of the latest version to the 'latest' directory
objects = s3.list_objects_v2(Bucket="ml-models", Prefix=source_prefix)
if "Contents" in objects:
    for obj in objects["Contents"]:
        src_key = obj["Key"]
        dest_key = src_key.replace(source_prefix, latest_prefix)
        print(f"Copying {src_key} → {dest_key}")
        s3.copy_object(Bucket="ml-models", CopySource={"Bucket": "ml-models", "Key": src_key}, Key=dest_key)


print(f"'latest' now points to {latest_uuid}")
```

Make sure to change the *model_name* variable to your model name. Otherwise it will not work. Include this script in your docker image and add it as the last step of your pipeline. This script will take the latest trained model and serve it, this is good when you developing a new model. But ofcourse you don't want to do this when in production.
When in production, create a script outside of the pipeline where you select a model you want to serve, MLFlow has tools to put models into production.

This puts the latest trained model inside of the `/latest` directory, where our model serving tool will take the model and serve it.

### 2. How is the model being served?

The platform uses a custom created image that works together with our MinIO setup, it connects to MinIO internaly to go into the ml-models bucket (this bucket gets created on first installation) and takes the model that is inside of the `latest` directory that we create in the previous step.

Everything from that point should work automaticly. If you don't have a model initalised inside of the `latest` directory yet, the pod that runs the serving container will crash on startup. This is not a problem though, whenever you have a model ready inside of the latest directory (or a new one is available) go ahead and restart that pod to serve the model.

### 3. Using the API


```python
url = "http://<external-ip>/mlflow-model/invocations"
headers = {"Content-Type": "application/json", "Host": "mlflow-model.local" }
response = requests.post(url, data=payload, headers=headers)

# Print response
print("Model Prediction:", response.json())
```

The most important parts when using our served model is:
- Using the right url, this should be  `http://<external-ip>/mlflow-model/invocations`
- Setting the `Host` header to `mlflow-model.local` If you don't do this, you will not be able to access your model.
- If you are using a Minikube cluster make sure to use `minikube tunnel`
