from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.container_component
def load_data(
    output_dataset_train_X: Output[Dataset],
    output_dataset_test_X: Output[Dataset],
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

@dsl.container_component
def train_model(
    dataset_train_X: Input[Dataset],
    dataset_test_X: Input[Dataset],
    dataset_train_y: Input[Dataset],
    dataset_test_y: Input[Dataset]
):
    return dsl.ContainerSpec(
        image='timsmans/metro-mlops-demo:latest', 
        command=['python', '/app/train.py'],
        args=[
            '--dataset_train_X', dataset_train_X.path,
            '--dataset_train_y', dataset_train_y.path,
            '--dataset_test_X', dataset_test_X.path,
            '--dataset_test_y', dataset_test_y.path,
        ],
    )

@dsl.container_component
def serve_latest_model(
):
    return dsl.ContainerSpec(
        image='timsmans/metro-mlops-demo:latest', 
        command=['python', '/app/serve_model.py'],
    )


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

    serve_model_task = serve_latest_model()
    
    train_task.set_caching_options(False)
    serve_model_task.set_caching_options(False)

    # Define execution order
    train_task.after(load_data_task)
    serve_model_task.after(train_task)

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=demo_pipeline,
        package_path="pipeline/demo_pipeline.yaml"
    )