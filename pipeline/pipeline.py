from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.container_component
def load_data(
    output_dataset_train: Output[Dataset],
    output_dataset_test: Output[Dataset],
    output_dataset_validation: Output[Dataset],
):
    return dsl.ContainerSpec(
        image='timsmans/metro-mlops-mood-machine:latest', 
        command=['python', '/app/load_data.py'],
        args=[
            '--output_dataset_train', output_dataset_train.path,
            '--output_dataset_test', output_dataset_test.path,
            '--output_dataset_validation', output_dataset_validation.path,
        ]
    )

@dsl.container_component
def train_model(
    input_dataset_train: Input[Dataset],
    input_dataset_validation: Input[Dataset],
    input_dataset_test: Input[Dataset],
    output_model: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='timsmans/metro-mlops-mood-machine:latest', 
        command=['python', '/app/train_model.py'],
        args=[
            '--input_dataset_train', input_dataset_train.path,
            '--input_dataset_validation', input_dataset_validation.path,
            '--input_dataset_test', input_dataset_test.path,
            '--output_model', output_model.path,
        ],
    )

@dsl.container_component
def serve_latest_model(
):
    return dsl.ContainerSpec(
        image='timsmans/metro-mlops-mood-machine:latest', 
        command=['python', '/app/serve_model.py'],
    )



@dsl.pipeline(
    name="mood-machine-pipeline",
    description="End-to-end mood-machine demo pipeline using tweet_eval dataset"
)
def demo_pipeline():
    # Data loading
    load_data_task = load_data()
    
    # Training
    train_task = train_model(
        input_dataset_train=load_data_task.outputs['output_dataset_train'],
        input_dataset_test=load_data_task.outputs['output_dataset_test'],
        input_dataset_validation=load_data_task.outputs['output_dataset_validation'],
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
        package_path="pipeline/mood_machine_pipeline.yaml"
    )