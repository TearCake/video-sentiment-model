import builtins
import os

# Save Python's native file opener
original_open = builtins.open

def windows_sagemaker_v3_patch(file, mode='r', *args, **kwargs):
    # Intercept when SageMaker attempts to write its container bootstrap shell script
    if isinstance(file, str) and file.endswith('sm_train.sh') and 'w' in mode:
        kwargs['newline'] = '\n'  # Force strict Linux Line Endings
    return original_open(file, mode, *args, **kwargs)

# Overwrite the built-in system open method at runtime
builtins.open = windows_sagemaker_v3_patch


from sagemaker.train import ModelTrainer
from sagemaker.core.debugger import TensorBoardOutputConfig
from sagemaker.core import image_uris
from sagemaker.train.configs import Compute, SourceCode, InputData

def start_training():
    image_uri = image_uris.retrieve(
        framework="pytorch",
        region="ap-south-1",
        version="2.5.1",
        py_version="py311",
        image_scope="training",
        instance_type="ml.g5.xlarge"
    )
    
    compute_config = Compute(
        instance_type='ml.g5.xlarge',
        instance_count=1
    )
    source_config = SourceCode(
        source_dir='training',
        entry_script='train.py',
    )
    
    environment_variables = {
        "SAGEMAKER_TENSORBOARD_LOG_DIR": "/opt/ml/output/tensorboard",
        "SAGEMAKER_TENSORBOARD_S3_OUTPUT_PATH": "s3://bucket-name/tensorboard"
    }
    
    estimator = ModelTrainer(
        training_image=image_uri,
        source_code=source_config,
        compute=compute_config,
        role='aws-role',
        hyperparameters={
            "batch-size": 32,
            "epochs": 25,
        },
        environment=environment_variables
    )
    
    input_channels = [
        InputData(channel_name="training", data_source="s3://bucket-name/dataset/train"),
        InputData(channel_name="validation", data_source="s3://bucket-name/dataset/dev"),
        InputData(channel_name="test", data_source="s3://bucket-name/dataset/test")
    ]
    
    # START TRAINING
    print("Initiating Amazon SageMaker V3 Cloud Orchestration run...")
    estimator.train(input_data_config=input_channels)
    
if __name__ == "__main__":
    start_training()
