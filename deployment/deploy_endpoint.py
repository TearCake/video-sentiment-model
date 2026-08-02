import sagemaker
from sagemaker.pytorch import PyTorchModel

def deploy_endpoint():
    sagemaker.session()
    role = "aws-role-name"  # Replace with your SageMaker execution role ARN
    model_uri = "s3://your-bucket-name/path/to/model.tar.gz"
    
    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        entry_point="inference.py",
        framework_version="2.5.1",
        py_version="py311",
        name="sentiment-analysis-model"
    )
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analysis-endpoint"
    )
    
if __name__ == "__main__":
    deploy_endpoint()