from sagemaker.serve import ModelBuilder
from sagemaker.serve.configs import Resources
from sagemaker.core.enums import EndpointType

def deploy_endpoint():
    role = "aws-role"
    model_uri = "s3://bucket-name/path/to/model.tar.gz"
    
    resource_config = Resources(
        instance_type="ml.g5.xlarge",
        initial_instance_count=1
    )
    
    print("Packaging model via SageMaker V3 ModelBuilder...")
    model_builder = ModelBuilder(
        model_metadata={"model_data": model_uri},
        role_arn=role,
        entry_point="inference.py",
        source_dir="training",
        image_uri=None
    )
    print("Building server configurations...")
    model = model_builder.build()

    print("Deploying endpoint to AWS container cluster (this may take a few minutes)...")
    predictor = model.deploy(
        resources=resource_config,
        endpoint_name="sentiment-analysis-endpoint",
        endpoint_type=EndpointType.REAL_TIME
    )
    
    print(f"Endpoint successfully deployed! Active at: {predictor.endpoint_name}")
    return predictor

if __name__ == "__main__":
    deploy_endpoint()