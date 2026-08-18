import os
import shutil
import tarfile
import tempfile
import time
from dotenv import load_dotenv
import boto3

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
load_dotenv()
ROLE_ARN       = os.environ.get("ROLE_ARN")
REGION         = os.environ.get("REGION")
ACCOUNT_ID     = os.environ.get("ACCOUNT_ID")

ORIG_MODEL_S3  = os.environ.get("ORIG_MODEL_S3")
UPLOAD_BUCKET  = os.environ.get("UPLOAD_BUCKET")
UPLOAD_KEY     = os.environ.get("UPLOAD_KEY")

ENDPOINT_NAME  = os.environ.get("ENDPOINT_NAME")
MODEL_NAME     = os.environ.get("MODEL_NAME")
EP_CONFIG_NAME = os.environ.get("EP_CONFIG_NAME")
INSTANCE_TYPE  = os.environ.get("INSTANCE_TYPE")

IMAGE_URI = (
    os.environ.get("IMAGE_URI_ARN"),
    os.environ.get("IMAGE_URI_NAME")
)

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
INFERENCE_FILE = os.path.join(SCRIPT_DIR, "inference.py")
REQUIREMENTS   = os.path.join(SCRIPT_DIR, "requirements.txt")

# ---------------------------------------------------------------------------
# Step 1: Repackage model.tar.gz  (add code/ subdirectory)
# ---------------------------------------------------------------------------
def repackage_model() -> str:
    s3 = boto3.client("s3", region_name=REGION)

    src_bucket = ORIG_MODEL_S3.split("/")[2]
    src_key    = "/".join(ORIG_MODEL_S3.split("/")[3:])

    with tempfile.TemporaryDirectory() as tmp:
        orig_tar = os.path.join(tmp, "orig.tar.gz")
        print(f"[1/4] Downloading {ORIG_MODEL_S3} …")
        s3.download_file(src_bucket, src_key, orig_tar)

        extract_dir = os.path.join(tmp, "model")
        os.makedirs(extract_dir)
        with tarfile.open(orig_tar, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

        # Add code/ directory
        code_dir = os.path.join(extract_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        shutil.copy(INFERENCE_FILE, os.path.join(code_dir, "inference.py"))
        shutil.copy(REQUIREMENTS,   os.path.join(code_dir, "requirements.txt"))

        new_tar = os.path.join(tmp, "model_v3.tar.gz")
        print("[2/4] Repacking tarball …")
        with tarfile.open(new_tar, "w:gz") as tar:
            for name in os.listdir(extract_dir):
                tar.add(os.path.join(extract_dir, name), arcname=name)

        print(f"[3/4] Uploading to s3://{UPLOAD_BUCKET}/{UPLOAD_KEY} …")
        s3.upload_file(new_tar, UPLOAD_BUCKET, UPLOAD_KEY)

    return f"s3://{UPLOAD_BUCKET}/{UPLOAD_KEY}"


# ---------------------------------------------------------------------------
# Step 2: Deploy via raw boto3  (Model → EndpointConfig → Endpoint)
# ---------------------------------------------------------------------------
def deploy(model_s3_uri: str):
    sm = boto3.client("sagemaker", region_name=REGION)

    # Delete old endpoint / config / model if they exist
    for fn, kwargs in [
        (sm.delete_endpoint,        {"EndpointName": ENDPOINT_NAME}),
        (sm.delete_endpoint_config, {"EndpointConfigName": EP_CONFIG_NAME}),
        (sm.delete_model,           {"ModelName": MODEL_NAME}),
    ]:
        try:
            fn(**kwargs)
            print(f"Deleted: {list(kwargs.values())[0]}")
            time.sleep(3)
        except Exception:
            pass  # resource didn't exist — fine

    print(f"[4/4] Creating SageMaker Model: {MODEL_NAME}")
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "MMS_DEFAULT_RESPONSE_TIMEOUT": "500",
            },
        },
        ExecutionRoleArn=ROLE_ARN,
    )

    print(f"Creating EndpointConfig: {EP_CONFIG_NAME}")
    sm.create_endpoint_config(
        EndpointConfigName=EP_CONFIG_NAME,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": MODEL_NAME,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }],
    )

    print(f"Creating Endpoint: {ENDPOINT_NAME}  (this takes ~5–10 min) …")
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=EP_CONFIG_NAME,
    )

    # Poll until InService
    while True:
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        print(f"  Status: {status}")
        if status == "InService":
            print(f"\n✅ Endpoint is InService: {ENDPOINT_NAME}")
            break
        if status in ("Failed", "OutOfService"):
            reason = resp.get("FailureReason", "unknown")
            raise RuntimeError(f"Endpoint creation failed: {reason}")
        time.sleep(30)


if __name__ == "__main__":
    model_uri = repackage_model()
    deploy(model_uri)