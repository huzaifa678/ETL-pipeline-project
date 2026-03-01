import boto3
import json
import os
import joblib
from datetime import datetime
from airflow.models import Variable

s3 = boto3.client("s3")

def upload_model_to_s3(
    model,
    bucket: str,
    stage: str,
    metrics: dict,
    model_name: str = "worldbank_population"
):
    assert stage in ["staging", "production"]
    
    aws_access_key_id = Variable.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = Variable.get("AWS_SECRET_ACCESS_KEY")
    region_name = Variable.get("AWS_REGION", default_var="us-east-1")
    
    s3 = session.client("s3")
    
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    tmp_path = "/model/model.pkl"
    joblib.dump(model, tmp_path)

    base_key = f"models/{stage}/{model_name}"
    
    s3.upload_file(
        tmp_path,
        bucket,
        f"{base_key}/model.pkl"
    )

    metadata = {
        "model_name": model_name,
        "stage": stage,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

    s3.put_object(
        Bucket=bucket,
        Key=f"{base_key}/metadata.json",
        Body=json.dumps(metadata)
    )

    print(f"Uploaded {stage} model to s3://{bucket}/{base_key}/")