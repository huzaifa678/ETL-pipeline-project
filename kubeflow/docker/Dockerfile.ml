FROM python:3.11-slim

WORKDIR /app

# Build tools occasionally needed by scientific wheels; slim afterwards.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# requirements.txt pins the project deps; add the extras the cluster steps need.
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir pyarrow s3fs joblib

COPY src /app/src
COPY setup.py /app/setup.py
# The Katib trial script.
COPY kubeflow/katib/train.py /app/katib/train.py
# S3-wired KFP pipeline steps.
COPY kubeflow/pipelines/steps /app/steps

ENV PYTHONPATH=/app
ENV BASE_URL=http://api.worldbank.org/v2/countries

CMD ["python", "-m", "src.pipeline"]
