# ETL Pipeline using Kubeflow Stack

Adds a Kubernetes-native ML platform on top of the existing Airflow/MLflow/DVC project:

| Capability        | Tool            | Install method                          |
|-------------------|-----------------|-----------------------------------------|
| ETL compute       | Spark Operator  | **Helm** (`kubeflow/spark-operator`)    |
| Model serving     | KServe (RawDeployment) | **Helm** (`oci://ghcr.io/kserve/charts`, pinned `v0.15.0`) |
| Cert management   | cert-manager (KServe webhook dependency) | **Helm** (`jetstack/cert-manager`) |
| Pipeline DAG      | Kubeflow Pipelines (KFP) | kustomize (no official Helm chart) |
| Hyperparam tuning | Katib           | kustomize (no official Helm chart)      |
| Experiment tracking | MLflow        | **Helm** (`community-charts/mlflow`)    |
| Observability     | Prometheus + Grafana + Pushgateway | **Helm** (`kube-prometheus-stack`, `prometheus-pushgateway`) |
| Workloads    | SparkApplication, InferenceService, Katib Experiment | **Helm chart `charts/worldbank`** |


## Pre-requisites

- Docker, `kubectl`, `helm`, `kustomize`, plus the cluster runtime (see below)
- ~16 GB RAM for the full stack (KFP + monitoring + a Spark job is the peak)
- AWS creds that can read/write `s3://etl-dvc-bucket`

## Cluster runtime: k3s (default) or kind

The Makefile targets a cluster runtime via `RUNTIME` (default **`k3s`**). The full
stack outgrew a typical laptop, so the default path is a cheap cloud VM running
[k3s](https://k3s.io/); local [`kind`](https://kind.sigs.k8s.io/) is still
supported with `make RUNTIME=kind …`.

**k3s on a cloud VM (recommended, e.g. Hetzner CPX42 — 8 vCPU / 16 GB / 320 GB):**

```bash
# on the VM (Ubuntu): install k3s + build/chart tooling
curl -sfL https://get.k3s.io | sh -s - --disable traefik --disable servicelb --write-kubeconfig-mode 644
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml      # add to ~/.bashrc
apt-get update && apt-get install -y docker.io git make python3-pip
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

git clone <your-repo> && cd <repo>/kubeflow
make all          # RUNTIME defaults to k3s
```

On k3s the node has real internet, so images pull directly — the `preload-images`
/ `kind-load.sh` workarounds are no-ops, and locally-built images are imported
straight into k3s's containerd (`docker save | k3s ctr images import`). k3s ships
the **local-path** StorageClass, so the MinIO/MySQL/MLflow PVCs auto-provision.

**Local kind:** prepend `RUNTIME=kind` to every command (`make RUNTIME=kind all`).
Needs `kind` installed and ~8 GB RAM free; see *Slow / timing-out image pulls*.

## Quick start

Fill in `manifests/secrets.yaml` first (see Secrets below), then run the whole
sequence with one command:

```bash
cd kubeflow
cp manifests/secrets.example.yaml manifests/secrets.yaml
$EDITOR manifests/secrets.yaml      # fill in every REPLACE_ME (see Secrets below)

make all          # cluster → operators → images → secrets → platform → deploy → seed
```

…or run the stages individually, in this order:

```bash
cd kubeflow

make cluster      # k3s: verify cluster reachable + namespaces (+ Katib label)
                  # kind: create the kind cluster too

make operators    # spark-operator, cert-manager, KServe, Katib, KFP

make images       # build worldbank-ml / worldbank-spark and load into the cluster

cp manifests/secrets.example.yaml manifests/secrets.yaml
$EDITOR manifests/secrets.yaml      # fill in every REPLACE_ME (see Secrets below)
make secrets

make platform     # MLflow + Prometheus/Grafana/Pushgateway

make deploy       # Spark ETL (cron) + Katib Experiment + KServe InferenceService

make seed         # one-off Spark ETL + training run → fills S3 + model.joblib
                  # (KServe becomes Ready once the model is uploaded)

make forward      # port-forward Grafana/Prometheus/Pushgateway/MLflow UIs

# optional: Kubeflow Pipelines
pip install -r requirements-kfp.txt
make pipeline                         # compile the KFP pipeline
kubectl -n kubeflow port-forward svc/ml-pipeline 8888:8888 &
make pipeline-schedule                # hourly recurring run

# remove everything when done
make clean
```

> `make seed` is required on a fresh cluster: the Katib trials and KServe both
> depend on data/model in S3 that don't exist until the first Spark ETL +
> training run produce them. `make all` includes it.

## Slow / timing-out image pulls (kind only)

> On the default **k3s** runtime the node pulls directly, so this section does not
> apply — `preload-images` is a no-op. This is only relevant for `RUNTIME=kind` on
> a slow/restricted local network.

Some images (notably `ghcr.io/kubeflow/spark-operator/controller`) time out when
pulled from **inside** kind on a slow/restricted network, leaving pods in
`ImagePullBackOff`. The fix is to pull them on the host (working internet) and
load them onto the kind nodes:

```bash
make preload-images        # pulls + loads everything in heavy-images.txt
```

- `heavy-images.txt` is the list — add any image you see stuck:
  `kubectl -n <ns> get pod <pod> -o jsonpath='{.spec.containers[*].image}'`
- `scripts/kind-load.sh <cluster> <image>` loads a single image (per-node `ctr`
  import, so it works with multi-arch images where `kind load` fails).
- The spark-operator image is auto-loaded by `make spark-operator`, so the
  default `make operators` / `make all` flow already handles it. Run
  `make preload-images` after `make cluster` if you hit other timeouts.

## Secrets

All passwords/credentials live in K8s Secrets — none are plaintext in the Helm
values. `manifests/secrets.example.yaml` defines them; copy to `secrets.yaml`
(gitignored), fill in every `REPLACE_ME`, and `make secrets` applies them (it also
pre-creates the `worldbank` / `kubeflow` / `monitoring` namespaces so the apply
succeeds before those charts install).

| Secret | Namespace | Used by | Keys |
|--------|-----------|---------|------|
| `aws-credentials` | `worldbank`, `kubeflow` | Spark/Katib/KFP S3 access, KServe storage-initializer, MLflow S3 artifacts | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| `mlflow-postgresql` | `worldbank` | MLflow server **and** its Postgres backend (shared) | `postgres-password` (admin), `password` (mlflow user) |
| `grafana-admin` | `monitoring` | Grafana admin login | `admin-user`, `admin-password` |

> The MLflow DB secret **must** be named `mlflow-postgresql` — that's the name the
> MLflow deployment hardcodes for its password ref, and the Bitnami Postgres
> subchart is pointed at the same secret so both sides stay in sync.

## Dynamic dates & scheduling (no hardcoded years)

Years are resolved at run time, not baked in:

- `endYear: 0` in `charts/worldbank/values.yaml` → the **current year**.
- `incremental: true` → ingest reads the last ingested year from the S3 state
  object (`s3://etl-dvc-bucket/worldbank/state/last_ingested_year.txt`) and only
  fetches newer years. Processed parquet is **partitioned by year** with dynamic
  partition overwrite, so re-runs upsert per-year partitions instead of
  duplicating or wiping history.

Two schedulers, pick per layer:

| Layer        | Mechanism                       | Where                                   |
|--------------|---------------------------------|-----------------------------------------|
| Spark ETL    | `ScheduledSparkApplication` cron | `spark.schedule.cron` (default hourly)  |
| KFP pipeline | recurring run cron               | `make pipeline-schedule` / `KFP_CRON`   |

Set `spark.schedule.enabled=false` for a one-off `SparkApplication` per Helm revision instead.

## What each piece does

- **`spark/etl_job.py`** — PySpark job submitted as a `SparkApplication`. Driver fetches
  World Bank data, Spark cleans it and writes parquet to `s3a://etl-dvc-bucket/worldbank/processed`.
- **`katib/train.py`** — trains a `Ridge` model reading `--alpha`, prints `r2=`/`rmse=` for
  Katib's stdout metrics collector. Katib searches `alpha` to maximize `r2`, then the best
  model is written to `s3://etl-dvc-bucket/models/staging/worldbank_population`.
- **`pipelines/worldbank_pipeline.py`** — KFP v2 pipeline tying ingest → transform → train
  together. **End-to-end via S3:** each step reads/writes S3 (scripts in
  `pipelines/steps/`), so data actually passes between the separate step pods.
  AWS creds are injected from the `aws-credentials` secret; train logs to the
  in-cluster MLflow and uploads `model.joblib` to S3 for KServe.
- **`charts/worldbank`** — Helm chart rendering the `SparkApplication`, KServe
  `InferenceService`, Katib `Experiment`, MLflow Deployment, and the S3/MLflow config.

## Accessing UIs

On a remote k3s VM these `port-forward`s bind to the VM's localhost, so tunnel them
over SSH first (then open `localhost:<port>` on your laptop):

```bash
ssh -N -L 5000:localhost:5000 -L 3000:localhost:3000 -L 9090:localhost:9090 root@<VM_IP>
```

```bash
kubectl -n kubeflow   port-forward svc/ml-pipeline-ui 8080:80                       # KFP
kubectl -n kubeflow   port-forward svc/katib-ui 8081:80                             # Katib
kubectl -n worldbank  port-forward svc/mlflow 5000:5000                             # MLflow
kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80                   # Grafana 
kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090  # Prometheus
```

## Observability

`make monitoring` installs **kube-prometheus-stack** (Prometheus Operator,
Prometheus, Grafana, Alertmanager, node/kube-state metrics) and the **Pushgateway**,
all via Helm. Batch workloads push their metrics to
`pushgateway.monitoring.svc.cluster.local:9091` — the KFP training step ships
`ml_model_rmse` / `ml_model_r2` there (via `PUSHGATEWAY_URL`), and the Pushgateway's
bundled ServiceMonitor lets Prometheus scrape it automatically. Grafana ships with
the standard Kubernetes dashboards out of the box; add a panel on the
`ml_model_*` series to track model quality over runs.

## Calling the served model

```bash
kubectl -n worldbank port-forward svc/worldbank-population-predictor 8082:80
curl -s http://localhost:8082/v1/models/worldbank-population:predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[2025, 1, 0, 0]]}'   # [year, country_China, country_Germany, country_United States]
```
