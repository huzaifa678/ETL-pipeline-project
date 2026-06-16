CLUSTER      := worldbank
NAMESPACE    := worldbank
ROOT         := $(shell dirname $(CURDIR))  
ML_IMAGE     := worldbank-ml:latest
SPARK_IMAGE  := worldbank-spark:latest


RUNTIME      ?= k3s


KATIB_REF      := v0.17.0
KFP_VERSION    := 2.2.0
KSERVE_VERSION := v0.15.0


VENDOR     := .vendor
KFP_REPO   := https://github.com/kubeflow/pipelines
KATIB_REPO := https://github.com/kubeflow/katib
HELM_TIMEOUT   := 30m


SPARK_OP_IMAGE := ghcr.io/kubeflow/spark-operator/controller:2.5.1
MLFLOW_IMAGE   := burakince/mlflow:3.13.0

# Namespaces the stack uses.
APP_NAMESPACES := $(NAMESPACE) kubeflow monitoring

.PHONY: help all forward cluster namespaces operators cert-manager spark-operator \
        load-spark-operator-image preload-images kserve katib kfp images secrets \
        platform mlflow monitoring deploy seed pipeline pipeline-schedule status clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

all: cluster operators images secrets platform deploy seed 
	@echo ""
	@echo ">> Stack is up. Open the dashboards with: make forward"

forward: 
	@echo "Grafana    http://localhost:3000  (admin / your grafana-admin password)"
	@echo "Prometheus http://localhost:9090"
	@echo "Pushgateway http://localhost:9091"
	@echo "MLflow     http://localhost:5000"
	@echo "KServe     http://localhost:8082  (POST /v1/models/worldbank-population:predict)"
	@echo "Katib      http://localhost:8081/katib/?ns=worldbank"
	@trap 'kill 0' EXIT; \
	kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80 & \
	kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 & \
	kubectl -n monitoring port-forward svc/pushgateway 9091:9091 & \
	kubectl -n worldbank port-forward svc/mlflow 5000:5000 & \
	kubectl -n worldbank port-forward svc/worldbank-population-predictor 8082:80 & \
	kubectl -n kubeflow port-forward svc/katib-ui 8081:80 & \
	wait

ifeq ($(RUNTIME),kind)
cluster:
	kind create cluster --config kind/kind-config.yaml
	$(MAKE) namespaces
else
cluster: ## Ensure a cluster is reachable (k3s on the VM) and create namespaces
	@kubectl get nodes >/dev/null 2>&1 || { \
	  echo "No cluster reachable. On the VM run:"; \
	  echo "  curl -sfL https://get.k3s.io | sh -s - --disable traefik --disable servicelb"; \
	  exit 1; }
	$(MAKE) namespaces
endif

namespaces: ## Create app namespaces and label worldbank for Katib injection
	@for ns in $(APP_NAMESPACES); do \
	  kubectl create namespace $$ns --dry-run=client -o yaml | kubectl apply -f - ; \
	done
	@# Katib only injects its metrics-collector into namespaces carrying this label.
	kubectl label namespace $(NAMESPACE) \
	  katib.kubeflow.org/metrics-collector-injection=enabled --overwrite

operators: spark-operator cert-manager kserve katib kfp ## Install all platform operators

spark-operator: load-spark-operator-image
	helm repo add spark-operator https://kubeflow.github.io/spark-operator
	helm repo update
	helm upgrade --install spark-operator spark-operator/spark-operator \
	  --namespace spark-operator --create-namespace \
	  -f helm-values/spark-operator-values.yaml --wait --timeout $(HELM_TIMEOUT)

ifeq ($(RUNTIME),kind)
load-spark-operator-image: ## Pre-load the spark-operator image onto kind nodes (avoids ghcr.io pull timeouts)
	./scripts/kind-load.sh $(CLUSTER) $(SPARK_OP_IMAGE)

load-mlflow-image: ## Pre-load the mlflow image onto kind nodes (avoids ghcr.io pull timeouts)
	./scripts/kind-load.sh $(CLUSTER) $(MLFLOW_IMAGE)

preload-images: ## Pre-pull every image in heavy-images.txt and load it into kind
	@while IFS= read -r img; do \
	  case "$$img" in ''|\#*) continue ;; esac; \
	  ./scripts/kind-load.sh $(CLUSTER) "$$img" || echo "!! failed to load $$img (continuing)"; \
	done < heavy-images.txt
else

load-spark-operator-image:
	@echo ">> RUNTIME=k3s: skipping preload; k3s pulls $(SPARK_OP_IMAGE) directly"
load-mlflow-image:
	@echo ">> RUNTIME=k3s: skipping preload; k3s pulls $(MLFLOW_IMAGE) directly"
preload-images:
	@echo ">> RUNTIME=k3s: skipping preload; k3s pulls images directly"
endif

cert-manager: ##  required for Kserver webhook
	helm repo add jetstack https://charts.jetstack.io
	helm repo update
	helm upgrade --install cert-manager jetstack/cert-manager \
	  --namespace cert-manager --create-namespace \
	  --set crds.enabled=true --wait --timeout $(HELM_TIMEOUT)

kserve: cert-manager ## Install KServe (CRDs + controller) via Helm, RawDeployment mode
	# OCI charts require an explicit --version (registries can't list tags).
	helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
	  --version $(KSERVE_VERSION) \
	  --namespace kserve --create-namespace --wait --timeout $(HELM_TIMEOUT)
	@# First apply may fail creating the bundled ClusterServingRuntimes before the
	@# controller's webhook is serving — that's an expected race, so tolerate it...
	-helm upgrade --install kserve oci://ghcr.io/kserve/charts/kserve \
	  --version $(KSERVE_VERSION) \
	  --namespace kserve -f helm-values/kserve-values.yaml --wait --timeout $(HELM_TIMEOUT)
	@# ...wait for the controller (and its webhook) to come up...
	kubectl -n kserve rollout status deploy/kserve-controller-manager --timeout=$(HELM_TIMEOUT)
	@# ...then reconcile; the serving runtimes now validate cleanly.
	helm upgrade --install kserve oci://ghcr.io/kserve/charts/kserve \
	  --version $(KSERVE_VERSION) \
	  --namespace kserve -f helm-values/kserve-values.yaml --wait --timeout $(HELM_TIMEOUT)

## no helm charts for katib and kfp; applied from locally-cloned kustomize manifests
$(VENDOR)/katib:
	git clone --depth 1 --branch $(KATIB_REF) $(KATIB_REPO) $@

$(VENDOR)/pipelines:
	git clone --depth 1 --branch $(KFP_VERSION) $(KFP_REPO) $@

katib: $(VENDOR)/katib
	kubectl apply -k $(VENDOR)/katib/manifests/v1beta1/installs/katib-standalone

kfp: $(VENDOR)/pipelines
	kubectl apply -k $(VENDOR)/pipelines/manifests/kustomize/cluster-scoped-resources
	kubectl wait --for condition=established --timeout=120s \
	  crd/applications.app.k8s.io || true
	kubectl apply -k $(VENDOR)/pipelines/manifests/kustomize/env/platform-agnostic

images: ## Build the ML + Spark images and load them into the cluster
	docker build -t $(ML_IMAGE)    -f docker/Dockerfile.ml    $(ROOT)
	docker build -t $(SPARK_IMAGE) -f docker/Dockerfile.spark $(ROOT)
ifeq ($(RUNTIME),kind)
	kind load docker-image $(ML_IMAGE)    --name $(CLUSTER)
	kind load docker-image $(SPARK_IMAGE) --name $(CLUSTER)
else
	docker save $(ML_IMAGE)    | sudo k3s ctr images import -
	docker save $(SPARK_IMAGE) | sudo k3s ctr images import -
endif

secrets: namespaces ## Apply all credentials secrets 
	@test -f manifests/secrets.yaml || \
	  (echo "Create manifests/secrets.yaml from secrets.example.yaml first" && exit 1)
	kubectl apply -f manifests/secrets.yaml

platform: mlflow monitoring ## Install MLflow + observability stack (Helm)

mlflow: load-mlflow-image
	helm repo add community-charts https://community-charts.github.io/helm-charts
	helm repo update
	helm upgrade --install mlflow community-charts/mlflow \
	  --namespace $(NAMESPACE) --create-namespace \
	  -f helm-values/mlflow-values.yaml --wait --timeout $(HELM_TIMEOUT)

monitoring: ## Install Prometheus + Grafana (kube-prometheus-stack) + Pushgateway via Helm
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
	  --namespace monitoring --create-namespace \
	  -f helm-values/monitoring-values.yaml --wait --timeout $(HELM_TIMEOUT)
	helm upgrade --install pushgateway prometheus-community/prometheus-pushgateway \
	  --namespace monitoring \
	  -f helm-values/pushgateway-values.yaml --wait --timeout $(HELM_TIMEOUT)

deploy: namespaces ## Helm-install the worldbank workloads (Spark ETL, Katib, KServe)
	helm upgrade --install worldbank charts/worldbank \
	  --namespace $(NAMESPACE) --create-namespace

seed: namespaces ## Bootstrap data: one-off Spark ETL + a training run (fills S3 + model)
	@echo ">> Triggering one-off Spark ETL (populates S3 processed parquet)..."
	-kubectl -n $(NAMESPACE) delete sparkapplication worldbank-etl-1 --ignore-not-found
	helm template worldbank charts/worldbank --set spark.schedule.enabled=false \
	  -s templates/spark-etl.yaml | kubectl -n $(NAMESPACE) apply -f -
	@echo ">> Waiting for Spark ETL to complete..."
	kubectl -n $(NAMESPACE) wait --for=jsonpath='{.status.applicationState.state}'=COMPLETED \
	  sparkapplication/worldbank-etl-1 --timeout=$(HELM_TIMEOUT)
	@echo ">> Running training job (uploads model.joblib for KServe)..."
	-kubectl -n $(NAMESPACE) delete job worldbank-seed-train --ignore-not-found
	kubectl -n $(NAMESPACE) apply -f manifests/seed-train-job.yaml
	kubectl -n $(NAMESPACE) wait --for=condition=complete job/worldbank-seed-train \
	  --timeout=$(HELM_TIMEOUT)
	@echo ">> Seed complete. KServe will pull the model and become Ready shortly."

pipeline: ## Compile the KFP pipeline (submit via the KFP UI)
	cd pipelines && PIPELINE_IMAGE=$(ML_IMAGE) python worldbank_pipeline.py

pipeline-schedule: ## Create the hourly recurring KFP run (needs KFP API port-forward)
	cd pipelines && python create_recurring_run.py

status: ## Show the state of the workloads
	kubectl -n $(NAMESPACE) get sparkapplications,experiments.kubeflow.org,inferenceservices,pods

ifeq ($(RUNTIME),kind)
clean:
	kind delete cluster --name $(CLUSTER)
else
clean: ## Uninstall k3s from the VM (wipes the cluster)
	@echo "Run on the VM to tear down k3s: sudo /usr/local/bin/k3s-uninstall.sh"
endif
