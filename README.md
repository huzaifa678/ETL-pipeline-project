# ETL Pipeline project using the world bank dataset

## The ETL pipeline

* ### Data Ingestion (Extract): 

    1. This work fetches the world bank dataset using the API URL and uses pagination to    extract data from raw from each page until done. 

    2. Converts raw JSON to the CSV file

* ### Data Transformation (Transform):

    1. Reads the CSV file and drops idle columns to keep the remove garbage values which add no insight to the model.

    2. Saves the transformed CSV

* ### Data Loading (Load):

    1. Loads the data for training the model

    2. Prepares the data from the CSV

    3. Uses the perpared data to train the model


## Tools used 

### MLOps

* #### Apache Airflow: Orchestration tool that uses jobs for the automation of ETL pipeline defined in code

* #### MLflow: Model tracking server used for logging the model parameters and tracking the model performance. Based on the model performance it versions the model selecting the best model adding stable versioning to it

* #### DVC: Data versioning tool for versioning the data the model trained and performed on. Fetching the history based on it's version and model behaviour/performance


### CI/CD

* #### Github Actions: Streamlining the data fetching from the DVC repository


### Containerization

* #### Docker: Containerizing Airflow defined Docker file and providing multi container environment for running the infrastructure containers (MLflow, Prometheus Grafana, PushGateway)


### Monitoring and Observability

* #### Prometheus: Monitoring tool for monitoring ETL pipeline progress and monitoring the model performance using RMSE and R2 loss metrics.

* #### PushGateway: Collects pushed metrics by the batch ETL jobs which is then pulled by prometheus.


## Guide

### Pre-requisites

* #### Docker MUST be installed and should be started

### Start 

```bash
docker compose up -d
```

### UI

  * #### Access the Airflow UI

      ``` bash
      http://localhost:8080
      ```

      * ##### Use default credentials or configure your own credentials

      * ##### scehdule the Extract Job in the Airflow UI

  * #### Access the Mlflow UI

      ```bash
      http://localhost:5000
      ```

  * #### Access the Grafana UI

      ```bash
      http://localhost:9090
      ```

      * ##### Access Prometheus from data sources
