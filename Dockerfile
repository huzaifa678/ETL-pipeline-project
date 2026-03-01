FROM astrocrpublic.azurecr.io/runtime:3.0-10

COPY ./dags /opt/airflow/dags
COPY ./src /opt/airflow/src
COPY ./requirements.txt /opt/airflow/requirements.txt

RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt