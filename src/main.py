from src.pipeline import run_pipeline
from src.monitoring.push import push_metrics

def main():
    run_pipeline()
    push_metrics(job_name="population_etl")

if __name__ == "__main__":
    main()