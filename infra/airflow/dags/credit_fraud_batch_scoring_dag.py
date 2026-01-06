from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipelines.batch_scoring import batch_score

default_args = {
    "owner": "raj",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="credit_fraud_batch_scoring",
    default_args=default_args,
    description="Batch scoring DAG using Production model from MLflow",
    schedule_interval="0 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["credit-fraud", "scoring"],
) as dag:

    def score_task(**context):
        batch_score(model_name="credit_fraud_xgb", stage="Production")

    score_op = PythonOperator(
        task_id="batch_score",
        python_callable=score_task,
    )
