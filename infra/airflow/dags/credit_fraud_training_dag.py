from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipelines.data_pipeline import ingest_raw_data, build_features
from pipelines.train_pipeline import train_and_log, promote_to_production

default_args = {
    "owner": "raj",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="credit_fraud_training",
    default_args=default_args,
    description="End-to-end training DAG for credit card fraud detection",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["credit-fraud", "training"],
) as dag:

    def ingest_task(**context):
        ingest_raw_data()

    def features_task(**context):
        build_features()

    def train_task(**context):
        run_id, _ = train_and_log(
            model_name="credit_fraud_xgb",
            experiment_name="credit_fraud_experiments",
        )

        context["ti"].xcom_push(key="run_id", value=run_id)

    def promote_task(**context):
        run_id = context["ti"].xcom_pull(
            key="run_id",
            task_ids="train_model",
        )
        promote_to_production(
            run_id,
            registered_name="credit_fraud_xgb",
        )

    ingest_op = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_task,
    )

    features_op = PythonOperator(
        task_id="build_features",
        python_callable=features_task,
    )

    train_op = PythonOperator(
        task_id="train_model",
        python_callable=train_task,
    )

    promote_op = PythonOperator(
        task_id="promote_model",
        python_callable=promote_task,
    )

    ingest_op >> features_op >> train_op >> promote_op
