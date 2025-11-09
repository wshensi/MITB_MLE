from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/app')

from utils.data_processing_tasks import (
    load_bronze_data,
    process_silver_data,
    process_gold_features
)
from utils.model_training_tasks import (
    prepare_training_data,
    train_logistic_regression,
    train_random_forest,
    evaluate_and_select_best_model,
    save_best_model
)
from utils.model_inference_tasks import (
    load_model_for_inference,
    prepare_inference_data,
    execute_predictions
)
from utils.model_monitoring_tasks import (
    load_monitoring_data,
    calculate_performance_metrics,
    calculate_drift_metrics,
    check_thresholds,
    generate_monitoring_report
)

snapshot_date_str = "2023-01-01"
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

training_cutoff_date_str = "2024-08-01"

inference_start_date_str = "2024-08-01"
inference_end_date_str = "2024-12-01"

monitoring_baseline_date_str = "2024-12-01"

performance_threshold = {
    'auc': 0.50,
    'precision': 0.50
}
drift_threshold = 0.2

BRONZE_DIR = '/app/datamart/bronze/'
SILVER_DIR = '/app/datamart/silver/'
GOLD_FEATURE_DIR = '/app/datamart/gold/feature_label_store/'
PREDICTIONS_DIR = '/app/datamart/gold/predictions/'
MONITORING_DIR = '/app/datamart/gold/monitoring_metrics/'

MODEL_ARTIFACTS_DIR = '/app/models/model_artifacts/'
MODEL_METADATA_DIR = '/app/models/model_metadata/'
MODEL_TEMP_DIR = '/app/models/temp/'

DATA_PREPARED_DIR = '/app/data/prepared/'
DATA_INFERENCE_DIR = '/app/data/inference/'
DATA_MONITORING_DIR = '/app/data/monitoring/'

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

with DAG(
    dag_id='ml_pipeline_complete',
    default_args=default_args,
    description='Complete ML Pipeline with Centralized Date Configuration',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'credit_risk', 'production', 'v2.0'],
    max_active_runs=1,
) as dag:

    start = DummyOperator(
        task_id='start_pipeline',
    )

    with TaskGroup(
        group_id='data_processing',
        tooltip='Bronze -> Silver -> Gold data layers'
    ) as data_processing_group:

        task_bronze = PythonOperator(
            task_id='load_bronze_data',
            python_callable=load_bronze_data,
            op_kwargs={
                'start_date': start_date_str,
                'end_date': end_date_str,
                'bronze_directory': BRONZE_DIR,
            },
            provide_context=True,
        )

        task_silver = PythonOperator(
            task_id='process_silver_data',
            python_callable=process_silver_data,
            op_kwargs={
                'start_date': start_date_str,
                'end_date': end_date_str,
                'bronze_directory': BRONZE_DIR,
                'silver_directory': SILVER_DIR,
            },
            provide_context=True,
        )

        task_gold = PythonOperator(
            task_id='process_gold_features',
            python_callable=process_gold_features,
            op_kwargs={
                'start_date': start_date_str,
                'end_date': end_date_str,
                'silver_directory': SILVER_DIR,
                'gold_directory': GOLD_FEATURE_DIR,
            },
            provide_context=True,
        )

        task_bronze >> task_silver >> task_gold

    with TaskGroup(
        group_id='model_training',
        tooltip='Train and select best model'
    ) as training_group:

        task_prepare_data = PythonOperator(
            task_id='prepare_training_data',
            python_callable=prepare_training_data,
            op_kwargs={
                'gold_feature_path': GOLD_FEATURE_DIR,
                'training_cutoff_date': training_cutoff_date_str,
                'output_path': DATA_PREPARED_DIR,
            },
            provide_context=True,
        )

        task_train_lr = PythonOperator(
            task_id='train_logistic_regression',
            python_callable=train_logistic_regression,
            op_kwargs={
                'prepared_data_path': DATA_PREPARED_DIR,
                'model_output_path': MODEL_TEMP_DIR,
            },
            provide_context=True,
        )

        task_train_rf = PythonOperator(
            task_id='train_random_forest',
            python_callable=train_random_forest,
            op_kwargs={
                'prepared_data_path': DATA_PREPARED_DIR,
                'model_output_path': MODEL_TEMP_DIR,
            },
            provide_context=True,
        )

        task_evaluate = PythonOperator(
            task_id='evaluate_and_select_best',
            python_callable=evaluate_and_select_best_model,
            op_kwargs={
                'temp_model_path': MODEL_TEMP_DIR,
                'prepared_data_path': DATA_PREPARED_DIR,
            },
            provide_context=True,
        )

        task_save_model = PythonOperator(
            task_id='save_best_model',
            python_callable=save_best_model,
            op_kwargs={
                'temp_model_path': MODEL_TEMP_DIR,
                'model_artifacts_path': MODEL_ARTIFACTS_DIR,
                'model_metadata_path': MODEL_METADATA_DIR,
            },
            provide_context=True,
        )

        task_prepare_data >> [task_train_lr, task_train_rf] >> task_evaluate >> task_save_model

    with TaskGroup(
        group_id='model_inference',
        tooltip='Load model and make predictions'
    ) as inference_group:

        task_load_model = PythonOperator(
            task_id='load_model',
            python_callable=load_model_for_inference,
            op_kwargs={
                'model_artifacts_path': MODEL_ARTIFACTS_DIR,
                'model_metadata_path': MODEL_METADATA_DIR,
            },
            provide_context=True,
        )

        task_prepare_inference = PythonOperator(
            task_id='prepare_inference_data',
            python_callable=prepare_inference_data,
            op_kwargs={
                'gold_feature_path': GOLD_FEATURE_DIR,
                'inference_start_date': inference_start_date_str,
                'inference_end_date': inference_end_date_str,
                'output_path': DATA_INFERENCE_DIR,
            },
            provide_context=True,
        )

        task_predict = PythonOperator(
            task_id='execute_predictions',
            python_callable=execute_predictions,
            op_kwargs={
                'model_info_path': '/app/data/model_info.json',
                'inference_data_path': DATA_INFERENCE_DIR,
                'predictions_output_path': PREDICTIONS_DIR,
                'prediction_date': inference_end_date_str,
            },
            provide_context=True,
        )

        task_load_model >> task_prepare_inference >> task_predict

    with TaskGroup(
        group_id='model_monitoring',
        tooltip='Monitor performance and drift'
    ) as monitoring_group:

        task_load_monitoring_data = PythonOperator(
            task_id='load_monitoring_data',
            python_callable=load_monitoring_data,
            op_kwargs={
                'predictions_path': PREDICTIONS_DIR,
                'gold_feature_path': GOLD_FEATURE_DIR,
                'output_path': DATA_MONITORING_DIR,
            },
            provide_context=True,
        )

        task_performance = PythonOperator(
            task_id='calculate_performance_metrics',
            python_callable=calculate_performance_metrics,
            op_kwargs={
                'monitoring_data_path': DATA_MONITORING_DIR,
                'output_path': DATA_MONITORING_DIR,
            },
            provide_context=True,
        )

        task_drift = PythonOperator(
            task_id='calculate_drift_metrics',
            python_callable=calculate_drift_metrics,
            op_kwargs={
                'monitoring_data_path': DATA_MONITORING_DIR,
                'baseline_date': monitoring_baseline_date_str,
                'output_path': DATA_MONITORING_DIR,
            },
            provide_context=True,
        )

        task_check = PythonOperator(
            task_id='check_thresholds',
            python_callable=check_thresholds,
            op_kwargs={
                'monitoring_data_path': DATA_MONITORING_DIR,
                'performance_threshold': performance_threshold,
                'drift_threshold': drift_threshold,
            },
            provide_context=True,
        )

        task_report = PythonOperator(
            task_id='generate_report',
            python_callable=generate_monitoring_report,
            op_kwargs={
                'monitoring_data_path': DATA_MONITORING_DIR,
                'output_path': MONITORING_DIR,
                'report_date': end_date_str,
            },
            provide_context=True,
        )

        task_load_monitoring_data >> [task_performance, task_drift] >> task_check >> task_report

    def check_retraining_needed(**context):
        ti = context['ti']

        retraining_status = ti.xcom_pull(
            task_ids='model_monitoring.check_thresholds',
            key='needs_retraining'
        )

        if retraining_status:
            print("  Retraining needed - triggering alert")
            return 'send_retraining_alert'
        else:
            print(" Model performance is satisfactory")
            return 'pipeline_complete'

    branch_retraining = BranchPythonOperator(
        task_id='branch_check_retraining',
        python_callable=check_retraining_needed,
        provide_context=True,
    )

    send_alert = PythonOperator(
        task_id='send_retraining_alert',
        python_callable=lambda **context: print(
            " ALERT: Model retraining required! Check monitoring reports."
        ),
        provide_context=True,
    )

    end = DummyOperator(
        task_id='pipeline_complete',
        trigger_rule='none_failed_min_one_success',
    )

    start >> data_processing_group >> training_group >> inference_group >> monitoring_group
    monitoring_group >> branch_retraining >> [send_alert, end]
    send_alert >> end

print("=" * 60)
print("ML PIPELINE CONFIGURATION SUMMARY")
print("=" * 60)
print(f" Data Processing:")
print(f"   Snapshot Date:  {snapshot_date_str}")
print(f"   Start Date:     {start_date_str}")
print(f"   End Date:       {end_date_str}")
print()
print(f" Model Training:")
print(f"   Cutoff Date:    {training_cutoff_date_str}")
print()
print(f" Model Inference:")
print(f"   Start Date:     {inference_start_date_str}")
print(f"   End Date:       {inference_end_date_str}")
print()
print(f" Model Monitoring:")
print(f"   Baseline Date:  {monitoring_baseline_date_str}")
print(f"   AUC Threshold:  {performance_threshold['auc']}")
print(f"   Drift Threshold: {drift_threshold}")
print("=" * 60)