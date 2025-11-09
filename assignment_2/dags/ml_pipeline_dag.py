"""
ML Pipeline DAG
Orchestrates end-to-end ML workflow: Training -> Inference -> Monitoring
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/dags')

print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")

# Import configuration
try:
    from config import (
        DEFAULT_DAG_ARGS, 
        PATHS, 
        MODEL_CONFIG,
        TRAINING_CUTOFF_DATE,
        INFERENCE_START_DATE,
        END_DATE_STR,
        MONITORING_CONFIG,
        MODEL_REFRESH_POLICY
    )
    print("✓ Config imported successfully")
except ImportError as e:
    print(f"⚠️  Using default config due to import error: {e}")
    DEFAULT_DAG_ARGS = {
        'owner': 'ml_engineer',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
    PATHS = {
        "gold_features": "/app/datamart/gold/feature_label_store",
        "gold_predictions": "/app/datamart/gold/predictions",
        "gold_monitoring": "/app/datamart/gold/monitoring_metrics",
        "models": "/app/models/model_artifacts",
        "model_metadata": "/app/models/model_metadata",
    }
    MODEL_CONFIG = {
        "test_size": 0.2,
        "random_state": 42,
        "dpd_threshold": 30,
        "mob_threshold": 6
    }
    TRAINING_CUTOFF_DATE = "2024-06-01"
    INFERENCE_START_DATE = "2024-06-01"
    END_DATE_STR = "2024-12-01"
    MONITORING_CONFIG = {
        "baseline_date": "2024-06-01"
    }
    MODEL_REFRESH_POLICY = {
        "performance_threshold": {"auc": 0.65, "precision": 0.70},
        "drift_threshold": 0.1
    }

# Import utils - 尝试多种导入方式
HAS_UTILS = False
import_errors = []

# 方法 1: 从 /app/utils 导入
try:
    from utils.model_training import train_and_save_best_model
    from utils.model_inference import run_inference_for_period
    from utils.model_monitoring import run_monitoring
    HAS_UTILS = True
    print("✓ Utils imported from /app/utils/")
except ImportError as e:
    import_errors.append(f"Method 1 failed: {e}")

# 方法 2: 直接添加 utils 到路径
if not HAS_UTILS:
    try:
        sys.path.insert(0, '/app/utils')
        import model_training
        import model_inference
        import model_monitoring
        train_and_save_best_model = model_training.train_and_save_best_model
        run_inference_for_period = model_inference.run_inference_for_period
        run_monitoring = model_monitoring.run_monitoring
        HAS_UTILS = True
        print("✓ Utils imported via direct path")
    except ImportError as e:
        import_errors.append(f"Method 2 failed: {e}")

# 如果都失败了，打印详细信息
if not HAS_UTILS:
    print("❌ All import methods failed!")
    for error in import_errors:
        print(f"  - {error}")
    
    # 打印目录内容用于调试
    print("\nDirectory listing:")
    for path in ['/app', '/app/utils', '/app/dags']:
        try:
            print(f"\n{path}:")
            for item in os.listdir(path):
                print(f"  - {item}")
        except Exception as e:
            print(f"  Error listing {path}: {e}")


def task_train_model(**context):
    """Task: Train ML models and select best one"""
    print("="*60)
    print("TASK: Training ML Models")
    print("="*60)
    
    if not HAS_UTILS:
        error_msg = "Utils modules not available! Import errors: " + "; ".join(import_errors)
        print(f"❌ {error_msg}")
        raise ImportError(error_msg)
    
    try:
        print(f"Loading data from: {PATHS['gold_features']}")
        print(f"Saving model to: {PATHS['models']}")
        print(f"Training cutoff date: {TRAINING_CUTOFF_DATE}")
        
        result = train_and_save_best_model(
            gold_feature_path=PATHS['gold_features'],
            model_dir=PATHS['models'],
            config=MODEL_CONFIG,
            training_cutoff_date=TRAINING_CUTOFF_DATE
        )
        
        # Push results to XCom
        if result and isinstance(result, dict):
            context['task_instance'].xcom_push(key='model_path', value=result.get('model_path'))
            context['task_instance'].xcom_push(key='best_model_name', value=result.get('best_model_name'))
            context['task_instance'].xcom_push(key='training_metrics', value=result.get('metrics'))
            
            print(f"\n✅ Training completed!")
            print(f"  Best model: {result.get('best_model_name')}")
            print(f"  Model path: {result.get('model_path')}")
            if 'metrics' in result:
                print(f"  Metrics: {result['metrics']}")
        
        return result
    
    except Exception as e:
        print(f"❌ Error in training: {e}")
        import traceback
        traceback.print_exc()
        raise


def task_run_inference(**context):
    """Task: Run inference on new data"""
    print("="*60)
    print("TASK: Running Model Inference")
    print("="*60)
    
    if not HAS_UTILS:
        error_msg = "Utils modules not available!"
        print(f"❌ {error_msg}")
        raise ImportError(error_msg)
    
    try:
        print(f"Model directory: {PATHS['models']}")
        print(f"Metadata directory: {PATHS['model_metadata']}")
        print(f"Feature path: {PATHS['gold_features']}")
        print(f"Output path: {PATHS['gold_predictions']}")
        print(f"Date range: {INFERENCE_START_DATE} to {END_DATE_STR}")
        
        result = run_inference_for_period(
            model_dir=PATHS['models'],
            metadata_dir=PATHS['model_metadata'],
            gold_feature_path=PATHS['gold_features'],
            predictions_output_path=PATHS['gold_predictions'],
            start_date=INFERENCE_START_DATE,
            end_date=END_DATE_STR
        )
        
        # Push results to XCom
        if result and isinstance(result, dict):
            context['task_instance'].xcom_push(key='inference_result', value=result)
            
            print(f"\n✅ Inference completed!")
            print(f"  Output file: {result.get('output_file')}")
            print(f"  Predictions: {result.get('num_predictions')}")
            print(f"  Default rate: {result.get('default_rate', 0)*100:.2f}%")
        
        return result
    
    except Exception as e:
        print(f"❌ Error in inference: {e}")
        import traceback
        traceback.print_exc()
        raise


def task_monitor_model(**context):
    """Task: Monitor model performance and stability"""
    print("="*60)
    print("TASK: Monitoring Model Performance & Stability")
    print("="*60)
    
    if not HAS_UTILS:
        error_msg = "Utils modules not available!"
        print(f"❌ {error_msg}")
        raise ImportError(error_msg)
    
    try:
        print(f"Predictions path: {PATHS['gold_predictions']}")
        print(f"Features path: {PATHS['gold_features']}")
        print(f"Output path: {PATHS['gold_monitoring']}")
        print(f"Baseline date: {MONITORING_CONFIG['baseline_date']}")
        
        result = run_monitoring(
            predictions_path=PATHS['gold_predictions'],
            gold_feature_path=PATHS['gold_features'],
            monitoring_output_path=PATHS['gold_monitoring'],
            baseline_date=MONITORING_CONFIG['baseline_date'],
            performance_threshold=MODEL_REFRESH_POLICY.get('performance_threshold', {'auc': 0.65}),
            drift_threshold=MODEL_REFRESH_POLICY.get('drift_threshold', 0.1)
        )
        
        # Push results to XCom
        if result and isinstance(result, dict):
            context['task_instance'].xcom_push(key='monitoring_result', value=result)
            
            print(f"\n✅ Monitoring completed!")
            if 'performance_file' in result:
                print(f"  Performance file: {result['performance_file']}")
            if 'drift_file' in result:
                print(f"  Drift file: {result['drift_file']}")
            
            # Check retraining recommendation
            if result.get('needs_retraining', False):
                print("\n⚠️  WARNING: Model retraining recommended!")
                if 'retraining_reasons' in result:
                    for reason in result['retraining_reasons']:
                        print(f"    - {reason}")
        
        return result
    
    except Exception as e:
        print(f"❌ Error in monitoring: {e}")
        import traceback
        traceback.print_exc()
        raise


# Define the DAG
dag = DAG(
    'ml_pipeline_loan_default',
    default_args=DEFAULT_DAG_ARGS,
    description='End-to-end ML pipeline for loan default prediction',
    schedule_interval=None,  # Manual trigger
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'credit_risk', 'loan_default'],
)

# Task 1: Data processing
task_data_processing = BashOperator(
    task_id='data_processing',
    bash_command='cd /app && python main.py',
    dag=dag,
)

# Task 2: Train models
task_training = PythonOperator(
    task_id='train_models',
    python_callable=task_train_model,
    provide_context=True,
    dag=dag,
)

# Task 3: Run inference
task_inference = PythonOperator(
    task_id='run_inference',
    python_callable=task_run_inference,
    provide_context=True,
    dag=dag,
)

# Task 4: Monitor model
task_monitoring = PythonOperator(
    task_id='monitor_model',
    python_callable=task_monitor_model,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
task_data_processing >> task_training >> task_inference >> task_monitoring