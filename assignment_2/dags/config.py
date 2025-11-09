"""
Configuration file for ML Pipeline DAG
"""
from datetime import datetime, timedelta

# Date range configuration
START_DATE_STR = "2023-01-01"
END_DATE_STR = "2024-12-01"
TRAINING_CUTOFF_DATE = "2024-06-01"  # Only train on data before this date
INFERENCE_START_DATE = "2024-06-01"  # Start inference from this date

# Model configuration
MODEL_CONFIG = {
    "dpd_threshold": 30,  # Days past due threshold for default label
    "mob_threshold": 6,   # Months on book for observation window
    "test_size": 0.2,     # Train-test split ratio
    "random_state": 42,
}

# Model types to train
MODEL_TYPES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm"
]

# Feature engineering configuration
FEATURE_CONFIG = {
    "clickstream_features": True,
    "attribute_features": True,
    "financial_features": True,
}

# Monitoring configuration
MONITORING_CONFIG = {
    "performance_metrics": ["auc", "precision", "recall", "f1"],
    "drift_detection": True,
    "stability_window_months": 3,
    "baseline_date": "2024-06-01",  # Baseline for drift detection
}

# Model governance SOP
MODEL_REFRESH_POLICY = {
    "performance_threshold": {
        "auc": 0.65,  # Retrain if AUC drops below this
        "precision": 0.70,
    },
    "drift_threshold": 0.1,  # PSI threshold
    "max_age_months": 6,  # Force retrain after 6 months
    "min_samples": 1000,  # Minimum samples for training
}

# Directory paths
PATHS = {
    "data": "/app/data",
    "bronze": "/app/datamart/bronze",
    "silver": "/app/datamart/silver",
    "gold_features": "/app/datamart/gold/feature_label_store",
    "gold_predictions": "/app/datamart/gold/predictions",
    "gold_monitoring": "/app/datamart/gold/monitoring_metrics",
    "models": "/app/models/model_artifacts",
    "model_metadata": "/app/models/model_metadata",
}

# Airflow DAG default arguments
DEFAULT_DAG_ARGS = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Deployment options
DEPLOYMENT_CONFIG = {
    "strategy": "blue_green",  # or "canary", "rolling"
    "approval_required": True,
    "automated_rollback": True,
    "rollback_threshold": {
        "auc_drop": 0.05,  # Rollback if AUC drops by 5%
        "error_rate": 0.1,  # Rollback if error rate > 10%
    }
}