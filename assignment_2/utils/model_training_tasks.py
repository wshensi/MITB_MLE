import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import glob

def prepare_training_data(gold_feature_path, training_cutoff_date, output_path, **context):
    print("\n" + "="*60)
    print("TASK 2.1: PREPARING TRAINING DATA")
    print("="*60)
    print(f"Input: {gold_feature_path}")
    print(f"Cutoff date: {training_cutoff_date}")
    print(f"Output: {output_path}")
    print("="*60)
    
    try:
        gold_records = context['ti'].xcom_pull(
            task_ids='data_processing.process_gold_features', 
            key='gold_records'
        )
        print(f"Gold records available: {gold_records:,}")
        
        print("\n Loading gold feature data...")
        files = glob.glob(os.path.join(gold_feature_path, "*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found in {gold_feature_path}")
        
        df_list = []
        for file in files:
            df = pd.read_parquet(file)
            df_list.append(df)
        
        df = pd.concat(df_list, ignore_index=True)
        print(f" Total samples loaded: {len(df):,}")
        
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
        
        print(f"\n⏰ Applying temporal split (cutoff: {training_cutoff_date})...")
        train_df = df[df['snapshot_date'] < pd.to_datetime(training_cutoff_date)].copy()
        print(f" Training samples: {len(train_df):,}")
        
        if len(train_df) < 100:
            raise ValueError(f"Insufficient training data: {len(train_df)} samples. Need at least 100.")
        
        possible_label_cols = ['label_default_30dpd', 'default_flag', 'label', 'target']
        label_col = None
        
        for col in possible_label_cols:
            if col in train_df.columns:
                label_col = col
                print(f" Found label column: {label_col}")
                break
        
        if label_col is None:
            raise ValueError(f"No label column found. Tried: {possible_label_cols}")
        
        print("\n Removing data leakage features...")
        data_leakage_features = [
            'dpd', 'overdue_amt', 'installments_missed', 'first_missed_date',
            'balance', 'paid_amt', 'payment_status', 'delinquency_status',
            'default_flag', 'charged_off', 'settlement_amt', 'recovery_amt',
            'mob', 'installment_num', 'days_since_last_payment',
            'last_payment_date', 'maturity_date', 'payment_delay_days',
            'late_payment_count', 'missed_payment_count'
        ]
        
        id_cols = ['user_id', 'loan_id', 'snapshot_date', 'Customer_ID', 'loan_start_date']
        
        found_leakage = [col for col in data_leakage_features if col in train_df.columns]
        if found_leakage:
            print(f"  Removing {len(found_leakage)} leakage features: {found_leakage[:5]}...")
        
        exclude_cols = set(id_cols + [label_col] + data_leakage_features)
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        print(f" Selected {len(feature_cols)} clean features")
        
        X = train_df[feature_cols].copy()
        y = train_df[label_col].copy()
        
        X = X.fillna(0)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            print(f"\n  Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
            feature_cols = [col for col in feature_cols if col not in constant_features]
        
        print("\n Checking for highly correlated features...")
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
        
        if to_drop:
            print(f"  Removing {len(to_drop)} highly correlated features (r > 0.95)")
            X = X.drop(columns=to_drop)
            feature_cols = [col for col in feature_cols if col not in to_drop]
        
        print(f"\n Final feature count: {len(feature_cols)}")
        print(f" Class distribution:")
        print(y.value_counts())
        
        print("\n  Splitting train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )
        
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Test:  {len(X_test):,} samples")
        
        print("\n  Applying SMOTE for class balance...")
        minority_class_count = min(y_train.value_counts())
        
        if minority_class_count >= 6:
            smote = SMOTE(random_state=42, k_neighbors=min(5, minority_class_count - 1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"   Balanced to {len(X_train_balanced):,} samples")
        else:
            print(f"    Skipping SMOTE (only {minority_class_count} minority samples)")
            X_train_balanced, y_train_balanced = X_train, y_train
        
        os.makedirs(output_path, exist_ok=True)
        
        print("\n Saving prepared data...")
        
        X_train_balanced.to_parquet(os.path.join(output_path, 'X_train.parquet'), index=False)
        X_test.to_parquet(os.path.join(output_path, 'X_test.parquet'), index=False)
        pd.DataFrame(y_train_balanced).to_parquet(os.path.join(output_path, 'y_train.parquet'), index=False)
        pd.DataFrame(y_test).to_parquet(os.path.join(output_path, 'y_test.parquet'), index=False)
        
        metadata = {
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'n_train': len(X_train_balanced),
            'n_test': len(X_test),
            'class_distribution_train': y_train_balanced.value_counts().to_dict(),
            'class_distribution_test': y_test.value_counts().to_dict(),
            'training_cutoff_date': training_cutoff_date,
            'prepared_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(" Data saved successfully")
        
        print("\n" + "="*60)
        print(" DATA PREPARATION COMPLETED")
        print("="*60)
        print(f"Features: {len(feature_cols)}")
        print(f"Train samples: {len(X_train_balanced):,}")
        print(f"Test samples: {len(X_test):,}")
        print("="*60)
        
        context['ti'].xcom_push(key='prepared_data_path', value=output_path)
        context['ti'].xcom_push(key='n_features', value=len(feature_cols))
        context['ti'].xcom_push(key='n_train', value=len(X_train_balanced))
        
        return {
            'status': 'success',
            'n_features': len(feature_cols),
            'n_train': len(X_train_balanced),
            'n_test': len(X_test)
        }
        
    except Exception as e:
        print(f" Error in data preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def train_logistic_regression(prepared_data_path, model_output_path, **context):
    print("\n" + "="*60)
    print("TASK 2.2A: TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    try:
        print(" Loading prepared data...")
        X_train = pd.read_parquet(os.path.join(prepared_data_path, 'X_train.parquet'))
        X_test = pd.read_parquet(os.path.join(prepared_data_path, 'X_test.parquet'))
        y_train = pd.read_parquet(os.path.join(prepared_data_path, 'y_train.parquet')).squeeze()
        y_test = pd.read_parquet(os.path.join(prepared_data_path, 'y_test.parquet')).squeeze()
        
        print(f" Train: {len(X_train):,} samples, {len(X_train.columns)} features")
        print(f" Test:  {len(X_test):,} samples")
        
        print("\n Training Logistic Regression...")
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )
        
        model.fit(X_train, y_train)
        print(" Training completed")
        
        print("\n Evaluating on test set...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        metrics = {
            'auc': float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else 0.5,
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        print("="*60)
        print("LOGISTIC REGRESSION RESULTS:")
        print("="*60)
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print("="*60)
        
        os.makedirs(model_output_path, exist_ok=True)
        
        model_file = os.path.join(model_output_path, 'logistic_regression.joblib')
        metrics_file = os.path.join(model_output_path, 'logistic_regression_metrics.json')
        
        joblib.dump(model, model_file)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n Model saved: {model_file}")
        print(f" Metrics saved: {metrics_file}")
        
        context['ti'].xcom_push(key='lr_auc', value=metrics['auc'])
        context['ti'].xcom_push(key='lr_metrics', value=metrics)
        
        return {
            'status': 'success',
            'model': 'logistic_regression',
            'metrics': metrics
        }
        
    except Exception as e:
        print(f" Error training Logistic Regression: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def train_random_forest(prepared_data_path, model_output_path, **context):
    print("\n" + "="*60)
    print("TASK 2.2B: TRAINING RANDOM FOREST")
    print("="*60)
    
    try:
        print(" Loading prepared data...")
        X_train = pd.read_parquet(os.path.join(prepared_data_path, 'X_train.parquet'))
        X_test = pd.read_parquet(os.path.join(prepared_data_path, 'X_test.parquet'))
        y_train = pd.read_parquet(os.path.join(prepared_data_path, 'y_train.parquet')).squeeze()
        y_test = pd.read_parquet(os.path.join(prepared_data_path, 'y_test.parquet')).squeeze()
        
        print(f" Train: {len(X_train):,} samples, {len(X_train.columns)} features")
        print(f" Test:  {len(X_test):,} samples")
        
        print("\n Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print(" Training completed")
        
        print("\n Evaluating on test set...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        metrics = {
            'auc': float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else 0.5,
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        print("="*60)
        print("RANDOM FOREST RESULTS:")
        print("="*60)
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print("="*60)
        
        os.makedirs(model_output_path, exist_ok=True)
        
        model_file = os.path.join(model_output_path, 'random_forest.joblib')
        metrics_file = os.path.join(model_output_path, 'random_forest_metrics.json')
        
        joblib.dump(model, model_file)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n Model saved: {model_file}")
        print(f" Metrics saved: {metrics_file}")
        
        context['ti'].xcom_push(key='rf_auc', value=metrics['auc'])
        context['ti'].xcom_push(key='rf_metrics', value=metrics)
        
        return {
            'status': 'success',
            'model': 'random_forest',
            'metrics': metrics
        }
        
    except Exception as e:
        print(f" Error training Random Forest: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def evaluate_and_select_best_model(temp_model_path, prepared_data_path, **context):
    print("\n" + "="*60)
    print("TASK 2.3: EVALUATING AND SELECTING BEST MODEL")
    print("="*60)
    
    try:
        lr_metrics = context['ti'].xcom_pull(task_ids='model_training.train_logistic_regression', key='lr_metrics')
        rf_metrics = context['ti'].xcom_pull(task_ids='model_training.train_random_forest', key='rf_metrics')
        
        print("\n MODEL COMPARISON:")
        print("="*60)
        print(f"Logistic Regression:")
        print(f"  AUC: {lr_metrics['auc']:.4f}")
        print(f"  Precision: {lr_metrics['precision']:.4f}")
        print(f"  Recall: {lr_metrics['recall']:.4f}")
        print(f"  F1: {lr_metrics['f1']:.4f}")
        print()
        print(f"Random Forest:")
        print(f"  AUC: {rf_metrics['auc']:.4f}")
        print(f"  Precision: {rf_metrics['precision']:.4f}")
        print(f"  Recall: {rf_metrics['recall']:.4f}")
        print(f"  F1: {rf_metrics['f1']:.4f}")
        print("="*60)
        
        if lr_metrics['auc'] >= rf_metrics['auc']:
            best_model_name = 'logistic_regression'
            best_metrics = lr_metrics
            print(f"\n WINNER: Logistic Regression (AUC: {lr_metrics['auc']:.4f})")
        else:
            best_model_name = 'random_forest'
            best_metrics = rf_metrics
            print(f"\n WINNER: Random Forest (AUC: {rf_metrics['auc']:.4f})")
        
        with open(os.path.join(prepared_data_path, 'metadata.json'), 'r') as f:
            data_metadata = json.load(f)
        
        selection_result = {
            'best_model_name': best_model_name,
            'best_metrics': best_metrics,
            'all_models': {
                'logistic_regression': lr_metrics,
                'random_forest': rf_metrics
            },
            'selection_criteria': 'auc',
            'selected_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_columns': data_metadata['feature_columns'],
            'n_features': data_metadata['n_features']
        }
        
        result_file = os.path.join(temp_model_path, 'model_selection.json')
        with open(result_file, 'w') as f:
            json.dump(selection_result, f, indent=4, default=str)
        
        print(f"\n Selection result saved: {result_file}")
        
        context['ti'].xcom_push(key='best_model_name', value=best_model_name)
        context['ti'].xcom_push(key='best_metrics', value=best_metrics)
        context['ti'].xcom_push(key='selection_result', value=selection_result)
        
        print("\n" + "="*60)
        print(" MODEL EVALUATION COMPLETED")
        print("="*60)
        
        return selection_result
        
    except Exception as e:
        print(f" Error in model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def save_best_model(temp_model_path, model_artifacts_path, model_metadata_path, **context):
    print("\n" + "="*60)
    print("TASK 2.4: SAVING BEST MODEL TO PRODUCTION")
    print("="*60)
    
    try:
        selection_result = context['ti'].xcom_pull(
            task_ids='model_training.evaluate_and_select_best',
            key='selection_result'
        )
        
        best_model_name = selection_result['best_model_name']
        best_metrics = selection_result['best_metrics']
        
        print(f"\n Saving: {best_model_name.upper()}")
        print(f"   AUC: {best_metrics['auc']:.4f}")
        
        os.makedirs(model_artifacts_path, exist_ok=True)
        os.makedirs(model_metadata_path, exist_ok=True)
        
        source_model_file = os.path.join(temp_model_path, f'{best_model_name}.joblib')
        model = joblib.load(source_model_file)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{best_model_name}_{timestamp}.joblib"
        model_path = os.path.join(model_artifacts_path, model_filename)
        
        joblib.dump(model, model_path)
        print(f" Model artifact saved: {model_path}")
        
        metadata = {
            'model_name': best_model_name,
            'model_path': model_path,
            'timestamp': timestamp,
            'metrics': best_metrics,
            'feature_columns': selection_result['feature_columns'],
            'n_features': selection_result['n_features'],
            'all_models_comparison': selection_result['all_models'],
            'selection_criteria': 'auc',
            'data_leakage_prevention': {
                'temporal_split': True,
                'leakage_features_removed': True,
                'description': 'Removed dpd, overdue_amt, and other post-loan features'
            },
            'deployment_ready': True,
            'version': '1.0'
        }
        
        metadata_filename = f"{best_model_name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(model_metadata_path, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f" Model metadata saved: {metadata_path}")
        
        print("\n" + "="*60)
        print(" BEST MODEL SAVED TO PRODUCTION")
        print("="*60)
        print(f"Model: {best_model_name}")
        print(f"AUC: {best_metrics['auc']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"F1: {best_metrics['f1']:.4f}")
        print("="*60)
        
        context['ti'].xcom_push(key='production_model_path', value=model_path)
        context['ti'].xcom_push(key='production_metadata_path', value=metadata_path)
        
        return {
            'status': 'success',
            'model_path': model_path,
            'metadata_path': metadata_path,
            'best_model': best_model_name,
            'auc': best_metrics['auc']
        }
        
    except Exception as e:
        print(f" Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise