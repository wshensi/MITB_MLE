import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import glob

def load_model_for_inference(model_artifacts_path, model_metadata_path, **context):
    print("\n" + "="*60)
    print("TASK 3.1: LOADING MODEL FOR INFERENCE")
    print("="*60)
    
    try:
        model_files = glob.glob(os.path.join(model_artifacts_path, "*.joblib"))
        
        if not model_files:
            production_model_path = context['ti'].xcom_pull(
                task_ids='model_training.save_best_model',
                key='production_model_path'
            )
            
            if production_model_path and os.path.exists(production_model_path):
                latest_model_path = production_model_path
                print(f" Using newly trained model: {latest_model_path}")
            else:
                raise FileNotFoundError(f"No model files found in {model_artifacts_path}")
        else:
            latest_model_path = max(model_files, key=os.path.getctime)
            print(f" Found latest model: {latest_model_path}")
        
        model_basename = os.path.basename(latest_model_path).replace('.joblib', '_metadata.json')
        metadata_path = os.path.join(model_metadata_path, model_basename)
        
        print("\n Loading model...")
        model = joblib.load(latest_model_path)
        print(f" Model loaded successfully")
        
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"\n Model Information:")
            print(f"  Name: {metadata.get('model_name', 'unknown')}")
            print(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")
            print(f"  AUC: {metadata.get('metrics', {}).get('auc', 'N/A')}")
            print(f"  Features: {metadata.get('n_features', 'N/A')}")
        else:
            print(f"  Warning: Metadata file not found: {metadata_path}")
            metadata = {
                'model_name': 'unknown',
                'feature_columns': None
            }
        
        model_info = {
            'model_path': latest_model_path,
            'metadata_path': metadata_path,
            'model_name': metadata.get('model_name', 'unknown'),
            'feature_columns': metadata.get('feature_columns', None),
            'metrics': metadata.get('metrics', {})
        }
        
        temp_info_path = '/app/data/model_info.json'
        os.makedirs(os.path.dirname(temp_info_path), exist_ok=True)
        
        with open(temp_info_path, 'w') as f:
            json.dump(model_info, f, indent=4, default=str)
        
        print(f"\n Model info saved: {temp_info_path}")
        
        print("\n" + "="*60)
        print(" MODEL LOADED SUCCESSFULLY")
        print("="*60)
        
        context['ti'].xcom_push(key='model_info', value=model_info)
        context['ti'].xcom_push(key='model_loaded', value=True)
        
        return {
            'status': 'success',
            'model_path': latest_model_path,
            'model_name': metadata.get('model_name', 'unknown')
        }
        
    except Exception as e:
        print(f" Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def prepare_inference_data(gold_feature_path, inference_start_date, inference_end_date, 
                          output_path, **context):
    print("\n" + "="*60)
    print("TASK 3.2: PREPARING INFERENCE DATA")
    print("="*60)
    print(f"Date range: {inference_start_date} to {inference_end_date}")
    print(f"Input: {gold_feature_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    try:
        model_info = context['ti'].xcom_pull(
            task_ids='model_inference.load_model',
            key='model_info'
        )
        
        feature_columns = model_info.get('feature_columns', None)
        print(f"\n Model expects {len(feature_columns) if feature_columns else 'unknown'} features")
        
        print("\n Loading gold feature data...")
        files = glob.glob(os.path.join(gold_feature_path, "*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found in {gold_feature_path}")
        
        df_list = []
        for file in files:
            df = pd.read_parquet(file)
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            
            mask = (df['snapshot_date'] >= pd.to_datetime(inference_start_date)) & \
                   (df['snapshot_date'] <= pd.to_datetime(inference_end_date))
            df_filtered = df[mask]
            
            if len(df_filtered) > 0:
                df_list.append(df_filtered)
        
        if not df_list:
            raise ValueError(f"No data found in date range {inference_start_date} to {inference_end_date}")
        
        df = pd.concat(df_list, ignore_index=True)
        print(f" Loaded {len(df):,} samples for inference")
        
        possible_customer_ids = ['Customer_ID', 'user_id', 'customer_id', 'cust_id']
        customer_id_col = None
        
        for col in possible_customer_ids:
            if col in df.columns:
                customer_id_col = col
                break
        
        if customer_id_col is None:
            raise ValueError(f"Could not find customer ID column. Tried: {possible_customer_ids}")
        
        print(f" Using customer ID column: {customer_id_col}")
        
        id_cols = [customer_id_col, 'loan_id', 'snapshot_date']
        id_cols = [col for col in id_cols if col in df.columns]
        
        id_df = df[id_cols].copy()
        
        if customer_id_col != 'Customer_ID':
            id_df = id_df.rename(columns={customer_id_col: 'Customer_ID'})
        
        print("\n Preparing features...")
        
        if feature_columns:
            print(f"  Using {len(feature_columns)} features from training")
            
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                print(f"    Adding {len(missing_cols)} missing features with zero values")
                for col in missing_cols:
                    df[col] = 0
            
            X = df[feature_columns].copy()
        else:
            print("    No feature list found, using all non-ID columns")
            exclude_cols = id_cols + ['label_default_30dpd', 'default_flag', 'label', 'target']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols].copy()
        
        X = X.fillna(0)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        print(f" Prepared {len(X):,} samples with {len(X.columns)} features")
        
        os.makedirs(output_path, exist_ok=True)
        
        X.to_parquet(os.path.join(output_path, 'X_inference.parquet'), index=False)
        id_df.to_parquet(os.path.join(output_path, 'id_data.parquet'), index=False)
        
        inference_metadata = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'inference_start_date': inference_start_date,
            'inference_end_date': inference_end_date,
            'prepared_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_path, 'inference_metadata.json'), 'w') as f:
            json.dump(inference_metadata, f, indent=4)
        
        print(f"\n Data saved to {output_path}")
        
        print("\n" + "="*60)
        print(" INFERENCE DATA PREPARATION COMPLETED")
        print("="*60)
        print(f"Samples: {len(X):,}")
        print(f"Features: {len(X.columns)}")
        print("="*60)
        
        context['ti'].xcom_push(key='inference_samples', value=len(X))
        context['ti'].xcom_push(key='inference_data_ready', value=True)
        
        return {
            'status': 'success',
            'n_samples': len(X),
            'n_features': len(X.columns)
        }
        
    except Exception as e:
        print(f" Error preparing inference data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def execute_predictions(model_info_path, inference_data_path, predictions_output_path,
                       prediction_date, **context):
    print("\n" + "="*60)
    print("TASK 3.3: EXECUTING PREDICTIONS")
    print("="*60)
    print(f"Prediction date: {prediction_date}")
    print(f"Output: {predictions_output_path}")
    print("="*60)
    
    try:
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        print(f"\n Using model: {model_info['model_name']}")
        
        model = joblib.load(model_info['model_path'])
        print(" Model loaded")
        
        print("\n Loading inference data...")
        X = pd.read_parquet(os.path.join(inference_data_path, 'X_inference.parquet'))
        id_df = pd.read_parquet(os.path.join(inference_data_path, 'id_data.parquet'))
        
        print(f" Loaded {len(X):,} samples")
        
        print("\n Making predictions...")
        
        pred_proba = model.predict_proba(X)[:, 1]
        pred_binary = model.predict(X)
        
        print(f" Predictions completed")
        print(f"  Mean probability: {pred_proba.mean():.4f}")
        print(f"  Predicted defaults: {pred_binary.sum():,} ({pred_binary.mean()*100:.2f}%)")
        
        predictions_df = id_df.copy()
        predictions_df['prediction_probability'] = pred_proba
        predictions_df['prediction_binary'] = pred_binary
        predictions_df['model_name'] = model_info['model_name']
        predictions_df['model_version'] = model_info.get('metrics', {}).get('auc', 'unknown')
        predictions_df['prediction_date'] = prediction_date
        predictions_df['predicted_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        os.makedirs(predictions_output_path, exist_ok=True)
        
        output_file = os.path.join(
            predictions_output_path,
            f"predictions_{prediction_date.replace('-', '')}.parquet"
        )
        
        predictions_df.to_parquet(output_file, index=False)
        
        print(f"\n Predictions saved: {output_file}")
        
        summary = {
            'prediction_date': prediction_date,
            'n_predictions': len(predictions_df),
            'mean_probability': float(pred_proba.mean()),
            'std_probability': float(pred_proba.std()),
            'predicted_defaults': int(pred_binary.sum()),
            'default_rate': float(pred_binary.mean()),
            'model_name': model_info['model_name'],
            'model_auc': model_info.get('metrics', {}).get('auc', 'N/A')
        }
        
        summary_file = os.path.join(predictions_output_path, f"summary_{prediction_date.replace('-', '')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f" Summary saved: {summary_file}")
        
        print("\n" + "="*60)
        print(" PREDICTIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total predictions: {len(predictions_df):,}")
        print(f"Average probability: {pred_proba.mean():.4f}")
        print(f"Predicted defaults: {pred_binary.sum():,} ({pred_binary.mean()*100:.2f}%)")
        print("="*60)
        
        context['ti'].xcom_push(key='predictions_file', value=output_file)
        context['ti'].xcom_push(key='n_predictions', value=len(predictions_df))
        context['ti'].xcom_push(key='default_rate', value=float(pred_binary.mean()))
        
        return {
            'status': 'success',
            'output_file': output_file,
            'n_predictions': len(predictions_df),
            'mean_probability': float(pred_proba.mean()),
            'default_rate': float(pred_binary.mean())
        }
        
    except Exception as e:
        print(f" Error executing predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise