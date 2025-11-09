"""
Model Inference Module
Loads trained model and makes predictions on new data
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import glob


class ModelInference:
    """Load model and make predictions"""
    
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_columns = None
    
    def load_latest_model(self, model_dir, metadata_dir):
        """Load the most recent model from directory"""
        # Find latest model file
        model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        latest_model_path = max(model_files, key=os.path.getctime)
        
        # Find corresponding metadata
        model_basename = os.path.basename(latest_model_path).replace('.joblib', '_metadata.json')
        metadata_path = os.path.join(metadata_dir, model_basename)
        
        print(f"Loading latest model: {latest_model_path}")
        self.load_model(latest_model_path, metadata_path)
        
        return latest_model_path, metadata_path
    
    def load_model(self, model_path, metadata_path):
        """Load model and metadata"""
        # Load model
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_columns = self.metadata['feature_columns']
            print(f"Model: {self.metadata['model_name']}")
            print(f"Training metrics: {self.metadata['metrics']}")
        else:
            print(f"Warning: Metadata file not found: {metadata_path}")
    
    def load_inference_data(self, gold_feature_path, start_date, end_date):
        """Load data for inference within date range"""
        print(f"Loading inference data from {start_date} to {end_date}")
        
        # Read all parquet files
        files = glob.glob(os.path.join(gold_feature_path, "*.parquet"))
        
        if not files:
            print(f"No parquet files found in {gold_feature_path}")
            return None
        
        df_list = []
        for file in files:
            df = pd.read_parquet(file)
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            
            # Filter by date range
            mask = (df['snapshot_date'] >= pd.to_datetime(start_date)) & \
                   (df['snapshot_date'] <= pd.to_datetime(end_date))
            df_filtered = df[mask]
            
            if len(df_filtered) > 0:
                df_list.append(df_filtered)
        
        if not df_list:
            print(f"No data found in date range {start_date} to {end_date}")
            return None
        
        df = pd.concat(df_list, ignore_index=True)
        print(f"Total samples for inference: {len(df)}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for inference"""
        # Store ID columns
        id_cols = ['user_id', 'loan_id', 'snapshot_date']
        id_df = df[id_cols].copy()
        
        # Get features in same order as training
        if self.feature_columns:
            # Use exact features from training
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in inference data: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            
            X = df[self.feature_columns].copy()
        else:
            # Fallback: use all columns except IDs
            feature_cols = [col for col in df.columns if col not in id_cols]
            X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        return X, id_df
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get probability scores
        pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Get binary predictions (default threshold 0.5)
        pred_binary = self.model.predict(X)
        
        return pred_proba, pred_binary
    
    def save_predictions(self, id_df, pred_proba, pred_binary, output_path, snapshot_date):
        """Save predictions to gold table"""
        # Create predictions dataframe
        predictions_df = id_df.copy()
        predictions_df['prediction_probability'] = pred_proba
        predictions_df['prediction_binary'] = pred_binary
        predictions_df['model_name'] = self.metadata['model_name'] if self.metadata else 'unknown'
        predictions_df['model_version'] = self.metadata['timestamp'] if self.metadata else 'unknown'
        predictions_df['prediction_date'] = datetime.now().strftime("%Y-%m-%d")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save as parquet
        output_file = os.path.join(
            output_path, 
            f"predictions_{snapshot_date.replace('-', '')}.parquet"
        )
        predictions_df.to_parquet(output_file, index=False)
        
        print(f"\nPredictions saved to: {output_file}")
        print(f"Number of predictions: {len(predictions_df)}")
        print(f"Average prediction probability: {pred_proba.mean():.4f}")
        print(f"Predicted defaults: {pred_binary.sum()} ({pred_binary.mean()*100:.2f}%)")
        
        return output_file, predictions_df


def run_inference_for_period(
    model_dir,
    metadata_dir,
    gold_feature_path,
    predictions_output_path,
    start_date,
    end_date
):
    """Run inference for a time period"""
    print("\n" + "="*60)
    print("STARTING MODEL INFERENCE")
    print("="*60)
    
    # Initialize inference engine
    inference = ModelInference()
    
    # Load latest model
    try:
        model_path, metadata_path = inference.load_latest_model(model_dir, metadata_dir)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Load inference data
    df = inference.load_inference_data(gold_feature_path, start_date, end_date)
    
    if df is None or len(df) == 0:
        print("No data available for inference.")
        return None
    
    # Prepare features
    X, id_df = inference.prepare_features(df)
    
    # Make predictions
    print("\nMaking predictions...")
    pred_proba, pred_binary = inference.predict(X)
    
    # Save predictions
    output_file, predictions_df = inference.save_predictions(
        id_df, pred_proba, pred_binary, 
        predictions_output_path, end_date
    )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return {
        'output_file': output_file,
        'num_predictions': len(predictions_df),
        'mean_probability': float(pred_proba.mean()),
        'predicted_defaults': int(pred_binary.sum()),
        'default_rate': float(pred_binary.mean())
    }