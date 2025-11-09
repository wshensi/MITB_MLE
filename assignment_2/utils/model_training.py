"""
Model Training Module
Trains multiple ML models and selects the best performer
"""
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
    f1_score, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import glob


class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
        
    def load_training_data(self, gold_feature_path):
        """Load features and labels from gold table"""
        print(f"Loading training data from {gold_feature_path}")
        
        # Read all parquet files
        files = glob.glob(os.path.join(gold_feature_path, "*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found in {gold_feature_path}")
        
        df_list = []
        for file in files:
            df = pd.read_parquet(file)
            df_list.append(df)
        
        df = pd.concat(df_list, ignore_index=True)
        print(f"Total samples loaded: {len(df)}")
        
        return df
    
    def prepare_features(self, df, training_cutoff_date):
        """
        Prepare features and labels with temporal split to avoid data leakage
        """
        # Convert snapshot_date to datetime
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
        
        # Filter training data by cutoff date to avoid temporal leakage
        train_df = df[df['snapshot_date'] < pd.to_datetime(training_cutoff_date)].copy()
        
        print(f"Training samples before {training_cutoff_date}: {len(train_df)}")
        
        # Identify label column
        label_col = 'default_flag' if 'default_flag' in train_df.columns else 'label'
        
        # Separate features and labels
        id_cols = ['user_id', 'loan_id', 'snapshot_date']
        
        # Drop ID columns and get feature columns
        feature_cols = [col for col in train_df.columns 
                       if col not in id_cols + [label_col]]
        
        X = train_df[feature_cols].copy()
        y = train_df[label_col].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        return X, y, feature_cols
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple model types"""
        print("\n" + "="*50)
        print("Training Models")
        print("="*50)
        
        # Handle class imbalance with SMOTE
        print("Applying SMOTE for class balance...")
        smote = SMOTE(random_state=self.config['random_state'])
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Training samples: {len(X_train_balanced)}")
        
        # Define models
        models_to_train = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.config['random_state'],
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config['random_state'],
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state'],
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state'],
                class_weight='balanced',
                verbose=-1
            )
        }
        
        # Train each model
        for model_name, model in models_to_train.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train
                model.fit(X_train_balanced, y_train_balanced)
                
                # Predict
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                
                # Evaluate
                metrics = {
                    'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                }
                
                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")
                
                # Store model and results
                self.models[model_name] = model
                self.results[model_name] = metrics
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                continue
        
        return self.models, self.results
    
    def select_best_model(self, metric='auc'):
        """Select best model based on specified metric"""
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        best_model = self.models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        print(f"\n{'='*50}")
        print(f"Best Model: {best_model_name}")
        print(f"Best {metric.upper()}: {best_metrics[metric]:.4f}")
        print(f"{'='*50}\n")
        
        return best_model_name, best_model, best_metrics
    
    def save_model(self, model_name, model, metrics, feature_cols, model_dir):
        """Save model artifacts and metadata"""
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': timestamp,
            'metrics': metrics,
            'feature_columns': feature_cols,
            'config': self.config
        }
        
        metadata_dir = model_dir.replace('model_artifacts', 'model_metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_filename = f"{model_name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f"Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path


def train_and_save_best_model(
    gold_feature_path, 
    model_dir, 
    config,
    training_cutoff_date="2024-06-01"
):
    """
    Main function to train models and save the best one
    """
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Create model directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir.replace('model_artifacts', 'model_metadata'), exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Load and prepare data
    df = trainer.load_training_data(gold_feature_path)
    X, y, feature_cols = trainer.prepare_features(df, training_cutoff_date)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train models
    models, results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Select and save best model
    best_model_name, best_model, best_metrics = trainer.select_best_model(metric='auc')
    model_path, metadata_path = trainer.save_model(
        best_model_name, best_model, best_metrics, feature_cols, model_dir
    )
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return {
        'best_model_name': best_model_name,
        'model_path': model_path,
        'metadata_path': metadata_path,
        'metrics': best_metrics,
        'all_results': results
    }