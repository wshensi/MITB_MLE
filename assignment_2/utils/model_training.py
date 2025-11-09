"""
Model Training Module
Trains multiple ML models and selects the best performer
Modified to train only 2 models: Logistic Regression & Random Forest
Fixed to remove data leakage features
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
    f1_score, classification_report, confusion_matrix
)
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
        
        # Identify label column - try multiple possible names
        possible_label_cols = [
            'label_default_30dpd',  # From data_processing_gold_features.py
            'default_flag',         # Alternative
            'label',               # Alternative
            'target'               # Alternative
        ]
        
        label_col = None
        for col in possible_label_cols:
            if col in train_df.columns:
                label_col = col
                print(f"✓ Found label column: {label_col}")
                break
        
        if label_col is None:
            # If no label found, show available columns for debugging
            label_candidates = [col for col in train_df.columns 
                              if 'label' in col.lower() or 'default' in col.lower()]
            raise ValueError(
                f"Could not find label column. Tried: {possible_label_cols}\n"
                f"Label-like columns available: {label_candidates}\n"
                f"All columns: {train_df.columns.tolist()}"
            )
        
        # ============================================================
        # CRITICAL: Remove features that cause data leakage
        # ============================================================
        # These features contain information about the target that would
        # not be available at prediction time (loan application time)
        
        data_leakage_features = [
            # Direct leakage - contains target information
            'dpd',                      # Days past due - this IS the default indicator
            'overdue_amt',              # Overdue amount
            'installments_missed',      # Number of missed installments
            'first_missed_date',        # When first missed payment occurred
            'balance',                  # Current balance (after defaults)
            'paid_amt',                 # Total paid (includes future payments)
            
            # Indirect leakage - loan performance metrics
            'payment_status',           # Current payment status
            'delinquency_status',       # Delinquency indicator
            'default_flag',             # Another default indicator
            'charged_off',              # Charge-off status
            'settlement_amt',           # Settlement amount
            'recovery_amt',             # Recovery amount
            
            # Time-based leakage
            'mob',                      # Months on book - can indicate loan maturity
            'installment_num',          # Current installment number
            'days_since_last_payment',  # Time since last payment
            'last_payment_date',        # Last payment date
            'maturity_date',            # Loan maturity date
            
            # Behavioral after loan start
            'payment_delay_days',       # Payment delay indicators
            'late_payment_count',       # Count of late payments
            'missed_payment_count',     # Count of missed payments
        ]
        
        # ID columns to exclude
        id_cols = ['user_id', 'loan_id', 'snapshot_date', 'Customer_ID', 'loan_start_date']
        
        # Print leakage features found (for debugging)
        found_leakage = [col for col in data_leakage_features if col in train_df.columns]
        if found_leakage:
            print(f"\n⚠️  WARNING: Removing {len(found_leakage)} potential data leakage features:")
            for feat in found_leakage:
                print(f"   - {feat}")
        
        # Select only valid features (exclude IDs, labels, and leakage features)
        exclude_cols = set(id_cols + [label_col] + data_leakage_features)
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        print(f"\n✓ Selected {len(feature_cols)} clean features (after removing leakage)")
        
        # Prepare feature matrix
        X = train_df[feature_cols].copy()
        y = train_df[label_col].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert to numeric and handle any remaining issues
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        # Remove constant features (no variance)
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            print(f"\n⚠️  Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
            feature_cols = [col for col in feature_cols if col not in constant_features]
        
        # Remove highly correlated features (correlation > 0.95)
        print("\nChecking for highly correlated features...")
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
        
        if to_drop:
            print(f"⚠️  Removing {len(to_drop)} highly correlated features (r > 0.95)")
            X = X.drop(columns=to_drop)
            feature_cols = [col for col in feature_cols if col not in to_drop]
        
        print(f"\n✓ Final feature count: {len(feature_cols)}")
        print(f"✓ Class distribution:")
        print(y.value_counts())
        
        return X, y, feature_cols
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train 2 models: Logistic Regression and Random Forest"""
        print("\n" + "="*60)
        print("Training Models (Logistic Regression & Random Forest)")
        print("="*60)
        
        # Handle class imbalance with SMOTE
        print("\nApplying SMOTE for class balance...")
        
        # Check if we have enough samples for SMOTE
        minority_class_count = min(y_train.value_counts())
        if minority_class_count < 6:
            print(f"⚠️  WARNING: Only {minority_class_count} minority samples. Skipping SMOTE.")
            X_train_balanced, y_train_balanced = X_train, y_train
        else:
            smote = SMOTE(random_state=self.config['random_state'], k_neighbors=min(5, minority_class_count-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"✓ After SMOTE - Training samples: {len(X_train_balanced)}")
            print(f"✓ Class distribution after SMOTE:")
            print(pd.Series(y_train_balanced).value_counts())
        
        # Define only 2 models with stronger regularization to prevent overfitting
        models_to_train = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.config['random_state'],
                class_weight='balanced',
                solver='liblinear',
                C=0.1,  # Strong regularization
                penalty='l2'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,              # Reduced from 10 to prevent overfitting
                min_samples_split=20,     # Increased from 10
                min_samples_leaf=10,      # Increased from 5
                max_features='sqrt',      # Use sqrt of features
                random_state=self.config['random_state'],
                class_weight='balanced',
                n_jobs=-1,
                max_samples=0.8           # Use only 80% of data per tree
            )
        }
        
        # Train each model
        for model_name, model in models_to_train.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()}")
            print(f"{'='*60}")
            
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
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Print results
                print(f"\n{'─'*60}")
                print(f"RESULTS:")
                print(f"{'─'*60}")
                print(f"  AUC:       {metrics['auc']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1 Score:  {metrics['f1']:.4f}")
                print(f"\n  Confusion Matrix:")
                print(f"                Predicted")
                print(f"                0        1")
                print(f"  Actual 0   {cm[0][0]:6d}   {cm[0][1]:6d}")
                print(f"  Actual 1   {cm[1][0]:6d}   {cm[1][1]:6d}")
                print(f"{'─'*60}")
                
                # Store results
                metrics['confusion_matrix'] = cm.tolist()
                self.models[model_name] = model
                self.results[model_name] = metrics
                
            except Exception as e:
                print(f"  ❌ Error training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return self.models, self.results
    
    def select_best_model(self, metric='auc'):
        """Select best model based on specified metric"""
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        best_model = self.models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL SELECTION")
        print(f"{'='*60}")
        print(f"  Winner: {best_model_name.upper()}")
        print(f"  Best {metric.upper()}: {best_metrics[metric]:.4f}")
        print(f"{'='*60}\n")
        
        return best_model_name, best_model, best_metrics
    
    def save_model(self, model_name, model, metrics, feature_cols, model_dir):
        """Save model artifacts and metadata"""
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': timestamp,
            'metrics': metrics,
            'feature_columns': feature_cols,
            'config': self.config,
            'data_leakage_prevention': {
                'temporal_split': True,
                'leakage_features_removed': True,
                'description': 'Removed dpd, overdue_amt, and other post-loan features'
            }
        }
        
        metadata_dir = model_dir.replace('model_artifacts', 'model_metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_filename = f"{model_name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path


def train_and_save_best_model(
    gold_feature_path, 
    model_dir, 
    config,
    training_cutoff_date="2024-06-01"
):
    """
    Main function to train models and save the best one
    Trains 2 models: Logistic Regression and Random Forest
    """
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE")
    print("Training: Logistic Regression & Random Forest")
    print("With Data Leakage Prevention")
    print("="*60)
    
    # Create model directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir.replace('model_artifacts', 'model_metadata'), exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Load and prepare data
    df = trainer.load_training_data(gold_feature_path)
    X, y, feature_cols = trainer.prepare_features(df, training_cutoff_date)
    
    # Check if we have enough data
    if len(X) < 100:
        raise ValueError(f"Not enough training samples: {len(X)}. Need at least 100.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"\n{'='*60}")
    print(f"DATA SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Train set: {len(X_train):,} samples")
    print(f"  Test set:  {len(X_test):,} samples")
    print(f"\n  Train label distribution:")
    for label, count in pd.Series(y_train).value_counts().items():
        print(f"    Class {label}: {count:,} ({count/len(y_train)*100:.1f}%)")
    print(f"\n  Test label distribution:")
    for label, count in pd.Series(y_test).value_counts().items():
        print(f"    Class {label}: {count:,} ({count/len(y_test)*100:.1f}%)")
    print(f"{'='*60}")
    
    # Train models
    models, results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    if not models:
        raise ValueError("No models were successfully trained!")
    
    # Select and save best model
    best_model_name, best_model, best_metrics = trainer.select_best_model(metric='auc')
    model_path, metadata_path = trainer.save_model(
        best_model_name, best_model, best_metrics, feature_cols, model_dir
    )
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"  Best Model: {best_model_name}")
    print(f"  Best AUC:   {best_metrics['auc']:.4f}")
    print(f"  Precision:  {best_metrics['precision']:.4f}")
    print(f"  Recall:     {best_metrics['recall']:.4f}")
    print(f"  F1 Score:   {best_metrics['f1']:.4f}")
    print("="*60)
    
    return {
        'best_model_name': best_model_name,
        'model_path': model_path,
        'metadata_path': metadata_path,
        'metrics': best_metrics,
        'all_results': results
    }