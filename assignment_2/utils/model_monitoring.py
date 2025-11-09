"""
Model Monitoring Module
Monitors model performance and data drift over time
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class ModelMonitor:
    """Monitor model performance and stability"""
    
    def __init__(self):
        self.performance_metrics = None
        self.drift_metrics = None
    
    def load_predictions(self, predictions_path):
        """Load all prediction files"""
        files = glob.glob(os.path.join(predictions_path, "*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No prediction files found in {predictions_path}")
        
        df_list = []
        for file in files:
            df = pd.read_parquet(file)
            df_list.append(df)
        
        predictions_df = pd.concat(df_list, ignore_index=True)
        predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date'])
        
        print(f"Loaded {len(predictions_df)} predictions from {len(files)} files")
        
        return predictions_df
    
    def load_actuals(self, gold_feature_path):
        """Load actual labels from gold feature store"""
        files = glob.glob(os.path.join(gold_feature_path, "*.parquet"))
        
        df_list = []
        for file in files:
            df = pd.read_parquet(file)
            # Keep ID and label columns
            label_col = 'default_flag' if 'default_flag' in df.columns else 'label'
            keep_cols = ['user_id', 'loan_id', 'snapshot_date', label_col]
            df = df[[col for col in keep_cols if col in df.columns]].copy()
            if label_col in df.columns and label_col != 'default_flag':
                df = df.rename(columns={label_col: 'default_flag'})
            df_list.append(df)
        
        actuals_df = pd.concat(df_list, ignore_index=True)
        actuals_df['snapshot_date'] = pd.to_datetime(actuals_df['snapshot_date'])
        
        print(f"Loaded {len(actuals_df)} actual labels")
        
        return actuals_df
    
    def calculate_performance_metrics(self, predictions_df, actuals_df):
        """Calculate performance metrics by joining predictions with actuals"""
        # Merge predictions with actuals
        merged_df = predictions_df.merge(
            actuals_df,
            on=['user_id', 'loan_id', 'snapshot_date'],
            how='inner'
        )
        
        print(f"Matched {len(merged_df)} predictions with actuals")
        
        if len(merged_df) == 0:
            print("Warning: No predictions matched with actuals")
            return None
        
        # Group by snapshot_date and calculate metrics
        results = []
        
        for date in sorted(merged_df['snapshot_date'].unique()):
            date_df = merged_df[merged_df['snapshot_date'] == date]
            
            y_true = date_df['default_flag']
            y_pred_proba = date_df['prediction_probability']
            y_pred_binary = date_df['prediction_binary']
            
            # Skip if only one class present
            if len(y_true.unique()) < 2:
                print(f"Skipping {date}: Only one class present")
                continue
            
            # Calculate metrics
            metrics = {
                'snapshot_date': date.strftime('%Y-%m-%d'),
                'num_samples': len(date_df),
                'actual_default_rate': float(y_true.mean()),
                'predicted_default_rate': float(y_pred_binary.mean()),
                'auc': float(roc_auc_score(y_true, y_pred_proba)),
                'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred_binary, zero_division=0))
            }
            
            results.append(metrics)
        
        if not results:
            return None
        
        performance_df = pd.DataFrame(results)
        self.performance_metrics = performance_df
        
        return performance_df
    
    def calculate_psi(self, expected, actual, bins=10):
        """Calculate Population Stability Index (PSI)"""
        # Create bins
        breakpoints = np.linspace(0, 1, bins + 1)
        
        # Bin the data
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Add small value to avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def calculate_drift_metrics(self, predictions_df, baseline_date):
        """Calculate data drift metrics using PSI"""
        baseline_date = pd.to_datetime(baseline_date)
        
        # Get baseline predictions
        baseline_df = predictions_df[
            predictions_df['snapshot_date'] == baseline_date
        ]
        
        if len(baseline_df) == 0:
            print(f"Warning: No baseline data found for {baseline_date}")
            return None
        
        baseline_scores = baseline_df['prediction_probability'].values
        
        # Calculate PSI for each date
        results = []
        
        for date in sorted(predictions_df['snapshot_date'].unique()):
            if date == baseline_date:
                continue
            
            date_df = predictions_df[predictions_df['snapshot_date'] == date]
            actual_scores = date_df['prediction_probability'].values
            
            psi = self.calculate_psi(baseline_scores, actual_scores)
            
            results.append({
                'snapshot_date': date.strftime('%Y-%m-%d'),
                'psi': psi,
                'drift_severity': self.interpret_psi(psi)
            })
        
        if not results:
            return None
        
        drift_df = pd.DataFrame(results)
        self.drift_metrics = drift_df
        
        return drift_df
    
    def interpret_psi(self, psi):
        """Interpret PSI value"""
        if psi < 0.1:
            return 'No significant drift'
        elif psi < 0.2:
            return 'Minor drift'
        else:
            return 'Major drift - action required'
    
    def visualize_metrics(self, output_path):
        """Create visualizations of performance and drift metrics"""
        os.makedirs(output_path, exist_ok=True)
        
        # Plot 1: Performance metrics over time
        if self.performance_metrics is not None and len(self.performance_metrics) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Performance Over Time', fontsize=16)
            
            metrics_to_plot = ['auc', 'precision', 'recall', 'f1']
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx // 2, idx % 2]
                self.performance_metrics.plot(
                    x='snapshot_date', 
                    y=metric, 
                    ax=ax, 
                    marker='o',
                    title=f'{metric.upper()} Over Time'
                )
                ax.set_xlabel('Date')
                ax.set_ylabel(metric.upper())
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            performance_plot_path = os.path.join(output_path, 'performance_metrics.png')
            plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Performance plot saved: {performance_plot_path}")
        
        # Plot 2: Drift metrics over time
        if self.drift_metrics is not None and len(self.drift_metrics) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            self.drift_metrics.plot(
                x='snapshot_date',
                y='psi',
                ax=ax,
                marker='o',
                title='Population Stability Index (PSI) Over Time'
            )
            ax.axhline(y=0.1, color='orange', linestyle='--', label='Minor drift threshold')
            ax.axhline(y=0.2, color='red', linestyle='--', label='Major drift threshold')
            ax.set_xlabel('Date')
            ax.set_ylabel('PSI')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            drift_plot_path = os.path.join(output_path, 'drift_metrics.png')
            plt.savefig(drift_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Drift plot saved: {drift_plot_path}")
        
        return True
    
    def save_metrics(self, output_path):
        """Save metrics to parquet files"""
        os.makedirs(output_path, exist_ok=True)
        
        # Save performance metrics
        if self.performance_metrics is not None:
            perf_path = os.path.join(output_path, 'performance_metrics.parquet')
            self.performance_metrics.to_parquet(perf_path, index=False)
            print(f"Performance metrics saved: {perf_path}")
        
        # Save drift metrics
        if self.drift_metrics is not None:
            drift_path = os.path.join(output_path, 'drift_metrics.parquet')
            self.drift_metrics.to_parquet(drift_path, index=False)
            print(f"Drift metrics saved: {drift_path}")
        
        return True
    
    def check_retraining_needed(self, performance_threshold, drift_threshold):
        """Check if model retraining is needed based on governance policy"""
        needs_retraining = False
        reasons = []
        
        if self.performance_metrics is not None and len(self.performance_metrics) > 0:
            latest_auc = self.performance_metrics['auc'].iloc[-1]
            if latest_auc < performance_threshold.get('auc', 0.65):
                needs_retraining = True
                reasons.append(f"AUC below threshold: {latest_auc:.4f} < {performance_threshold['auc']}")
        
        if self.drift_metrics is not None and len(self.drift_metrics) > 0:
            latest_psi = self.drift_metrics['psi'].iloc[-1]
            if latest_psi > drift_threshold:
                needs_retraining = True
                reasons.append(f"PSI above threshold: {latest_psi:.4f} > {drift_threshold}")
        
        return needs_retraining, reasons


def run_monitoring(
    predictions_path,
    gold_feature_path,
    monitoring_output_path,
    baseline_date,
    performance_threshold={'auc': 0.65, 'precision': 0.70},
    drift_threshold=0.1
):
    """Run monitoring pipeline"""
    print("\n" + "="*60)
    print("STARTING MODEL MONITORING")
    print("="*60)
    
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Load data
    try:
        predictions_df = monitor.load_predictions(predictions_path)
        actuals_df = monitor.load_actuals(gold_feature_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    performance_df = monitor.calculate_performance_metrics(predictions_df, actuals_df)
    
    if performance_df is not None:
        print("\nPerformance Metrics Summary:")
        print(performance_df.to_string())
    
    # Calculate drift metrics
    print("\nCalculating drift metrics...")
    drift_df = monitor.calculate_drift_metrics(predictions_df, baseline_date)
    
    if drift_df is not None:
        print("\nDrift Metrics Summary:")
        print(drift_df.to_string())
    
    # Visualize metrics
    print("\nCreating visualizations...")
    monitor.visualize_metrics(monitoring_output_path)
    
    # Save metrics
    print("\nSaving metrics...")
    monitor.save_metrics(monitoring_output_path)
    
    # Check if retraining needed
    needs_retraining, reasons = monitor.check_retraining_needed(
        performance_threshold, drift_threshold
    )
    
    if needs_retraining:
        print("\n" + "="*60)
        print("⚠️  RETRAINING RECOMMENDED")
        print("="*60)
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("\n✓ Model performance is satisfactory")
    
    print("\n" + "="*60)
    print("MONITORING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return {
        'performance_file': os.path.join(monitoring_output_path, 'performance_metrics.parquet'),
        'drift_file': os.path.join(monitoring_output_path, 'drift_metrics.parquet'),
        'performance_plot': os.path.join(monitoring_output_path, 'performance_metrics.png'),
        'drift_plot': os.path.join(monitoring_output_path, 'drift_metrics.png'),
        'needs_retraining': needs_retraining,
        'retraining_reasons': reasons
    }