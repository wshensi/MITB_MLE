"""
Model Monitoring Module
Monitors model performance and data drift over time
Fixed: Column name consistency (Customer_ID, label_default_30dpd)
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
        
        print(f"✓ Loaded {len(predictions_df):,} predictions from {len(files)} files")
        
        return predictions_df
    
    def load_actuals(self, gold_feature_path):
        """Load actual labels from gold feature store"""
        files = glob.glob(os.path.join(gold_feature_path, "*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No files found in {gold_feature_path}")
        
        df_list = []
        for file in files:
            df = pd.read_parquet(file)
            
            # Smart detection of customer ID column
            possible_customer_ids = ['Customer_ID', 'user_id', 'customer_id', 'cust_id']
            customer_id_col = None
            for col in possible_customer_ids:
                if col in df.columns:
                    customer_id_col = col
                    break
            
            if customer_id_col is None:
                print(f"⚠️  Warning: No customer ID column found in {file}")
                continue
            
            # Smart detection of label column
            possible_label_cols = ['label_default_30dpd', 'default_flag', 'label', 'target']
            label_col = None
            for col in possible_label_cols:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                print(f"⚠️  Warning: No label column found in {file}")
                continue
            
            # Keep only necessary columns
            keep_cols = [customer_id_col, 'loan_id', 'snapshot_date', label_col]
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols].copy()
            
            # Standardize column names
            if customer_id_col != 'Customer_ID':
                df = df.rename(columns={customer_id_col: 'Customer_ID'})
            
            if label_col != 'actual_label':
                df = df.rename(columns={label_col: 'actual_label'})
            
            df_list.append(df)
        
        if not df_list:
            raise ValueError("No valid data found in gold feature store")
        
        actuals_df = pd.concat(df_list, ignore_index=True)
        actuals_df['snapshot_date'] = pd.to_datetime(actuals_df['snapshot_date'])
        
        print(f"✓ Loaded {len(actuals_df):,} actual labels")
        
        return actuals_df
    
    def calculate_performance_metrics(self, predictions_df, actuals_df):
        """Calculate performance metrics by joining predictions with actuals"""
        print(f"\n{'='*60}")
        print("Calculating Performance Metrics")
        print(f"{'='*60}")
        
        # Check if Customer_ID exists in predictions
        if 'Customer_ID' not in predictions_df.columns:
            print("⚠️  Customer_ID not in predictions, checking for alternatives...")
            # Try to find alternative customer ID column
            for col in ['user_id', 'customer_id', 'cust_id']:
                if col in predictions_df.columns:
                    predictions_df = predictions_df.rename(columns={col: 'Customer_ID'})
                    print(f"✓ Renamed {col} to Customer_ID")
                    break
        
        # Merge predictions with actuals
        merge_cols = ['Customer_ID', 'loan_id', 'snapshot_date']
        
        # Check which columns are available for merging
        available_merge_cols = [col for col in merge_cols if col in predictions_df.columns and col in actuals_df.columns]
        
        if not available_merge_cols:
            raise ValueError(
                f"No common columns for merging.\n"
                f"Predictions columns: {predictions_df.columns.tolist()}\n"
                f"Actuals columns: {actuals_df.columns.tolist()}"
            )
        
        print(f"✓ Merging on: {available_merge_cols}")
        
        merged_df = predictions_df.merge(
            actuals_df,
            on=available_merge_cols,
            how='inner'
        )
        
        print(f"✓ Matched {len(merged_df):,} predictions with actuals")
        
        if len(merged_df) == 0:
            print("❌ Warning: No predictions matched with actuals")
            return None
        
        # Group by snapshot_date and calculate metrics
        results = []
        
        for date in sorted(merged_df['snapshot_date'].unique()):
            date_df = merged_df[merged_df['snapshot_date'] == date]
            
            y_true = date_df['actual_label']
            y_pred_proba = date_df['prediction_probability']
            y_pred_binary = date_df['prediction_binary']
            
            # Skip if only one class present
            if len(y_true.unique()) < 2:
                print(f"⚠️  Skipping {date}: Only one class present")
                continue
            
            try:
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
                
            except Exception as e:
                print(f"⚠️  Error calculating metrics for {date}: {str(e)}")
                continue
        
        if not results:
            print("❌ No valid metrics calculated")
            return None
        
        performance_df = pd.DataFrame(results)
        self.performance_metrics = performance_df
        
        print(f"✓ Calculated metrics for {len(performance_df)} time periods")
        
        return performance_df
    
    def calculate_psi(self, expected, actual, bins=10):
        """Calculate Population Stability Index (PSI)"""
        try:
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
        except Exception as e:
            print(f"⚠️  Error calculating PSI: {str(e)}")
            return None
    
    def calculate_drift_metrics(self, predictions_df, baseline_date):
        """Calculate data drift metrics using PSI"""
        print(f"\n{'='*60}")
        print("Calculating Drift Metrics")
        print(f"{'='*60}")
        
        baseline_date = pd.to_datetime(baseline_date)
        
        # Get baseline predictions
        baseline_df = predictions_df[
            predictions_df['snapshot_date'] == baseline_date
        ]
        
        if len(baseline_df) == 0:
            print(f"⚠️  Warning: No baseline data found for {baseline_date}")
            # Use first available date as baseline
            baseline_date = predictions_df['snapshot_date'].min()
            baseline_df = predictions_df[predictions_df['snapshot_date'] == baseline_date]
            print(f"✓ Using {baseline_date} as baseline instead")
        
        baseline_scores = baseline_df['prediction_probability'].values
        
        # Calculate PSI for each date
        results = []
        
        for date in sorted(predictions_df['snapshot_date'].unique()):
            if date == baseline_date:
                continue
            
            date_df = predictions_df[predictions_df['snapshot_date'] == date]
            actual_scores = date_df['prediction_probability'].values
            
            psi = self.calculate_psi(baseline_scores, actual_scores)
            
            if psi is not None:
                results.append({
                    'snapshot_date': date.strftime('%Y-%m-%d'),
                    'psi': psi,
                    'drift_severity': self.interpret_psi(psi)
                })
        
        if not results:
            print("❌ No drift metrics calculated")
            return None
        
        drift_df = pd.DataFrame(results)
        self.drift_metrics = drift_df
        
        print(f"✓ Calculated drift metrics for {len(drift_df)} time periods")
        
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
        
        print(f"\n{'='*60}")
        print("Creating Visualizations")
        print(f"{'='*60}")
        
        # Plot 1: Performance metrics over time
        if self.performance_metrics is not None and len(self.performance_metrics) > 0:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Model Performance Over Time', fontsize=16, fontweight='bold')
                
                metrics_to_plot = ['auc', 'precision', 'recall', 'f1']
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
                
                for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
                    ax = axes[idx // 2, idx % 2]
                    
                    ax.plot(
                        self.performance_metrics['snapshot_date'],
                        self.performance_metrics[metric],
                        marker='o',
                        linewidth=2,
                        markersize=8,
                        color=color,
                        label=metric.upper()
                    )
                    
                    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
                    ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
                    ax.set_title(f'{metric.upper()} Over Time', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add horizontal line for reference
                    if metric == 'auc':
                        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
                        ax.legend()
                
                plt.tight_layout()
                performance_plot_path = os.path.join(output_path, 'performance_metrics.png')
                plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Performance plot saved: {performance_plot_path}")
                
            except Exception as e:
                print(f"⚠️  Error creating performance plot: {str(e)}")
        
        # Plot 2: Drift metrics over time
        if self.drift_metrics is not None and len(self.drift_metrics) > 0:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(
                    self.drift_metrics['snapshot_date'],
                    self.drift_metrics['psi'],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    color='#FF6B35',
                    label='PSI'
                )
                
                ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, 
                          label='Minor drift threshold (0.1)', alpha=0.7)
                ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, 
                          label='Major drift threshold (0.2)', alpha=0.7)
                
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('PSI', fontsize=12, fontweight='bold')
                ax.set_title('Population Stability Index (PSI) Over Time', 
                            fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                drift_plot_path = os.path.join(output_path, 'drift_metrics.png')
                plt.savefig(drift_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Drift plot saved: {drift_plot_path}")
                
            except Exception as e:
                print(f"⚠️  Error creating drift plot: {str(e)}")
        
        return True
    
    def save_metrics(self, output_path):
        """Save metrics to parquet files"""
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Saving Metrics")
        print(f"{'='*60}")
        
        # Save performance metrics
        if self.performance_metrics is not None:
            perf_path = os.path.join(output_path, 'performance_metrics.parquet')
            self.performance_metrics.to_parquet(perf_path, index=False)
            print(f"✓ Performance metrics saved: {perf_path}")
        
        # Save drift metrics
        if self.drift_metrics is not None:
            drift_path = os.path.join(output_path, 'drift_metrics.parquet')
            self.drift_metrics.to_parquet(drift_path, index=False)
            print(f"✓ Drift metrics saved: {drift_path}")
        
        return True
    
    def check_retraining_needed(self, performance_threshold, drift_threshold):
        """Check if model retraining is needed based on governance policy"""
        needs_retraining = False
        reasons = []
        
        if self.performance_metrics is not None and len(self.performance_metrics) > 0:
            latest_auc = self.performance_metrics['auc'].iloc[-1]
            threshold_auc = performance_threshold.get('auc', 0.65)
            
            if latest_auc < threshold_auc:
                needs_retraining = True
                reasons.append(
                    f"AUC below threshold: {latest_auc:.4f} < {threshold_auc}"
                )
        
        if self.drift_metrics is not None and len(self.drift_metrics) > 0:
            latest_psi = self.drift_metrics['psi'].iloc[-1]
            
            if latest_psi > drift_threshold:
                needs_retraining = True
                reasons.append(
                    f"PSI above threshold: {latest_psi:.4f} > {drift_threshold}"
                )
        
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
    print("MODEL MONITORING PIPELINE")
    print("="*60)
    print(f"Baseline date: {baseline_date}")
    print(f"Performance thresholds: {performance_threshold}")
    print(f"Drift threshold: {drift_threshold}")
    print("="*60)
    
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Load data
    try:
        predictions_df = monitor.load_predictions(predictions_path)
        actuals_df = monitor.load_actuals(gold_feature_path)
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate performance metrics
    performance_df = monitor.calculate_performance_metrics(predictions_df, actuals_df)
    
    if performance_df is not None and len(performance_df) > 0:
        print(f"\n{'='*60}")
        print("Performance Metrics Summary")
        print(f"{'='*60}")
        print(performance_df.to_string(index=False))
    else:
        print("\n⚠️  No performance metrics calculated")
    
    # Calculate drift metrics
    drift_df = monitor.calculate_drift_metrics(predictions_df, baseline_date)
    
    if drift_df is not None and len(drift_df) > 0:
        print(f"\n{'='*60}")
        print("Drift Metrics Summary")
        print(f"{'='*60}")
        print(drift_df.to_string(index=False))
    else:
        print("\n⚠️  No drift metrics calculated")
    
    # Visualize metrics
    monitor.visualize_metrics(monitoring_output_path)
    
    # Save metrics
    monitor.save_metrics(monitoring_output_path)
    
    # Check if retraining needed
    needs_retraining, reasons = monitor.check_retraining_needed(
        performance_threshold, drift_threshold
    )
    
    print(f"\n{'='*60}")
    if needs_retraining:
        print("⚠️  RETRAINING RECOMMENDED")
        print(f"{'='*60}")
        for reason in reasons:
            print(f"  • {reason}")
    else:
        print("✅ MODEL PERFORMANCE IS SATISFACTORY")
        print(f"{'='*60}")
        print("  • All metrics within acceptable range")
        print("  • No significant drift detected")
    
    print(f"\n{'='*60}")
    print("✅ MONITORING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return {
        'performance_file': os.path.join(monitoring_output_path, 'performance_metrics.parquet'),
        'drift_file': os.path.join(monitoring_output_path, 'drift_metrics.parquet'),
        'performance_plot': os.path.join(monitoring_output_path, 'performance_metrics.png'),
        'drift_plot': os.path.join(monitoring_output_path, 'drift_metrics.png'),
        'needs_retraining': needs_retraining,
        'retraining_reasons': reasons
    }