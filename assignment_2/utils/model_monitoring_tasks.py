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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_monitoring_data(predictions_path, gold_feature_path, output_path, **context):
    print("\n" + "="*60)
    print("TASK 4.1: LOADING MONITORING DATA")
    print("="*60)
    print(f"Predictions: {predictions_path}")
    print(f"Actuals: {gold_feature_path}")
    print("="*60)

    try:
        print("\n Loading predictions...")
        pred_files = glob.glob(os.path.join(predictions_path, "*.parquet"))

        if not pred_files:
            predictions_file = context['ti'].xcom_pull(
                task_ids='model_inference.execute_predictions',
                key='predictions_file'
            )

            if predictions_file and os.path.exists(predictions_file):
                pred_files = [predictions_file]
                print(f" Using newly generated predictions: {predictions_file}")
            else:
                raise FileNotFoundError(f"No prediction files found in {predictions_path}")

        pred_list = []
        for file in pred_files:
            df = pd.read_parquet(file)
            pred_list.append(df)

        predictions_df = pd.concat(pred_list, ignore_index=True)
        predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date'])

        print(f" Loaded {len(predictions_df):,} predictions from {len(pred_files)} files")

        print("\n Loading actual labels...")
        actual_files = glob.glob(os.path.join(gold_feature_path, "*.parquet"))

        if not actual_files:
            raise FileNotFoundError(f"No files found in {gold_feature_path}")

        actual_list = []
        for file in actual_files:
            df = pd.read_parquet(file)

            possible_customer_ids = ['Customer_ID', 'user_id', 'customer_id', 'cust_id']
            customer_id_col = None
            for col in possible_customer_ids:
                if col in df.columns:
                    customer_id_col = col
                    break

            if customer_id_col is None:
                continue

            possible_label_cols = ['label_default_30dpd', 'default_flag', 'label', 'target']
            label_col = None
            for col in possible_label_cols:
                if col in df.columns:
                    label_col = col
                    break

            if label_col is None:
                continue

            keep_cols = [customer_id_col, 'loan_id', 'snapshot_date', label_col]
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols].copy()

            if customer_id_col != 'Customer_ID':
                df = df.rename(columns={customer_id_col: 'Customer_ID'})
            if label_col != 'actual_label':
                df = df.rename(columns={label_col: 'actual_label'})

            actual_list.append(df)

        if not actual_list:
            raise ValueError("No valid actual labels found")

        actuals_df = pd.concat(actual_list, ignore_index=True)
        actuals_df['snapshot_date'] = pd.to_datetime(actuals_df['snapshot_date'])

        print(f" Loaded {len(actuals_df):,} actual labels")

        print("\n Merging predictions with actuals...")

        merge_cols = ['Customer_ID', 'loan_id', 'snapshot_date']
        available_merge_cols = [col for col in merge_cols
                               if col in predictions_df.columns and col in actuals_df.columns]

        if not available_merge_cols:
            raise ValueError(
                f"No common columns for merging.\n"
                f"Predictions: {predictions_df.columns.tolist()}\n"
                f"Actuals: {actuals_df.columns.tolist()}"
            )

        print(f"  Merging on: {available_merge_cols}")

        merged_df = predictions_df.merge(
            actuals_df,
            on=available_merge_cols,
            how='inner'
        )

        print(f" Matched {len(merged_df):,} predictions with actuals")

        if len(merged_df) == 0:
            raise ValueError("No predictions matched with actuals")

        os.makedirs(output_path, exist_ok=True)

        merged_file = os.path.join(output_path, 'merged_data.parquet')
        merged_df.to_parquet(merged_file, index=False)

        print(f"\n Merged data saved: {merged_file}")

        summary = {
            'n_predictions': len(predictions_df),
            'n_actuals': len(actuals_df),
            'n_matched': len(merged_df),
            'match_rate': len(merged_df) / len(predictions_df) if len(predictions_df) > 0 else 0,
            'date_range': {
                'min': merged_df['snapshot_date'].min().strftime('%Y-%m-%d'),
                'max': merged_df['snapshot_date'].max().strftime('%Y-%m-%d')
            }
        }

        summary_file = os.path.join(output_path, 'data_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, default=str)

        print("\n" + "="*60)
        print(" MONITORING DATA LOADED")
        print("="*60)
        print(f"Matched samples: {len(merged_df):,}")
        print(f"Match rate: {summary['match_rate']*100:.1f}%")
        print("="*60)

        context['ti'].xcom_push(key='merged_data_file', value=merged_file)
        context['ti'].xcom_push(key='n_matched', value=len(merged_df))

        return {
            'status': 'success',
            'merged_file': merged_file,
            'n_matched': len(merged_df)
        }

    except Exception as e:
        print(f" Error loading monitoring data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def calculate_performance_metrics(monitoring_data_path, output_path, **context):
    print("\n" + "="*60)
    print("TASK 4.2: CALCULATING PERFORMANCE METRICS")
    print("="*60)

    try:
        merged_file = context['ti'].xcom_pull(
            task_ids='model_monitoring.load_monitoring_data',
            key='merged_data_file'
        )

        merged_df = pd.read_parquet(merged_file)
        print(f" Loaded {len(merged_df):,} samples")

        print("\n Calculating metrics by snapshot date...")

        results = []

        for date in sorted(merged_df['snapshot_date'].unique()):
            date_df = merged_df[merged_df['snapshot_date'] == date]

            y_true = date_df['actual_label']
            y_pred_proba = date_df['prediction_probability']
            y_pred_binary = date_df['prediction_binary']

            if len(y_true.unique()) < 2:
                print(f"    Skipping {date.strftime('%Y-%m-%d')}: Only one class")
                continue

            try:
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

                cm = confusion_matrix(y_true, y_pred_binary)
                metrics['confusion_matrix'] = cm.tolist()

                results.append(metrics)

                print(f"   {date.strftime('%Y-%m-%d')}: AUC={metrics['auc']:.4f}, "
                      f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

            except Exception as e:
                print(f"    Error for {date.strftime('%Y-%m-%d')}: {str(e)}")
                continue

        if not results:
            raise ValueError("No valid metrics calculated")

        performance_df = pd.DataFrame(results)

        os.makedirs(output_path, exist_ok=True)

        perf_file = os.path.join(output_path, 'performance_metrics.parquet')
        performance_df.to_parquet(perf_file, index=False)

        print(f"\n Performance metrics saved: {perf_file}")

        print("\n" + "="*60)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*60)
        print(f"Time periods analyzed: {len(performance_df)}")
        print(f"Average AUC: {performance_df['auc'].mean():.4f}")
        print(f"Average Precision: {performance_df['precision'].mean():.4f}")
        print(f"Average Recall: {performance_df['recall'].mean():.4f}")
        print(f"Average F1: {performance_df['f1'].mean():.4f}")
        print("="*60)

        context['ti'].xcom_push(key='performance_file', value=perf_file)
        context['ti'].xcom_push(key='performance_metrics', value=results)
        context['ti'].xcom_push(key='avg_auc', value=performance_df['auc'].mean())

        return {
            'status': 'success',
            'performance_file': perf_file,
            'n_periods': len(performance_df),
            'avg_auc': float(performance_df['auc'].mean())
        }

    except Exception as e:
        print(f" Error calculating performance metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def calculate_drift_metrics(monitoring_data_path, baseline_date, output_path, **context):
    print("\n" + "="*60)
    print("TASK 4.3: CALCULATING DRIFT METRICS (PSI)")
    print("="*60)
    print(f"Baseline date: {baseline_date}")
    print("="*60)

    try:
        merged_file = context['ti'].xcom_pull(
            task_ids='model_monitoring.load_monitoring_data',
            key='merged_data_file'
        )

        merged_df = pd.read_parquet(merged_file)
        print(f" Loaded {len(merged_df):,} samples")

        baseline_df = merged_df[
            merged_df['snapshot_date'] == pd.to_datetime(baseline_date)
        ]

        if len(baseline_df) == 0:
            print(f"  No data for baseline date {baseline_date}, using earliest date")
            baseline_date = merged_df['snapshot_date'].min()
            baseline_df = merged_df[merged_df['snapshot_date'] == baseline_date]

        baseline_probs = baseline_df['prediction_probability'].values
        print(f" Baseline: {len(baseline_probs):,} samples from {baseline_date}")

        print("\n Calculating PSI for each date...")

        def calculate_psi(expected, actual, bins=10):
            try:
                breakpoints = np.linspace(0, 1, bins + 1)

                expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
                actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

                expected_percents = expected_percents + 1e-10
                actual_percents = actual_percents + 1e-10

                psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
                psi = np.sum(psi_values)

                return psi
            except Exception as e:
                print(f"    PSI calculation error: {str(e)}")
                return None

        drift_results = []

        for date in sorted(merged_df['snapshot_date'].unique()):
            if date == baseline_date:
                continue

            date_df = merged_df[merged_df['snapshot_date'] == date]
            actual_probs = date_df['prediction_probability'].values

            psi = calculate_psi(baseline_probs, actual_probs)

            if psi is not None:
                drift_status = 'stable'
                if psi >= 0.2:
                    drift_status = 'major_drift'
                elif psi >= 0.1:
                    drift_status = 'minor_drift'

                drift_results.append({
                    'snapshot_date': date.strftime('%Y-%m-%d'),
                    'psi': float(psi),
                    'drift_status': drift_status,
                    'num_samples': len(date_df)
                })

                print(f"   {date.strftime('%Y-%m-%d')}: PSI={psi:.4f} ({drift_status})")

        if not drift_results:
            print("  No drift metrics calculated")
            drift_df = pd.DataFrame()
        else:
            drift_df = pd.DataFrame(drift_results)

        os.makedirs(output_path, exist_ok=True)

        if len(drift_df) > 0:
            drift_file = os.path.join(output_path, 'drift_metrics.parquet')
            drift_df.to_parquet(drift_file, index=False)
            print(f"\n Drift metrics saved: {drift_file}")

            print("\n" + "="*60)
            print("DRIFT METRICS SUMMARY")
            print("="*60)
            print(f"Baseline date: {baseline_date}")
            print(f"Periods analyzed: {len(drift_df)}")
            print(f"Average PSI: {drift_df['psi'].mean():.4f}")
            print(f"Max PSI: {drift_df['psi'].max():.4f}")

            drift_counts = drift_df['drift_status'].value_counts()
            for status, count in drift_counts.items():
                print(f"  {status}: {count}")
            print("="*60)
        else:
            drift_file = None
            print("\n  No drift metrics to save")

        context['ti'].xcom_push(key='drift_file', value=drift_file)
        context['ti'].xcom_push(key='drift_metrics', value=drift_results)
        if len(drift_df) > 0:
            context['ti'].xcom_push(key='max_psi', value=drift_df['psi'].max())

        return {
            'status': 'success',
            'drift_file': drift_file,
            'n_periods': len(drift_df),
            'max_psi': float(drift_df['psi'].max()) if len(drift_df) > 0 else None
        }

    except Exception as e:
        print(f" Error calculating drift metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def check_thresholds(monitoring_data_path, performance_threshold, drift_threshold, **context):
    print("\n" + "="*60)
    print("TASK 4.4: CHECKING THRESHOLDS")
    print("="*60)
    print(f"Performance threshold (AUC): {performance_threshold['auc']}")
    print(f"Drift threshold (PSI): {drift_threshold}")
    print("="*60)

    try:
        needs_retraining = False
        reasons = []

        print("\n Checking performance metrics...")
        avg_auc = context['ti'].xcom_pull(
            task_ids='model_monitoring.calculate_performance_metrics',
            key='avg_auc'
        )

        if avg_auc is not None:
            print(f"  Average AUC: {avg_auc:.4f}")
            threshold_auc = performance_threshold.get('auc', 0.65)

            if avg_auc < threshold_auc:
                needs_retraining = True
                reasons.append(f"AUC below threshold: {avg_auc:.4f} < {threshold_auc}")
                print(f"    AUC below threshold!")
            else:
                print(f"   AUC within acceptable range")

        print("\n Checking drift metrics...")
        max_psi = context['ti'].xcom_pull(
            task_ids='model_monitoring.calculate_drift_metrics',
            key='max_psi'
        )

        if max_psi is not None:
            print(f"  Max PSI: {max_psi:.4f}")

            if max_psi > drift_threshold:
                needs_retraining = True
                reasons.append(f"PSI above threshold: {max_psi:.4f} > {drift_threshold}")
                print(f"    Significant drift detected!")
            else:
                print(f"   No significant drift detected")

        os.makedirs(monitoring_data_path, exist_ok=True)

        threshold_result = {
            'needs_retraining': needs_retraining,
            'reasons': reasons,
            'checked_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'thresholds': {
                'performance': performance_threshold,
                'drift': drift_threshold
            },
            'current_metrics': {
                'avg_auc': avg_auc,
                'max_psi': max_psi
            }
        }

        result_file = os.path.join(monitoring_data_path, 'threshold_check.json')
        with open(result_file, 'w') as f:
            json.dump(threshold_result, f, indent=4, default=str)

        print(f"\n Threshold check saved: {result_file}")

        print("\n" + "="*60)
        if needs_retraining:
            print("  RETRAINING RECOMMENDED")
            print("="*60)
            for reason in reasons:
                print(f"  • {reason}")
        else:
            print(" MODEL PERFORMANCE SATISFACTORY")
            print("="*60)
            print("  • All metrics within acceptable range")
        print("="*60)

        context['ti'].xcom_push(key='needs_retraining', value=needs_retraining)
        context['ti'].xcom_push(key='retraining_reasons', value=reasons)

        return {
            'status': 'success',
            'needs_retraining': needs_retraining,
            'reasons': reasons
        }

    except Exception as e:
        print(f" Error checking thresholds: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_monitoring_report(monitoring_data_path, output_path, report_date, **context):
    print("\n" + "="*60)
    print("TASK 4.5: GENERATING MONITORING REPORT")
    print("="*60)
    print(f"Report date: {report_date}")
    print(f"Output: {output_path}")
    print("="*60)

    try:
        os.makedirs(output_path, exist_ok=True)

        print("\n Creating visualizations...")
        perf_file = context['ti'].xcom_pull(
            task_ids='model_monitoring.calculate_performance_metrics',
            key='performance_file'
        )

        if perf_file and os.path.exists(perf_file):
            perf_df = pd.read_parquet(perf_file)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Performance Over Time', fontsize=16, fontweight='bold')

            metrics_to_plot = ['auc', 'precision', 'recall', 'f1']
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

            for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
                ax = axes[idx // 2, idx % 2]
                ax.plot(
                    perf_df['snapshot_date'],
                    perf_df[metric],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    color=color
                )
                ax.set_xlabel('Date', fontsize=11, fontweight='bold')
                ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
                ax.set_title(f'{metric.upper()} Over Time', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(axis='x', rotation=45)

                if metric == 'auc':
                    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
                    ax.legend()

            plt.tight_layout()
            perf_plot = os.path.join(output_path, 'performance_metrics.png')
            plt.savefig(perf_plot, dpi=300, bbox_inches='tight')
            plt.close()

            print(f" Performance plot saved: {perf_plot}")

        drift_file = context['ti'].xcom_pull(
            task_ids='model_monitoring.calculate_drift_metrics',
            key='drift_file'
        )

        if drift_file and os.path.exists(drift_file):
            drift_df = pd.read_parquet(drift_file)

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(
                drift_df['snapshot_date'],
                drift_df['psi'],
                marker='o',
                linewidth=2,
                markersize=8,
                color='#FF6B35',
                label='PSI'
            )

            ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=2,
                      label='Minor drift (0.1)', alpha=0.7)
            ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2,
                      label='Major drift (0.2)', alpha=0.7)

            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('PSI', fontsize=12, fontweight='bold')
            ax.set_title('Population Stability Index Over Time',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(rotation=45)

            plt.tight_layout()
            drift_plot = os.path.join(output_path, 'drift_metrics.png')
            plt.savefig(drift_plot, dpi=300, bbox_inches='tight')
            plt.close()

            print(f" Drift plot saved: {drift_plot}")

        if perf_file and os.path.exists(perf_file):
            final_perf = os.path.join(output_path, 'performance_metrics.parquet')
            perf_df.to_parquet(final_perf, index=False)
            print(f" Performance metrics copied: {final_perf}")

        if drift_file and os.path.exists(drift_file):
            final_drift = os.path.join(output_path, 'drift_metrics.parquet')
            drift_df.to_parquet(final_drift, index=False)
            print(f" Drift metrics copied: {final_drift}")

        summary = {
            'report_date': report_date,
            'generated_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'performance_summary': {
                'avg_auc': float(perf_df['auc'].mean()) if 'perf_df' in locals() else None,
                'avg_precision': float(perf_df['precision'].mean()) if 'perf_df' in locals() else None,
                'avg_recall': float(perf_df['recall'].mean()) if 'perf_df' in locals() else None,
                'avg_f1': float(perf_df['f1'].mean()) if 'perf_df' in locals() else None
            },
            'drift_summary': {
                'max_psi': float(drift_df['psi'].max()) if 'drift_df' in locals() and len(drift_df) > 0 else None,
                'avg_psi': float(drift_df['psi'].mean()) if 'drift_df' in locals() and len(drift_df) > 0 else None
            },
            'needs_retraining': context['ti'].xcom_pull(
                task_ids='model_monitoring.check_thresholds',
                key='needs_retraining'
            ),
            'retraining_reasons': context['ti'].xcom_pull(
                task_ids='model_monitoring.check_thresholds',
                key='retraining_reasons'
            )
        }

        report_file = os.path.join(output_path, f'monitoring_report_{report_date.replace("-", "")}.json')
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=4, default=str)

        print(f"\n Summary report saved: {report_file}")

        print("\n" + "="*60)
        print(" MONITORING REPORT GENERATED")
        print("="*60)
        print(f"Performance plots: {output_path}")
        print(f"Drift plots: {output_path}")
        print(f"Summary report: {report_file}")
        print("="*60)

        return {
            'status': 'success',
            'report_file': report_file,
            'output_path': output_path
        }

    except Exception as e:
        print(f" Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        raise