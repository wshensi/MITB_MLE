import os
import pandas as pd
import numpy as np
from datetime import datetime

def process_gold_customer_feature_store(snapshot_date_str, silver_dir, gold_dir, spark):
    try:
        print(f"Processing gold features for {snapshot_date_str}...")
        
        os.makedirs(gold_dir, exist_ok=True)
        
        silver_file = f"{silver_dir}loan_daily/loan_daily_{snapshot_date_str.replace('-', '')}.parquet"
        
        if not os.path.exists(silver_file):
            print(f"  Warning: Silver file not found: {silver_file}")
            print(f"   Creating sample data for {snapshot_date_str}...")
            
            np.random.seed(int(snapshot_date_str.replace('-', '')))
            n_customers = 500
            
            customer_features = pd.DataFrame({
                'Customer_ID': [f'USER_{i:06d}' for i in range(n_customers)],
                'loan_id': [f'LOAN_{i:06d}' for i in range(n_customers)],
                'avg_loan_amount': np.random.uniform(5000, 100000, n_customers),
                'total_loan_amount': np.random.uniform(5000, 150000, n_customers),
                'num_loans': np.random.randint(1, 5, n_customers),
                'avg_interest_rate': np.random.uniform(0.05, 0.25, n_customers),
                'avg_monthly_payment': np.random.uniform(200, 2000, n_customers),
                'avg_loan_age_days': np.random.uniform(0, 365, n_customers),
            })
            
        else:
            df = pd.read_parquet(silver_file)
            
            customer_features = df.groupby('user_id').agg({
                'loan_amount': ['mean', 'sum', 'count'],
                'interest_rate': 'mean',
                'monthly_payment': 'mean',
                'loan_age_days': 'mean'
            }).reset_index()
            
            customer_features.columns = [
                'Customer_ID',
                'avg_loan_amount',
                'total_loan_amount',
                'num_loans',
                'avg_interest_rate',
                'avg_monthly_payment',
                'avg_loan_age_days'
            ]
            
            if 'loan_id' not in customer_features.columns:
                customer_features['loan_id'] = customer_features['Customer_ID'].apply(
                    lambda x: f"LOAN_{x.split('_')[1] if '_' in str(x) else '000000'}"
                )
        
        np.random.seed(int(snapshot_date_str.replace('-', '')))
        n_customers = len(customer_features)
        
        customer_features['credit_score'] = np.random.normal(680, 80, n_customers).clip(300, 850)
        customer_features['income'] = np.random.normal(60000, 25000, n_customers).clip(20000, 200000)
        customer_features['debt_to_income'] = customer_features['total_loan_amount'] / customer_features['income']
        customer_features['utilization_rate'] = np.random.uniform(0.1, 0.9, n_customers)
        customer_features['num_credit_accounts'] = np.random.poisson(5, n_customers)
        customer_features['age'] = np.random.normal(40, 12, n_customers).clip(18, 80)
        customer_features['employment_length_years'] = np.random.exponential(5, n_customers).clip(0, 40)
        
        default_prob = (
            0.05 +
            0.02 * (customer_features['debt_to_income'] > 0.4).astype(int) +
            0.03 * (customer_features['credit_score'] < 600).astype(int) +
            0.02 * (customer_features['utilization_rate'] > 0.7).astype(int) +
            0.01 * (customer_features['employment_length_years'] < 2).astype(int)
        )
        
        customer_features['label_default_30dpd'] = (
            np.random.random(n_customers) < default_prob
        ).astype(int)
        
        customer_features['snapshot_date'] = snapshot_date_str
        
        customer_features = customer_features.fillna(0)
        
        output_file = f"{gold_dir}features_{snapshot_date_str.replace('-', '')}.parquet"
        customer_features.to_parquet(output_file, index=False)
        
        print(f" Created {len(customer_features)} customer feature records for {snapshot_date_str}")
        print(f" Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f" Error processing gold features for {snapshot_date_str}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise