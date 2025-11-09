import os
import pandas as pd
import numpy as np

def process_bronze_financials(date_str, bronze_directory, spark):
    try:
        np.random.seed(int(date_str.replace('-', '')) + 3)
        
        n_users = 500
        
        data = {
            'user_id': [f'USER_{i:06d}' for i in range(n_users)],
            'annual_income': np.random.lognormal(11, 0.5, n_users).clip(20000, 500000),
            'monthly_expenses': np.random.lognormal(8.5, 0.4, n_users).clip(1000, 20000),
            'savings_balance': np.random.lognormal(9, 1.2, n_users).clip(0, 100000),
            'checking_balance': np.random.lognormal(7.5, 0.8, n_users).clip(0, 50000),
            'credit_card_balance': np.random.lognormal(8, 1, n_users).clip(0, 30000),
            'credit_card_limit': np.random.lognormal(9.5, 0.5, n_users).clip(1000, 100000),
            'num_credit_cards': np.random.poisson(3, n_users).clip(0, 15),
            'num_bank_accounts': np.random.poisson(2, n_users).clip(1, 10),
            'has_mortgage': np.random.choice([0, 1], n_users, p=[0.6, 0.4]),
            'has_auto_loan': np.random.choice([0, 1], n_users, p=[0.7, 0.3]),
            'credit_score': np.random.normal(680, 80, n_users).clip(300, 850).astype(int),
            'snapshot_date': date_str
        }
        
        df = pd.DataFrame(data)
        
        os.makedirs(bronze_directory, exist_ok=True)
        
        output_file = f"{bronze_directory}financials_{date_str.replace('-', '')}.parquet"
        df.to_parquet(output_file, index=False)
        
        return True
    except Exception as e:
        print(f"Error processing bronze financials: {str(e)}")
        return False

def process_silver_financials(date_str, bronze_directory, silver_directory, spark):
    try:
        bronze_file = f"{bronze_directory}financials_{date_str.replace('-', '')}.parquet"
        df = pd.read_parquet(bronze_file)
        
        df['debt_to_income'] = df['credit_card_balance'] / (df['annual_income'] / 12)
        df['savings_to_income'] = df['savings_balance'] / (df['annual_income'] / 12)
        df['credit_utilization'] = df['credit_card_balance'] / df['credit_card_limit']
        df['monthly_discretionary_income'] = (df['annual_income'] / 12) - df['monthly_expenses']
        df['total_liquid_assets'] = df['savings_balance'] + df['checking_balance']
        df['net_worth_estimate'] = df['total_liquid_assets'] - df['credit_card_balance']
        
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        financials_dir = f"{silver_directory}financials/"
        os.makedirs(financials_dir, exist_ok=True)
        
        output_file = f"{financials_dir}financials_{date_str.replace('-', '')}.parquet"
        df.to_parquet(output_file, index=False)
        
        return True
    except Exception as e:
        print(f"Error processing silver financials: {str(e)}")
        return False