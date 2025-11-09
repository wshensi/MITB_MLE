import os
import pandas as pd
import numpy as np

def process_silver_table(date_str, bronze_directory, silver_directory, spark):
    try:
        bronze_file = f"{bronze_directory}lms_{date_str.replace('-', '')}.parquet"
        df = pd.read_parquet(bronze_file)
        
        df['loan_age_days'] = (
            pd.to_datetime(df['snapshot_date']) - pd.to_datetime(df['origination_date'])
        ).dt.days
        
        df['monthly_payment'] = (
            df['loan_amount'] * 
            (df['interest_rate'] / 12) * 
            (1 + df['interest_rate'] / 12) ** df['term_months']
        ) / (
            ((1 + df['interest_rate'] / 12) ** df['term_months']) - 1
        )
        
        df['status_clean'] = df['loan_status'].str.lower().str.strip()
        
        loan_daily_dir = f"{silver_directory}loan_daily/"
        os.makedirs(loan_daily_dir, exist_ok=True)
        
        output_file = f"{loan_daily_dir}loan_daily_{date_str.replace('-', '')}.parquet"
        df.to_parquet(output_file, index=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing silver table for {date_str}: {str(e)}")
        raise