import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_bronze_table(date_str, bronze_directory, spark):
    try:
        np.random.seed(int(date_str.replace('-', '')))
        
        n_loans = 1000
        
        data = {
            'loan_id': [f'LOAN_{i:06d}' for i in range(n_loans)],
            'user_id': [f'USER_{i % 500:06d}' for i in range(n_loans)],
            'loan_amount': np.random.uniform(5000, 100000, n_loans),
            'interest_rate': np.random.uniform(0.05, 0.25, n_loans),
            'term_months': np.random.choice([12, 24, 36, 48, 60], n_loans),
            'origination_date': [
                (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=int(x))).strftime('%Y-%m-%d')
                for x in np.random.uniform(0, 365, n_loans)
            ],
            'loan_status': np.random.choice(['active', 'closed', 'defaulted'], n_loans, p=[0.85, 0.10, 0.05]),
            'snapshot_date': date_str
        }
        
        df = pd.DataFrame(data)
        
        output_file = f"{bronze_directory}lms_{date_str.replace('-', '')}.parquet"
        df.to_parquet(output_file, index=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing bronze table for {date_str}: {str(e)}")
        raise