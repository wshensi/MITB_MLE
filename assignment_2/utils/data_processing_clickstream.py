import os
import pandas as pd
import numpy as np

def process_bronze_clickstream(date_str, bronze_directory, spark):
    try:
        np.random.seed(int(date_str.replace('-', '')) + 1)
        
        n_events = 5000
        
        data = {
            'event_id': [f'EVT_{i:08d}' for i in range(n_events)],
            'user_id': [f'USER_{np.random.randint(0, 500):06d}' for _ in range(n_events)],
            'event_type': np.random.choice(
                ['page_view', 'click', 'form_submit', 'login', 'logout'],
                n_events,
                p=[0.5, 0.3, 0.1, 0.05, 0.05]
            ),
            'event_timestamp': pd.date_range(
                start=date_str,
                periods=n_events,
                freq='17S'
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'snapshot_date': date_str
        }
        
        df = pd.DataFrame(data)
        
        os.makedirs(bronze_directory, exist_ok=True)
        
        output_file = f"{bronze_directory}clickstream_{date_str.replace('-', '')}.parquet"
        df.to_parquet(output_file, index=False)
        
        return True
    except Exception as e:
        print(f"Error processing bronze clickstream: {str(e)}")
        return False

def process_silver_clickstream(date_str, bronze_directory, silver_directory, spark):
    try:
        bronze_file = f"{bronze_directory}clickstream_{date_str.replace('-', '')}.parquet"
        df = pd.read_parquet(bronze_file)
        
        user_activity = df.groupby('user_id').agg({
            'event_id': 'count',
            'event_type': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        user_activity.columns = ['user_id', 'total_events', 'event_breakdown']
        user_activity['snapshot_date'] = date_str
        
        clickstream_dir = f"{silver_directory}clickstream/"
        os.makedirs(clickstream_dir, exist_ok=True)
        
        output_file = f"{clickstream_dir}clickstream_{date_str.replace('-', '')}.parquet"
        user_activity.to_parquet(output_file, index=False)
        
        return True
    except Exception as e:
        print(f"Error processing silver clickstream: {str(e)}")
        return False