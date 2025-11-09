import os
import pandas as pd
import numpy as np

def process_bronze_attributes(date_str, bronze_directory, spark):
    try:
        np.random.seed(int(date_str.replace('-', '')) + 2)
        
        n_users = 500
        
        data = {
            'user_id': [f'USER_{i:06d}' for i in range(n_users)],
            'age': np.random.normal(40, 12, n_users).clip(18, 80).astype(int),
            'gender': np.random.choice(['M', 'F', 'Other'], n_users, p=[0.48, 0.48, 0.04]),
            'education': np.random.choice(
                ['High School', 'Bachelor', 'Master', 'PhD'],
                n_users,
                p=[0.3, 0.45, 0.20, 0.05]
            ),
            'employment_status': np.random.choice(
                ['Employed', 'Self-Employed', 'Unemployed', 'Retired'],
                n_users,
                p=[0.70, 0.15, 0.10, 0.05]
            ),
            'marital_status': np.random.choice(
                ['Single', 'Married', 'Divorced', 'Widowed'],
                n_users,
                p=[0.35, 0.50, 0.12, 0.03]
            ),
            'num_dependents': np.random.poisson(1.5, n_users).clip(0, 10),
            'snapshot_date': date_str
        }
        
        df = pd.DataFrame(data)
        
        os.makedirs(bronze_directory, exist_ok=True)
        
        output_file = f"{bronze_directory}attributes_{date_str.replace('-', '')}.parquet"
        df.to_parquet(output_file, index=False)
        
        return True
    except Exception as e:
        print(f"Error processing bronze attributes: {str(e)}")
        return False

def process_silver_attributes(date_str, bronze_directory, silver_directory, spark):
    try:
        bronze_file = f"{bronze_directory}attributes_{date_str.replace('-', '')}.parquet"
        df = pd.read_parquet(bronze_file)
        
        df['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1, 'Other': 2})
        df['education_years'] = df['education'].map({
            'High School': 12,
            'Bachelor': 16,
            'Master': 18,
            'PhD': 21
        })
        df['is_employed'] = (df['employment_status'] == 'Employed').astype(int)
        df['is_married'] = (df['marital_status'] == 'Married').astype(int)
        
        attributes_dir = f"{silver_directory}attributes/"
        os.makedirs(attributes_dir, exist_ok=True)
        
        output_file = f"{attributes_dir}attributes_{date_str.replace('-', '')}.parquet"
        df.to_parquet(output_file, index=False)
        
        return True
    except Exception as e:
        print(f"Error processing silver attributes: {str(e)}")
        return False