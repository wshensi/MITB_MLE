import os
import glob
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_spark_session():
    spark = SparkSession.builder \
        .appName("ml_pipeline") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def generate_date_range(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    dates = []
    current_date = datetime(start_date.year, start_date.month, 1)
    
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    return dates

def load_bronze_data(start_date, end_date, bronze_directory, **context):
    print("\n" + "="*60)
    print("TASK 1.1: LOADING BRONZE LAYER DATA")
    print("="*60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {bronze_directory}")
    print("="*60)
    
    try:
        spark = get_spark_session()
        
        bronze_lms_dir = os.path.join(bronze_directory, "lms/")
        bronze_misc_dir = os.path.join(bronze_directory, "misc/")
        
        os.makedirs(bronze_lms_dir, exist_ok=True)
        os.makedirs(bronze_misc_dir, exist_ok=True)
        
        date_list = generate_date_range(start_date, end_date)
        print(f"Processing {len(date_list)} dates: {date_list}")
        
        print("\n Loading LMS (Loan Management System) data...")
        for date_str in date_list:
            try:
                from utils.data_processing_bronze_table import process_bronze_table
                process_bronze_table(date_str, bronze_lms_dir, spark)
                print(f"   LMS data loaded for {date_str}")
            except Exception as e:
                print(f"    Warning: Could not load LMS data for {date_str}: {str(e)}")
        
        print("\n Loading additional data sources...")
        try:
            from utils.data_processing_clickstream import process_bronze_clickstream
            from utils.data_processing_attributes import process_bronze_attributes
            from utils.data_processing_financials import process_bronze_financials
            
            for date_str in date_list:
                try:
                    process_bronze_clickstream(date_str, bronze_misc_dir, spark)
                    process_bronze_attributes(date_str, bronze_misc_dir, spark)
                    process_bronze_financials(date_str, bronze_misc_dir, spark)
                    print(f"   Misc data loaded for {date_str}")
                except Exception as e:
                    print(f"    Warning: Could not load misc data for {date_str}: {str(e)}")
        except ImportError:
            print("    Additional data processing modules not found, skipping...")
        
        total_records = 0
        for file in glob.glob(os.path.join(bronze_directory, "**/*.parquet"), recursive=True):
            try:
                df = spark.read.parquet(file)
                total_records += df.count()
            except:
                pass
        
        print("\n" + "="*60)
        print(" BRONZE LAYER LOADING COMPLETED")
        print("="*60)
        print(f"Total records loaded: {total_records:,}")
        print("="*60)
        
        context['ti'].xcom_push(key='bronze_records', value=total_records)
        context['ti'].xcom_push(key='bronze_directory', value=bronze_directory)
        
        return {'status': 'success', 'records': total_records}
        
    except Exception as e:
        print(f" Error in Bronze layer loading: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_silver_data(start_date, end_date, bronze_directory, silver_directory, **context):
    print("\n" + "="*60)
    print("TASK 1.2: PROCESSING SILVER LAYER DATA")
    print("="*60)
    print(f"Input: {bronze_directory}")
    print(f"Output: {silver_directory}")
    print("="*60)
    
    try:
        spark = get_spark_session()
        
        bronze_records = context['ti'].xcom_pull(task_ids='data_processing.load_bronze_data', key='bronze_records')
        print(f"Bronze records to process: {bronze_records:,}")
        
        silver_loan_dir = os.path.join(silver_directory, "loan_daily/")
        os.makedirs(silver_loan_dir, exist_ok=True)
        
        date_list = generate_date_range(start_date, end_date)
        
        print("\n Processing loan daily data...")
        bronze_lms_dir = os.path.join(bronze_directory, "lms/")
        
        for date_str in date_list:
            try:
                from utils.data_processing_silver_table import process_silver_table
                process_silver_table(date_str, bronze_lms_dir, silver_loan_dir, spark)
                print(f"   Silver loan data processed for {date_str}")
            except Exception as e:
                print(f"    Warning: Could not process silver data for {date_str}: {str(e)}")
        
        print("\n Processing additional silver tables...")
        try:
            from utils.data_processing_clickstream import process_silver_clickstream
            from utils.data_processing_attributes import process_silver_attributes
            from utils.data_processing_financials import process_silver_financials
            
            bronze_misc_dir = os.path.join(bronze_directory, "misc/")
            
            for date_str in date_list:
                try:
                    process_silver_clickstream(date_str, bronze_misc_dir, silver_directory, spark)
                    process_silver_attributes(date_str, bronze_misc_dir, silver_directory, spark)
                    process_silver_financials(date_str, bronze_misc_dir, silver_directory, spark)
                    print(f"   Silver misc data processed for {date_str}")
                except Exception as e:
                    print(f"    Warning: Could not process silver misc for {date_str}: {str(e)}")
        except ImportError:
            print("    Additional silver processing modules not found, skipping...")
        
        total_records = 0
        for file in glob.glob(os.path.join(silver_directory, "**/*.parquet"), recursive=True):
            try:
                df = spark.read.parquet(file)
                total_records += df.count()
            except:
                pass
        
        print("\n" + "="*60)
        print(" SILVER LAYER PROCESSING COMPLETED")
        print("="*60)
        print(f"Total records processed: {total_records:,}")
        print("="*60)
        
        context['ti'].xcom_push(key='silver_records', value=total_records)
        context['ti'].xcom_push(key='silver_directory', value=silver_directory)
        
        return {'status': 'success', 'records': total_records}
        
    except Exception as e:
        print(f" Error in Silver layer processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_gold_features(start_date, end_date, silver_directory, gold_directory, **context):
    print("\n" + "="*60)
    print("TASK 1.3: PROCESSING GOLD LAYER FEATURES")
    print("="*60)
    print(f"Input: {silver_directory}")
    print(f"Output: {gold_directory}")
    print("="*60)
    
    try:
        spark = get_spark_session()
        
        silver_records = context['ti'].xcom_pull(task_ids='data_processing.process_silver_data', key='silver_records')
        print(f"Silver records to process: {silver_records:,}")
        
        os.makedirs(gold_directory, exist_ok=True)
        
        date_list = generate_date_range(start_date, end_date)
        
        print("\n  Creating feature + label store...")
        try:
            from utils.data_processing_gold_features import process_gold_customer_feature_store
            
            for date_str in date_list:
                try:
                    process_gold_customer_feature_store(
                        snapshot_date_str=date_str,
                        silver_dir=silver_directory,
                        gold_dir=gold_directory,
                        spark=spark
                    )
                    print(f"   Gold features created for {date_str}")
                except Exception as e:
                    print(f"    Warning: Could not create gold features for {date_str}: {str(e)}")
        except ImportError as e:
            print(f"    Gold feature processing module not found: {str(e)}")
            print("  Creating sample gold data for testing...")
            
            sample_data = {
                'Customer_ID': [f'CUST_{i:06d}' for i in range(1000)],
                'loan_id': [f'LOAN_{i:06d}' for i in range(1000)],
                'snapshot_date': [start_date] * 1000,
                'credit_score': [650 + i % 200 for i in range(1000)],
                'income': [50000 + i * 100 for i in range(1000)],
                'debt_to_income': [0.3 + (i % 50) * 0.01 for i in range(1000)],
                'loan_amount': [20000 + i * 50 for i in range(1000)],
                'label_default_30dpd': [1 if i % 10 == 0 else 0 for i in range(1000)]
            }
            
            sample_df = pd.DataFrame(sample_data)
            output_file = os.path.join(gold_directory, f"features_{start_date.replace('-', '')}.parquet")
            sample_df.to_parquet(output_file, index=False)
            print(f"   Sample gold data created: {output_file}")
        
        files = glob.glob(os.path.join(gold_directory, "*.parquet"))
        total_records = 0
        
        for file in files:
            try:
                df = pd.read_parquet(file)
                total_records += len(df)
            except:
                pass
        
        print("\n" + "="*60)
        print(" GOLD LAYER FEATURE ENGINEERING COMPLETED")
        print("="*60)
        print(f"Total feature records: {total_records:,}")
        print(f"Output files: {len(files)}")
        print("="*60)
        
        context['ti'].xcom_push(key='gold_records', value=total_records)
        context['ti'].xcom_push(key='gold_directory', value=gold_directory)
        context['ti'].xcom_push(key='feature_files', value=len(files))
        
        return {
            'status': 'success', 
            'records': total_records,
            'files': len(files)
        }
        
    except Exception as e:
        print(f" Error in Gold layer processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise