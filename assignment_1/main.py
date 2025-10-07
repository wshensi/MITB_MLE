import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# ----------------- Bronze: loan_daily -----------------
bronze_lms_directory = "datamart/bronze/lms/"
if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)

# ----------------- Silver: loan_daily -----------------
silver_loan_daily_directory = "datamart/silver/loan_daily/"
if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# ----------------- Bronze & Silver: clickstream / attributes / financials -----------------
from utils.data_processing_clickstream import process_bronze_clickstream, process_silver_clickstream
from utils.data_processing_attributes import process_bronze_attributes, process_silver_attributes
from utils.data_processing_financials import process_bronze_financials, process_silver_financials

bronze_misc_directory = "datamart/bronze/misc/"
if not os.path.exists(bronze_misc_directory):
    os.makedirs(bronze_misc_directory)

silver_root_directory = "datamart/silver/"

for date_str in dates_str_lst:
    process_bronze_clickstream(date_str, bronze_misc_directory, spark)
    process_silver_clickstream(date_str, bronze_misc_directory, silver_root_directory, spark)

    process_bronze_attributes(date_str, bronze_misc_directory, spark)
    process_silver_attributes(date_str, bronze_misc_directory, silver_root_directory, spark)

    process_bronze_financials(date_str, bronze_misc_directory, spark)
    process_silver_financials(date_str, bronze_misc_directory, silver_root_directory, spark)

# ----------------- Gold: Feature + Label Store -----------------
from utils.data_processing_gold_features import process_gold_customer_feature_store

gold_feature_label_store_directory = "datamart/gold/feature_label_store/"
if not os.path.exists(gold_feature_label_store_directory):
    os.makedirs(gold_feature_label_store_directory)

silver_base_directory = "datamart/silver/"  

for date_str in dates_str_lst:
    process_gold_customer_feature_store(
        snapshot_date_str=date_str,
        silver_dir=silver_base_directory,
        gold_dir=gold_feature_label_store_directory,
        spark=spark
    )
folder_path = gold_feature_label_store_directory
files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("[GOLD] feature_label_store total row_count:", df.count())
df.show()
