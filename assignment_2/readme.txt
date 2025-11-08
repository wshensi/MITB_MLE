Readme
============================

Description
-----------
This project builds a complete Bronze → Silver → Gold data pipeline using PySpark inside Docker.
It processes monthly data from 2023-01 to 2024-12 for four datasets: Loan Daily (main), Clickstream, Attributes, and Financials.
The pipeline cleans, transforms, joins, and labels data for downstream modeling.

How to run
----------
Step 1: Start Docker container
Step 2: Open terminal inside Docker container
Step 3: Run the following command:
python main.py

Data flow
---------
Bronze layer
- Raw CSV files are stored in:
  datamart/bronze/lms/ for Loan data
  datamart/bronze/misc/ for Clickstream, Attributes, Financials data

Silver layer
- Cleaned and transformed Parquet files are stored in:
  datamart/silver/loan_daily/
  datamart/silver/clickstream/
  datamart/silver/attributes/
  datamart/silver/financials/

Gold layer
- All Silver tables are joined by Customer_ID and snapshot_date
- A label column named label_default_30dpd is generated (dpd >= 30 is labeled as 1, else 0)
- The final wide table is stored in:
  datamart/gold/feature_label_store/

Output summary
--------------
Bronze layer produces raw CSV files
Silver layer produces cleaned Parquet files
Gold layer produces the final feature and label dataset

Make sure you have the following files:
------------------------
data folder
utils folder
main.py
docker-compose.yaml
Dockerfile
requirements.txt
readme.txt

Notes
-----
The date range is set in main.py using start_date_str and end_date_str
The Gold layer output is used for modeling
If needed, delete the datamart folder to rerun the entire pipeline from scratch
