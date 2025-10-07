import os
import pyspark.sql.functions as F
from pyspark.sql.types import NumericType

def process_gold_customer_feature_store(snapshot_date_str, silver_dir, gold_dir, spark):
    """
    Gold Layer: Join four Silver tables (loan, clickstream, attributes, financials)
    into a wide feature store table for downstream modeling.
    """
    suffix = snapshot_date_str.replace('-', '_')

    # Read Silver tables
    loan = spark.read.parquet(os.path.join(silver_dir, "loan_daily", f"silver_loan_daily_{suffix}.parquet"))
    click = spark.read.parquet(os.path.join(silver_dir, "clickstream", f"silver_clickstream_{suffix}.parquet"))
    attr = spark.read.parquet(os.path.join(silver_dir, "attributes", f"silver_attributes_{suffix}.parquet"))
    fin = spark.read.parquet(os.path.join(silver_dir, "financials", f"silver_financials_{suffix}.parquet"))

    # Join logic
    # loan is the main table, granularity: Customer_ID × snapshot_date
    df = (
        loan
        .join(click, on=["Customer_ID", "snapshot_date"], how="left")
        .join(attr, on=["Customer_ID", "snapshot_date"], how="left")
        .join(fin, on=["Customer_ID", "snapshot_date"], how="left")
    )

    # Label column (optional)
    # Example: 30-day delinquency label
    if "dpd" in df.columns:
        df = df.withColumn("label_default_30dpd", F.when(F.col("dpd") >= 30, 1).otherwise(0))

    # Write to Gold layer
    gold_file = f"gold_feature_store_{suffix}.parquet"
    output_path = os.path.join(gold_dir, gold_file)
    df.write.mode("overwrite").parquet(output_path)
    print(f"[GOLD] Feature store saved to: {output_path}, rows: {df.count()}, columns: {len(df.columns)}")

    return df
