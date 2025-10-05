import os
import pyspark.sql.functions as F
from pyspark.sql.types import NumericType

def process_gold_customer_feature_store(snapshot_date_str, silver_dir, gold_dir, spark):
    """
    Gold Layer: Join four Silver tables (loan, clickstream, attributes, financials)
    into a wide feature store table for downstream modeling.
    """

    suffix = snapshot_date_str.replace('-', '_')

    # 1️⃣ Read Silver tables
    loan = spark.read.parquet(os.path.join(silver_dir, f"silver_loan_daily_{suffix}.parquet"))
    click = spark.read.parquet(os.path.join(silver_dir, f"silver_clickstream_{suffix}.parquet"))
    attr = spark.read.parquet(os.path.join(silver_dir, f"silver_attributes_{suffix}.parquet"))
    fin = spark.read.parquet(os.path.join(silver_dir, f"silver_financials_{suffix}.parquet"))

    # 2️⃣ Standardize column naming (avoid conflicts)
    # clickstream
    for col_name in click.columns:
        if col_name not in ["Customer_ID", "snapshot_date"]:
            click = click.withColumnRenamed(col_name, f"click_{col_name}")

    # attributes (static)
    for col_name in attr.columns:
        if col_name != "Customer_ID":
            attr = attr.withColumnRenamed(col_name, f"attr_{col_name}")

    # financials
    for col_name in fin.columns:
        if col_name not in ["Customer_ID", "snapshot_date"]:
            fin = fin.withColumnRenamed(col_name, f"fin_{col_name}")

    # 3️⃣ Join logic
    # loan is the main table, granularity: loan × snapshot_date
    df = (
        loan
        .join(click, on="Customer_ID", how="left")  # <-- changed here
        .join(attr, on="Customer_ID", how="left")
        .join(fin, on=["Customer_ID", "snapshot_date"], how="left")
    )

    # 4️⃣ Handle missing values (adjustable based on business requirements)
    # Example: clickstream missing → 0, financials → 0, attributes → Unknown
    for col_name, dtype in df.dtypes:
        if col_name.startswith("click_") or col_name.startswith("fin_"):
            df = df.fillna({col_name: 0})
        elif col_name.startswith("attr_"):
            # if numeric column, fill with 0 or None; if string, fill with "Unknown"
            if dtype in ("int", "bigint", "double", "float", "decimal"):
                df = df.fillna({col_name: 0})
            else:
                df = df.fillna({col_name: "Unknown"})

    # 5️⃣ Label column (optional)
    # Example: 30-day delinquency label
    if "dpd" in df.columns:
        df = df.withColumn("label_default_30dpd", F.when(F.col("dpd") >= 30, 1).otherwise(0))

    # 6️⃣ Write to Gold layer
    gold_file = f"gold_feature_store_{suffix}.parquet"
    output_path = os.path.join(gold_dir, gold_file)
    df.write.mode("overwrite").parquet(output_path)

    print(f"[GOLD] Feature store saved to: {output_path}, rows: {df.count()}, columns: {len(df.columns)}")

    return df