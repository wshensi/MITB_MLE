import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# ================================
# Bronze Layer: Raw ingestion
# ================================
def process_bronze_financials(snapshot_date_str, bronze_dir, spark):
    """
    Read features_financials.csv, optionally filter by snapshot_date if present,
    and save it to the bronze layer as a CSV file.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/features_financials.csv"  # Adjust this path if needed

    # Load raw CSV file
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Filter by snapshot_date if the column exists
    if 'snapshot_date' in df.columns:
        df = df.filter(F.col('snapshot_date') == snapshot_date)

    # Save as bronze CSV file
    file_name = f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv"
    output_path = os.path.join(bronze_dir, file_name)
    df.toPandas().to_csv(output_path, index=False)

    print(f"[BRONZE][financials] saved to: {output_path}, row count: {df.count()}")
    return df


# ================================
# Silver Layer: Cleaning & Feature Engineering
# ================================
def process_silver_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Read financials data from the bronze layer, perform cleaning, type casting,
    and feature engineering for non-numeric columns, then save as Parquet.
    """
    file_name = f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv"
    input_path = os.path.join(bronze_dir, file_name)
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    print(f"[SILVER][financials] loaded from: {input_path}, row count: {df.count()}")

    # --- Basic type casting ---
    if 'Customer_ID' in df.columns:
        df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))
    if 'snapshot_date' in df.columns:
        df = df.withColumn("snapshot_date", F.to_date("snapshot_date"))

    # --- Handle Credit_Mix (categorical ordinal mapping) ---
    if 'Credit_Mix' in df.columns:
        df = df.withColumn(
            "Credit_Mix_Score",
            F.when(F.col("Credit_Mix") == "Bad", F.lit(0))
             .when(F.col("Credit_Mix") == "Standard", F.lit(1))
             .when(F.col("Credit_Mix") == "Good", F.lit(2))
             .otherwise(F.lit(None).cast(IntegerType()))
        )

    # --- Handle Credit_History_Age (e.g., "31 Years and 0 Months") ---
    if 'Credit_History_Age' in df.columns:
        df = df.withColumn(
            "Credit_History_Months",
            (
                F.regexp_extract(F.col("Credit_History_Age"), r'(\d+)\s+Years', 1).cast(IntegerType()) * 12 +
                F.regexp_extract(F.col("Credit_History_Age"), r'(\d+)\s+Months', 1).cast(IntegerType())
            )
        )

    # --- Handle Payment_of_Min_Amount ---
    if 'Payment_of_Min_Amount' in df.columns:
        df = df.withColumn(
            "Payment_Min_Amount_Flag",
            F.when(F.col("Payment_of_Min_Amount") == "Yes", F.lit(1))
             .when(F.col("Payment_of_Min_Amount") == "No", F.lit(0))
             .otherwise(F.lit(None).cast(IntegerType()))
        )

    # --- Handle Payment_Behaviour ---
    if 'Payment_Behaviour' in df.columns:
        df = df.withColumn(
            "Payment_Behaviour_Clean",
            F.when(F.col("Payment_Behaviour").rlike("^[A-Za-z_]+$"), F.col("Payment_Behaviour"))
             .otherwise(F.lit("Unknown"))
        )

    # --- Handle Type_of_Loan ---
    if 'Type_of_Loan' in df.columns:
        df = df.withColumn("Type_of_Loan_Text", F.col("Type_of_Loan").cast(StringType()))
        df = df.withColumn(
            "Type_of_Loan_Count",
            F.when(
                F.col("Type_of_Loan").isNotNull(),
                F.size(
                    F.split(
                        F.regexp_replace(F.col("Type_of_Loan"), r'\s+and\s+', ','), r'\s*,\s*'
                    )
                )
            ).otherwise(F.lit(0)))

    # --- Cast remaining purely numeric columns ---
    exclude_cols = {
        "Customer_ID", "snapshot_date",
        "Credit_Mix", "Credit_Mix_Score",
        "Credit_History_Age", "Credit_History_Months",
        "Payment_of_Min_Amount", "Payment_Min_Amount_Flag",
        "Payment_Behaviour", "Payment_Behaviour_Clean",
        "Type_of_Loan", "Type_of_Loan_Text", "Type_of_Loan_Count"}
    
    numeric_cols = [c for c in df.columns if c not in exclude_cols]

    # --- Robust cleaning for numeric columns ---
    for col_name in numeric_cols:
        cleaned = F.regexp_replace(F.col(col_name), r'[^0-9.\-]', '')
        df = df.withColumn(
            col_name,
            F.when(F.trim(cleaned) == "", None)
             .otherwise(cleaned.cast(FloatType())))


    # --- Save as Parquet ---
    silver_file = f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    output_dir = os.path.join(silver_dir, "financials") 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, silver_file)
    df.write.mode("overwrite").parquet(output_path)

    print(f"[SILVER][financials] saved to: {output_path}, columns: {len(df.columns)}")
    return df

