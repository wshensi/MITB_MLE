import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# Bronze Layer: Raw ingestion and storage
def process_bronze_attributes(snapshot_date_str, bronze_dir, spark):
    """
    Read features_attributes.csv, optionally filter by snapshot_date if present,
    and save it to the bronze layer as a CSV file.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/features_attributes.csv"  # Adjust this path if needed

    # Load raw CSV file
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Filter by snapshot_date if the column exists
    if 'snapshot_date' in df.columns:
        df = df.filter(F.col('snapshot_date') == snapshot_date)

    # Save as bronze CSV file
    file_name = f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv"
    output_path = os.path.join(bronze_dir, file_name)
    df.toPandas().to_csv(output_path, index=False)

    print(f"[BRONZE][attributes] saved to: {output_path}, row count: {df.count()}")
    return df


# Silver Layer: Data cleaning, type casting, and feature enrichment
def process_silver_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Read attributes data from the bronze layer, perform cleaning, type casting,
    and feature engineering, then save as a Parquet file in the silver layer.
    """
    file_name = f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv"
    input_path = os.path.join(bronze_dir, file_name)
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    print(f"[SILVER][attributes] loaded from: {input_path}, row count: {df.count()}")

    #  Type casting based on expected columns (adjust depending on actual data)
    if 'Customer_ID' in df.columns:
        df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))
    # Clean up Age column before casting
    if 'Age' in df.columns:
        df = df.withColumn("Age", F.regexp_replace(F.col("Age"), "[^0-9]", "").cast(IntegerType()))
    if 'Income_Level' in df.columns:
        df = df.withColumn("Income_Level", F.col("Income_Level").cast(FloatType()))
    if 'snapshot_date' in df.columns:
        df = df.withColumn("snapshot_date", F.to_date("snapshot_date"))

    #  Feature engineering example: create age groups
    if 'Age' in df.columns:
        df = df.withColumn(
            "Age_Group",
            F.when(F.col("Age") < 25, "Youth")
             .when((F.col("Age") >= 25) & (F.col("Age") < 45), "Adult")
             .when((F.col("Age") >= 45) & (F.col("Age") < 65), "Middle-aged")
             .otherwise("Senior")
        )

    #  Save cleaned and enriched data as a Parquet file
    silver_file = f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    output_dir = os.path.join(silver_dir, "attributes")  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, silver_file)
    df.write.mode("overwrite").parquet(output_path)

    print(f"[SILVER][attributes] saved to: {output_path}, columns: {len(df.columns)}")
    return df

