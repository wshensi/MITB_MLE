import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# Bronze Layer: Raw ingestion and storage
def process_bronze_clickstream(snapshot_date_str, bronze_dir, spark):
    """
    Read feature_clickstream.csv, optionally filter by snapshot_date if present,
    and save it to the bronze layer as a CSV file.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/feature_clickstream.csv"  # Adjust this path if needed

    # Load raw CSV file
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Filter by snapshot_date if the column exists
    if 'snapshot_date' in df.columns:
        df = df.filter(F.col('snapshot_date') == snapshot_date)

    # Save as bronze CSV file
    file_name = f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    output_path = os.path.join(bronze_dir, file_name)
    df.toPandas().to_csv(output_path, index=False)

    print(f"[BRONZE][clickstream] saved to: {output_path}, row count: {df.count()}")
    return df


# Silver Layer: Data cleaning, type casting, and feature enrichment
def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Read clickstream data from the bronze layer, perform cleaning, type casting,
    and feature aggregation, then save the result as a Parquet file in the silver layer.
    """
    file_name = f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    input_path = os.path.join(bronze_dir, file_name)
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    print(f"[SILVER][clickstream] loaded from: {input_path}, row count: {df.count()}")

    # Type casting for expected columns
    if 'Customer_ID' in df.columns:
        df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))
    # Convert snapshot_date to DateType to ensure consistency with other tables during joins
    df = df.withColumn("snapshot_date", F.to_date(F.lit(snapshot_date_str)))

    # Print schema to check the available fields
    df.printSchema()

    # Feature aggregation
    # Count the total number of clickstream records for each Customer_ID and snapshot_date
    df = df.groupBy("Customer_ID", "snapshot_date").agg(
        F.count("*").alias("total_clicks")
    )

    # Ensure the output directory exists
    output_dir = os.path.join(silver_dir, "clickstream")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the cleaned and aggregated data as a Parquet file
    silver_file = f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    output_path = os.path.join(output_dir, silver_file)
    df.write.mode("overwrite").parquet(output_path)

    print(f"[SILVER][clickstream] saved to: {output_path}, columns: {len(df.columns)}")
    return df
