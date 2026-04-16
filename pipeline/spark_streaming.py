import os
import sys
import logging
import pyspark
from feast import FeatureStore
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count, approx_count_distinct
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql import DataFrame
# --- THE ULTIMATE WINDOWS PYSPARK PATCH ---
# These lines force Windows to behave like Linux. ---
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin" + ";" + os.environ["PATH"]
os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"
# ------------------------------------------

logging.getLogger("py4j").setLevel(logging.ERROR)

def main() -> None:
    print("🚀 Initializing Spark Session...")
    
    #spark session initialization
    #master("local[*]"): Use all CPU cores on your laptop
    #spark.jars.packages:	Downloads the Kafka connector from Maven so Spark can talk to Red Panda
    #spark.driver.host / bindAddress:	Forces Spark to use localhost (prevents Windows firewall popups)

    spark = SparkSession.builder \
        .appName("StreamGuard-Feature-Engine") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    #Purpose: This is the contract for what JSON messages look like coming from Red Panda.
    #Without this schema, Spark treats the JSON as a generic string and you can't query nested fields like properties.ip
    event_schema = StructType([
        StructField("event_id", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("timestamp", TimestampType(), True), 
        StructField("session_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("properties", StructType([
            StructField("ip", StringType(), True)
        ]), True)
    ])

    print("🎧 Connecting to Redpanda/Kafka topic 'shop-events'...")
    
    #localhost:19092 : Red Panda broker address
    #startingOffsets, "latest"	Ignore old messages; only process new ones that arrive after this job starts
    #earliest: read all messages that were already in red panda.
    
    #result of raw_stream is a DataFrame with columns key, value, topic, partition, offset. The actual JSON is in the value column as binary.
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:19092") \
        .option("subscribe", "shop-events") \
        .option("startingOffsets", "latest") \
        .load()

    parsed_stream = raw_stream \
        .selectExpr("CAST(value AS STRING) as json_payload") \
        .select(from_json(col("json_payload"), event_schema).alias("data")) \
        .select("data.*")

    # Calculate features and extract the window end time for Feast
    feature_stream = parsed_stream \
        .withWatermark("timestamp", "1 minute") \
        .groupBy(
            window(col("timestamp"), "10 seconds"),
            col("user_id")
        ) \
        .agg(
            count("event_id").alias("click_velocity"),
            approx_count_distinct("properties.ip").alias("ip_entropy")
        ) \
        .withColumn("event_timestamp", col("window.end")) # Feast requires this exact column name

    # --- THE BRIDGE TO FEAST ---
    # This function runs every time Spark finishes a 10-second math batch
    # --- THE BRIDGE TO FEAST & PARQUET ---
    def write_to_feast(batch_df: DataFrame, batch_id: int) -> None:
        pdf = batch_df.toPandas()
        
        if not batch_df.isEmpty():
            # 1. Push to Redis (Online Store for Real-Time)
            store = FeatureStore(repo_path="feature_repo/feature_repo")
            store.write_to_online_store(feature_view_name="user_activity", df=pdf)
            
            # 2. Push to Parquet (Offline Store for Training/Testing)
            # This satisfies your syllabus requirement!
            batch_df.write.mode("append").parquet("feature_repo/feature_repo/data/offline_features")
            
            print(f"✅ Pushed {len(pdf)} profiles to Redis AND Parquet!")
    # ---------------------------
    # ---------------------------

    print("🧮 Spark is now calculating and writing to Redis...")
    
    # Send the stream to our custom function instead of the console
    query = feature_stream.writeStream \
        .outputMode("update") \
        .foreachBatch(write_to_feast) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()