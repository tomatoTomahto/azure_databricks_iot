# Databricks notebook source
# MAGIC %md # Manufacturing IoT Analytics on Azure Databricks
# MAGIC ## Part 1: Data Engineering
# MAGIC This notebook demonstrates the following architecture for IIoT Ingest, Processing and Analytics on Azure. The following architecture is implemented for the demo. 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Manufacturing_architecture.png" width=800>
# MAGIC 
# MAGIC The notebook is broken into sections following these steps:
# MAGIC 1. **Data Ingest** - stream real-time raw sensor data from Azure IoT Hubs into the Delta format in Azure Storage
# MAGIC 2. **Data Processing** - stream process sensor data from raw (Bronze) to silver (aggregated) to gold (enriched) Delta tables on Azure Storage

# COMMAND ----------

# AzureML Workspace info (name, region, resource group and subscription ID) for model deployment
dbutils.widgets.text("Storage Account","<your ADLS Gen 2 account name>","Storage Account")

# COMMAND ----------

# MAGIC %md ## Step 1 - Environment Setup
# MAGIC 
# MAGIC The pre-requisites are listed below:
# MAGIC 
# MAGIC ### Azure Services Required
# MAGIC * Azure IoT Hub 
# MAGIC * [Azure IoT Simulator](https://azure-samples.github.io/raspberry-pi-web-simulator/) running with the code provided in [this github repo](https://github.com/tomatoTomahto/azure_databricks_iot/blob/master/Manufacturing%20IoT/iot_simulator.js) and configured for your IoT Hub
# MAGIC * ADLS Gen 2 Storage account with a container called `iot`
# MAGIC 
# MAGIC ### Azure Databricks Configuration Required
# MAGIC * 3-node (min) Databricks Cluster running **DBR 7.0+** and the following libraries:
# MAGIC  * **Azure Event Hubs Connector for Databricks** - Maven coordinates `com.microsoft.azure:azure-eventhubs-spark_2.12:2.3.17`
# MAGIC * The following Secrets defined in scope `iot`
# MAGIC  * `iothub-cs` - Connection string for your IoT Hub **(Important - use the [Event Hub Compatible](https://devblogs.microsoft.com/iotdev/understand-different-connection-strings-in-azure-iot-hub/) connection string)**
# MAGIC  * `adls_key` - Access Key to ADLS storage account **(Important - use the [Access Key](https://raw.githubusercontent.com/tomatoTomahto/azure_databricks_iot/master/bricks.com/blog/2020/03/27/data-exfiltration-protection-with-azure-databricks.html))**
# MAGIC * The following notebook widgets populated:
# MAGIC  * `Storage Account` - Name of your storage account

# COMMAND ----------

# Setup access to storage account for temp data when pushing to Synapse
storage_account = dbutils.widgets.get("Storage Account")
spark.conf.set(f"fs.azure.account.key.{storage_account}.dfs.core.windows.net", dbutils.secrets.get("iot","adls_key"))

# Setup storage locations for all data
ROOT_PATH = f"abfss://iot@{storage_account}.dfs.core.windows.net/manufacturing_demo/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"
SYNAPSE_PATH = ROOT_PATH + "synapse/"
CHECKPOINT_PATH = ROOT_PATH + "checkpoints/"

# Other initializations
IOT_CS = dbutils.secrets.get('iot','iothub-cs') # IoT Hub connection string (Event Hub Compatible)
ehConf = { 'eventhubs.connectionString':sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(IOT_CS) }

# Enable auto compaction and optimized writes in Delta
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled","true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled","true")

# Pyspark and ML Imports
import os, json, requests
from pyspark.sql import functions as F

# COMMAND ----------

# Make sure root path is empty
dbutils.fs.rm(ROOT_PATH, True)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Clean up tables & views
# MAGIC DROP TABLE IF EXISTS manufacturing.sensors_raw;
# MAGIC DROP TABLE IF EXISTS manufacturing.sensors_agg;
# MAGIC DROP TABLE IF EXISTS manufacturing.sensors_enriched;
# MAGIC DROP VIEW IF EXISTS manufacturing.facilities;
# MAGIC DROP VIEW IF EXISTS manufacturing.operating_limits;
# MAGIC DROP VIEW IF EXISTS manufacturing.parts_inventory;
# MAGIC DROP VIEW IF EXISTS manufacturing.ml_feature_view;
# MAGIC DROP TABLE IF EXISTS manufacturing.temperature_predictions;
# MAGIC DROP DATABASE IF EXISTS manufacturing;
# MAGIC 
# MAGIC -- Create a database to host our tables - Databricks stores the table/database metadata but all data (Delta/Parquet files) is stored in ADLS
# MAGIC CREATE DATABASE IF NOT EXISTS manufacturing;

# COMMAND ----------

# MAGIC %md ## Step 2 - Data Ingest from IoT Hubs
# MAGIC Azure Databricks provides a native connector to IoT and Event Hubs. Below, we will use PySpark Structured Streaming to read from an IoT Hub stream of data and write the data in it's raw format directly into Delta. 
# MAGIC 
# MAGIC Make sure that your IoT Simulator is sending payloads to IoT Hub as shown below.
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iot_simulator.gif" width=800>
# MAGIC 
# MAGIC We have one type of data payload in our IoT Hub:
# MAGIC 1. **Sensor readings** - this payload contains `date`,`timestamp`,`temperature`,`humidity`, `pressure`, `moisture`, `oxygen`, `radiation`, and `conductivity` fields
# MAGIC 
# MAGIC We write the raw data stream from IoT Hubs into Delta table formats on Azure Data Lake Storage. We are able to query this Bronze tables *immediately* as the data streams in.

# COMMAND ----------

# Schema of incoming data from IoT hub
schema = "facilityId string, timestamp timestamp, temperature double, humidity double, pressure double, moisture double, oxygen double, radiation double, conductivity double"

# Read directly from IoT Hub using the EventHubs library for Databricks
iot_stream = (
  spark.readStream.format("eventhubs")                                         # Read from IoT Hubs directly
    .options(**ehConf)                                                         # Use the Event-Hub-enabled connect string
    .load()                                                                    # Load the data
    .withColumn('reading', F.from_json(F.col('body').cast('string'), schema))  # Extract the "body" payload from the messages
    .select('reading.*', F.to_date('reading.timestamp').alias('date'))         # Create a "date" field for partitioning
)

# Write the stream into Delta locations on ADLS
write_iot_to_delta = ( iot_stream
    .select('date','facilityid','timestamp','temperature','humidity','pressure',
            'moisture','oxygen','radiation','conductivity')                    # Extract the fields of interest
    .writeStream.format('delta')                                               # Write our stream to the Delta format
    .partitionBy('date')                                                       # Partition our data by Date for performance
    .option("checkpointLocation", CHECKPOINT_PATH + "sensors_raw")             # Checkpoint so we can restart streams gracefully
    .start(BRONZE_PATH + "sensors_raw")                                        # Stream the data into an ADLS Path
)

# # Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS manufacturing.sensors_raw USING DELTA LOCATION "{BRONZE_PATH + "sensors_raw"}"')
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- We can query the data directly from storage immediately as soon as it starts streams into Delta 
# MAGIC SELECT * FROM manufacturing.sensors_raw

# COMMAND ----------

# MAGIC %md ## Step 2 - Data Processing in Delta
# MAGIC While our raw sensor data is being streamed into Bronze Delta tables on Azure Storage, we can create streaming pipelines on this data that flow it through Silver and Gold data sets.
# MAGIC 
# MAGIC We will use the following schema for Silver and Gold data sets:
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Manufacturing_dataflow.png" width=800>

# COMMAND ----------

# MAGIC %md ### 2a. Delta Bronze (Raw) to Delta Silver (Aggregated)
# MAGIC The first step of our processing pipeline will clean and aggregate the measurements to 5 minute intervals. 
# MAGIC 
# MAGIC Since we are aggregating time-series values and there is a likelihood of late-arriving data and data changes, we will use the [**MERGE**](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/language-manual/merge-into?toc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fazure-databricks%2Ftoc.json&bc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fbread%2Ftoc.json) functionality of Delta to upsert records into target tables. 
# MAGIC 
# MAGIC MERGE allows us to upsert source records into a target storage location. This is useful when dealing with time-series data as:
# MAGIC 1. Data often arrives late and requires aggregation states to be updated
# MAGIC 2. Historical data needs to be backfilled while streaming data is feeding into the table
# MAGIC 
# MAGIC When streaming source data, `foreachBatch()` can be used to perform a merges on micro-batches of data.

# COMMAND ----------

# Create functions to merge sensor data into it's target Delta table
def merge_delta(incremental, target): 
  incremental.dropDuplicates(['date','window','facilityid']).createOrReplaceTempView("incremental")
  
  try:
    # MERGE records into the target table using the specified join key
    incremental._jdf.sparkSession().sql(f"""
      MERGE INTO delta.`{target}` t
      USING incremental i
      ON i.date=t.date AND i.window = t.window AND i.facilityid = t.facilityid
      WHEN MATCHED THEN UPDATE SET *
      WHEN NOT MATCHED THEN INSERT *
    """)
  except:
    # If the †arget table does not exist, create one
    incremental.write.format("delta").partitionBy("date").save(target)
    
aggregate_sensors = (
  spark.readStream.format('delta').table("manufacturing.sensors_raw")          # Read data as a stream from our source Delta table
    .groupBy('facilityid','date',F.window('timestamp','5 minutes'))            # Aggregate readings to hourly intervals
    .agg(F.avg('temperature').alias('temperature'),F.avg('humidity').alias('humidity'),F.avg('pressure').alias('pressure'),
         F.avg('moisture').alias('moisture'),F.avg('oxygen').alias('oxygen'),F.avg('radiation').alias('radiation'),F.avg('conductivity').alias('conductivity'))
    .writeStream                                                               # Write the resulting stream
    .foreachBatch(lambda i, b: merge_delta(i, SILVER_PATH + "sensors_agg"))    # Pass each micro-batch to a function
    .outputMode("update")                                                      # Merge works with update mode
    .option("checkpointLocation", CHECKPOINT_PATH + "sensors_agg")             # Checkpoint so we can restart streams gracefully
    .start()
)

# # Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS manufacturing.sensors_agg USING DELTA LOCATION "{SILVER_PATH + "sensors_agg"}"')
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql
# MAGIC -- As data gets merged in real-time to our hourly table, we can query it immediately
# MAGIC SELECT * FROM manufacturing.sensors_agg WHERE facilityid = 'FAC-1' ORDER BY window ASC

# COMMAND ----------

# MAGIC %md ### 2b. Delta Silver (Aggregated) to Delta Gold (Enriched)
# MAGIC Next we perform a streaming join of sensor readings to enrichment data like facility information, capacity, location and inventory levels that we can use for data science and model training.

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Create dummy tables with fake data for enrichment
# MAGIC -- Facilities: geographical and capacity information for manufacturing facilities
# MAGIC CREATE OR REPLACE VIEW manufacturing.facilities AS
# MAGIC SELECT facilityid, 
# MAGIC   float(random()*20+30) as latitude, 
# MAGIC   float(-random()*40-80) as longitude,
# MAGIC   array('CA','OR','WA','TX','AZ','OH','AL','CO','FL','MN')[int(split(facilityid, '-')[1])] as state, 
# MAGIC   int(random()*1000+200) as capacity
# MAGIC FROM (SELECT DISTINCT facilityid FROM manufacturing.sensors_agg);
# MAGIC 
# MAGIC -- Operating Limits: temperature and pressure bounds for safe operations (components will fail when outside these bounds)
# MAGIC CREATE OR REPLACE VIEW manufacturing.operating_limits AS
# MAGIC SELECT facilityid, 
# MAGIC     float(approx_percentile(temperature,0.10)) AS min_temp,
# MAGIC     float(approx_percentile(temperature,0.90)) AS max_temp,
# MAGIC     float(approx_percentile(pressure,0.10)) AS min_pressure,
# MAGIC     float(approx_percentile(pressure,0.90)) AS max_pressure
# MAGIC FROM manufacturing.sensors_agg
# MAGIC GROUP BY facilityid;
# MAGIC 
# MAGIC -- Parts Inventory: daily inventory levels for parts at each facility
# MAGIC CREATE OR REPLACE VIEW manufacturing.parts_inventory AS
# MAGIC SELECT facilityid, 
# MAGIC   date,
# MAGIC   float(random()*500+200) as inventory
# MAGIC FROM (SELECT DISTINCT facilityid, date FROM manufacturing.sensors_agg);

# COMMAND ----------

# Read streams from Delta Silver tables and join them together on common columns (facilityid)
sensors_agg = spark.readStream.format('delta').option("ignoreChanges", True).table('manufacturing.sensors_agg')
sensors_enriched = (
  sensors_agg.join(spark.table('manufacturing.facilities'), 'facilityid')
    .join(spark.table('manufacturing.operating_limits'),'facilityid')
)

# Write the stream to a foreachBatch function which performs the MERGE as before
merge_gold_stream = (
  sensors_enriched
    .withColumn('window',sensors_enriched.window.start)
    .writeStream 
    .foreachBatch(lambda i, b: merge_delta(i, GOLD_PATH + "sensors_enriched"))
    .option("checkpointLocation", CHECKPOINT_PATH + "sensors_enriched")         
    .start()
)

# Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS manufacturing.sensors_enriched USING DELTA LOCATION "{GOLD_PATH + "sensors_enriched"}"')    
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql SELECT * FROM manufacturing.sensors_enriched WHERE facilityid="FAC-0" ORDER BY date, window

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Structure Optimization
# MAGIC Delta `OPTIMIZE` commands perform file compaction and multi-dimensional clustering (called ZORDER) on a set of columns. This is useful as IoT data typically generates many small files that need to be compacted, and querying the tables based on a facility or timestamp can be sped up by ordering the data on those columns for file-skipping. Delta does this automatically using [auto-optimize](https://docs.microsoft.com/en-us/azure/databricks/delta/optimizations/auto-optimize), or can be done periodically using the [optimize](https://docs.microsoft.com/en-us/azure/databricks/delta/optimizations/file-mgmt) command below. 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Optimize all 3 tables for querying and model training performance
# MAGIC OPTIMIZE manufacturing.sensors_raw ZORDER BY facilityid, timestamp;
# MAGIC OPTIMIZE manufacturing.sensors_agg ZORDER BY facilityid, window;
# MAGIC OPTIMIZE manufacturing.sensors_enriched ZORDER BY facilityid, window;

# COMMAND ----------

# MAGIC %md Our Delta Gold tables are now ready for predictive analytics! We now have aggregated and enriched sensor readings. Our next step is to predict the specific operating conditions that lead to component failures (ie. going out of operating limit bounds)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Benefits of Delta Lake on Time-Series Data
# MAGIC A key component of this architecture is the Azure Data Lake Store (ADLS), which enables the write-once, access-often analytics pattern in Azure. However, Data Lakes alone do not solve challenges that come with time-series streaming data. The Delta storage format provides a layer of resiliency and performance on all data sources stored in ADLS. Specifically for time-series data, Delta provides the following advantages over other storage formats on ADLS:
# MAGIC 
# MAGIC |**Required Capability**|**Other formats on ADLS**|**Delta Format on ADLS**|
# MAGIC |--------------------|-----------------------------|---------------------------|
# MAGIC |**Unified batch & streaming**|Data Lakes are often used in conjunction with a streaming store like CosmosDB, resulting in a complex architecture|ACID-compliant transactions enable data engineers to perform streaming ingest and historically batch loads into the same locations on ADLS|
# MAGIC |**Schema enforcement and evolution**|Data Lakes do not enforce schema, requiring all data to be pushed into a relational database for reliability|Schema is enforced by default. As new IoT devices are added to the data stream, schemas can be evolved safely so downstream applications don’t fail|
# MAGIC |**Efficient Upserts**|Data Lakes do not support in-line updates and merges, requiring deletion and insertions of entire partitions to perform updates|MERGE commands are effective for situations handling delayed IoT readings, modified dimension tables used for real-time enrichment, or if data needs to be reprocessed|
# MAGIC |**File Compaction**|Streaming time-series data into Data Lakes generate hundreds or even thousands of tiny files|Auto-compaction in Delta optimizes the file sizes to increase throughput and parallelism|
# MAGIC |**Multi-dimensional clustering**|Data Lakes provide push-down filtering on partitions only|ZORDERing time-series on fields like timestamp or sensor ID allows Databricks to filter and join on those columns up to 100x faster than simple partitioning techniques|

# COMMAND ----------

# MAGIC %md ## (Optional) - Stream Data to Synapse or Azure Data Explorer for Serving
# MAGIC The Data Lake is a great data store for historical analysis, data science, ML and ad-hoc visualization against *all hitorical* data using Databricks. However, a common use case in IoT projects is to serve an aggregated subset (3-6 months) or business level summary data to end users. Synapse SQL Pools provide *low latency, high concurrency* serving capabilities to BI tools for production-level reporting and BI. Follow the example notebook [here](https://databricks.com/notebooks/iiot/iiot-end-to-end-part-1.html) to stream our **GOLD** Delta table into a Synapse SQL Pool for reporting. 
# MAGIC 
# MAGIC Similarily, another common use case in IoT projects is to serve real-time time-series reading into an operational dashboard used by operational engineers. Azure Data Explorer provides a real-time database for building operational dashboards on *current* data. Databricks can be used to stream either the raw or aggregated sensor data into ADX for operational serving. Follow the example snippet in the blog article [here](https://databricks.com/blog/2020/08/20/modern-industrial-iot-analytics-on-azure-part-3.html) to stream data from Delta into ADX. 

# COMMAND ----------

