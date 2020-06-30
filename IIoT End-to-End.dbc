# Databricks notebook source
# MAGIC %md # End to End Industrial IoT (IIoT) on Azure Databricks
# MAGIC This notebook demonstrates the following architecture for IIoT Ingest, Processing and Analytics on Azure. The following architecture is implemented for the demo. 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/end_to_end_architecture.png" width=800>
# MAGIC 
# MAGIC The notebook is broken into sections following these steps:
# MAGIC 1. **Data Ingest** - stream real-time raw sensor data from Azure IoT Hubs into the Delta format in Azure Storage
# MAGIC 2. **Data Processing** - stream process sensor data from raw (Bronze) to silver (aggregated) to gold (enriched) Delta tables on Azure Storage
# MAGIC 3. **Machine Learning** - train XGBoost regression models using distributed ML to predict power output and asset remaining life on historical sensor data
# MAGIC 4. **Model Deployment** - deploy trained models for real-time serving in Azure ML services 
# MAGIC 5. **Model Inference** - score real data instantly against hosted models via REST API

# COMMAND ----------

# AzureML Workspace info (name, region, resource group and subscription ID) for model deployment
dbutils.widgets.text("Subscription ID","3f2e4d32-8e8d-46d6-82bc-5bb8d962328b","Subscription ID")
dbutils.widgets.text("Resource Group","sgupta_rg","Resource Group")
dbutils.widgets.text("Region","eastus2","Region")
dbutils.widgets.text("Storage Account","sguptaadlsg2","Storage Account")

# COMMAND ----------

# MAGIC %md ## Step 1 - Environment Setup
# MAGIC 
# MAGIC The pre-requisites are listed below:
# MAGIC 
# MAGIC ### Azure Services Required
# MAGIC * Azure IoT Hub 
# MAGIC * [Azure IoT Simulator](https://azure-samples.github.io/raspberry-pi-web-simulator/) running with the code provided in [this github repo] and configured for your IoT Hub
# MAGIC * ADLS Gen 2 Storage account with a container called `iot`
# MAGIC * Azure Synapse SQL Pool call `iot`
# MAGIC * Azure Machine Learning Workspace called `iot`
# MAGIC 
# MAGIC ### Azure Databricks Configuration Required
# MAGIC * 3-node (min) Databricks Cluster running **DBR 7.0ML+** and the following libraries:
# MAGIC  * **MLflow[AzureML]** - PyPI library `azureml-mlflow`
# MAGIC  * **Azure Event Hubs Connector for Databricks** - Maven coordinates `com.microsoft.azure:azure-eventhubs-spark_2.12:2.3.16`
# MAGIC * The following Secrets defined in scope `iot`
# MAGIC  * `iothub-cs` - Connection string for your IoT Hub **(Important - use the [Event Hub Compatible](https://devblogs.microsoft.com/iotdev/understand-different-connection-strings-in-azure-iot-hub/) connection string)**
# MAGIC  * `adls_key` - Access Key to ADLS storage account **(Important - use the [Access Key](https://raw.githubusercontent.com/tomatoTomahto/azure_databricks_iot/master/bricks.com/blog/2020/03/27/data-exfiltration-protection-with-azure-databricks.html))**
# MAGIC  * `synapse_cs` - JDBC connect string to your Synapse SQL Pool **(Important - use the [SQL Authentication](https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/azure/synapse-analytics#spark-driver-to-azure-synapse) with username/password connection string)**
# MAGIC * The following notebook widgets populated:
# MAGIC  * `Subscription ID` - subscription ID of your Azure ML Workspace
# MAGIC  * `Resource Group` - resource group name of your Azure ML Workspace
# MAGIC  * `Region` - Azure region of your Azure ML Workspace
# MAGIC  * `Storage Account` - Name of your storage account

# COMMAND ----------

# Setup access to storage account for temp data when pushing to Synapse
storage_account = dbutils.widgets.get("Storage Account")
spark.conf.set(f"fs.azure.account.key.{storage_account}.dfs.core.windows.net", dbutils.secrets.get("iot","adls_key"))

# Setup storage locations for all data
ROOT_PATH = f"abfss://iot@{storage_account}.dfs.core.windows.net/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"
SYNAPSE_PATH = ROOT_PATH + "synapse/"
CHECKPOINT_PATH = ROOT_PATH + "checkpoints/"

# Other initializations
IOT_CS = dbutils.secrets.get('iot','iothub-cs') # IoT Hub connection string
ehConf = { 'eventhubs.connectionString':sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(IOT_CS) }

# Enable auto compaction and optimized writes in Delta
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled","true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled","true")

# Pyspark and ML Imports
import os, json, requests
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np 
import pandas as pd
import xgboost as xgb
import mlflow.xgboost
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
import random, string
random_string = lambda length: ''.join(random.SystemRandom().choice(string.ascii_lowercase) for _ in range(length))

# COMMAND ----------

# Make sure root path is empty
dbutils.fs.rm(ROOT_PATH, True)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Clean up tables & views
# MAGIC DROP TABLE IF EXISTS turbine_raw;
# MAGIC DROP TABLE IF EXISTS weather_raw;
# MAGIC DROP TABLE IF EXISTS turbine_agg;
# MAGIC DROP TABLE IF EXISTS weather_agg;
# MAGIC DROP TABLE IF EXISTS turbine_enriched;
# MAGIC DROP TABLE IF EXISTS turbine_power;
# MAGIC DROP TABLE IF EXISTS turbine_maintenance;
# MAGIC DROP VIEW IF EXISTS turbine_combined;
# MAGIC DROP VIEW IF EXISTS feature_view;
# MAGIC DROP TABLE IF EXISTS turbine_life_predictions;
# MAGIC DROP TABLE IF EXISTS turbine_power_predictions;

# COMMAND ----------

# MAGIC %md ## Step 2 - Data Ingest from IoT Hubs
# MAGIC Azure Databricks provides a native connector to IoT and Event Hubs. Below, we will use PySpark Structured Streaming to read from an IoT Hub stream of data and write the data in it's raw format directly into Delta. 
# MAGIC 
# MAGIC Make sure that your IoT Simulator is sending payloads to IoT Hub as shown below.
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iot_simulator.gif" width=800>
# MAGIC 
# MAGIC We have two separate types of data payloads in our IoT Hub:
# MAGIC 1. **Turbine Sensor readings** - this payload contains `date`,`timestamp`,`deviceid`,`rpm` and `angle` fields
# MAGIC 2. **Weather Sensor readings** - this payload contains `date`,`timestamp`,`temperature`,`humidity`,`windspeed`, and `winddirection` fields
# MAGIC 
# MAGIC We split out the two payloads into separate streams and write them both into Delta locations on Azure Storage. We are able to query these two Bronze tables *immediately* as the data streams in.

# COMMAND ----------

# Schema of incoming data from IoT hub
schema = "timestamp timestamp, deviceId string, temperature double, humidity double, windspeed double, winddirection string, rpm double, angle double"

# Read directly from IoT Hub using the EventHubs library for Databricks
iot_stream = (
  spark.readStream.format("eventhubs")                                               # Read from IoT Hubs directly
    .options(**ehConf)                                                               # Use the Event-Hub-enabled connect string
    .load()                                                                          # Load the data
    .withColumn('reading', F.from_json(F.col('body').cast('string'), schema))        # Extract the "body" payload from the messages
    .select('reading.*', F.to_date('reading.timestamp').alias('date'))               # Create a "date" field for partitioning
)

# Split our IoT Hub stream into separate streams and write them both into their own Delta locations
write_turbine_to_delta = (
  iot_stream.filter('temperature is null')                                           # Filter out turbine telemetry from other data streams
    .select('date','timestamp','deviceId','rpm','angle')                             # Extract the fields of interest
    .writeStream.format('delta')                                                     # Write our stream to the Delta format
    .partitionBy('date')                                                             # Partition our data by Date for performance
    .option("checkpointLocation", CHECKPOINT_PATH + "turbine_raw")                   # Checkpoint so we can restart streams gracefully
    .start(BRONZE_PATH + "turbine_raw")                                              # Stream the data into an ADLS Path
)

write_weather_to_delta = (
  iot_stream.filter(iot_stream.temperature.isNotNull())                              # Filter out weather telemetry only
    .select('date','deviceid','timestamp','temperature','humidity','windspeed','winddirection') 
    .writeStream.format('delta')                                                     # Write our stream to the Delta format
    .partitionBy('date')                                                             # Partition our data by Date for performance
    .option("checkpointLocation", CHECKPOINT_PATH + "weather_raw")                   # Checkpoint so we can restart streams gracefully
    .start(BRONZE_PATH + "weather_raw")                                              # Stream the data into an ADLS Path
)

# Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS turbine_raw USING DELTA LOCATION "{BRONZE_PATH + "turbine_raw"}"')
    spark.sql(f'CREATE TABLE IF NOT EXISTS weather_raw USING DELTA LOCATION "{BRONZE_PATH + "weather_raw"}"')
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- We can query the data directly from storage immediately as soon as it starts streams into Delta 
# MAGIC SELECT * FROM turbine_raw WHERE deviceid = 'WindTurbine-1'

# COMMAND ----------

# MAGIC %md ## Step 2 - Data Processing in Delta
# MAGIC While our raw sensor data is being streamed into Bronze Delta tables on Azure Storage, we can create streaming pipelines on this data that flow it through Silver and Gold data sets.
# MAGIC 
# MAGIC We will use the following schema for Silver and Gold data sets:
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iot_delta_bronze_to_gold.png" width=800>

# COMMAND ----------

# MAGIC %md ### 2a. Delta Bronze (Raw) to Delta Silver (Aggregated)
# MAGIC The first step of our processing pipeline will clean and aggregate the measurements to 1 hour intervals. 
# MAGIC 
# MAGIC Since we are aggregating time-series values and there is a likelihood of late-arriving data and data changes, we will use the [**MERGE**](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/language-manual/merge-into?toc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fazure-databricks%2Ftoc.json&bc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fbread%2Ftoc.json) functionality of Delta to upsert records into target tables. 
# MAGIC 
# MAGIC MERGE allows us to upsert source records into a target storage location. This is useful when dealing with time-series data as:
# MAGIC 1. Data often arrives late and requires aggregation states to be updated
# MAGIC 2. Historical data needs to be backfilled while streaming data is feeding into the table
# MAGIC 
# MAGIC When streaming source data, `foreachBatch()` can be used to perform a merges on micro-batches of data.

# COMMAND ----------

# Create functions to merge turbine and weather data into their target Delta tables
def merge_delta(incremental, target): 
  incremental.dropDuplicates(['date','window','deviceid']).createOrReplaceTempView("incremental")
  
  try:
    # MERGE records into the target table using the specified join key
    incremental._jdf.sparkSession().sql(f"""
      MERGE INTO delta.`{target}` t
      USING incremental i
      ON i.date=t.date AND i.window = t.window AND i.deviceId = t.deviceid
      WHEN MATCHED THEN UPDATE SET *
      WHEN NOT MATCHED THEN INSERT *
    """)
  except:
    # If the †arget table does not exist, create one
    incremental.write.format("delta").partitionBy("date").save(target)
    
turbine_b_to_s = (
  spark.readStream.format('delta').table("turbine_raw")                        # Read data as a stream from our source Delta table
    .groupBy('deviceId','date',F.window('timestamp','5 minutes'))              # Aggregate readings to hourly intervals
    .agg(F.avg('rpm').alias('rpm'), F.avg("angle").alias("angle"))
    .writeStream                                                               # Write the resulting stream
    .foreachBatch(lambda i, b: merge_delta(i, SILVER_PATH + "turbine_agg"))    # Pass each micro-batch to a function
    .outputMode("update")                                                      # Merge works with update mode
    .option("checkpointLocation", CHECKPOINT_PATH + "turbine_agg")             # Checkpoint so we can restart streams gracefully
    .start()
)

weather_b_to_s = (
  spark.readStream.format('delta').table("weather_raw")                        # Read data as a stream from our source Delta table
    .groupBy('deviceid','date',F.window('timestamp','5 minutes'))              # Aggregate readings to hourly intervals
    .agg({"temperature":"avg","humidity":"avg","windspeed":"avg","winddirection":"last"})
    .selectExpr('date','window','deviceid','`avg(temperature)` as temperature','`avg(humidity)` as humidity',
                '`avg(windspeed)` as windspeed','`last(winddirection)` as winddirection')
    .writeStream                                                               # Write the resulting stream
    .foreachBatch(lambda i, b: merge_delta(i, SILVER_PATH + "weather_agg"))    # Pass each micro-batch to a function
    .outputMode("update")                                                      # Merge works with update mode
    .option("checkpointLocation", CHECKPOINT_PATH + "weather_agg")             # Checkpoint so we can restart streams gracefully
    .start()
)

# Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS turbine_agg USING DELTA LOCATION "{SILVER_PATH + "turbine_agg"}"')
    spark.sql(f'CREATE TABLE IF NOT EXISTS weather_agg USING DELTA LOCATION "{SILVER_PATH + "weather_agg"}"')
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql
# MAGIC -- As data gets merged in real-time to our hourly table, we can query it immediately
# MAGIC SELECT * FROM turbine_agg t JOIN weather_agg w ON (t.date=w.date AND t.window=w.window) WHERE t.deviceid='WindTurbine-1' ORDER BY t.window DESC

# COMMAND ----------

# MAGIC %md ### 2b. Delta Silver (Aggregated) to Delta Gold (Enriched)
# MAGIC Next we perform a streaming join of weather and turbine readings to create one enriched dataset we can use for data science and model training.

# COMMAND ----------

# Read streams from Delta Silver tables and join them together on common columns (date & window)
turbine_agg = spark.readStream.format('delta').option("ignoreChanges", True).table('turbine_agg')
weather_agg = spark.readStream.format('delta').option("ignoreChanges", True).table('weather_agg').drop('deviceid')
turbine_enriched = turbine_agg.join(weather_agg, ['date','window'])

# Write the stream to a foreachBatch function which performs the MERGE as before
merge_gold_stream = (
  turbine_enriched
    .selectExpr('date','deviceid','window.start as window','rpm','angle','temperature','humidity','windspeed','winddirection')
    .writeStream 
    .foreachBatch(lambda i, b: merge_delta(i, GOLD_PATH + "turbine_enriched"))
    .option("checkpointLocation", CHECKPOINT_PATH + "turbine_enriched")         
    .start()
)

# Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS turbine_enriched USING DELTA LOCATION "{GOLD_PATH + "turbine_enriched"}"')
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql SELECT * FROM turbine_enriched WHERE deviceid='WindTurbine-1'

# COMMAND ----------

# MAGIC %md ### 2c: Stream Delta GOLD Table to Synapse
# MAGIC Synapse Analytics provides on-demand SQL directly on Data Lake source formats. Databricks can also directly stream data to Synapse SQL Pools for Data Warehousing workloads like BI dashboarding and reporting. 
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/synapse_databricks_delta.png" width=800>

# COMMAND ----------

spark.conf.set("spark.databricks.sqldw.writeSemantics", "copy")                           # Use COPY INTO for faster loads to Synapse from Databricks

write_to_synapse = (
  spark.readStream.format('delta').option('ignoreChanges',True).table('turbine_enriched') # Read in Gold turbine readings from Delta as a stream
    .writeStream.format("com.databricks.spark.sqldw")                                     # Write to Synapse (SQL DW connector)
    .option("url",dbutils.secrets.get("iot","synapse_cs"))                                # SQL Pool JDBC connection (with SQL Auth) string
    .option("tempDir", SYNAPSE_PATH)                                                      # Temporary ADLS path to stage the data (with forwarded permissions)
    .option("forwardSparkAzureStorageCredentials", "true")
    .option("dbTable", "turbine_enriched")                                                # Table in Synapse to write to
    .option("checkpointLocation", CHECKPOINT_PATH+"synapse")                              # Checkpoint for resilient streaming
    .start()
)

# COMMAND ----------

# MAGIC %md ### 2d. Backfill Historical Data
# MAGIC In order to train a model, we will need to backfill our streaming data with historical data. The cell below generates 1 year of historical hourly turbine and weather data and inserts it into our Gold Delta table.

# COMMAND ----------

# Function to simulate generating time-series data given a baseline, slope, and some seasonality
def generate_series(time_index, baseline, slope=0.01, period=365*24*12):
  rnd = np.random.RandomState(time_index)
  season_time = (time_index % period) / period
  seasonal_pattern = np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))
  return baseline * (1 + 0.1 * seasonal_pattern + 0.1 * rnd.randn(len(time_index)))
  
# Get start and end dates for our historical data
dates = spark.sql('select max(date)-interval 365 days as start, max(date) as end from turbine_enriched').toPandas()
  
# Get the baseline readings for each sensor for backfilling data
turbine_enriched_pd = spark.table('turbine_enriched').toPandas()
baselines = turbine_enriched_pd.min()[3:8]
devices = turbine_enriched_pd['deviceid'].unique()

# Iterate through each device to generate historical data for that device
print("---Generating Historical Enriched Turbine Readings---")
for deviceid in devices:
  print(f'Backfilling device {deviceid}')
  windows = pd.date_range(start=dates['start'][0], end=dates['end'][0], freq='5T') # Generate a list of hourly timestamps from start to end date
  historical_values = pd.DataFrame({
    'date': windows.date,
    'window': windows, 
    'winddirection': np.random.choice(['N','NW','W','SW','S','SE','E','NE'], size=len(windows)),
    'deviceId': deviceid
  })
  time_index = historical_values.index.to_numpy()                                 # Generate a time index

  for sensor in baselines.keys():
    historical_values[sensor] = generate_series(time_index, baselines[sensor])    # Generate time-series data from this sensor

  # Write dataframe to enriched_readings Delta table
  spark.createDataFrame(historical_values).write.format("delta").mode("append").saveAsTable("turbine_enriched")
  
# Create power readings based on weather and operating conditions
print("---Generating Historical Turbine Power Readings---")
spark.sql(f'CREATE TABLE turbine_power USING DELTA PARTITIONED BY (date) LOCATION "{GOLD_PATH + "turbine_power"}" AS SELECT date, window, deviceId, 0.1 * (temperature/humidity) * (3.1416 * 25) * windspeed * rpm AS power FROM turbine_enriched')

# Create a maintenance records based on peak power usage
print("---Generating Historical Turbine Maintenance Records---")
spark.sql(f'CREATE TABLE turbine_maintenance USING DELTA LOCATION "{GOLD_PATH + "turbine_maintenance"}" AS SELECT DISTINCT deviceid, FIRST(date) OVER (PARTITION BY deviceid, year(date), month(date) ORDER BY power) AS date, True AS maintenance FROM turbine_power')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Optimize all 3 tables for querying and model training performance
# MAGIC OPTIMIZE turbine_enriched WHERE date<current_date() ZORDER BY deviceid, window;
# MAGIC OPTIMIZE turbine_power ZORDER BY deviceid, window;
# MAGIC OPTIMIZE turbine_maintenance ZORDER BY deviceid;

# COMMAND ----------

# MAGIC %md Our Delta Gold tables are now ready for predictive analytics! We now have hourly weather, turbine operating and power measurements, and daily maintenance logs going back one year. We can see that there is significant correlation between most of the variables.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query all 3 tables
# MAGIC CREATE OR REPLACE VIEW gold_readings AS
# MAGIC SELECT r.*, 
# MAGIC   p.power, 
# MAGIC   ifnull(m.maintenance,False) as maintenance
# MAGIC FROM turbine_enriched r 
# MAGIC   JOIN turbine_power p ON (r.date=p.date AND r.window=p.window AND r.deviceid=p.deviceid)
# MAGIC   LEFT JOIN turbine_maintenance m ON (r.date=m.date AND r.deviceid=m.deviceid);
# MAGIC   
# MAGIC SELECT * FROM gold_readings ORDER BY deviceid, window

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

# MAGIC %md ## Step 3 - Machine Learning
# MAGIC Now that our data is flowing reliably from our sensor devices into an enriched Delta table in Data Lake storage, we can start to build ML models to predict power output and remaining life of our assets using historical sensor, weather, power and maintenance data. 
# MAGIC 
# MAGIC We create two models ***for each Wind Turbine***:
# MAGIC 1. Turbine Power Output - using current readings for turbine operating parameters (angle, RPM) and weather (temperature, humidity, etc.), predict the expected power output 6 hours from now
# MAGIC 2. Turbine Remaining Life - predict the remaining life in days until the next maintenance event
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/turbine_models.png" width=800>
# MAGIC 
# MAGIC We will use the XGBoost framework to train regression models. Due to the size of the data and number of Wind Turbines, we will use Spark UDFs to distribute training across all the nodes in our cluster.

# COMMAND ----------

# MAGIC %md ### 3a. Feature Engineering
# MAGIC In order to predict power output 6 hours ahead, we need to first time-shift our data to create our label column. We can do this easily using Spark Window partitioning. 
# MAGIC 
# MAGIC In order to predict remaining life, we need to backtrace the remaining life from the maintenance events. We can do this easily using cross joins. The following diagram illustrates the ML Feature Engineering pipeline:
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/ml_pipeline.png" width=800>

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Calculate the age of each turbine and the remaining life in days
# MAGIC CREATE OR REPLACE VIEW turbine_age AS
# MAGIC WITH reading_dates AS (SELECT distinct date, deviceid FROM turbine_power),
# MAGIC   maintenance_dates AS (
# MAGIC     SELECT d.*, datediff(nm.date, d.date) as datediff_next, datediff(d.date, lm.date) as datediff_last 
# MAGIC     FROM reading_dates d LEFT JOIN turbine_maintenance nm ON (d.deviceid=nm.deviceid AND d.date<=nm.date)
# MAGIC     LEFT JOIN turbine_maintenance lm ON (d.deviceid=lm.deviceid AND d.date>=lm.date ))
# MAGIC SELECT date, deviceid, ifnull(min(datediff_last),0) AS age, ifnull(min(datediff_next),0) AS remaining_life
# MAGIC FROM maintenance_dates 
# MAGIC GROUP BY deviceid, date;
# MAGIC 
# MAGIC -- Calculate the power 6 hours ahead using Spark Windowing and build a feature_table to feed into our ML models
# MAGIC CREATE OR REPLACE VIEW feature_table AS
# MAGIC SELECT r.*, age, remaining_life,
# MAGIC   LEAD(power, 72, power) OVER (PARTITION BY r.deviceid ORDER BY window) as power_6_hours_ahead
# MAGIC FROM gold_readings r JOIN turbine_age a ON (r.date=a.date AND r.deviceid=a.deviceid)
# MAGIC WHERE r.date < CURRENT_DATE();

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT window, power, power_6_hours_ahead FROM feature_table WHERE deviceid='WindTurbine-1' AND date=CURRENT_DATE() - INTERVAL 1 DAYS

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT date, avg(age) as age, avg(remaining_life) as life FROM feature_table WHERE deviceid='WindTurbine-1' GROUP BY date ORDER BY date

# COMMAND ----------

# MAGIC %md ### 3b. Distributed Model Training - Predict Power Output
# MAGIC [Pandas UDFs](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/udf-python-pandas?toc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fazure-databricks%2Ftoc.json&bc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fbread%2Ftoc.json) allow us to vectorize Pandas code across multiple nodes in a cluster. Here we create a UDF to train an XGBoost Regressor model against all the historic data for a particular Wind Turbine. We use a Grouped Map UDF as we perform this model training on the Wind Turbine group level.

# COMMAND ----------

# Create a function to train a XGBoost Regressor on a turbine's data
def train_distributed_xgb(readings_pd, model_type, label_col, prediction_col):
  mlflow.xgboost.autolog()
  with mlflow.start_run():
    # Log the model type and device ID
    mlflow.log_param('deviceid', readings_pd['deviceid'][0])
    mlflow.log_param('model', model_type)

    # Train an XGBRegressor on the data for this Turbine
    alg = xgb.XGBRegressor() 
    train_dmatrix = xgb.DMatrix(data=readings_pd[feature_cols].astype('float'),label=readings_pd[label_col])
    params = {'learning_rate': 0.5, 'alpha':10, 'colsample_bytree': 0.5, 'max_depth': 5}
    model = xgb.train(params=params, dtrain=train_dmatrix, evals=[(train_dmatrix, 'train')])

    # Make predictions on the dataset and return the results
    readings_pd[prediction_col] = model.predict(train_dmatrix)
  return readings_pd

# Create a Spark Dataframe that contains the features and labels we need
non_feature_cols = ['date','window','deviceid','winddirection','remaining_life']
feature_cols = ['angle','rpm','temperature','humidity','windspeed','power','age']
label_col = 'power_6_hours_ahead'
prediction_col = label_col + '_predicted'

# Read in our feature table and select the columns of interest
feature_df = spark.table('feature_table').selectExpr(non_feature_cols + feature_cols + [label_col] + [f'0 as {prediction_col}'])

# Register a Pandas UDF to distribute XGB model training using Spark
@pandas_udf(feature_df.schema, PandasUDFType.GROUPED_MAP)
def train_power_models(readings_pd):
  return train_distributed_xgb(readings_pd, 'power_prediction', label_col, prediction_col)

# Run the Pandas UDF against our feature dataset - this will train 1 model for each turbine
power_predictions = feature_df.groupBy('deviceid').apply(train_power_models)

# Save predictions to storage
power_predictions.write.format("delta").mode("overwrite").partitionBy("date").saveAsTable("turbine_power_predictions")

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Plot actuals vs. predicted
# MAGIC SELECT date, deviceid, avg(power_6_hours_ahead) as actual, avg(power_6_hours_ahead_predicted) as predicted FROM turbine_power_predictions GROUP BY date, deviceid

# COMMAND ----------

# MAGIC %md #### Automated Model Tracking in Databricks
# MAGIC As you train the models, notice how Databricks-managed MLflow automatically tracks each run in the "Runs" tab of the notebook. You can open each run and view the parameters, metrics, models and model artifacts that are captured by MLflow Autologging. For XGBoost Regression models, MLflow tracks: 
# MAGIC 1. Any model parameters (alpha, colsample, learning rate, etc.) passed to the `params` variable
# MAGIC 2. Metrics specified in `evals` (RMSE by default)
# MAGIC 3. The trained XGBoost model file
# MAGIC 4. Feature importances
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iiot_mlflow_tracking.gif" width=800>

# COMMAND ----------

# MAGIC %md ### 3c. Distributed Model Training - Predict Remaining Life
# MAGIC Our second model predicts the remaining useful life of each Wind Turbine based on the current operating conditions. We have historical maintenance data that indicates when a replacement activity occured - this will be used to calculate the remaining life as our training label. 
# MAGIC 
# MAGIC Once again, we train an XGBoost model for each Wind Turbine to predict the remaining life given a set of operating parameters and weather conditions

# COMMAND ----------

# Create a Spark Dataframe that contains the features and labels we need
non_feature_cols = ['date','window','deviceid','winddirection','power_6_hours_ahead_predicted']
label_col = 'remaining_life'
prediction_col = label_col + '_predicted'

# Read in our feature table and select the columns of interest
feature_df = spark.table('turbine_power_predictions').selectExpr(non_feature_cols + feature_cols + [label_col] + [f'0 as {prediction_col}'])

# Register a Pandas UDF to distribute XGB model training using Spark
@pandas_udf(feature_df.schema, PandasUDFType.GROUPED_MAP)
def train_life_models(readings_pd):
  return train_distributed_xgb(readings_pd, 'life_prediction', label_col, prediction_col)

# Run the Pandas UDF against our feature dataset - this will train 1 model per turbine and write the predictions to a table
life_predictions = (
  feature_df.groupBy('deviceid').apply(train_life_models)
    .write.format("delta").mode("overwrite")
    .partitionBy("date")
    .saveAsTable("turbine_life_predictions")
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT date, avg(remaining_life) as Actual_Life, avg(remaining_life_predicted) as Predicted_Life 
# MAGIC FROM turbine_life_predictions 
# MAGIC WHERE deviceid='WindTurbine-1' 
# MAGIC GROUP BY date ORDER BY date

# COMMAND ----------

# MAGIC %md The models to predict remaining useful life have been trained and logged by MLflow. We can now move on to model deployment in AzureML.

# COMMAND ----------

# MAGIC %md ## Step 4 - Model Deployment to AzureML
# MAGIC Now that our models have been trained, we can deploy them in an automated way directly to a model serving environment like Azure ML. Below, we connect to an AzureML workspace, build a container image for the model, and deploy that image to Azure Container Instances (ACI) to be hosted for REST API calls. 
# MAGIC 
# MAGIC **Note:** This step can take up to 10 minutes to run due to images being created and deplyed in Azure ML.
# MAGIC 
# MAGIC **Important:** This step requires authentication to Azure - open the link provided in the output of the cell in a new browser tab and use the code provided.

# COMMAND ----------

# AML Workspace Information - replace with your workspace info
aml_resource_group = dbutils.widgets.get("Resource Group")
aml_subscription_id = dbutils.widgets.get("Subscription ID")
aml_region = dbutils.widgets.get("Region")
aml_workspace_name = "iot"
turbine = "WindTurbine-1"
power_model = "power_prediction"
life_model = "life_prediction"

# Connect to a workspace (replace widgets with your own workspace info)
workspace = Workspace.create(name = aml_workspace_name,
                             subscription_id = aml_subscription_id,
                             resource_group = aml_resource_group,
                             location = aml_region,
                             exist_ok=True)

# Retrieve the remaining_life and power_output experiments on WindTurbine-1, and get the best performing model (min RMSE)
best_life_model = mlflow.search_runs(filter_string=f'params.deviceid="{turbine}" and params.model="{life_model}"')\
  .dropna().sort_values("metrics.train-rmse")['artifact_uri'].iloc[0] + '/model'
best_power_model = mlflow.search_runs(filter_string=f'params.deviceid="{turbine}" and params.model="{power_model}"')\
  .dropna().sort_values("metrics.train-rmse")['artifact_uri'].iloc[0] + '/model'

scoring_uris = {}
for model, path in [('life',best_life_model),('power',best_power_model)]:
  # Build images for each of our two models in Azure Container Instances
  print(f"-----Building image for {model} model-----")
  model_image, azure_model = mlflow.azureml.build_image(model_uri=path, 
                                                        workspace=workspace, 
                                                        model_name=model,
                                                        image_name=model,
                                                        description=f"XGBoost model to predict {model} of a turbine", 
                                                        synchronous=True)
  model_image.wait_for_creation(show_output=True)

  # Deploy web services to host each model as a REST API
  print(f"-----Deploying image for {model} model-----")
  dev_webservice_name = model + random_string(10)
  dev_webservice_deployment_config = AciWebservice.deploy_configuration()
  dev_webservice = Webservice.deploy_from_image(name=dev_webservice_name, image=model_image, deployment_config=dev_webservice_deployment_config, workspace=workspace)
  dev_webservice.wait_for_deployment()

  # Get the URI for sending REST requests to
  scoring_uris[model] = dev_webservice.scoring_uri

print(f"-----Model URIs for Scoring:-----")
scoring_uris

# COMMAND ----------

# MAGIC %md You can view your model, it's deployments and URL endpoints by navigating to https://ml.azure.com/.
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iiot_azureml.gif" width=800>

# COMMAND ----------

# MAGIC %md ## Step 5 - Model Inference: Real-time Scoring
# MAGIC We can now make HTTP REST calls from a web app, PowerBI, or directly from Databricks to the hosted model URI to score data directly

# COMMAND ----------

# Retrieve the Scoring URL provided by AzureML
power_uri = scoring_uris['power'] 
life_uri = scoring_uris['life'] 

# Construct a payload to send with the request
payload = {
  'angle':8,
  'rpm':6,
  'temperature':25,
  'humidity':50,
  'windspeed':5,
  'power':150,
  'age':10
}

def score_data(uri, payload):
  rest_payload = json.dumps({"data": [list(payload.values())]})
  response = requests.post(uri, data=rest_payload, headers={"Content-Type": "application/json"})
  return json.loads(response.text)

print(f'Current Operating Parameters: {payload}')
print(f'Predicted power (in kwh) from model: {score_data(power_uri, payload)}')
print(f'Predicted remaining life (in days) from model: {score_data(life_uri, payload)}')

# COMMAND ----------

# MAGIC %md ### Step 6: Asset Optimization
# MAGIC We can now identify the optimal operating conditions for maximizing power output while also maximizing asset useful life. 
# MAGIC 
# MAGIC \\(Revenue = Price\displaystyle\sum_1^{365} Power_t\\)
# MAGIC 
# MAGIC \\(Cost = {365 \over Life_{rpm}} Price \displaystyle\sum_1^{24} Power_t \\)
# MAGIC 
# MAGIC Price\displaystyle\sum_{t=1}^{24})\\)
# MAGIC 
# MAGIC \\(Profit = Revenue - Cost\\)
# MAGIC 
# MAGIC \\(Power_t\\) and \\(Life\\) will be calculated by scoring many different RPM values in AzureML. The results can be visualized to identify the RPM that yields the highest profit.

# COMMAND ----------

# Construct a payload to send with the request
payload = {
  'angle':8,
  'rpm':6,
  'temperature':25,
  'humidity':50,
  'windspeed':5,
  'power':150,
  'age':10
}

# Iterate through 50 different RPM configurations and capture the predicted power and remaining life at each RPM
results = []
for rpm in range(1,15):
  payload['rpm'] = rpm
  expected_power = score_data(power_uri, payload)[0]
  payload['power'] = expected_power
  expected_life = -score_data(life_uri, payload)[0]
  results.append((rpm, expected_power, expected_life))
  
# Calculalte the Revenue, Cost and Profit generated for each RPM configuration
optimization_df = pd.DataFrame(results, columns=['RPM', 'Expected Power', 'Expected Life'])
optimization_df['Revenue'] = optimization_df['Expected Power'] * 24 * 365
optimization_df['Cost'] = optimization_df['Expected Power'] * 24 * 365 / optimization_df['Expected Life']
optimization_df['Profit'] = optimization_df['Revenue'] + optimization_df['Cost']

display(optimization_df)

# COMMAND ----------

# MAGIC %md The optimal operating parameters for **WindTurbine-1** given the specified weather conditions is **11 RPM** for generating a maximum profit of **$1.4M**! Your results may vary due to the random nature of the sensor readings. 

# COMMAND ----------

