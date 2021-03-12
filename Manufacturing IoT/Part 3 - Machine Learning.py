# Databricks notebook source
# MAGIC %md # Manufacturing IoT Analytics on Azure Databricks 
# MAGIC ## Part 3 - Machine Learning
# MAGIC This notebook demonstrates the following architecture for IIoT Ingest, Processing and Analytics on Azure. The following architecture is implemented for the demo. 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Manufacturing_architecture.png" width=800>
# MAGIC 
# MAGIC The notebook is broken into sections following these steps:
# MAGIC 3. **Machine Learning** - train XGBoost regression models using distributed ML to predict power output and asset remaining life on historical sensor data
# MAGIC 4. **Model Deployment** - deploy trained models for real-time serving in the Databricks Model Registry
# MAGIC 5. **Model Inference** - score real data instantly against hosted models via REST API

# COMMAND ----------

# MAGIC %md
# MAGIC The Lakehouse architecture pattern allows the *best tool for the job* to be brought *to the data*. We have performed data engineer, SQL analytics, and now data science & ML - **all on the same data living in our organization's Data Lake**. Delta + ADLS provides security, reliability and performance to all data sets - stream, batch, structured, unstructured - and opens the data up to any analytics workload. 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Lakehouse.png" width=800>

# COMMAND ----------

# AzureML Workspace info (name, region, resource group and subscription ID) for model deployment
dbutils.widgets.text("Storage Account", "<your storage account>", "Storage Account")

# COMMAND ----------

# MAGIC %md ## Step 1 - Environment Setup
# MAGIC 
# MAGIC The pre-requisites are listed below:
# MAGIC 
# MAGIC ### Azure Services Required
# MAGIC * ADLS Gen 2 Storage account with a container called `iot`
# MAGIC 
# MAGIC ### Azure Databricks Configuration Required
# MAGIC * 3-node (min) Databricks Cluster running **DBR 7.0ML+**
# MAGIC * The following Secrets defined in scope `iot`
# MAGIC  * `adls_key` - Access Key to ADLS storage account **(Important - use the [Access Key](https://raw.githubusercontent.com/tomatoTomahto/azure_databricks_iot/master/bricks.com/blog/2020/03/27/data-exfiltration-protection-with-azure-databricks.html))**
# MAGIC 
# MAGIC **Part 1 Notebook Run to generate and process the data** (notebook can be found [here](https://github.com/tomatoTomahto/azure_databricks_iot/blob/master/Manufacturing%20IoT/Part%201%20-%20Data%20Engineering.py)). Ensure all tables have been created on ADLS.

# COMMAND ----------

# Setup access to storage account for temp data when pushing to Synapse
storage_account = dbutils.widgets.get("Storage Account")
spark.conf.set(f"fs.azure.account.key.{storage_account}.dfs.core.windows.net", dbutils.secrets.get("iot","adls_key"))

# Setup storage locations for all data
ROOT_PATH = f"abfss://iot@{storage_account}.dfs.core.windows.net/manufacturing_demo/"

# Pyspark and ML Imports
import os, json, requests
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np 
import pandas as pd
import xgboost as xgb
import mlflow.xgboost

# COMMAND ----------

# MAGIC %md ## Step 3 - Machine Learning
# MAGIC Now that our data is flowing reliably from our sensor devices into an enriched Delta table in Data Lake storage, we can start to build ML models to predict when a maintenance event will occur. We do this by predicting future sensor values and identifying when those values go outside the safe operating bounds of the facility. 
# MAGIC 
# MAGIC We create a predictive temperature model ***for each manufacturing facility***
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/manufacturing_pred_maint.png" width=800>
# MAGIC 
# MAGIC We will use the XGBoost framework to train regression models. Due to the size of the data and number of Facilities, we will use Spark UDFs to distribute training across all the nodes in our cluster.

# COMMAND ----------

# MAGIC %md ### 3a. Feature Engineering
# MAGIC In order to predict temperature 5 minutes ahead, we need to first time-shift our data to create our label column. We can do this easily using Spark Window partitioning. 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW manufacturing.ml_feature_view AS
# MAGIC SELECT facilityid, temperature, humidity, pressure, moisture, oxygen, radiation, conductivity,
# MAGIC   LEAD(temperature, 1, temperature) OVER (PARTITION BY facilityid ORDER BY window) as next_temperature
# MAGIC FROM manufacturing.sensors_enriched;
# MAGIC 
# MAGIC SELECT * FROM manufacturing.ml_feature_view

# COMMAND ----------

# MAGIC %md ### 3b. Distributed Model Training - Predict Power Output
# MAGIC [Pandas UDFs](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/udf-python-pandas?toc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fazure-databricks%2Ftoc.json&bc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fbread%2Ftoc.json) allow us to vectorize Pandas code across multiple nodes in a cluster. Here we create a UDF to train an XGBoost Regressor model against all the historic data for a particular Manufacturing facility. We use a Grouped Map UDF as we perform this model training on the facility group level.

# COMMAND ----------

# Create a function to train a XGBoost Regressor on a facility's data
def train_distributed_xgb(readings_pd, label_col, prediction_col):
  mlflow.xgboost.autolog()
  with mlflow.start_run():
    # Log the model type and faciity ID
    mlflow.log_param('facilityid', readings_pd['facilityid'][0])

    # Train an XGBRegressor on the data for this facility
    alg = xgb.XGBRegressor() 
    train_dmatrix = xgb.DMatrix(data=readings_pd[feature_cols].astype('float'),label=readings_pd[label_col])
    params = {'learning_rate': 0.5, 'alpha':2, 'colsample_bytree': 0.5, 'max_depth': 5}
    model = xgb.train(params=params, dtrain=train_dmatrix, evals=[(train_dmatrix, 'train')])

    # Make predictions on the dataset and return the results
    readings_pd[prediction_col] = model.predict(train_dmatrix)
  return readings_pd

# Read in our feature table and select the columns of interest
feature_df = spark.table('manufacturing.ml_feature_view').selectExpr('facilityid','temperature', 'humidity', 'pressure', 'moisture', 'oxygen', 'radiation', 'conductivity','next_temperature','0 as next_temperature_predicted')

# Register a Pandas UDF to distribute XGB model training using Spark
@pandas_udf(feature_df.schema, PandasUDFType.GROUPED_MAP)
def train_temperature_model(readings_pd):
  return train_distributed_xgb(readings_pd, 'next_temperature', 'next_temperature_predicted')

# COMMAND ----------

# Run the Pandas UDF against our feature dataset - this will train 1 model for each facility
temperature_predictions = (
  feature_df.groupBy('facilityid')
    .apply(train_temperature_model)
    .write.format("delta").mode("overwrite")
    .option("path",ROOT_PATH + "gold/temperature_predictions")
    .saveAsTable("manufacturing.temperature_predictions")
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Plot actuals vs. predicted
# MAGIC SELECT * FROM manufacturing.temperature_predictions

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

# MAGIC %md The models to predict remaining temperature have been trained and logged by MLflow. We can now move on to model deployment in Databricks.

# COMMAND ----------

# MAGIC %md ## Step 4 - Model Deployment to MLflow
# MAGIC Now that our models have been trained, we can deploy them in an automated way directly to a model serving environment like Azure ML or MLflow. Below, we register the best performing models to the Databricks-native hosted MLflow model registry for deployment tracking and serving. Once registered, we can enable Serving to expose a REST API to the model. You can register models using the UI or via the MLflow API as shown in the next cell. 

# COMMAND ----------

# Choose a facility's model to deploy
facility = "FAC-0"

# Retrieve the best performing model (min RMSE)
best_model = mlflow.search_runs(filter_string=f'params.facilityid="{facility}"')\
  .dropna().sort_values("metrics.train-rmse")['artifact_uri'].iloc[0] + '/model'

# Register our best performing model in the Databricks model registry
mlflow.register_model(best_model, "temperature_prediction")

# COMMAND ----------

# MAGIC %md ## Step 5 - Model Inference: Real-time Scoring
# MAGIC Once our model is registered, we can track it in the Model Registry and enable model serving as shown below:
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/mlflow_register_serve.gif" width=800>
# MAGIC 
# MAGIC We can now make HTTP REST calls from a web app, PowerBI, or directly from Databricks to the hosted model URI to score data directly

# COMMAND ----------

import os
import requests
import pandas as pd

# Build a payload to score (ie. get the temperature 5 minute from now given the current sensor conditions)
payload = [{
  'temperature':25.4,
  'humidity':67.2, 
  'pressure':33.6, 
  'moisture':49.8, 
  'oxygen':27.7, 
  'radiation':116.3, 
  'conductivity':128.0
}]

# Call our API to score data
temp_prediction_uri = "https://adb-5016390217096892.12.azuredatabricks.net/model/temperature_prediction/1/invocations" # Replace with your model URI from MLflow or AzureML
databricks_token = "????" # Replace with your Databricks Personal Access Token

# Function to call the API using REST and return a result
def score_model(uri, payload):
  headers = {'Authorization': f'Bearer {databricks_token}'}
  data_json = payload
  response = requests.request(method='POST', headers=headers, url=uri, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

prediction = score_model(temp_prediction_uri, payload)

print(f'Temperature predicted from model: {int(prediction[0])}°C')

# COMMAND ----------

# MAGIC %md ### Step 6: Predictive Maintenance
# MAGIC We can now identify the operating conditions that lead to failures when the temperature goes above or below the safe operating threshholds. By running a sensitivity analysis by repeatedly calling our served temperature model, we can plot a heatmap of the pressure-temperature combinations that will lead to failures and maintenance event. 

# COMMAND ----------

min_temp = 25.0
max_temp = 27.0

# Construct a payload to send with the request
payload = [{
  'temperature':25.4,
  'humidity':67.2, 
  'pressure':33.6, 
  'moisture':49.8, 
  'oxygen':27.7, 
  'radiation':116.3, 
  'conductivity':128.0
}]

from numpy import arange

# Iterate through 50 different RPM configurations and capture the predicted power and remaining life at each RPM
results = []
for temperature in arange(24.0,26.0,0.2):
  for pressure in arange(34.0,36.0,0.2):
    payload[0]['temperature'] = temperature
    payload[0]['pressure'] = pressure
    expected_temperature = score_model(temp_prediction_uri, payload)[0]
    failure = 0 if expected_temperature < max_temp else 1
    results.append((temperature, pressure, expected_temperature, failure))
  
# Calculalte the Revenue, Cost and Profit generated for each RPM configuration
matrix_df = pd.DataFrame(results, columns=['Temperature', 'Pressure', 'Expected Temperature', 'Failure'])

display(matrix_df)

# COMMAND ----------

# MAGIC %md We can see that when temperature is between 25.2 - 25.6 and pressure is between 34 and 35.4, we have failures 5 minutes later due to the temperature falling outside the safe operating threshholds. 

# COMMAND ----------

# MAGIC %md ### Optional - Deployment to Azure ML
# MAGIC Many customers will want to use an central external ML hosting tool like AzureML to serve all models - whether developed in Databricks or outside (on laptops, in AzureML, AutoML frameworks, etc.). Models trained in Databricks can be easily deployed to AzureML using the [mlflow-azureml] library installed on the cluster. See the sample notebook [here](https://databricks.com/notebooks/iiot/iiot-end-to-end-part-2.html) for the code snippet to deploy the IoT model into AzureML for serving. 

# COMMAND ----------

