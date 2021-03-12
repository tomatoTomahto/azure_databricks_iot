-- Databricks notebook source
-- MAGIC %md # Manufacturing IoT Analytics on Azure Databricks
-- MAGIC ## Part 2: SQL on the Manufacturing Delta Lake
-- MAGIC Databricks [SQL Analytics](https://databricks.com/product/sql-analytics) can be used to query your Data Lake Delta tables using an intuitive analyst-focused interface. BI tools like PowerBI can also connect to our SQL Endpoints for fast, scalable and secure ad-hoc BI on our Manufacturing data without moving the data into a data silo. 
-- MAGIC 
-- MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/sql_dashboard_manu.gif" width=800>
-- MAGIC 
-- MAGIC As SQL Analyics in is private preview, the queries have been included in this notebook. In order to recreate the dashboard above:
-- MAGIC 1. Create a SQL Endpoint
-- MAGIC 2. Add a service principal to Data Source settings to access ADLS
-- MAGIC 3. Copy and execute the queries below into Queries in SQL Analytics
-- MAGIC 4. Create parameters for `facilities` and `date` range
-- MAGIC 5. Build counters, time-series charts, maps and other visualizations from the queries
-- MAGIC 6. Assemble a dashboard using the visualizations created
-- MAGIC 
-- MAGIC **Note**: The queries won't run as in in a Notebook because they are parameterized (ie. `{{ facilities }}`) for SQL Analytics use.

-- COMMAND ----------

-- MAGIC %md ### Inventory Anaysis

-- COMMAND ----------

select distinct i.facilityid, 
    last(inventory) over (partition by i.facilityid order by date) as inventory,
    f.state
from manufacturing.parts_inventory i join manufacturing.facilities f
on (i.facilityid = f.facilityid)
where i.facilityid in ({{ facilities }})

-- COMMAND ----------

-- MAGIC %md ### Enriched Analysis

-- COMMAND ----------

select * from manufacturing.sensors_enriched
where facilityid in ({{ facilityids }})
    and date between '{{ date.start }}' and '{{ date.end }}' 

-- COMMAND ----------

-- MAGIC %md ### Raw Time Series Plots

-- COMMAND ----------

select * from manufacturing.sensors_raw
where facilityid in ({{ facilityids }})
    and date between '{{ date.start }}' and '{{ date.end }}' 

-- COMMAND ----------

-- MAGIC %md ### Aggregate Readings

-- COMMAND ----------

select count(*) as Readings, 
    avg(temperature) as Temperature,
    avg(humidity) as Humidity,
    avg(pressure) as Pressure,
    avg(moisture) as Moisture,
    avg(oxygen) as Oxygen,
    avg(radiation) as Radiation,
    avg(conductivity) as Conductivity
from manufacturing.sensors_raw
where facilityid in ({{ facilityids }})
    and date between '{{ date.start }}' and '{{ date.end }}' 

-- COMMAND ----------

-- MAGIC %md ### Facilities List
-- MAGIC This list can be used to create a Query Based filter for the other queries

-- COMMAND ----------

select facilityid from manufacturing.facilities

-- COMMAND ----------

-- MAGIC %md ### Facility Maps
-- MAGIC This query can be used to create geo maps mapping capacity and locations of the facilities

-- COMMAND ----------

select * from manufacturing.facilities where facilityid in ({{ facilityids }})