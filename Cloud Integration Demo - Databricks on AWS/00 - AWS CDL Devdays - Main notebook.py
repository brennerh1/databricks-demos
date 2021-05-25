# Databricks notebook source
# MAGIC %md ### These notebooks require integration with your AWS account, services and roles.   If you would like to try them in your environment, please contact us at <awsbricksters@databricks.com> for assistance.
# MAGIC 
# MAGIC 0. We will cover the Access Management requirements which includes the IAM policies and permissions
# MAGIC 0. Review the cluster configuration options for enabling the Glue Catalog
# MAGIC 0. Optionally, setup a Glue crawler as used in this demo 
# MAGIC 0. Cover the end-to-end requirements for establishing a secure Redshift connection
# MAGIC 0. And finally, the Redshift library prerequisite

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##AWS Cloud Data Lake demo Overview
# MAGIC 
# MAGIC Components and steps we will follow in today's demo:
# MAGIC <br />
# MAGIC <br />
# MAGIC 1. AWS Glue as our Central Metastore
# MAGIC 2. We will launch 1 Kinesis Stream ie. **User click stream**
# MAGIC 3. Join an already existing user Profile Delta table registered in our Glue metastore 
# MAGIC 4. We will execute a crawler job to pull in an S3 datasets into our AWS Glue metastore
# MAGIC 5. The pipeline consists of a Data Lake medallion appproach
# MAGIC 6. We will demonstrate the Full DML support of Delta Lake while curating the Data Lake.  
# MAGIC 6. The curated GOLD dataset will be available to Athena and pushed to Redshift for later consumption
# MAGIC 7. Finally, a QuickSight dashboard  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/AWS - CDL Data Lake Architecture.png" height="800" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/AWS CDL Devday - Delta Lake.png" height="600" width="1000"/> 
# MAGIC 
# MAGIC ## more information can be found at [Delta.io](https://delta.io/)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###Step 1: [Prep / Clean Up](https://field-eng.cloud.databricks.com/#notebook/1564553/command/1564554)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 2: [Launch the Clicks Kinesis Stream](https://field-eng.cloud.databricks.com/#notebook/1564618/command/1572786)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 3: [Receive the Bronze and prepare the Silver Clicks Data](https://field-eng.cloud.databricks.com/#notebook/1565307/command/1566755)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 4: [Produce the Gold Data and make it available for Amazon Athena](https://field-eng.cloud.databricks.com/#notebook/1564635)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 5: [Final Stage: Push the Gold Data to Redshift](https://field-eng.cloud.databricks.com/#notebook/1573648/command/1573649)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 6: [QuickSight Dashboard](https://us-west-2.quicksight.aws.amazon.com/sn/dashboards/bd3941d4-2783-4e46-98c9-b813d73edf36)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##We encourage you to explore the documentation to learn more about the various features of the Databricks platform and notebooks.
# MAGIC 
# MAGIC <br />
# MAGIC * <a href="https://docs.databricks.com/user-guide/index.html" target="_blank">User Guide</a>
# MAGIC * <a href="https://docs.databricks.com/user-guide/notebooks/index.html" target="_blank">User Guide / Notebooks</a>
# MAGIC * <a href="https://docs.databricks.com/administration-guide/index.html" target="_blank">Administration Guide</a>
# MAGIC * <a href="https://docs.databricks.com/api/index.html" target="_blank">REST API</a>
# MAGIC * <a href="https://docs.databricks.com/release-notes/index.html" target="_blank">Release Notes</a>
# MAGIC * <a href="https://docs.databricks.com" target="_blank">And much more!</a>
