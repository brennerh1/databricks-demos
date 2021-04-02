# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Life Expectancy from Health Data
# MAGIC 
# MAGIC ![WHO](https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/gartner_2020/WHO.png)
# MAGIC 
# MAGIC **Note**: You should be able to run the notebook starting from section 2, or section 3, or the "what if?" cells at the end without running prior cells.
# MAGIC 
# MAGIC ## 1. Data Preparation
# MAGIC 
# MAGIC The first major task is to access and evaluate the data. This work might be performed by a data engineering team, who is more familiar with Scala than, for example, Python. Within Databricks, both teams can choose the best language for their work and still collaborate easily even within a notebook.
# MAGIC 
# MAGIC Data comes from several sources.
# MAGIC 
# MAGIC - The WHO Health Indicators (primary data) for:
# MAGIC   - the USA: https://data.humdata.org/dataset/who-data-for-united-states-of-america
# MAGIC   - similarly for other developed nations: Australia, Denmark, Finland, France, Germany, Iceland, Italy, New Zealand, Norway, Portugal, Spain, Sweden, the UK
# MAGIC - The World Bank Health Indicators (supplementary data) for:
# MAGIC   - the USA: https://data.humdata.org/dataset/world-bank-combined-indicators-for-united-states
# MAGIC   - similarly for other developed nations
# MAGIC - Our World In Data (Drug Use)
# MAGIC   - https://ourworldindata.org/drug-use
# MAGIC   
# MAGIC ### Health Indicators primary data
# MAGIC 
# MAGIC The "health indicators" datasets are the primary data sets. They are CSV files, and are easily read by Spark. However, they don't have a consistent schema. Some contains extra "DATASOURCE" columns, which can be ignored.

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.DataFrame
# MAGIC 
# MAGIC val rawDataPath = "/mnt/databricks-datasets-private/ML/gartner_2020/"
# MAGIC 
# MAGIC def transformCSV(path: String): DataFrame = {
# MAGIC   // Second row contains odd "comment" lines that intefere with schema validation. Filter, then parse as CSV
# MAGIC   val withoutComment = spark.read.text(path).filter(!$"value".startsWith("#")).as[String]
# MAGIC   spark.read.option("inferSchema", true).option("header", true).csv(withoutComment)
# MAGIC }
# MAGIC 
# MAGIC // Some "Health Indicators" files have three extra "DATASOURCE" columns; ignore them
# MAGIC var rawHealthIndicators =
# MAGIC   transformCSV(rawDataPath + "health_indicators/format2").union(
# MAGIC   transformCSV(rawDataPath + "health_indicators/format1").drop("DATASOURCE (CODE)", "DATASOURCE (DISPLAY)", "DATASOURCE (URL)"))
# MAGIC 
# MAGIC display(rawHealthIndicators)

# COMMAND ----------

# MAGIC %md
# MAGIC Save the descriptions for codes for later use. The data contains 109 distinct indicators for the 14 countries:

# COMMAND ----------

# MAGIC %fs rm --recurse=true /tmp/KnowledgeRepo/ML/gartner_2020/descriptions

# COMMAND ----------

# MAGIC %scala
# MAGIC rawHealthIndicators.select("GHO (CODE)", "GHO (DISPLAY)").distinct().toDF("Code", "Description").
# MAGIC   write.format("delta").save("/tmp/KnowledgeRepo/ML/gartner_2020/descriptions")
# MAGIC display(spark.read.format("delta").load("/tmp/KnowledgeRepo/ML/gartner_2020/descriptions").orderBy("Code"))

# COMMAND ----------

# MAGIC %md
# MAGIC The data needs some basic normalization and filtering:
# MAGIC - Remove any variables that are effectively variations on life expectancy, as this is the variable to be explained
# MAGIC - Use published data only
# MAGIC - For now, use data for both sexes, not male/female individually
# MAGIC - Correctly parse the Value / Display Value, which are inconsistently available
# MAGIC - Flatten year ranges like "2012-2017" to individual years
# MAGIC - Keep only data from 2000 onwards
# MAGIC 
# MAGIC Finally, the data needs to be 'pivoted' to contain indicator values as columns.

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions.{explode, when, udf}
# MAGIC import org.apache.spark.sql.types.FloatType
# MAGIC 
# MAGIC // Can't use life expectancy at 60
# MAGIC rawHealthIndicators = rawHealthIndicators.filter($"GHO (CODE)" =!= "WHOSIS_000015")
# MAGIC 
# MAGIC // Keep just PUBLISHED data, not VOID
# MAGIC rawHealthIndicators = rawHealthIndicators.filter($"PUBLISHSTATE (CODE)" === "PUBLISHED").drop("PUBLISHSTATE (CODE)")
# MAGIC 
# MAGIC // Use stats for both sexes now, not male/female separately. It's either NULL or BTSX
# MAGIC rawHealthIndicators = rawHealthIndicators.filter(($"SEX (CODE)".isNull) || ($"SEX (CODE)" === "BTSX")).drop("SEX (CODE)")
# MAGIC 
# MAGIC // Use Numeric where available, otherwise Display Value, as value. Low/High/StdErr/StdDev are unevenly available, so drop
# MAGIC rawHealthIndicators = rawHealthIndicators.
# MAGIC   withColumn("Value", when($"Numeric".isNull, $"Display Value").otherwise($"Numeric"))
# MAGIC 
# MAGIC // Some "year" values are like 2012-2017. Explode to a value for each year in the range
# MAGIC val yearsToRangeUDF = udf { (s: String) =>
# MAGIC     if (s.contains("-")) {
# MAGIC       val Array(start, end) = s.split("-")
# MAGIC       (start.toInt to end.toInt).toArray
# MAGIC     } else {
# MAGIC       Array(s.toInt)
# MAGIC     }
# MAGIC   }
# MAGIC rawHealthIndicators = rawHealthIndicators.withColumn("Year", explode(yearsToRangeUDF($"YEAR (CODE)")))
# MAGIC 
# MAGIC // Rename columns, while dropping everything but Year, Country, GHO CODE, and Value
# MAGIC rawHealthIndicators = rawHealthIndicators.select(
# MAGIC   $"GHO (CODE)".alias("GHO"), $"Year", $"COUNTRY (CODE)".alias("Country"), $"Value".cast(FloatType))
# MAGIC 
# MAGIC // Keep only 2000-2018 at most
# MAGIC rawHealthIndicators = rawHealthIndicators.filter("Year >= 2000 AND Year <= 2018")
# MAGIC 
# MAGIC // avg() because some values will exist twice because of WORLDBANKINCOMEGROUP; value is virtually always the same
# MAGIC val healthIndicatorsDF = rawHealthIndicators.groupBy("Country", "Year").pivot("GHO").avg("Value")
# MAGIC healthIndicatorsDF.createOrReplaceTempView("healthIndicators")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM healthIndicators ORDER BY Country, Year

# COMMAND ----------

# MAGIC %md
# MAGIC This data set contains life expectancy (`WHOSIS_000001`), so we can already compare life expectancy from 2000-2016 across countries. The USA is an outlier, it seems.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Year, Country, WHOSIS_000001 AS LifeExpectancy FROM healthIndicators WHERE Year <= 2016

# COMMAND ----------

# MAGIC %md
# MAGIC ### Indicators supplementary data
# MAGIC The data from the World Bank can likewise be normalized, filtered and analyzed.

# COMMAND ----------

# MAGIC %scala
# MAGIC var rawIndicators = transformCSV(rawDataPath + "indicators")
# MAGIC display(rawIndicators)

# COMMAND ----------

# MAGIC %md
# MAGIC This data set has 2,283 (!) features for these countries, by year. As above, many are highly correlated, and even redundant for comparative purposes. For example, figures in local currency are less useful for comparison than the ones expressed in US$. Likewise to limit the scale of the feature set, male/female figures reported separately are removed.

# COMMAND ----------

# MAGIC %scala
# MAGIC rawIndicators.select("Indicator Code", "Indicator Name").distinct().toDF("Code", "Description").
# MAGIC   write.format("delta").mode("append").save("/tmp/KnowledgeRepo/ML/gartner_2020/descriptions")
# MAGIC display(spark.read.format("delta").load("/tmp/KnowledgeRepo/ML/gartner_2020/descriptions").orderBy("Code"))

# COMMAND ----------

# MAGIC %scala
# MAGIC // Keep only 2000-2018 at most
# MAGIC rawIndicators = rawIndicators.filter("Year >= 2000 AND Year <= 2018")
# MAGIC 
# MAGIC // Can't use life expectancy from World Bank, or mortality rates or survival rates -- too closely related to life expectancy
# MAGIC rawIndicators = rawIndicators.
# MAGIC   filter(!$"Indicator Code".startsWith("SP.DYN.LE")).
# MAGIC   filter(!$"Indicator Code".startsWith("SP.DYN.AMRT")).filter(!$"Indicator Code".startsWith("SP.DYN.TO"))
# MAGIC 
# MAGIC // Don't use gender columns separately for now
# MAGIC rawIndicators = rawIndicators.
# MAGIC   filter(!$"Indicator Code".endsWith(".FE") && !$"Indicator Code".endsWith(".MA")).
# MAGIC   filter(!$"Indicator Code".contains(".FE.") && !$"Indicator Code".contains(".MA."))
# MAGIC 
# MAGIC // Don't use local currency variants
# MAGIC rawIndicators = rawIndicators.
# MAGIC   filter(!$"Indicator Code".endsWith(".CN") && !$"Indicator Code".endsWith(".KN")).
# MAGIC   filter(!$"Indicator Code".startsWith("PA.") && !$"Indicator Code".startsWith("PX."))
# MAGIC 
# MAGIC rawIndicators = rawIndicators.select($"Country ISO3".alias("Country"), $"Year", $"Indicator Code".alias("Indicator"), $"Value")
# MAGIC 
# MAGIC val indicatorsDF = rawIndicators.groupBy("Country", "Year").pivot("Indicator").avg("Value")
# MAGIC indicatorsDF.createOrReplaceTempView("indicators")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM indicators

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overdoses
# MAGIC 
# MAGIC One issue that comes to mind when thinking about life expectancy, given the unusual downward trend in life expectancy in the USA, is drug-related deaths. These have been a newsworthy issue for the USA for several years. Our World In Data provides drug overdose data by country, year, and type:

# COMMAND ----------

# MAGIC %scala
# MAGIC var rawOverdoses = spark.read.option("inferSchema", true).option("header", true).csv(rawDataPath + "overdoses_world")
# MAGIC display(rawOverdoses)

# COMMAND ----------

# MAGIC %scala
# MAGIC // Rename some columns for compatibility
# MAGIC rawOverdoses = rawOverdoses.drop("Entity").
# MAGIC   toDF("Country", "Year", "CocaineDeaths", "IllicitDrugDeaths", "OpioidsDeaths", "AlcoholDeaths", "OtherIllicitDeaths", "AmphetamineDeaths")
# MAGIC rawOverdoses = rawOverdoses.filter("Year >= 2000 AND Year <= 2018")
# MAGIC rawOverdoses.createOrReplaceTempView("rawOverdoses")

# COMMAND ----------

# MAGIC %md
# MAGIC These three data sets, having been filtered and normalized, can now be joined by country and year, to produce the raw input for further analysis. Join and write to a Delta table as a 'silver' table of cleaned data.

# COMMAND ----------

# MAGIC %fs rm --recurse=true /tmp/KnowledgeRepo/ML/gartner_2020/input

# COMMAND ----------

# MAGIC %scala
# MAGIC spark.sql("SELECT * FROM healthIndicators LEFT OUTER JOIN indicators USING (Country, Year) LEFT OUTER JOIN rawOverdoses USING (Country, Year)").
# MAGIC   write.format("delta").save("/tmp/KnowledgeRepo/ML/gartner_2020/input")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS gartner;
# MAGIC USE gartner;
# MAGIC CREATE TABLE IF NOT EXISTS gartner_2020 USING DELTA LOCATION '/tmp/KnowledgeRepo/ML/gartner_2020/input';
# MAGIC CREATE TABLE IF NOT EXISTS descriptions USING DELTA LOCATION '/tmp/KnowledgeRepo/ML/gartner_2020/descriptions';

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC The data is now available to data scientists as [`gartner.gartner_2020`](https://demo.cloud.databricks.com/#table/gartner/gartner_2020)
# MAGIC 
# MAGIC ## 2. Analysis and Modeling
# MAGIC 
# MAGIC At this point, the cleaned and joined data might be handed over to a data scientist for analysis. It can be re-read from the Delta table. At this point, it may be data scientists taking over, and they can continue in Python using the same data set.

# COMMAND ----------

input_df = spark.read.table("gartner.gartner_2020")
display(input_df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering
# MAGIC Many columns' values are mostly missing, or entirely empty. However, because the data is organized by time, it's reasonable to forward/back fill data by country and year to impute missing values. This isn't is as good as having actual values, but as the dimensions here are generally slow-changing over years, it is likely to help the analysis.

# COMMAND ----------

import pandas as pd

input_pd = input_df.orderBy("Year").toPandas()
input_pd = pd.concat([input_pd['Country'], input_pd.groupby('Country').ffill()], axis=1)
input_pd = pd.concat([input_pd['Country'], input_pd.groupby('Country').bfill()], axis=1)
input_df = spark.createDataFrame(input_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC First, take a look at some suggested features and how they correlate, especially with the target (life expectancy, `WHOSIS_000001`, the left/top column/row). In Databricks it's easy to use standard libraries like `seaborn` for this.

# COMMAND ----------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

selected_indicators = [
  "WHOSIS_000001",       # Target - life expectancy
  "WHS9_85",             # Literacy rate among adults aged >= 15 years (%)
  "WHS9_96",             # Population living in urban areas (%)
  "WHS7_156",            # Per capita total expenditure on health at average exchange rate (US$)
  "NY.GDP.PCAP.CD",      # GDP per capita (current US$)
  "BAR.NOED.15UP.ZS",    # Barro-Lee: Percentage of population age 15+ with no education
  "OpioidsDeaths"
]

def hide_current_axis(*args, **kwds):
  plt.gca().set_visible(False)

pairplot_pd = input_df.select(['Country'] + list(map(lambda c: f"`{c}`", selected_indicators))).toPandas()
g = sns.pairplot(pairplot_pd, hue='Country', vars=selected_indicators, dropna=True, palette='Paired')
g.map_upper(hide_current_axis)

# COMMAND ----------

# MAGIC %md
# MAGIC In the scatterplot, points are years and countries. Data from different countries are colored differently. Note that the USA (dark blue) stands out easily on many dimensions, particularly in the last row: opioid-related deaths.
# MAGIC 
# MAGIC Many key features aren't particularly correlated. A few are, like "GDP per capita (current US$)" vs "Per capita total expenditure on health at average exchange rate (US$)"; naturally nations with more economic production per capita spend more on health care. Again, the US stands out for spending relatively _more_ per capita than would be expected from GDP.
# MAGIC 
# MAGIC To make further sense of this, it's necessary to prepare the data for a machine learning model that can attempt to relate these many input features to the desired outcome, life expectancy.

# COMMAND ----------

# MAGIC %fs rm --recurse=true /tmp/KnowledgeRepo/ML/gartner_2020/featurized

# COMMAND ----------

from pyspark.sql.functions import col

# Simple one-hot encoding
countries = sorted(map(lambda r: r['Country'], input_df.select("Country").distinct().collect()))

with_countries_df = input_df
for country in countries:
  with_countries_df = with_countries_df.withColumn(f"Country_{country}", col("Country") == country)
  
with_countries_df = with_countries_df.drop("Country")
with_countries_df.write.format("delta").save("/tmp/KnowledgeRepo/ML/gartner_2020/featurized")

# COMMAND ----------

# MAGIC %sql
# MAGIC USE gartner;
# MAGIC CREATE TABLE IF NOT EXISTS gartner_2020_featurized USING DELTA LOCATION '/tmp/KnowledgeRepo/ML/gartner_2020/featurized';

# COMMAND ----------

# MAGIC %md
# MAGIC We're going to go ahead and enable MLflow autologging for Spark data sources now, before they're used; this will be explained later.

# COMMAND ----------

import mlflow.spark
mlflow.spark.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now, a 'gold' table of featurized data is available for data scientists and modelers to use for the modeling.
# MAGIC 
# MAGIC ### Modeling
# MAGIC 
# MAGIC From here, data scientists may work in other frameworks like `pandas` to handle this small data set. In fact, those familiar with `pandas` can even manipulate and query data using Spark using the same API, via `koalas`, and then return to `pandas` DataFrames using the same API.

# COMMAND ----------

import databricks.koalas as ks

input_ks = spark.read.table("gartner.gartner_2020_featurized").to_koalas()
input_ks = input_ks[input_ks['Year'] <= 2016]

# Train/test split on <= 2014 vs 2015-2016
input_ks_train = input_ks[input_ks['Year'] <= 2014]
input_ks_test = input_ks[input_ks['Year'] > 2014]

X_ks_train = input_ks_train.drop('WHOSIS_000001', axis=1)
y_ks_train = input_ks_train['WHOSIS_000001']
X_ks_test = input_ks_test.drop('WHOSIS_000001', axis=1)
y_ks_test = input_ks_test['WHOSIS_000001']

X = input_ks.drop('WHOSIS_000001', axis=1).to_pandas()
y = input_ks['WHOSIS_000001'].to_pandas()
X_train = X_ks_train.to_pandas()
X_test =  X_ks_test.to_pandas()
y_train = y_ks_train.to_pandas()
y_test =  y_ks_test.to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC The data set is actually quite small -- 255 rows by about 1000 columns -- and consumes barely 2MB of memory. It's trivial to fit a model to this data with standard packages like `scikit-learn` or `xgboost`. However each of these models requires tuning, and needs building of 100 or more models to find the best combination.
# MAGIC 
# MAGIC In Databricks, the tool `hyperopt` can be use to build these models on a Spark cluster in parallel. The results are logged automatically to `mlflow`.

# COMMAND ----------

from math import exp
import xgboost as xgb
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
import mlflow
import numpy as np

def params_to_xgb(params):
  return {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'max_depth':        int(params['max_depth']),
    'learning_rate':    exp(params['log_learning_rate']), # exp() here because hyperparams are in log space
    'reg_alpha':        exp(params['log_reg_alpha']),
    'reg_lambda':       exp(params['log_reg_lambda']),
    'gamma':            exp(params['log_gamma']),
    'min_child_weight': exp(params['log_min_child_weight']),
    #'importance_type':  'total_gain',
    'seed':             0
  }

def train_model(params):
  train = xgb.DMatrix(data=X_train, label=y_train)
  test = xgb.DMatrix(data=X_test, label=y_test)
  booster = xgb.train(params=params_to_xgb(params), dtrain=train, num_boost_round=1000,\
                      evals=[(test, "test")], early_stopping_rounds=50)
  mlflow.log_param('best_iteration', booster.attr('best_iteration'))
  return {'status': STATUS_OK, 'loss': booster.best_score, 'booster': booster.attributes()}

search_space = {
  'max_depth':            hp.quniform('max_depth', 20, 60, 1),
  # use uniform over loguniform here simply to make metrics show up better in mlflow comparison, in logspace
  'log_learning_rate':    hp.uniform('log_learning_rate', -3, 0),
  'log_reg_alpha':        hp.uniform('log_reg_alpha', -5, -1),
  'log_reg_lambda':       hp.uniform('log_reg_lambda', 1, 8),
  'log_gamma':            hp.uniform('log_gamma', -6, -1),
  'log_min_child_weight': hp.uniform('log_min_child_weight', -1, 4)
}

spark_trials = SparkTrials(parallelism=12)
best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=96, trials=spark_trials, rstate=np.random.RandomState(123))

# COMMAND ----------

# MAGIC %md
# MAGIC The resulting runs and their hyperparameters can be visualized in the mlflow tracking server, via Databricks:
# MAGIC 
# MAGIC <img width="800" src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/gartner_2020/hyperopt.png"/>
# MAGIC 
# MAGIC Root mean squared error was about 0.3 years, compared to life expectancies ranging from about 77 to 83 years. With a best set of hyperparameters chosen, the final model is re-fit and logged with `mlflow`, along with an analysis of feature importance from `shap`:

# COMMAND ----------

code_lookup_df = spark.read.table("gartner.descriptions")
code_lookup = dict([(r['Code'], r['Description']) for r in code_lookup_df.collect()])
display_cols = [code_lookup[c] if c in code_lookup else c for c in X.columns]

# COMMAND ----------

import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

plt.close()

with mlflow.start_run() as run:
  best_iteration = int(spark_trials.best_trial['result']['booster']['best_iteration'])
  booster = xgb.train(params=params_to_xgb(best_params), dtrain=xgb.DMatrix(data=X, label=y), num_boost_round=best_iteration)
  mlflow.log_params(best_params)
  mlflow.log_param('best_iteration', best_iteration)
  mlflow.xgboost.log_model(booster, "xgboost", input_example=X.head(), signature=infer_signature(X, y))

  shap_values = shap.TreeExplainer(booster).shap_values(X, y=y)
  shap.summary_plot(shap_values, X, feature_names=display_cols, plot_size=(14,6), max_display=10, show=False)
  plt.savefig("summary_plot.png", bbox_inches="tight")
  plt.close()
  mlflow.log_artifact("summary_plot.png")
  
  best_run = run.info

# COMMAND ----------

# MAGIC %md
# MAGIC This model can then be registered as the current candidate model for further evaluation in the Model Registry:
# MAGIC 
# MAGIC <img width="800" src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/gartner_2020/registry.png"/>

# COMMAND ----------

import time

model_name = "gartner_2020"
client = mlflow.tracking.MlflowClient()
try:
  client.create_registered_model(model_name)
except Exception as e:
  pass

model_version = client.create_model_version(model_name, f"{best_run.artifact_uri}/xgboost", best_run.run_id)

time.sleep(5) # Just to make sure it's had a second to register
client.transition_model_version_stage(model_name, model_version.version, stage="Staging")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Interpretation
# MAGIC 
# MAGIC From there the question is, what features seem to predict life expectancy? Given the relatively limited span of data, and the limitations of what models can tell us about causality, interpretation requires some care. We applied the package `shap` using Databricks to attempt to explain what the model learned in a principled way, and can now view the logged plot:

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
latest_model_detail = client.get_latest_versions("gartner_2020", stages=['Staging'])[0]
client.download_artifacts(latest_model_detail.run_id, "summary_plot.png", "/dbfs/FileStore/tmp/KnowledgeRepo/ML/")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/tmp/KnowledgeRepo/ML/summary_plot.png)

# COMMAND ----------

# MAGIC %md
# MAGIC The two most important features that stand out by far are:
# MAGIC - Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)
# MAGIC - Year
# MAGIC 
# MAGIC The horizontal axis units are years of life expectancy. It is not the rate of change of life expectancy with respect to the feature value, but the average effect (positive or negative) that the feature's particular value explains per country and year over the data set. Each country and year is a dot, and red dots indicate high values of the feature.
# MAGIC 
# MAGIC Year is self-explanatory; clearly there is generally an upward trend in life expectancy per time, with an average absolute of effect of 0.3 years. But mortality from cardiac diseases, cancer, and diabetes explains even more. Higher %s obviously explain lower life expectancy, as seen at the left.
# MAGIC 
# MAGIC None of these necessarily cause life expectancy directly, but as a first pass, these are suggestive of factors that at least correlate over the last 20 years.
# MAGIC 
# MAGIC SHAP can produce an interaction plot to further study the effect of the most-significant feature. Its built-in `matplotlib`-based plots render directly.

# COMMAND ----------

import numpy as np
import mlflow
import mlflow.xgboost
import shap

model = mlflow.xgboost.load_model(latest_model_detail.source)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X, y=y)

feature = "SH.DYN.NCOM.ZS"
feature_loc = X.columns.get_loc(feature)
interactions = explainer.shap_interaction_values(X).mean(axis=0)
interactions[feature_loc] = 0 # don't consider feature itself as interaction
max_interaction = np.argmax(np.abs(interactions[feature_loc]))

def abbrev(c):
  return c if len(c) < 32 else c[0:32]+"..."
abbrev_display_cols = [abbrev(c) for c in display_cols]
display(shap.dependence_plot(abbrev(code_lookup["SH.DYN.NCOM.ZS"]), shap_values, X, x_jitter=0.5, alpha=0.5, interaction_index=max_interaction, feature_names=abbrev_display_cols))

# COMMAND ----------

# MAGIC %md
# MAGIC Each point is a country and year. This plots the mortality rate mentioned above (`SH.DYN.NCOM.ZS`) versus SHAP value -- the effect on predicted life expectancy that this particular mortality rate has in that time and place. Of course, higher mortality rates are associated with lower predicted life expectancy.
# MAGIC 
# MAGIC Colors correspond to years, which was selected as a feature that most strongly interacts with mortality rate. It's also not surprising that in later years (red), mortality rate is lower and thus life expectancy higher. There is a mild secondary trend here, seen if comparing the curve of blue points (longer go) to red point (more recent). Predicted life expectancy, it might be said, varies less with this mortality rate recently than in the past.

# COMMAND ----------

# MAGIC %md
# MAGIC The United States stood out as an outlier in the life expectancy plot above. We might instead ask, how is the USA different relative to other countries. SHAP can help explain how features explain predicted life expectancy differently.

# COMMAND ----------

us_delta = shap_values[X['Country_USA']].mean(axis=0) - shap_values[~X['Country_USA']].mean(axis=0)
importances = list(zip([float(f) for f in us_delta], display_cols))
top_importances = sorted(importances, key=lambda p: abs(p[0]), reverse=True)[:10]
display(spark.createDataFrame(top_importances, ["Mean SHAP delta", "Feature"]))

# COMMAND ----------

# MAGIC %md
# MAGIC Mortality rate due to cardiac disease, diabetes and cancer stands out in the USA. On average, it explains almost a year less life expectancy than in other countris.
# MAGIC 
# MAGIC This model can now be moved to Production, for consumption and deployment for inference:

# COMMAND ----------

client.transition_model_version_stage(model_name, latest_model_detail.version, stage="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC PS we're going to save off a JSON representation of an input for usage a little later:

# COMMAND ----------

print(X.head(1).to_json(orient='records'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Moving to Production

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from pyspark.sql.functions import col
from pyspark.sql.types import StringType

# Get latest model
model_udf = mlflow.pyfunc.spark_udf(spark, "models:/gartner_2020/production")

future_input_df = spark.read.table("gartner.gartner_2020_featurized").drop("WHOSIS_000001").filter("Year > 2016")

quoted_cols = list(map(lambda c: f"`{c}`", future_input_df.columns))
with_prediction_df = future_input_df.withColumn("WHOSIS_000001", model_udf(*quoted_cols))

# COMMAND ----------

# Unencode country for display
country_cols = [c for c in with_prediction_df.columns if c.startswith("Country_")]
def unencode_country(*is_country):
  for i in range(len(country_cols)):
    if is_country[i]:
      return country_cols[i][-3:]
    
unencode_country_udf = udf(unencode_country, StringType())

country_unencoded_df = with_prediction_df.withColumn("Country", unencode_country_udf(*country_cols)).drop(*country_cols)

display(country_unencoded_df.select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy")).orderBy("Year", "Country"))

# COMMAND ----------

input_df = spark.read.table("gartner.gartner_2020")
display(input_df.filter("Year <= 2016").select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy")).union(
  country_unencoded_df.select(col("Year"), col("Country"), col("WHOSIS_000001").alias("LifeExpectancy"))))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The projected life expectancy can be plotted with previous known values. If the trends seem a little flat, it's because most of the real data is not yet available in the years 2017-2018 at this time. Imputed data fills the gaps, and causes predictions to be relatively similar to 2016.
# MAGIC 
# MAGIC ## Real-Time model serving
# MAGIC 
# MAGIC The model can also be served as a REST API, accepting JSON-formatted requests to an endpoint that can be run on Azure ML, AWS SageMaker, or, even within Databricks for testing and low-throughput use cases.
# MAGIC 
# MAGIC Enable Serving of the model in the Model Registry. You can send the JSON snippet above to the service this way:
# MAGIC 
# MAGIC <img width="800" src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/gartner_2020/serving.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## "What-If?" and Dashboarding
# MAGIC 
# MAGIC Finally, this model can power a simple dashboard, where an analyst can change values and see updated predictions reflected in a plot. In this case, the dashboard shows predicted life expectancy over time for the USA, where a single feature is varied over a range.
# MAGIC 
# MAGIC Switch to the View for Dashboard "WhatIfGartner2020".
# MAGIC 
# MAGIC Try changing the Feature widget above and changing the range of values. The heatmap re-renders.

# COMMAND ----------

import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

model = mlflow.xgboost.load_model("models:/gartner_2020/production")

future_input_pd = spark.read.table("gartner.gartner_2020_featurized").drop("WHOSIS_000001").filter("Country_USA").orderBy("Year").toPandas()

dbutils.widgets.removeAll()
dbutils.widgets.dropdown("Feature", "SH.DYN.NCOM.ZS", ["WHS9_96", "SH.DYN.NCOM.ZS"], "Feature")
dbutils.widgets.text("From", "0", "From")
dbutils.widgets.text("To", "30", "To")

# COMMAND ----------

feature = dbutils.widgets.get("Feature")
from_val = float(dbutils.widgets.get("From"))
to_val = float(dbutils.widgets.get("To"))

count = 10
range_val = [to_val - (to_val - from_val) * i / count for i in range(count+1)]
range_year = [2000 + i for i in range(19)]

predictions = np.zeros((len(range_val), len(range_year)))
for i in range(len(range_val)):
  widget_input_pd = future_input_pd.copy()
  widget_input_pd[feature] = range_val[i]
  predictions[i,:] = model.predict(xgb.DMatrix(data=widget_input_pd))

plt.close()
plt.figure(figsize=(14, 7))
yticks = [f"{r:.3f}".rstrip('0').rstrip('.') for r in range_val]
display(sns.heatmap(pd.DataFrame(predictions, index=range_val, columns=range_year), cmap='RdYlBu', annot=True, square=True, yticklabels=yticks))
