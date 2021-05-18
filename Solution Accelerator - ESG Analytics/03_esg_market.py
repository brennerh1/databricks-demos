# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/databricks_fsi_white.png width=600px>

# COMMAND ----------

# MAGIC %md
# MAGIC # ESG - market risk
# MAGIC 
# MAGIC The future of finance goes hand in hand with social responsibility, environmental stewardship and corporate ethics. In order to stay competitive, Financial Services Institutions (FSI)  are increasingly  disclosing more information about their **environmental, social and governance** (ESG) performance. By better understanding and quantifying the sustainability and societal impact of any investment in a company or business, FSIs can mitigate reputation risk and maintain the trust with both their clients and shareholders. At Databricks, we increasingly hear from our customers that ESG has become a C-suite priority. This is not solely driven by altruism but also by economics: [Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/). In this demo, we offer a novel approach to sustainable finance by combining NLP techniques and graph analytics to extract key strategic ESG initiatives and learn companies' relationships in a global market and their impact to market risk calculations.
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_esg_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_esg_report">STAGE1</a>: Using NLP to extract key ESG initiatives PDF reports
# MAGIC + <a href="$./02_esg_scoring">STAGE2</a>: Introducing a novel approach to ESG scoring using graph analytics
# MAGIC + <a href="$./03_esg_market">STAGE3</a>: Applying ESG to market risk calculations
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC As reported in previous notebook (see <a href="$./02_esg_scoring">STAGE2</a>), our framework for ESG scoring is generic enough to accommodate multiple use cases. Whilst core FSIs may consider their own company as a landmark to Page Rank in order to better evaluate **reputational risks**, asset managers could consider all their positions as landmarks to better **assess the sustainability** relative to each of their investments. In order to validate our initial assumption that [...] [higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/), we create a synthetic portfolio made of random equities that we run through our ESG framework and **combine with actual stock information** retrieved from Yahoo Finance.
# MAGIC 
# MAGIC ### Dependencies
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. **NOTE** The next cell assumes you are running this notebook on a Databricks cluster that does not make use of the ML runtime.  If using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment. 

# COMMAND ----------

# DBTITLE 0,Install needed libraries
dbutils.library.installPyPI('yfinance')
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # `STEP1`: Synthetic portfolio
# MAGIC We create a synthetic portfolio made of random equities that we run through our ESG framework and combine with actual stock information retrieved from Yahoo Finance. Note that we've manually cleansed this portfolio upfront to have company name matching our GDELT dataset.

# COMMAND ----------

# DBTITLE 1,Create our portfolio
import pandas as pd
from io import StringIO

portfolio = """
symbol,organisation
STI,suntrust banks
BIG,big lots
PANW,palo alto networks
IT,gartner
AGN,allergan
FL,foot locker
STI.B,suntrust banks
PRH,prudential financial
CAT,caterpillar
VMW,vmware
CLGX,corelogic
MET,metlife
JWN,nordstrom
PJH,prudential financial
RHT,red hat
PLD,prologis
JCI,johnson controls
PEP,pepsico
NEE,nextera energy
MO,altria group
STI.A,suntrust banks
TSN,tyson foods
DAL,delta air lines
TVE,tennessee valley authority
AIG,american international group
TOT,total sa
PFE,pfizer
CMG,chipotle mexican grill
EFX,equifax
TSNU,tyson foods
UPS,united parcel service
PSO,pearson
AIG.W,american international group
CAH,cardinal health
MTN,vail resorts
BR,broadridge financial solutions
BLK,blackrock
EXPR,express
PRU,prudential financial
SPG,simon property group"""

portfolio_df = pd.read_csv(StringIO(portfolio))
spark.createDataFrame(portfolio_df).createOrReplaceTempView('portfolio')
display(portfolio_df)

# COMMAND ----------

# DBTITLE 1,Retrieve ESG from previous notebook
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW esg_nyse_scores AS
# MAGIC SELECT 
# MAGIC   g.organisation,
# MAGIC   g.theme,
# MAGIC   g.total,
# MAGIC   g.days,
# MAGIC   g.esg,
# MAGIC   p.symbol
# MAGIC FROM esg.scores g
# MAGIC JOIN portfolio p
# MAGIC ON p.organisation = g.organisation;
# MAGIC 
# MAGIC SELECT * FROM esg_nyse_scores;

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_var_scores.png">

# COMMAND ----------

# MAGIC %md
# MAGIC # `STEP2`: Rebuild our global market network and compute ESG
# MAGIC In this section, we apply the exact same framework as reported in <a href="$./02_esg_scoring">STAGE2</a> so don't be surprised by the lack of comments. Due to the nature of news analytics, it is not surprising to observe news publishing companies (such as Thomson Reuters or Bloomberg) or social networks (Facebook, Twitter) as strongly connected organisations. Not reflecting the true connections of a given business but rather explained by a simple co-occurrence in news articles, we filter them out prior to our page rank process.

# COMMAND ----------

# DBTITLE 0,Create our ESG network
# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import org.graphframes.GraphFrame
# MAGIC 
# MAGIC // GDELT may consider multiple organisations from a NLP standpoint that we would like to ingore.
# MAGIC val blacklist = Set("united states", "european union", "reuters", "twitter", "facebook", "thomson reuters", "associated press", "new york times", "bloomberg")
# MAGIC val blacklist_b = spark.sparkContext.broadcast(blacklist)
# MAGIC 
# MAGIC // Cartesian products of organisations mentioned in a same news article
# MAGIC val buildTuples = udf((xs: Seq[String]) => {
# MAGIC   val organisations = xs.filter(x => !blacklist_b.value.contains(x))
# MAGIC   organisations.flatMap(x1 => {
# MAGIC     organisations.map(x2 => {
# MAGIC       (x1, x2)
# MAGIC     })
# MAGIC   }).toSeq.filter({ case (x1, x2) =>
# MAGIC     x1 != x2 // remove self edges
# MAGIC   })
# MAGIC })
# MAGIC 
# MAGIC // Generate our vertex dataset (organisations mentioned in news articles)
# MAGIC val nodes = spark
# MAGIC   .read
# MAGIC   .table("esg.scores")
# MAGIC   .select(col("organisation").as("id"))
# MAGIC   .distinct()
# MAGIC 
# MAGIC // Generate our edges (organisations shating common news articles more than 200 times)
# MAGIC val edges = spark.read.table("esg.gdelt_silver")
# MAGIC   .groupBy("url")
# MAGIC   .agg(collect_list(col("organisation")).as("organisations"))
# MAGIC   .withColumn("tuples", buildTuples(col("organisations")))
# MAGIC   .withColumn("tuple", explode(col("tuples")))
# MAGIC   .withColumn("src", col("tuple._1"))
# MAGIC   .withColumn("dst", col("tuple._2"))
# MAGIC   .groupBy("src", "dst")
# MAGIC   .agg(sum(lit(1)).as("relationship"))
# MAGIC   .filter(col("relationship") > 200)
# MAGIC 
# MAGIC // Create our graph object
# MAGIC val esgGraph = GraphFrame(nodes, edges)
# MAGIC println("Number of nodes : " + esgGraph.vertices.count()) //2,611
# MAGIC println("Number of edges : " + esgGraph.edges.count()) //97,212

# COMMAND ----------

# MAGIC %md
# MAGIC We want to compute the importance of each organisation relative to the different instruments in our portfolio. We will consider each of our instrument as a `landmark` to compute shortest path and page rank, leading to our weighted propagated ESG score introduced in previous notebook.

# COMMAND ----------

# DBTITLE 0,Create our new landmarks
# MAGIC %scala
# MAGIC 
# MAGIC // Consider our portfolio items as new landmarks
# MAGIC // These will be used to compute node influence via personalised page rank
# MAGIC val landmarks = spark
# MAGIC   .sql("SELECT DISTINCT organisation FROM esg_nyse_scores")
# MAGIC   .rdd
# MAGIC   .map(_.getAs[String]("organisation"))
# MAGIC   .collect()
# MAGIC 
# MAGIC // Make these landmarks available at executor level
# MAGIC val landmarks_b = spark.sparkContext.broadcast(landmarks)

# COMMAND ----------

# DBTITLE 0,Limit our graph to a depth of 4
# MAGIC %scala
# MAGIC 
# MAGIC // Run shortest path algorithm with our portfolio items as landmarks
# MAGIC val shortestPaths = esgGraph
# MAGIC   .shortestPaths
# MAGIC   .landmarks(landmarks)
# MAGIC   .run()
# MAGIC 
# MAGIC // Limit our graph to at most 4 hops away from our landmarks
# MAGIC val filterDepth = udf((distances: Map[String, Int]) => {
# MAGIC   distances.values.exists(_ < 5)
# MAGIC })
# MAGIC 
# MAGIC // Filter graph
# MAGIC val esgDenseGraph = GraphFrame(shortestPaths, edges)
# MAGIC   .filterVertices(filterDepth(col("distances")))
# MAGIC   .cache()

# COMMAND ----------

# DBTITLE 0,Run a personalised page rank to find relation importance
# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.ml.linalg.Vector
# MAGIC 
# MAGIC // Run personalised page rank with our portfolio as landmark
# MAGIC // This retrieve connections importance relative to our investments
# MAGIC val prNodes = esgDenseGraph
# MAGIC   .parallelPersonalizedPageRank
# MAGIC   .resetProbability(0.15)
# MAGIC   .maxIter(100)
# MAGIC   .sourceIds(landmarks.asInstanceOf[Array[Any]])
# MAGIC   .run()
# MAGIC 
# MAGIC // Retrieve importance to each of our investment
# MAGIC val importances = udf((pr: Vector) => {
# MAGIC   pr.toArray.zipWithIndex.map({ case (importance, id) =>
# MAGIC     (landmarks_b.value(id), importance)
# MAGIC   })
# MAGIC })
# MAGIC 
# MAGIC // Extract list of connections and their relative importance to our investments
# MAGIC val connections = prNodes
# MAGIC   .vertices
# MAGIC   .withColumn("importances", importances(col("pageranks")))
# MAGIC   .withColumn("importance", explode(col("importances")))
# MAGIC   .select(
# MAGIC     col("importance._1").as("organisation"),
# MAGIC     col("id").as("connection"),
# MAGIC     col("importance._2").as("importance")
# MAGIC   )

# COMMAND ----------

# DBTITLE 0,Compute our ESG contribution
# MAGIC %scala
# MAGIC 
# MAGIC // Create our weighted propagated ESG average 
# MAGIC // By bringing internal ESG for each connection proportional to their importance
# MAGIC spark
# MAGIC   .read
# MAGIC   .table("esg.scores")
# MAGIC   .withColumnRenamed("organisation", "connection")
# MAGIC   .join(connections, List("connection"))
# MAGIC   .withColumn("weightedEsg", col("esg") * col("importance"))
# MAGIC   .groupBy("organisation", "theme")
# MAGIC   .agg(sum("weightedEsg").as("totalWeightedEsg"), sum("importance").as("totalImportance"))
# MAGIC   .withColumn("weightedEsg", col("totalWeightedEsg") / col("totalImportance"))
# MAGIC   .select(col("organisation"), col("theme"), col("weightedEsg").as("esg"))
# MAGIC   .filter(col("esg").isNotNull)
# MAGIC   .createOrReplaceTempView("esg_portfolio_scores")
# MAGIC   
# MAGIC display(spark.read.table("esg_portfolio_scores"))

# COMMAND ----------

# MAGIC %md
# MAGIC # `STEP3`: Access stock market data
# MAGIC In this section, we retrieve stock market data for the last 2 years for each equity in our defined portfolio. The aim is to search for correlation between good ESG score and profitability and bad ESG score and market volatility. For more information about the use of Yahoo Finance at scale, please refer to a [notebook](https://databricks.com/notebooks/01_market_etl.html) published recently as part of our risk management solution accelerator

# COMMAND ----------

# DBTITLE 0,Compute ESG quantiles for each position
from pyspark.sql import functions as F
import pandas as pd

# retrieve all ESG scores
esg_df = spark.read.table('esg_portfolio_scores') \
  .groupBy("organisation") \
  .agg(F.avg("esg").alias("esg")) \
  .join(spark.read.table("portfolio"), ['organisation']) \
  .toPandas()

# classify ESG into quantiles
esg_df['class'] = pd.qcut(esg_df['esg'], q=5, labels=['BBB', 'BB', 'A', 'AA', 'AAA'])
symbol_esg = spark.createDataFrame(esg_df)

# COMMAND ----------

# DBTITLE 0,Download stock market data
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import yfinance as yf

# define the output schema of our pandas dataframe
schema = StructType([
    StructField('symbol', StringType(), True), 
    StructField('date', DateType(), True),
    StructField('close', DoubleType(), True)
  ])

# for each symbol, we download last 2 years worth of history
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def fetch_tick(group, pdf):
  tick = group[0]
  try:
    msft = yf.Ticker(tick)
    raw = msft.history(period="2y")[['Close']]
    # fill in missing business days
    # use last observation carried forward for missing value
    output_df = raw.asfreq(freq='B', method='pad')
    # Pandas does not keep index (date) when converted into spark dataframe
    output_df['date'] = output_df.index
    output_df['symbol'] = tick    
    output_df = output_df.rename(columns={"Close": "close"})
    return output_df
  except:
    return pd.DataFrame(columns = ['symbol', 'date', 'close'])

# Download yahoo finance historical data in parallel
spark \
  .read.table('portfolio') \
  .groupBy("symbol") \
  .apply(fetch_tick) \
  .createOrReplaceTempView("portfolio_stocks")

# COMMAND ----------

# DBTITLE 1,Download stock market data
display(spark.read.table("portfolio_stocks").filter(F.col("symbol") == "BLK").orderBy(F.asc("date")))

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_var_stock.png">

# COMMAND ----------

# MAGIC %md
# MAGIC For each instrument in our portfolio, we compute daily log returns via a simple [window partition](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html) function.

# COMMAND ----------

# DBTITLE 0,Compute instrument returns
from pyspark.sql import Window
from pyspark.sql.functions import udf
import numpy as np

# Create UDF for computing daily log returns
@udf("double")
def compute_return(first, close):
  return float(np.log(close / first))

# Apply a tumbling 1 day window on each instrument
window = Window.partitionBy('symbol').orderBy('date').rowsBetween(-1, 0)

# Compute returns on investment
stock_esg = spark \
  .read \
  .table('portfolio_stocks') \
  .filter(F.col('close').isNotNull()) \
  .withColumn("first", F.first('close').over(window)) \
  .withColumn("return", compute_return('first', 'close')) \
  .join(symbol_esg, ['symbol']) \
  .select('symbol', 'class', 'date', 'close', 'return') \
  .toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # `STEP4`: ESG to market risk
# MAGIC Following up on our recent blog post about [modernizing risk management](https://databricks.com/blog/2020/05/27/modernizing-risk-management-part-1-streaming-data-ingestion-rapid-model-development-and-monte-carlo-simulations-at-scale.html), we can use this new information available to us to drive better risk calculations. Splitting our portfolio into 2 distinct books, composed of the best and worst 10% of our ESG rated instruments, we compute below the historical returns and its corresponding 95% value-at-risk (historical VaR). Please refer to our previous blog for more information about value at risk calculation. 
# MAGIC The aim is to confirm our initial assumption that *there appears to be a link between ESG—Environment, Social, and Governance—and financial performance* ([source](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/)). With our ESG scores computed for each of our instruments, we would like to access instrument performance and compute market volatility.

# COMMAND ----------

# DBTITLE 0,Retrieve 2 extreme ESG instruments
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import seaborn as sns

# we sort our positions by their computed ESG score
# and take our lowest and highest ESG rated instruments
esg_df = esg_df.sort_values(by='esg', ascending=True)
w_esg = esg_df.iloc[0]
b_esg = esg_df.iloc[-1]

organisations = [w_esg.organisation, b_esg.organisation]
symbols = [w_esg.symbol, b_esg.symbol]

# we retrieve sentiment analysis data for these 2 organisations
gdelt_data = spark \
  .read \
  .table('esg.gdelt_gold') \
  .filter(F.col('organisation').isin(organisations)) \
  .groupBy('date', 'organisation') \
  .agg(F.sum("tone").alias("tone"), F.sum("total").alias("total")) \
  .withColumn("tone", F.col("tone") / F.col("total")) \
  .select('date', 'organisation', 'tone') \
  .toPandas()

# COMMAND ----------

# DBTITLE 1,Retrieve 2 extreme ESG instruments
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]}, figsize=(12, 8))
colors = ['lightcoral', 'teal']

for i in range(0,2):
  
  # retrieve sentiment time series
  organisation = organisations[i]
  org_df = gdelt_data[gdelt_data['organisation'] == organisation].drop('organisation', axis=1).sort_values('date')
  org_df = org_df.set_index('date')
  org_df = org_df.asfreq(freq = 'D', method = 'pad')

  # retrieve stock time series
  symbol = symbols[i]
  sym_df = stock_esg[stock_esg['symbol'] == symbol].drop(['symbol', 'class'], axis=1).sort_values('date')
  sym_df = sym_df.set_index('date')
  sym_df = sym_df.asfreq(freq = 'D', method = 'pad')
  sym_df['close'] = 100 * sym_df['close'] / sym_df.iloc[0].close
  
  # align 2 dates
  sym_df = sym_df.reindex(org_df.index)

  # plot both series in subplots
  axs[0].plot(org_df.index, org_df.rolling(window=30).mean().tone, linewidth=2, label=organisation, color=colors[i])
  axs[1].plot(sym_df.index, sym_df.rolling(window=7).mean().close, linewidth=2, label="{} ({})".format(organisation, symbol), color=colors[i])
  
# plot graph
axs[0].title.set_text('Sentiment analysis (as proxy for ESG)')
axs[1].title.set_text('Stock performance (normalised)')
axs[0].tick_params(axis='x', which='both', bottom='off', labelbottom='off')
axs[0].legend(loc="upper left")
axs[1].legend(loc="upper left")

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_var_performance.png">

# COMMAND ----------

# MAGIC %md
# MAGIC As reported in the graph above, despite an evident lack of data to draw scientific conclusions, it would appear that our highest and lowest ESG rated companies (we report the sentiment analysis as a proxy of ESG in the top graph) are respectively the best or worst profitable instruments in our portfolio over the last 18 months. Interestingly, [csrhub](https://www.csrhub.com/) reports the exact opposite, `pearson` being 10 points above `prologis`, highlighting the subjectivity of ESG and its inconsistency between what was communicated and what is actually observed. 

# COMMAND ----------

# MAGIC %md
# MAGIC In the section below, we run an hypothetical scenario where our synthetic portfolio is split into 2 books where each one is respectively composed of the best and worst 10% of our ESG rated instruments. For each book, we report its historical returns and its corresponding 95% value-at-risk (historical VaR).

# COMMAND ----------

# DBTITLE 1,Historical value at risk for different ESG scores
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize = (12, 6))

# create 5 different books for each ESG quantile, aggregating individual returns
grouped_assets = stock_esg.groupby(['class', 'date']).sum()

# retrieve poorly rated ESG portfolio and compute historical value at risk
bbb_portfolio = grouped_assets.loc['BBB']
bbb_var = np.quantile(bbb_portfolio['return'], 5 / 100)
plt.hist(bbb_portfolio['return'], bins=200, alpha=0.25, color='lightcoral', density=True, label='BBB portfolio')
plt.axvline(bbb_var, color='lightcoral', linestyle='--', label="VAR95 {:.2f}".format(bbb_var))

# retrieve highly rated ESG portfolio and compute historical value at risk
aaa_portfolio = grouped_assets.loc['AAA']
aaa_var = np.quantile(aaa_portfolio['return'], 5 / 100)
plt.hist(aaa_portfolio['return'], bins=200, alpha=0.25, color='teal', density=True, label='AAA portfolio')
plt.axvline(aaa_var, color='teal', linestyle='--', label="VAR95 {:.2f}".format(aaa_var))

# plot graph
plt.xlim(-0.5,0.5)
plt.legend(loc='upper right')
plt.xlabel('Portfolio historical returns')
plt.ylabel('Density')
plt.title('Historical 95% VaR')

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_var_risk.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Without any prior knowledge of our instruments beyond the metrics we extracted through our framework, we can observe a risk exposure to be 2 times higher for a portfolio made of poor ESG rated companies, supporting the assumptions found in the literature that ”*poor ESG [...] correlates with higher market volatility*”, hence to a greater value-at-risk. Using the flexibility of cloud compute and the level of interactivity in your data enabled through our Databricks runtime, risk analysts can better understand the risk facing their business by slicing and dicing market risk calculations at different industries, countries, segments, and now at different ESGs ratings. This data-driven ESG framework enables businesses to ask new questions such as: how much of your risk would be decreased by bringing the environmental rating of this company up 10 points? How much more exposure would you face by investing in these instruments given their low ESG scores?
# MAGIC 
# MAGIC Using the flexibility and scale of cloud compute and the level of interactivity in your data enabled through our Databricks runtime, risk analysts can better understand the risks facing their business by slicing and dicing market risk calculations at different industries, countries, segments, **and now at different ESG ratings**. What is your market risk exposure for environmental? Social? Governance?
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_var_evolution.png" alt="esg_var" width="1000">
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_esg_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_esg_report">STAGE1</a>: Using NLP to extract key ESG initiatives PDF reports
# MAGIC + <a href="$./02_esg_scoring">STAGE2</a>: Introducing a novel approach to ESG scoring using graph analytics
# MAGIC + <a href="$./03_esg_market">STAGE3</a>: Applying ESG to market risk calculations
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | PyPDF2                                 | PDF toolkit.            | CUSTOM     | http://mstamy2.github.io/PyPDF2/                    |
# MAGIC | graphframes:graphframes:0.8.1          | Graph library           | Apache2    | https://github.com/graphframes/graphframes          |
# MAGIC | Yfinance                               | Yahoo finance           | Apache2    | https://github.com/ranaroussi/yfinance              |
# MAGIC | com.aamend.spark:gdelt:3.0             | GDELT wrapper           | Apache2    | https://github.com/aamend/spark-gdelt               |
# MAGIC | PyLDAvis                               | LDA visualizer          | MIT        | https://github.com/bmabey/pyLDAvis                  |
# MAGIC | Gensim                                 | Topic modelling         | L-GPL2     | https://radimrehurek.com/gensim/                    |
# MAGIC | Wordcloud                              | Visualization library   | MIT        | https://github.com/amueller/word_cloud              |

# COMMAND ----------


