// Databricks notebook source
// MAGIC %md
// MAGIC <img src=https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/databricks_fsi_white.png width=600px>

// COMMAND ----------

// MAGIC %md
// MAGIC # ESG - data driven ESG score
// MAGIC 
// MAGIC The future of finance goes hand in hand with social responsibility, environmental stewardship and corporate ethics. In order to stay competitive, Financial Services Institutions (FSI)  are increasingly  disclosing more information about their **environmental, social and governance** (ESG) performance. By better understanding and quantifying the sustainability and societal impact of any investment in a company or business, FSIs can mitigate reputation risk and maintain the trust with both their clients and shareholders. At Databricks, we increasingly hear from our customers that ESG has become a C-suite priority. This is not solely driven by altruism but also by economics: [Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/). In this demo, we offer a novel approach to sustainable finance by combining NLP techniques and graph analytics to extract key strategic ESG initiatives and learn companies' relationships in a global market and their impact to market risk calculations.
// MAGIC 
// MAGIC ---
// MAGIC + <a href="$./00_esg_context">STAGE0</a>: Home page
// MAGIC + <a href="$./01_esg_report">STAGE1</a>: Using NLP to extract key ESG initiatives PDF reports
// MAGIC + <a href="$./02_esg_scoring">STAGE2</a>: Introducing a novel approach to ESG scoring using graph analytics
// MAGIC + <a href="$./03_esg_market">STAGE3</a>: Applying ESG to market risk calculations
// MAGIC ---
// MAGIC <antoine.amend@databricks.com>

// COMMAND ----------

// MAGIC %md
// MAGIC ## Context
// MAGIC As covered in the previous notebook (see <a href="$./01_esg_report">STAGE1</a>), we were able to compare businesses side by side across 9 different ESG initiatives. Although we could attempt to derive an ESG score, we want our score **not to be subjective but truly data driven**. In other terms, we do not want to solely base our assumptions on companies’ official disclosures but rather on how companies' reputations are perceived in the media, across all 3 environmental, social and governance variables. For that purpose, we use [GDELT](https://www.gdeltproject.org/), the global database of event location and tones.

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Internal ESG
// MAGIC We assume that the overall **tone captured from financial news articles is a good proxy for companies' ESG scores**. For instance, a series of bad press articles related to maritime disasters and oil spills would strongly affect a company's environmental performance. On the opposite, news articles about "[...] *financing needs of women-owned businesses in developing countries ([source](https://www.bloomberg.com/news/articles/2019-10-08/goldman-sachs-helps-loan-1-45-billion-to-women-entrepreneurs))*" with a more positive tone would positively contribute to a better ESG score. Our approach is to look at the difference between a company sentiment and its industry average; how much more "positive" or "negative" a company is perceived across all financial services news articles, over time. 
// MAGIC 
// MAGIC #### Network ESG
// MAGIC In the second part of this notebook, we will use **graph analytics to learn connections** between organisations and the impact a company may have to a business ESG rating. When looking at this problem from a network perspective, the "true" ESG score should be derived from the contribution of a company's connections. As an example, if a firm keeps investing in companies directly or indirectly related with environmental issues (and as such mentioned in negative tone articles), this will be captured, should be quantified and must be reflected back on companies' overall ESG. We can cite Barclays reputation being impacted in late 2018 because of its indirect connections to tar sand projects ([source](https://www.theguardian.com/business/2018/dec/05/barclays-customers-threaten-leave-en-masse-tar-sands-investment-greenpeace)). 

// COMMAND ----------

// MAGIC %md
// MAGIC ### Dependencies
// MAGIC 
// MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment. 
// MAGIC 
// MAGIC In order to efficiently process GDELT files, we make use of an open source scala based library I personally open sourced as a pet project (`com.aamend.spark:spark-gdelt:2.x`). Note that it does not represent Databricks in any way as this was developed a long time ago and maintained on a best effort basis. In addition, we also bring `graphframes:graphframes:0.6.0-spark2.3-s_2.11` as an abstraction layer to core `graphx` functionality.

// COMMAND ----------

// DBTITLE 0,Install needed libraries
// MAGIC %python
// MAGIC dbutils.library.installPyPI('wordcloud')
// MAGIC dbutils.library.restartPython()

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP1`: Retrieve news from 60,000 publishers
// MAGIC *Supported by Google Jigsaw, the [GDELT](https://www.gdeltproject.org/) Project monitors the world's broadcast, print, and web news from nearly every corner of every country in over 100 languages and identifies the people, locations, organizations, themes, sources, emotions, counts, quotes, images and events driving our global society every second of every day, creating a free open platform for computing on the entire world.* Although it is convenient to scrape for [master URL]((http://data.gdeltproject.org/gdeltv2/lastupdate.txt) file to process latest GDELT increment, processing 2 years backlog is time consuming and resource intensive. Below bash script is for illustration purpose mainly, so please **proceed with caution**. 

// COMMAND ----------

// DBTITLE 0,Create dbfs scratch space
try {
  dbutils.fs.rm("/tmp/gdelt", true)
} finally {
  dbutils.fs.mkdirs("/tmp/gdelt")
}

// COMMAND ----------

// DBTITLE 0,Download Jan 2020 data - CAUTION (long process)
// MAGIC %sh
// MAGIC 
// MAGIC MASTER_URL=http://data.gdeltproject.org/gdeltv2/masterfilelist.txt
// MAGIC 
// MAGIC if [[ -e /tmp/gdelt ]] ; then
// MAGIC   rm -rf /tmp/gdelt
// MAGIC fi
// MAGIC mkdir /tmp/gdelt
// MAGIC 
// MAGIC echo "Retrieve 2020 archives to date"
// MAGIC URLS=`curl ${MASTER_URL} 2>/dev/null | awk '{print $3}' | grep gkg.csv.zip | grep gdeltv2/20200731`
// MAGIC for URL in $URLS; do
// MAGIC   echo "Downloading ${URL}"
// MAGIC   wget $URL -O /tmp/gdelt/gdelt.csv.zip > /dev/null 2>&1
// MAGIC   unzip /tmp/gdelt/gdelt.csv.zip -d /tmp/gdelt/ > /dev/null 2>&1
// MAGIC   LATEST_FILE=`ls -1rt /tmp/gdelt/*.csv | head -1`
// MAGIC   LATEST_NAME=`basename ${LATEST_FILE}`
// MAGIC   cp $LATEST_FILE /dbfs/tmp/gdelt/$LATEST_NAME
// MAGIC   rm -rf /tmp/gdelt/gdelt.csv.zip
// MAGIC   rm $LATEST_FILE
// MAGIC done

// COMMAND ----------

// DBTITLE 0,Parse GDELT events using 3rd party library
import com.aamend.spark.gdelt._
val gdeltDF = spark.read.gdeltGkg("/tmp/gdelt")
gdeltDF.write.format("delta").mode("append").saveAsTable("esg.gdelt_bronze")

// COMMAND ----------

// MAGIC %md
// MAGIC Given the volume of data available in GDELT (100 million records for the last 18 months only), we leverage the [lakehouse](https://databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html) paradigm by moving data from raw, to filtered and enriched, respectively from Bronze, to Silver and Gold layers, and extend our process to operate in near real time (GDELT files are published every 15mn)

// COMMAND ----------

// DBTITLE 0,200 millions records for the last 18 months
// MAGIC %sql
// MAGIC WITH gdelt_timeline AS (
// MAGIC SELECT CONCAT(year(to_date(publishDate)), '-', month(to_date(publishDate))) AS date FROM esg.gdelt_bronze
// MAGIC )
// MAGIC 
// MAGIC SELECT date, COUNT(*) 
// MAGIC FROM gdelt_timeline
// MAGIC GROUP BY date
// MAGIC ORDER BY date ASC

// COMMAND ----------

// MAGIC %md
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_count.png">

// COMMAND ----------

// MAGIC %md
// MAGIC Our Delta Lake table may be composed of many small files (one per 15mn window at least). To improve the performance of queries, we run the `OPTIMIZE` command as follows

// COMMAND ----------

// DBTITLE 0,Compact small files
// MAGIC %sql
// MAGIC OPTIMIZE esg.gdelt_bronze

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP2`: Extracting ESG specific news
// MAGIC Whilst GDELT captures over 2000+ themes (keywords based), we want to focus only on specific themes to our problem statement and filter for ESG related articles to a silver table.

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC - We assume all `ENV_*` themes to be related to **environment**
// MAGIC - We assume all `UNGP_*` themes to be related to **social** ([United nations guiding principles for human right](https://en.wikipedia.org/wiki/United_Nations_Guiding_Principles_on_Business_and_Human_Rights))
// MAGIC - Any other financial news (captured via `ECON_*`) would affect the company **governance** and conduct strategy.

// COMMAND ----------

// DBTITLE 0,Dedupe FSI records
import org.apache.spark.sql.functions._

// Organisations may be called slightly differently, so we want to retrieve our ESG specific records using alternative names
val organisationAltNames = Map(
  "standard chartered"       -> Seq("standard chartered"),
  "rbc"                      -> Seq("rbc ", "royal bank of canada"),
  "credit suisse"            -> Seq("credit suisse"),
  "lloyds"                   -> Seq("lloyds bank"),
  "jp morgan chase"          -> Seq("jpmorgan", "jp morgan"),
  "goldman sachs"            -> Seq("goldman sachs"),
  "santander"                -> Seq("santander", "banco santander"),
  "lazard"                   -> Seq("lazard"),
  "macquarie"                -> Seq("macquarie group", "macquarie bank", "macquarie management", "macquarie investment", "macquarie capital"),
  "barclays"                 -> Seq("barclays"),
  "northern trust"           -> Seq("northern trust"),
  "citi"                     -> Seq("citigroup"),
  "morgan stanley"           -> Seq("morgan stanley")
)

// Broadcast of dictionay of names
val organisationAltNamesB = spark.sparkContext.broadcast(organisationAltNames)

// Clean organisation name. If match our ESG list, get the clean name, else, return the original organisation name
val cleanOrganisation = udf((s: String) => {
  organisationAltNamesB.value.find({ case (organisation, alts) =>
    alts.exists(alt => {
      s.startsWith(alt)
    })
  }).map(_._1).getOrElse(s)
})

// COMMAND ----------

// MAGIC %md
// MAGIC GDELT follows `FIPS 10-4` standard as a list of two-character country codes which are used for geopolitical entity identification. This, however, is not practical with modern visualisations or geo frameworks for which ISO standards are the norm. In this demo, it is key for FIPS `UK` to be considered as ISO `GB` and not as Ukraine. In order to avoid any confusion, we'll be using 3 letter code (`GBR` for United Kingdom of Great Britain and Northern Ireland) using mapping table below.

// COMMAND ----------

// DBTITLE 0,ISO3 country codes
val dat = """
ABW  Aruba
AFG  Afghanistan
AGO  Angola
AIA  Anguilla
ALA  Aland Islands
ALB  Albania
AND  Andorra
ARE  United Arab Emirates
ARG  Argentina
ARM  Armenia
ASM  American Samoa
ATA  Antarctica
ATF  French Southern Territories
ATG  Antigua and Barbuda
AUS  Australia
AUT  Austria
AZE  Azerbaijan
BDI  Burundi
BEL  Belgium
BEN  Benin
BES  Bonaire, Sint Eustatius and Saba
BFA  Burkina Faso
BGD  Bangladesh
BGR  Bulgaria
BHR  Bahrain
BHS  Bahamas
BIH  Bosnia and Herzegovina
BLM  Saint Barthélemy
BLR  Belarus
BLZ  Belize
BMU  Bermuda
BOL  Bolivia (Plurinational State of)
BRA  Brazil
BRB  Barbados
BRN  Brunei Darussalam
BTN  Bhutan
BVT  Bouvet Island
BWA  Botswana
CAF  Central African Republic
CAN  Canada
CCK  Cocos (Keeling) Islands
CHE  Switzerland
CHL  Chile
CHN  China
CIV  Côte d'Ivoire
CMR  Cameroon
COD  Congo, Democratic Republic of the
COG  Congo
COK  Cook Islands
COL  Colombia
COM  Comoros
CPV  Cabo Verde
CRI  Costa Rica
CUB  Cuba
CUW  Curaçao
CXR  Christmas Island
CYM  Cayman Islands
CYP  Cyprus
CZE  Czechia
DEU  Germany
DJI  Djibouti
DMA  Dominica
DNK  Denmark
DOM  Dominican Republic
DZA  Algeria
ECU  Ecuador
EGY  Egypt
ERI  Eritrea
ESH  Western Sahara
ESP  Spain
EST  Estonia
ETH  Ethiopia
FIN  Finland
FJI  Fiji
FLK  Falkland Islands (Malvinas)
FRA  France
FRO  Faroe Islands
FSM  Micronesia (Federated States of)
GAB  Gabon
GBR  United Kingdom of Great Britain and Northern Ireland
GEO  Georgia
GGY  Guernsey
GHA  Ghana
GIB  Gibraltar
GIN  Guinea
GLP  Guadeloupe
GMB  Gambia
GNB  Guinea-Bissau
GNQ  Equatorial Guinea
GRC  Greece
GRD  Grenada
GRL  Greenland
GTM  Guatemala
GUF  French Guiana
GUM  Guam
GUY  Guyana
HKG  Hong Kong
HMD  Heard Island and McDonald Islands
HND  Honduras
HRV  Croatia
HTI  Haiti
HUN  Hungary
IDN  Indonesia
IMN  Isle of Man
IND  India
IOT  British Indian Ocean Territory
IRL  Ireland
IRN  Iran (Islamic Republic of)
IRQ  Iraq
ISL  Iceland
ISR  Israel
ITA  Italy
JAM  Jamaica
JEY  Jersey
JOR  Jordan
JPN  Japan
KAZ  Kazakhstan
KEN  Kenya
KGZ  Kyrgyzstan
KHM  Cambodia
KIR  Kiribati
KNA  Saint Kitts and Nevis
KOR  Korea, Republic of
KWT  Kuwait
LAO  Lao People's Democratic Republic
LBN  Lebanon
LBR  Liberia
LBY  Libya
LCA  Saint Lucia
LIE  Liechtenstein
LKA  Sri Lanka
LSO  Lesotho
LTU  Lithuania
LUX  Luxembourg
LVA  Latvia
MAC  Macao
MAF  Saint Martin (French part)
MAR  Morocco
MCO  Monaco
MDA  Moldova, Republic of
MDG  Madagascar
MDV  Maldives
MEX  Mexico
MHL  Marshall Islands
MKD  North Macedonia
MLI  Mali
MLT  Malta
MMR  Myanmar
MNE  Montenegro
MNG  Mongolia
MNP  Northern Mariana Islands
MOZ  Mozambique
MRT  Mauritania
MSR  Montserrat
MTQ  Martinique
MUS  Mauritius
MWI  Malawi
MYS  Malaysia
MYT  Mayotte
NAM  Namibia
NCL  New Caledonia
NER  Niger
NFK  Norfolk Island
NGA  Nigeria
NIC  Nicaragua
NIU  Niue
NLD  Netherlands
NOR  Norway
NPL  Nepal
NRU  Nauru
NZL  New Zealand
OMN  Oman
PAK  Pakistan
PAN  Panama
PCN  Pitcairn
PER  Peru
PHL  Philippines
PLW  Palau
PNG  Papua New Guinea
POL  Poland
PRI  Puerto Rico
PRK  Korea (Democratic People's Republic of)
PRT  Portugal
PRY  Paraguay
PSE  Palestine, State of
PYF  French Polynesia
QAT  Qatar
REU  Réunion
ROU  Romania
RUS  Russian Federation
RWA  Rwanda
SAU  Saudi Arabia
SDN  Sudan
SEN  Senegal
SGP  Singapore
SGS  South Georgia and the South Sandwich Islands
SHN  Saint Helena, Ascension and Tristan da Cunha
SJM  Svalbard and Jan Mayen
SLB  Solomon Islands
SLE  Sierra Leone
SLV  El Salvador
SMR  San Marino
SOM  Somalia
SPM  Saint Pierre and Miquelon
SRB  Serbia
SSD  South Sudan
STP  Sao Tome and Principe
SUR  Suriname
SVK  Slovakia
SVN  Slovenia
SWE  Sweden
SWZ  Eswatini
SXM  Sint Maarten (Dutch part)
SYC  Seychelles
SYR  Syrian Arab Republic
TCA  Turks and Caicos Islands
TCD  Chad
TGO  Togo
THA  Thailand
TJK  Tajikistan
TKL  Tokelau
TKM  Turkmenistan
TLS  Timor-Leste
TON  Tonga
TTO  Trinidad and Tobago
TUN  Tunisia
TUR  Turkey
TUV  Tuvalu
TWN  Taiwan, Province of China
TZA  Tanzania, United Republic of
UGA  Uganda
UKR  Ukraine
UMI  United States Minor Outlying Islands
URY  Uruguay
USA  United States of America
UZB  Uzbekistan
VAT  Holy See
VCT  Saint Vincent and the Grenadines
VEN  Venezuela (Bolivarian Republic of)
VGB  Virgin Islands (British)
VIR  Virgin Islands (U.S.)
VNM  Viet Nam
VUT  Vanuatu
WLF  Wallis and Futuna
WSM  Samoa
YEM  Yemen
ZAF  South Africa
ZMB  Zambia
ZWE  Zimbabwe"""

val countries = dat.split("\n").map(_.split("\\W").head).toSet
val countriesB = spark.sparkContext.broadcast(countries)

// COMMAND ----------

// DBTITLE 0,Enrich with country information
// Create dictionary of FIPS code vs. ISO code for country
import com.aamend.spark.gdelt._
val fips2iso = spark.loadCountryCodes.rdd.map(x => (x.fips, x.iso3)).collectAsMap()
val fips2isoB = spark.sparkContext.broadcast(fips2iso)

// Extract mentions of countries, converted to ISO3
val getCountryCode = udf((xs: Seq[Row]) => {
  xs.map(x => {
    fips2isoB.value.getOrElse(x.getAs[String]("countryCode"), "")
  }).filter(x => {
    countriesB.value.contains(x)
  }).distinct
})

// COMMAND ----------

// DBTITLE 0,Filter ESG themes
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row

// Only search for ECON, ENV or UNGP related themes
// We assume all ENV_ themes to be related to environment
// We assume all UNGP_ themes to be related to social (United nations guiding principles for human right)
// Any other financial news (captured via ECON_) would affect the company governance and conduct strategy.
val filterThemes = udf((xs: Seq[String]) => {
  val themes = xs.flatMap(x => {
    x.split("_").head match {
      case "ENV"  => Some("E")
      case "ECON" => Some("G")
      case "UNGP" => Some("S")
      case _      => None: Option[String]
    }
  })
  // Any article, regardless of Environmental or Social would need to be ECON_ related to be used in that demo
  if(themes.exists(theme => theme == "G"))
    themes.distinct
  else
    Seq.empty[String]
})

// COMMAND ----------

// DBTITLE 0,Process GDELT increment to silver table
import com.aamend.spark.gdelt._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger

val gdeltStreamDf = spark
  .readStream                                                   // Reading as a stream, processing record since last check point
  .format("delta")                                              // Reading from a delta table
  .table("esg.gdelt_bronze")                                    // Bronze table to read data from, then enrich and filter
  .withColumn("themes", filterThemes(col("themes")))
  .filter(size(col("themes")) > 0)
  .withColumn("organisation", explode(col("organisations")))
  .withColumn("organisation", cleanOrganisation(lower(col("organisation"))))
  .select(
    col("publishDate"),
    col("organisation"),
    col("documentIdentifier").as("url"),
    getCountryCode(col("locations")).as("countries"),
    col("themes"),
    col("tone.tone")
  )

gdeltStreamDf
  .writeStream                                                   // Writing data as a stream
  .trigger(Trigger.Once)                                         // Create a streaming job triggered only once...
  .option("checkpointLocation", "/tmp/gdelt_checkpoint")         // ... that only processes data since last checkpoint
  .format("delta")                                               // write to delta table
  .table("esg.gdelt_silver")                                     // write enriched / cleansed data to silver layer

// COMMAND ----------

// DBTITLE 0,ESG news for JPMC
// MAGIC %sql
// MAGIC SELECT publishDate, url, themes, tone FROM esg.gdelt_silver
// MAGIC WHERE organisation = 'jp morgan chase'

// COMMAND ----------

// MAGIC %md
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_articles.png">

// COMMAND ----------

// MAGIC %md
// MAGIC We want to use GDELT as a proxy for ESG scoring, looking at sentiment analysis (captured by native GDELT with no need to create our own NLP pipeline) across different themes of interest. We will be aggregating sentiment across themes and organisations for each of our the companies we captured in previous notebook (companies for which we have an existing ESG report).

// COMMAND ----------

// DBTITLE 0,Create aggregated view by theme
// MAGIC %sql
// MAGIC 
// MAGIC -- aggregate sentiment, count by themes for each organisation
// MAGIC -- we prefer SUM of sentiment over AVG so that the average can still be computed for different slices of data
// MAGIC CREATE TABLE esg.gdelt_gold USING delta AS
// MAGIC SELECT 
// MAGIC   u.date,
// MAGIC   u.organisation,
// MAGIC   u.theme,
// MAGIC   SUM(u.tone) AS tone,
// MAGIC   COUNT(*) AS total
// MAGIC FROM (
// MAGIC   SELECT 
// MAGIC     to_date(g.publishDate) AS date,
// MAGIC     g.organisation,
// MAGIC     explode(g.themes) AS theme,
// MAGIC     g.tone
// MAGIC   FROM esg.gdelt_silver g
// MAGIC   WHERE length(g.organisation) > 0
// MAGIC ) u
// MAGIC GROUP BY
// MAGIC   u.date,
// MAGIC   u.organisation,
// MAGIC   u.theme;
// MAGIC 
// MAGIC -- display table
// MAGIC SELECT * FROM esg.gdelt_gold;

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP3`: Creating an internal ESG score
// MAGIC Our simple approach is to look at the difference between a company sentiment and its industry average; how much more "positive" or "negative" a company is perceived across all financial services news articles. By looking at the average of that difference over day, and normalizing across industries, we create a score internal to a company across its 'E', 'S' and 'G' dimensions. We will later understand the companies connections to normalize this score by the contribution of its connected components (mentioned in introduction as influential nodes).

// COMMAND ----------

// DBTITLE 0,Create industry average
// MAGIC %python
// MAGIC from pyspark.sql import functions as F
// MAGIC 
// MAGIC # access sentiment across all companies mentioned in financial news
// MAGIC # as our table was using SUM(tone) instead of AVG(tone), we can easily access the average accross the E, S and G
// MAGIC industry_avg = spark \
// MAGIC   .read \
// MAGIC   .table('esg.gdelt_gold') \
// MAGIC   .groupBy('date') \
// MAGIC   .agg(
// MAGIC     F.sum("tone").alias("tone"),
// MAGIC     F.sum("total").alias("total")
// MAGIC   ) \
// MAGIC   .withColumn("tone", F.col("tone") / F.col("total")) \
// MAGIC   .select('date', 'tone') \
// MAGIC   .toPandas() \
// MAGIC   .sort_values('date') \
// MAGIC   .set_index('date') \
// MAGIC   .asfreq(freq = 'D', method = 'pad')

// COMMAND ----------

// DBTITLE 0,Sentiment analysis as a proxy for ESG
// MAGIC %python
// MAGIC import pandas as pd
// MAGIC from pyspark.sql import functions as F
// MAGIC import matplotlib.pyplot as plt
// MAGIC 
// MAGIC # retrieve Barclays average sentiments across its E, S and G news articles
// MAGIC # we convert as a timeseries to Pandas for visualisation
// MAGIC barclays_df = spark \
// MAGIC   .read \
// MAGIC   .table('esg.gdelt_gold') \
// MAGIC   .filter(F.col('organisation') == 'bank of america') \
// MAGIC   .groupBy('date', 'organisation') \
// MAGIC   .agg(
// MAGIC     F.sum("tone").alias("tone"),
// MAGIC     F.sum("total").alias("total")
// MAGIC   ) \
// MAGIC   .withColumn("tone", F.col("tone") / F.col("total")) \
// MAGIC   .select('date', 'tone') \
// MAGIC   .toPandas() \
// MAGIC   .sort_values('date') \
// MAGIC   .set_index('date') \
// MAGIC   .asfreq(freq = 'D', method = 'pad')
// MAGIC 
// MAGIC # we can join Barclays series with industry average
// MAGIC # and compute daily difference (being positive or negative)
// MAGIC diff_df = industry_avg.merge(barclays_df, left_index=True, right_index=True)
// MAGIC diff_df['delta'] = diff_df['tone_y'] - diff_df['tone_x']
// MAGIC 
// MAGIC # visualize Barclays diff compare to industry average
// MAGIC plt.figure(figsize=(20,10))
// MAGIC 
// MAGIC # plot raw data as well as 7 days moving average for better interpretation
// MAGIC plt.plot(diff_df.index, diff_df.delta, color='lightblue', label='')
// MAGIC plt.plot(diff_df.index, diff_df.rolling(window=7).mean().delta, color='dodgerblue', label='sentiment benchmark')
// MAGIC plt.legend(loc='upper left', frameon=False)
// MAGIC plt.axhline(0, linewidth=.5, color='grey')
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %md
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_sentiment.png">

// COMMAND ----------

// MAGIC %md
// MAGIC With a (un-normalised) score of 0.9 (average of daily difference), Barclays positively deviate from the industry by +0.9 in average, indicative of good ESG score

// COMMAND ----------

// DBTITLE 0,Scoring any organization
// MAGIC %sql
// MAGIC 
// MAGIC -- we stored the sum of tone and count of articles to enable AVG operations on different slices of data
// MAGIC -- create view at an organisation level
// MAGIC CREATE OR REPLACE TEMPORARY VIEW organisation_day AS (
// MAGIC   SELECT organisation, theme, date, SUM(tone) / SUM(total) AS tone, SUM(total) AS total FROM esg.gdelt_gold
// MAGIC   GROUP BY organisation, theme, date
// MAGIC );
// MAGIC 
// MAGIC -- we stored the sum of tone and count of articles to enable AVG operations on different slices of data
// MAGIC -- create a view across all organisations
// MAGIC CREATE OR REPLACE TEMPORARY VIEW industry_day AS (
// MAGIC   SELECT date, theme, SUM(tone) / SUM(total) AS tone, SUM(total) AS total FROM esg.gdelt_gold
// MAGIC   GROUP BY date, theme
// MAGIC );
// MAGIC 
// MAGIC -- our crude ESG score is the average difference between organisation sentiment vs. industries
// MAGIC -- we apply heuristic filters to remove noise from GDELT, only looking at clear connections
// MAGIC DROP TABLE IF EXISTS esg.scores;
// MAGIC CREATE TABLE esg.scores USING delta AS
// MAGIC SELECT
// MAGIC   t.organisation,
// MAGIC   t.theme,
// MAGIC   SUM(t.total) AS total,
// MAGIC   COUNT(*) AS days,
// MAGIC   AVG(t.diff) AS esg
// MAGIC FROM (
// MAGIC   SELECT 
// MAGIC     o.date, 
// MAGIC     o.organisation, 
// MAGIC     o.tone - i.tone AS diff, 
// MAGIC     o.total,
// MAGIC     o.theme
// MAGIC   FROM organisation_day o
// MAGIC   JOIN industry_day i
// MAGIC   ON o.date = i.date AND o.theme = i.theme
// MAGIC ) t
// MAGIC GROUP BY t.organisation, t.theme
// MAGIC HAVING days > 300 AND total > 1000; 
// MAGIC 
// MAGIC SELECT organisation, theme, esg FROM esg.scores 
// MAGIC ORDER BY esg 
// MAGIC DESC;

// COMMAND ----------

// MAGIC %md
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_scores.png">

// COMMAND ----------

// MAGIC %md
// MAGIC **We are no longer limited to the handful of FSIs** we do have an ESG report for and can capture an internal ESG score for each and every organisation (or entity from an NLP standpoint) mentioned in news articles (~ 60,000 entities), being public or private companies.

// COMMAND ----------

// MAGIC %md
// MAGIC ## `STEP4`: Detecting ESG influence
// MAGIC 
// MAGIC Using graph analytics, we want to learn connections across businesses and the influence they may have to an ESG rating. [Page Rank](https://en.wikipedia.org/wiki/PageRank) is a common technique used to identify nodes importance in large network. Democratized for web indexing, it can be applied in various setup to detect influential nodes. In this example, we use a variant of Page Rank, **Personalised Page Rank**, where we want to identify influencial nodes relative to our core companies we would like to score. As "influencer" nodes, these important connections will strongly contribute to an ESG coverage for a given organisation. 
// MAGIC 
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_pagerank.png" alt="logical_flow" width="500">

// COMMAND ----------

// MAGIC %md
// MAGIC Using [Graphframes](https://graphframes.github.io/graphframes/docs/_site/index.html), we can easily create a network of companies sharing financial news articles. The more companies are mentioned altogether, the stronger their link will be (edge weight). This graph will allow us to find companies importance relative to our core FSIs we would like to assess, where each connection will contribute to an ESG score - positively or negatively - proportional to its importance. 

// COMMAND ----------

// DBTITLE 0,Create our nodes dataframe
import org.apache.spark.sql.functions._

// Companies will be considered as vertex in our network, edge will contain the number of shared news articles
val nodes = spark
  .read
  .table("esg.scores")
  .select(col("organisation").as("id"))
  .distinct()

// COMMAND ----------

// DBTITLE 1,Create company relationships
import org.apache.spark.sql.functions._

// GDELT has nasty habit to categorize united states or european as organisations
// We can also remove nodes we know are common, such as reuters
// This list is obviously not exhaustive and may be tweaked depending on your strategy
val blacklist = spark.sparkContext.broadcast(Set("united states", "european union", "reuters"))

// Given mentions of multiple organisations within a given URL, build all combinations as tuples
// Graph will be undirected so we register both directions - doubling our graph size!
val buildTuples = udf((xs: Seq[String]) => {
  val organisations = xs.filter(x => !blacklist.value.contains(x))
  organisations.flatMap(x1 => {
    organisations.map(x2 => {
      (x1, x2)
    })
  }).toSeq.filter({ case (x1, x2) =>
    x1 != x2 // remove self edges
  })
})

// build organisations tuples
// Our graph follows a power of law distributions in term of edge weights
// more than 90% of the connections have no more than 100 articles in common
// Reducing the number of edges in our graph from 51,679,930 down to 61,143 using 200 filter
val edges = spark.read.table("esg.gdelt_silver")
  .groupBy("url")
  .agg(collect_list(col("organisation")).as("organisations"))
  .withColumn("tuples", buildTuples(col("organisations")))
  .withColumn("tuple", explode(col("tuples")))
  .withColumn("src", col("tuple._1"))
  .withColumn("dst", col("tuple._2"))
  .groupBy("src", "dst")
  .agg(sum(lit(1)).as("relationship"))
  .filter(col("relationship") > 200)

display(edges)

// COMMAND ----------

// DBTITLE 0,Create our Graph object
import org.graphframes.GraphFrame
val esgGraph = GraphFrame(nodes, edges).cache()
println("Number of nodes : " + esgGraph.vertices.count()) //2,611
println("Number of edges : " + esgGraph.edges.count()) //107,894

// COMMAND ----------

// DBTITLE 0,Consider our FSIs as Landmarks
// Whether we run a personalised page rank or shortest path, we do so relative to our core FSI we would like to score
// These nodes will be considered as landmarks for the following graph analytics
// Let's make sure any of our FSIs are included in our graph
val landmarks = esgGraph
  .vertices
  .select("id")
  //`isin` takes a vararg, not a list, so expand our list to args
  .filter(col("id").isin(organisationAltNames.keys.toArray:_*))
  .rdd
  .map(_.getAs[String]("id"))
  .collect

// COMMAND ----------

// MAGIC %md
// MAGIC The [depth](https://www.sciencedirect.com/science/article/pii/S0022000077800329) of a graph is the maximum of all its shortest paths. To put it another way, it as how many hops at most do you need to reach two seperate companies of our network. In our case, we want to limit our network to at most 4 connections to our FSI nodes. To do so, we run a [shortestpath](https://en.wikipedia.org/wiki/Shortest_path_problem) algorithm first using our FSIs as 'landmarks'. This returns a dataframe of vertices (companies) with their associated distances to each of our landmark (that we can filter for distance < 5)

// COMMAND ----------

// DBTITLE 0,Limit our graph depth to max 4
// As our network can be fairly big, we want to first filter nodes we know would not contribute much to ESG score
// Either because they are too "far away" (using shortest path)
// Or not reachable from our ESG nodes (connected component)
val shortestPaths = esgGraph
  .shortestPaths
  .landmarks(landmarks)
  .run()

// Either way, we run a shortest path algorithm and filter for maximum 5 hops from our core FSIs 
// Note that we chose 5 as a starting point, can be confirmed by looking at distribution of distances
val filterDepth = udf((distances: Map[String, Int]) => {
  distances.values.exists(distance => distance < 5)
})

// By applying this filter upfront, we reduced number of edges by 2
// Filtering upfront will allow us to use personalised page rank with more iterations and faster 
val esgDenseGraph = GraphFrame(shortestPaths, edges).filterVertices(filterDepth(col("distances"))).cache()
println("Number of nodes : " + esgDenseGraph.vertices.count()) //2,308
println("Number of edges : " + esgDenseGraph.edges.count()) //54,150

// COMMAND ----------

// MAGIC %md
// MAGIC With our graph filtered to maximum 4 hops, we can afford to be more greedy with page rank algorithm by increasing the number of iterations required to better estimate company's connections relative importances. We will use a personalized page rank algorithm, natively supported by `graphframe` (and underlying `GraphX`).

// COMMAND ----------

// DBTITLE 0,Run a personalised page rank to find relation importance
val prNodes = esgDenseGraph
  .parallelPersonalizedPageRank
  .resetProbability(0.15)
  // with our graph reduced to max 4 hops, we run 100 iterations to better estimate importance
  .maxIter(500)
  // interestingly, Graphframes complains if not type of `Array[Any]`
  .sourceIds(landmarks.asInstanceOf[Array[Any]])
  .run()

// COMMAND ----------

// DBTITLE 0,Get connections importance
import org.apache.spark.ml.linalg.Vector

// Page rank returns output as a vector containing personalised page rank score for each landmark
// We extract the relevant organisation and score from the returned vector
val landmarksB = spark.sparkContext.broadcast(landmarks)
val importances = udf((pr: Vector) => {
  pr.toArray.zipWithIndex.map({ case (importance, id) =>
    (landmarksB.value(id), importance)
  })
})

// We explode our page rank importances into tuple of [organisation <importance> connection]
val connections = prNodes
  .vertices
  .withColumn("importances", importances(col("pageranks")))
  .withColumn("importance", explode(col("importances")))
  .select(
    col("importance._1").as("organisation"),
    col("id").as("connection"),
    col("importance._2").as("importance")
  )

// COMMAND ----------

// DBTITLE 0,Get a weighted ESG contribution based on connection importance
// We join the page rank score (importance) with ESG internal score for each connection
val connectionsContributions = spark
  .read
  .table("esg.scores")
  .withColumnRenamed("organisation", "connection")
  .join(connections, List("connection"))
  .select("organisation", "connection", "theme", "esg", "importance")

// Save our results back to delta
connectionsContributions
  .write
  .mode("overwrite")
  .format("delta")
  .saveAsTable("esg.connections")

// COMMAND ----------

// MAGIC %md
// MAGIC We can directly visualize the top 100 influential nodes to a specific business (in this case Barclays PLC) as per below graph. Without any surprise, Barclays is well connected with most of our core FSIs (such as JP Morgan Chase, Goldman Sachs or Credit Suisse), but also to the Security Exchange Commission, Federal Reserve and International Monetary Fund. Further down this distribution, we find public and private companies such as Huawei, Chevron, Starbucks or Johnson and Johnson. Strongly or loosely related, directly or indirectly connected, all these businesses (or entities from an NLP standpoint) could theoretically affect Barclays ESG performance, either positively or negatively, and as such impact Barclays reputation.

// COMMAND ----------

// DBTITLE 0,Show JPMC important connections
// MAGIC %sql
// MAGIC SELECT connection, importance FROM esg.connections
// MAGIC WHERE organisation = 'jp morgan chase'
// MAGIC AND connection != 'jp morgan chase'
// MAGIC ORDER BY importance DESC
// MAGIC LIMIT 200

// COMMAND ----------

// MAGIC %md
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_influence.png">

// COMMAND ----------

// MAGIC %md
// MAGIC Due to the nature of news analytics, it is not surprising to observe news publishing companies (such as Thomson Reuters or Bloomberg) or social networks (Facebook, Twitter) as strongly connected organisations. Not reflecting the true connections of a given business but rather explained by a simple co-occurrence in news articles, we should consider filtering them out prior to our page rank process by removing nodes with a high degree of connections. However, this additional noise seems constant across our FSIs and as such does not seem to disadvantage one organisation over another. By combining our ESG score captured earlier with the importance of each of these entities, it becomes easy to apply a weighted average on the “JPMC network” where each business contributes to Barclays ESG score proportionally to its relative importance. We call this approach a **propagated weighted ESG score**. 

// COMMAND ----------

// DBTITLE 1,Compute weighted ESG score
// weighted ESG as SUM[organisation_esg x organisation_importance] / SUM[organisation_importance]
val scoresNorm = spark.read.table("esg.connections")
  .withColumn("weightedEsg", col("esg") * col("importance"))
  .groupBy("organisation", "theme")
  .agg(
    sum("weightedEsg").as("totalWeightedEsg"),
    sum("importance").as("totalImportance")
  )
  .withColumn("weightedEsg", col("totalWeightedEsg") / col("totalImportance"))
  .select(col("organisation"), col("theme"), col("weightedEsg").as("esg"))

// persist PW-ESG scores to delta for MI/BI purposes
scoresNorm
  .write
  .mode("overwrite")
  .format("delta")
  .saveAsTable("esg.scores_fsi")

// Display weighted scores
display(spark.read.table("esg.scores_fsi"))

// COMMAND ----------

// MAGIC %md
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_scores_norm.png">

// COMMAND ----------

// MAGIC %md
// MAGIC We observe the negative or positive influence of any company’s network using a word cloud visualization. In the picture below, we show the negative influence (entities contributing negatively to ESG relative to) for jp morgan chase. 

// COMMAND ----------

// DBTITLE 1,The "JPMC" detractors network
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.sql import functions as F
// MAGIC import matplotlib.pyplot as plt
// MAGIC from wordcloud import WordCloud
// MAGIC import numpy as np
// MAGIC import random
// MAGIC 
// MAGIC organisation = 'jp morgan chase'
// MAGIC 
// MAGIC # retrieve all connections for a given organisation
// MAGIC connections = spark.read.table("esg.connections") \
// MAGIC     .filter(F.col("theme") == 'E').filter(F.col("organisation") == organisation) \
// MAGIC     .toPandas().set_index('connection')[['importance', 'esg']]
// MAGIC 
// MAGIC # retrieve organisation ESG score
// MAGIC esg = connections.loc[organisation].esg
// MAGIC 
// MAGIC # get all companies contributing negatively
// MAGIC detractors = connections[connections['esg'] < esg]
// MAGIC 
// MAGIC # create a dictionary for each detractor with esg influence
// MAGIC detractors_importance = dict(zip(detractors.index, detractors.importance))
// MAGIC 
// MAGIC # build a wordcloud object
// MAGIC detractors_wc = WordCloud(
// MAGIC       background_color="white",
// MAGIC       max_words=5000, 
// MAGIC       width=600, 
// MAGIC       height=400, 
// MAGIC       contour_width=3, 
// MAGIC       contour_color='steelblue'
// MAGIC   ).generate_from_frequencies(detractors_importance)
// MAGIC 
// MAGIC # plot wordcloud
// MAGIC plt.figure(figsize=(10, 10))
// MAGIC plt.imshow(detractors_wc)
// MAGIC plt.axis('off')

// COMMAND ----------

// MAGIC %md
// MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_gdelt_detractors.png">

// COMMAND ----------

// MAGIC %md
// MAGIC Using news analytics, we demonstrated how to compare sentiment across multiple themes and accross industries in order to derive an internal ESG score. And although this score could be a crude approach to ESG, the key message was to show how to leverage graph analytics to understand the connection importance and their negative or positive contribution. In real life, one would need to augment this framework with the internal data they have about their different investments in order to build stronger connections and extract similar patterns before a news has been made public, hence mitigating serious reputation risks upfront.
// MAGIC 
// MAGIC ---
// MAGIC + <a href="$./00_esg_context">STAGE0</a>: Home page
// MAGIC + <a href="$./01_esg_report">STAGE1</a>: Using NLP to extract key ESG initiatives PDF reports
// MAGIC + <a href="$./02_esg_scoring">STAGE2</a>: Introducing a novel approach to ESG scoring using graph analytics
// MAGIC + <a href="$./03_esg_market">STAGE3</a>: Applying ESG to market risk calculations
// MAGIC ---

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
// MAGIC 
// MAGIC | library                                | description             | license    | source                                              |
// MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
// MAGIC | PyPDF2                                 | PDF toolkit.            | CUSTOM     | http://mstamy2.github.io/PyPDF2/                    |
// MAGIC | graphframes:graphframes:0.8.1          | Graph library           | Apache2    | https://github.com/graphframes/graphframes          |
// MAGIC | Yfinance                               | Yahoo finance           | Apache2    | https://github.com/ranaroussi/yfinance              |
// MAGIC | com.aamend.spark:gdelt:3.0             | GDELT wrapper           | Apache2    | https://github.com/aamend/spark-gdelt               |
// MAGIC | PyLDAvis                               | LDA visualizer          | MIT        | https://github.com/bmabey/pyLDAvis                  |
// MAGIC | Gensim                                 | Topic modelling         | L-GPL2     | https://radimrehurek.com/gensim/                    |
// MAGIC | Wordcloud                              | Visualization library   | MIT        | https://github.com/amueller/word_cloud              |

// COMMAND ----------


