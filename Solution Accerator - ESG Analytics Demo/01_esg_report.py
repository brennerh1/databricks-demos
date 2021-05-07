# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/databricks_fsi_white.png width=600px>

# COMMAND ----------

# MAGIC %md
# MAGIC # ESG - reports
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
# MAGIC Financial services organisations are now facing more and more pressure from their shareholders to disclose more information about their environmental, social and governance strategies. Typically released on their websites on a yearly basis as a form of a PDF document, companies communicate on their key ESG initiatives across multiple themes such as how they value their employees, clients or customers, how they positively contribute back to society or even how they reduce  (or commit to reduce) their carbon emissions. Consumed by third parties agencies, these reports are usually consolidated and benchmarked across industries to create ESG metrics. In this notebook, we would like to programmatically access 40+ ESG reports from top tier financial services institutions and learn key ESG initiatives across different topics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dependencies
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. You need to attach this notebook to a cluster running the Databricks standalone 7.X **ML runtime** cluster with the following libraries installed

# COMMAND ----------

# MAGIC %pip install PyPDF2==1.26.0 gensim==3.8.3 nltk==3.5 pyLDAvis==2.1.2 wordcloud

# COMMAND ----------

# MAGIC %md
# MAGIC In addition to those dependencies, we'll make use of existing trained models for nltk (for pre-processing steps). These models would need to be accessible from every worker, hence downloaded to a distributed file storage (mounted under `/dbfs`)

# COMMAND ----------

# DBTITLE 1,Download NLTK model to distributed filesystem
import nltk
nltk.download('wordnet', download_dir="/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/wordnet")
nltk.download('punkt', download_dir="/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/punkt")

# model can be loaded as follows
# nltk.data.path.append('/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/wordnet')
# nltk.data.path.append('/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/punkt')

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1`: Extract ESG reports
# MAGIC In this section, we manually search for publicly available ESG reports from top tier FSIs ([example](https://home.barclays/content/dam/home-barclays/documents/citizenship/ESG/Barclays-PLC-ESG-Report-2019.pdf)). As of today, I am not aware of any central repository that would consolidate all these reports across companies and industries, so we have to provide all URLs to download ESG reports from for specific companies. As all PDFs may be of different formats, we have to spend a lot of time consolidating reports into well defined statements, then extracting grammatically valid sentences using `spacy`. Although our dataset is relatively small, loading and executing spacy models is an expensive process. We leverage the `pandasUDF` paradigm to load models only once so that our process can easily scale for a larger collection of ESG documents across all your investments.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://cdn.onlinewebfonts.com/svg/img_260551.png" width=50/>
# MAGIC Should you need to apply this framework to a different industry, make sure to replace URLs with reports specific to your industry.

# COMMAND ----------

# DBTITLE 0,Add URLs of ESG reports to train model against
import pandas as pd

esg_urls_rows = [
  ['wells fargo', 'https://www08.wellsfargomedia.com/assets/pdf/about/corporate-responsibility/environmental-social-governance-report.pdf'],
  ['barclays', 'https://home.barclays/content/dam/home-barclays/documents/citizenship/ESG/Barclays-PLC-ESG-Report-2019.pdf'],
  ['jp morgan chase', 'https://impact.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/jpmc-cr-esg-report-2019.pdf'],
  ['morgan stanley', 'https://www.morganstanley.com/pub/content/dam/msdotcom/sustainability/Morgan-Stanley_2019-Sustainability-Report_Final.pdf'],
  ['us bancorp', 'https://www.usbank.com/annual-report/2019/pdf/2019_USBancorp_ESG.pdf'],
  ['pnc', 'https://www.pnc.com/content/dam/pnc-com/pdf/aboutpnc/CSR/PNC_2018_CSR_Report.pdf'],
  ['goldman sachs', 'https://www.goldmansachs.com/what-we-do/sustainable-finance/documents/reports/2019-sustainability-report.pdf'],
  ['hsbc', 'https://www.hsbc.com/-/files/hsbc/our-approach/measuring-our-impact/pdfs/190408-esg-update-april-2019-eng.pdf'],
  ['citi', 'https://www.citigroup.com/citi/about/esg/download/2019/Global-ESG-Report-2019.pdf'],
  ['td bank', 'https://www.td.com/document/PDF/corporateresponsibility/2018-ESG-Report.pdf'],
  ['bank of america', 'http://investor.bankofamerica.com/annual-reports-proxy-statements/2019_Annual_Report'],
  ['rbc', 'https://www.rbc.com/community-social-impact/_assets-custom/pdf/2019-ESG-Report.PDF'],
  ['macquarie', 'https://www.macquarie.com/assets/macq/investor/reports/2020/sections/Macquarie-Group-FY20-ESG.pdf'],
  ['lloyds', 'https://www.lloydsbankinggroup.com/globalassets/documents/investors/2020/2020feb20_lbg_esg_approach.pdf'],
  ['santander', 'https://www.santander.co.uk/assets/s3fs-public/documents/2019_santander_esg_supplement.pdf'],
  ['bluebay', 'https://www.bluebay.com/globalassets/documents/bluebay-annual-esg-investment-report-2018.pdf'],
  ['lasalle', 'https://www.lasalle.com/documents/ESG_Policy_2019.pdf'],
  ['riverstone', 'https://www.riverstonellc.com/media/1196/riverstone_esg_report.pdf'],
  ['aberdeen standard', 'https://www.standardlifeinvestments.com/RI_Report.pdf'],
  ['apollo', 'https://www.apollo.com/~/media/Files/A/Apollo-V2/documents/apollo-2018-esg-summary-annual-report.pdf'],
  ['bmogan', 'https://www.bmogam.com/gb-en/intermediary/wp-content/uploads/2019/02/cm16148-esg-profile-and-impact-report-2018_v33_digital.pdf'],
  ['vanguard', 'https://personal.vanguard.com/pdf/ISGESG.pdf'],
  ['ruffer', 'https://www.ruffer.co.uk/-/media/Ruffer-Website/Files/Downloads/ESG/2018_Ruffer_report_on_ESG.pdf'],
  ['northern trust', 'https://cdn.northerntrust.com/pws/nt/documents/fact-sheets/mutual-funds/institutional/annual-stewardship-report.pdf'],
  ['hermes investments', 'https://www.hermes-investment.com/ukw/wp-content/uploads/sites/80/2017/09/Hermes-Global-Equities-ESG-Dashboard-Overview_NB.pdf'],
  ['abri capital', 'http://www.abris-capital.com/sites/default/files/Abris%20ESG%20Report%202018.pdf'],
  ['schroders', 'https://www.schroders.com/en/sysglobalassets/digital/insights/2019/pdfs/sustainability/sustainable-investment-report/sustainable-investment-report-q2-2019.pdf'],
  ['lazard', 'https://www.lazardassetmanagement.com/docs/-m0-/54142/LazardESGIntegrationReport_en.pdf'],
  ['credit suisse', 'https://www.credit-suisse.com/pwp/am/downloads/marketing/br_esg_capabilities_uk_csam_en.pdf'],
  ['coller capital', 'https://www.collercapital.com/sites/default/files/Coller%20Capital%20ESG%20Report%202019-Digital%20copy.pdf'],
  ['cinven', 'https://www.cinven.com/media/2086/81-cinven-esg-policy.pdf'],
  ['warburg pircus', 'https://www.warburgpincus.com/content/uploads/2019/07/Warburg-Pincus-ESG-Brochure.pdf'],
  ['exponent', 'https://www.exponentpe.com/sites/default/files/2020-01/Exponent%20ESG%20Report%202018.pdf'],
  ['silverfleet capital', 'https://www.silverfleetcapital.com/media-centre/silverfleet-esg-report-2020.pdf'],
  ['kkr', 'https://www.kkr.com/_files/pdf/KKR_2018_ESG_Impact_and_Citizenship_Report.pdf'],
  ['cerberus', 'https://www.cerberus.com/media/2019/07/Cerberus-2018-ESG-Report_FINAL_WEB.pdf'],
  ['standard chartered', 'https://av.sc.com/corp-en/others/2018-sustainability-summary2.pdf'],
]

# create a Pandas dataframe of ESG report URLs
esg_urls_df = pd.DataFrame(esg_urls_rows, columns=['company', 'url'])
esg_df = spark.createDataFrame(esg_urls_df).repartition(10)

# COMMAND ----------

# DBTITLE 1,Extract ESG pdf content
import requests
import PyPDF2
import io

from pyspark.sql import functions as F
from pyspark.sql.functions import udf

@udf("string")
def extract_content_udf(url):
  try:
    response = requests.get(url)
    open_pdf_file = io.BytesIO(response.content)
    pdf = PyPDF2.PdfFileReader(open_pdf_file)  
    text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
    return "\n".join(text)
  except:
    # some PDFs may be malformed or encrypted
    return None

# COMMAND ----------

# DBTITLE 1,Extract statements
import string
import re
import nltk

from pyspark.sql.functions import pandas_udf, PandasUDFType

def remove_non_ascii(text):
  printable = set(string.printable)
  return ''.join(filter(lambda x: x in printable, text))

def extract_statements(text):
  
  # remove non ASCII characters
  text = remove_non_ascii(text)
  
  lines = []
  prev = ""
  for line in text.split('\n'):
    # aggregate consecutive lines where text may be broken down
    # only if next line starts with a space or previous does not end with a dot.
    if(line.startswith(' ') or not prev.endswith('.')):
        prev = prev + ' ' + line
    else:
        # new paragraph
        lines.append(prev)
        prev = line
        
  # don't forget left-over paragraph
  lines.append(prev)

  # clean paragraphs from extra space, unwanted characters, urls, etc.
  # best effort clean up, consider a more versatile cleaner
  sentences = []
  
  for line in lines:
      # removing header number
      line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
      # removing trailing spaces
      line = line.strip()
      # words may be split between lines, ensure we link them back together
      line = re.sub(r'\s?-\s?', '-', line)
      # remove space prior to punctuation
      line = re.sub(r'\s?([,:;\.])', r'\1', line)
      # ESG contains a lot of figures that are not relevant to grammatical structure
      line = re.sub(r'\d{5,}', r' ', line)
      # remove mentions of URLs
      line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
      # remove multiple spaces
      line = re.sub(r'\s+', ' ', line)
      # remove multiple dot
      line = re.sub(r'\.+', '.', line)
      
      # split paragraphs into well defined sentences using nltk
      for part in nltk.sent_tokenize(line):
        sentences.append(str(part).strip())

  return sentences


@pandas_udf('array<string>', PandasUDFType.SCALAR_ITER)
def extract_statements_udf(content_series_iter):
  
  # load nltk model
  nltk.data.path.append('/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/punkt')
  
  # clean and tokenize a batch of PDF text content 
  for content_series in content_series_iter:
    yield content_series.map(extract_statements)

# COMMAND ----------

# DBTITLE 1,Extract statements
esg_corpus = (
  esg_df
    .withColumn("content", extract_content_udf(F.col("url")))
    .filter(F.col("content").isNotNull())
    .withColumn("statement", F.explode(extract_statements_udf(F.col("content"))))
    .select("company", "url", "statement")
    .filter(F.length(F.col("statement")) > 50)
)

esg_corpus.write.format("delta").mode("overwrite").saveAsTable("esg.csr")
display(spark.read.table("esg.csr"))

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_urls.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2`: Learn ESG topics
# MAGIC 
# MAGIC In this section, we apply [latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to learn topics descriptive to CSR reports. We want to be able to better understand and eventually sumarize complex CSR reports into a specific ESG related themes (such as 'valuing employees'). Note that this section is highly interactive and requires multiple iterations based on LDA output.

# COMMAND ----------

# DBTITLE 1,Retrieve data from Delta Lake
esg_corpus = spark.read.table("esg.csr")

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://cdn.onlinewebfonts.com/svg/img_260551.png" width=50/>
# MAGIC Should you need to apply this framework to different industry, make sure to add relevant stopwords

# COMMAND ----------

# DBTITLE 1,Define specific stop words
import gensim
from gensim.parsing.preprocessing import STOPWORDS

# context specific keywords not to include in topic modelling
org_stop_words = [
  'plc', 'group', 'target',
  'track', 'capital', 'holding',
  'report', 'annual',
  'esg', 'bank', 'report', 'csr',
  'environment', 'social', 'governance',
  'corporate', 'responsibility',
  'million', 'billion',
]

# add company names as stop words
organisations = esg_corpus.select("company").distinct().toPandas().company
for organisation in organisations:
    for t in organisation.split(' '):
        org_stop_words.append(t)

# our list contains all english stop words + companies names + specific keywords
stop_words = STOPWORDS.union(org_stop_words)

# COMMAND ----------

# DBTITLE 1,Lemmatize content
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from pyspark.sql.functions import pandas_udf, PandasUDFType
from gensim.utils import simple_preprocess

def lemmatize(text):
  
  results = []
  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer()
  for token in simple_preprocess(text):
    stem = stemmer.stem(lemmatizer.lemmatize(token))
    if (len(stem) > 3):
      results.append(stem)

  return ' '.join(results)

@pandas_udf('string', PandasUDFType.SCALAR_ITER)
def lemmatize_udf(content_series_iter):
  
  # specify where to find NLTK model
  nltk.data.path.append('/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/wordnet')
  
  # stem and lemmatize sentences
  for content_series in content_series_iter:
    yield content_series.map(lemmatize)

# COMMAND ----------

# DBTITLE 1,Extract sentences
from pyspark.sql import functions as F

esg = (
  esg_corpus
    .withColumn("lemma", lemmatize_udf(F.col("statement")))
    .select("company", "statement", "lemma")
    .filter(F.length(F.col("lemma")) > 100)
).toPandas()

display(esg)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_statements.png">

# COMMAND ----------

# MAGIC %md
# MAGIC The challenge is to extract good quality of topics that are clear, segregated and meaningful. This depends heavily on the quality of text preprocessing (above) and the strategy of finding the optimal number of topics (below). Although our models can be tuned much further using e.g. `learning_decay`, we want to keep this framework really simple to use in order accomodate most of use cases in a pseudo seamless manner. Therefore, we will be choosing number of topics based on business context rather than empirical evidence (such as model log likelihood). For that purpose, we'll be evaluating the quality of topics using visualizations such as `pyLDAvis`

# COMMAND ----------

# DBTITLE 1,Train a LDA model
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

import mlflow
import mlflow.sklearn
import json

with mlflow.start_run(run_name='topic_modeling') as run:

  vectorizer = CountVectorizer(
    stop_words = stop_words,     # our list of stop words ESG specific
    ngram_range = (1,1)          # training on words, or bi-grams or trigrams 
  )

  # train model
  lda = LatentDirichletAllocation(
    random_state = 42, 
    learning_method='online',
    learning_decay = 0.3,        # learning decay
    evaluate_every = -1,         # compute perplexity every n iters, default: Don't
    n_components = 9             # Number of topics
  )
  
  mlflow.log_param("n_components", 9)
  mlflow.log_param("learning_decay", 0.3)
  
  # train pipeline
  pipeline = make_pipeline(vectorizer, lda)
  pipeline.fit(esg.lemma)

  # log model
  mlflow.sklearn.log_model(pipeline, 'pipeline')
  
  # Mlflow run ID
  lda_run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://cdn.onlinewebfonts.com/svg/img_260551.png" width=50/>
# MAGIC After reviewing the generated topics using below visualization, you may decide to refine the LDA model with different parameters. 

# COMMAND ----------

# DBTITLE 1,Machine learning topics
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, vectorizer.transform(esg.lemma), vectorizer, mds='tsne')

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_lda.png">

# COMMAND ----------

# MAGIC %md
# MAGIC The **left panel** visualise the topics as circles in the two-dimensional plane whose centres are determined by computing the [Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) between topics, and then by using multidimensional scaling to project the inter-topic distances onto two dimensions. Each topic’s overall prevalence is encoded using the areas of the circles.
# MAGIC 
# MAGIC The **right panel** depicts a horizontal bar chart whose bars represent the individual terms that are the most useful for interpreting the currently selected topic on the left. A pair of overlaid bars represent both the corpus-wide frequency of a given term as well as the topic-specific frequency of the term.

# COMMAND ----------

# DBTITLE 1,Explore ESG policies
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# We ensure relevance of our topics using simple wordcloud visualisation
def word_cloud(model, tf_feature_names, index):
    
    imp_words_topic=""
    comp = model.components_[index]
    tfs = ['_'.join(t.split(' ')) for t in tf_feature_names]
    vocab_comp = zip(tfs, comp)
    sorted_words = sorted(vocab_comp, key = lambda x:x[1], reverse=True)[:200]
    
    for word in sorted_words:
        imp_words_topic = imp_words_topic + " " + word[0]
    
    return WordCloud(
        background_color="white",
        width=300, 
        height=300, 
        contour_width=2, 
        contour_color='steelblue'
    ).generate(imp_words_topic)
    
topics = len(lda.components_)
tf_feature_names = vectorizer.get_feature_names()
fig = plt.figure(figsize=(40, 40 * topics / 3))

# Display wordcloud for each extracted topic
for i, topic in enumerate(lda.components_):
    ax = fig.add_subplot(topics, 3, i + 1)
    wordcloud = word_cloud(lda, tf_feature_names, i)
    ax.imshow(wordcloud)
    ax.axis('off')

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_wordcloud.png">

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://cdn.onlinewebfonts.com/svg/img_260551.png" width=50/>
# MAGIC For different industries, or a new set of PDF documents, you may have to look at both wordcloud and pyLDAvis visualizations above and rename below topics accordingly.

# COMMAND ----------

# DBTITLE 1,Name our topics
topic_names = [
  "ethical finance",
  "driving performance",
  "energy transition",
  "developing communities",
  "risk management",
  "focusing on people",
  "code of conduct",
  "valuing employees",
  "impact investing"
]

# COMMAND ----------

import mlflow
pipeline = mlflow.sklearn.load_model("models:/esg_lda/production")

# COMMAND ----------

# DBTITLE 1,Describing documents against ESG policies
import numpy as np
import pandas as pd

# score our original dataset to attach topic distribution to each ESG statement
transformed = pipeline.transform(esg.lemma)

# find principal topic from distribution...
a = [topic_names[np.argmax(distribution)] for distribution in transformed]

# ... with associated probability
b = [np.max(distribution) for distribution in transformed]

# consolidate LDA output into a handy dataframe 
df2 = pd.DataFrame(zip(a, b, transformed), columns=['topic', 'probability', 'probabilities'])
esg_group = pd.concat([esg, df2], axis=1).drop(["lemma", "probabilities"], axis=1)
display(esg_group.sort_values(by="probability", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_summarize.png">

# COMMAND ----------

# DBTITLE 1,Compare companies core ESG initiatives
# create a simple pivot table of number of occurence of each topic across organisations
esg_group_filter = esg_group[esg_group['probability'] > 0.5]
esg_focus = pd.crosstab(esg_group_filter.company, esg_group_filter.topic)

# scale topic frequency between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

# normalize pivot table
esg_focus_norm = pd.DataFrame(scaler.fit_transform(esg_focus), columns=esg_focus.columns)
esg_focus_norm.index = esg_focus.index

# plot heatmap, showing main area of focus for each FSI across topics we learned
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(esg_focus_norm, annot=False, linewidths=.5, cmap='Blues')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/esg_scoring/esg_compare.png">

# COMMAND ----------

# MAGIC %md
# MAGIC This matrix offers a quick lense through ESG strategies across FSIs. Whilst some companies would focus more on employees and peole, some put more focus on ethical investments.

# COMMAND ----------

# DBTITLE 1,Compute industry ESG focus average
# extract quantiles for each category
industry = esg_focus.quantile(np.arange(0.1, 1, 0.1))
industry['quantile'] = industry.index

# serialize industry average
industry_path = "/dbfs/tmp/industry.json"
industry_json = industry.to_json(orient='records')
with open(industry_path, "w") as f:
  f.write(industry_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP3`: Creating APIs through `pyFunc`
# MAGIC In previous section, we've seen how to wrap business logic into a `pyFunc` object. In this section, we'll be extending our logic further to automatically scrape, score and benchmark ESG initiative from URLs via a simple HTTP call. 

# COMMAND ----------

# DBTITLE 1,Export scraping + benchmark business logic
class ESGBenchmarkAPI(mlflow.pyfunc.PythonModel):
    
  def __init__(self, pipeline):
    self.pipeline = pipeline  
  
  
  def load_context(self, context): 
    import nltk
    import json
    nltk.data.path.append(context.artifacts['wordnet'])
    nltk.data.path.append(context.artifacts['punkt'])
    
    with open(context.artifacts['topic_path']) as f:
      topics = pd.DataFrame(json.load(f), columns=["topic"])
      topics['id'] = topics.index
      self.topics = topics
      
    with open(context.artifacts['industry_path'], "r") as f:
      self.industry = pd.read_json(f.read())
  
  
  def _download_content(self, url):
    import requests
    import io
    import PyPDF2
    response = requests.get(url)
    open_pdf_file = io.BytesIO(response.content)
    pdf = PyPDF2.PdfFileReader(open_pdf_file)  
    text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
    return "\n".join(text)
  
  
  def _extract_statements(self, text):
    import string
    import re
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    lines = []
    prev = ""
    for line in text.split('\n'):
      if(line.startswith(' ') or line.startswith('-') or not prev.endswith('.')):
          prev = prev + ' ' + line
      else:
          lines.append(prev)
          prev = line

    lines.append(prev)
    sentences = []
    for line in lines:
        line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
        line = line.strip()
        line = re.sub(r'\s?-\s?', '-', line)
        line = re.sub(r'\s?([,:;\.])', r'\1', line)
        line = re.sub(r'\d{5,}', r' ', line)
        line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'\.+', '.', line)
        for part in nltk.sent_tokenize(line):
          sentences.append(str(part).strip())

    return sentences
  
  
  def _lemmatize(self, text):
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from gensim.utils import simple_preprocess
    results = []
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for token in simple_preprocess(text):
      stem = stemmer.stem(lemmatizer.lemmatize(token))
      if (len(stem) > 3):
        results.append(stem)

    return ' '.join(results)
   

  def _quantile(self, topic, score):
    co = self.industry[[topic, 'quantile']]
    fi = co[co[topic] < score]
    if (fi.shape[0] == 0):
      return 0.1
    else:
      return fi['quantile'].iloc[-1]
    

  def predict(self, context, df):
    import pandas as pd
    import numpy as np
    
    # our UDF expects 1 record, the URL to scrape
    url = df.url.iloc[0]
    
    # scrape text content
    content = self._download_content(url)
    
    # extract sentences
    statements = self._extract_statements(content)
    
    # lemmatize statements
    lemmas = [self._lemmatize(statement) for statement in statements]
    
    # extract distribution for each statement
    distributions = self.pipeline.transform(lemmas)
    principal_topics = [np.argmax(distribution) for distribution in distributions]
    principal_probas = [np.max(distribution) for distribution in distributions]
    df = pd.DataFrame(zip(principal_topics, principal_probas), columns=['id', 'probability'])
    df['lemma'] = 1
    
    # Filter dataframe for most descriptive statemnts (at least 50% described by one topic)
    mask = df['probability'] > 0.5
    esg_df = df.loc[mask].merge(self.topics, on='id')[['topic', 'lemma']]

    # Compare focus with industry average
    count_df = esg_df.groupby(['topic']).count()
    
    scores = []
    for topic in count_df.index:
      score = int(100 * self._quantile(topic, count_df.loc[topic].lemma))
      scores.append([topic, score])
      
    return pd.DataFrame(scores, columns=['topic', 'quantile'])

# COMMAND ----------

# DBTITLE 1,Create pyfunc model
with mlflow.start_run(run_name='esg_scraper_api'):

  import json
  topic_path = '/dbfs/tmp/topics.json'
  with open(topic_path, "w") as f:
    f.write(json.dumps(topic_names))
  
  conda_env = mlflow.pyfunc.get_default_conda_env()
  conda_env['dependencies'][2]['pip'] += ['PyPDF2==1.26.0']
  conda_env['dependencies'][2]['pip'] += ['nltk==3.5']
  conda_env['dependencies'][2]['pip'] += ['gensim==3.8.3']
  conda_env['dependencies'][2]['pip'] += ['scikit-learn']
  
  artifacts = {
    'industry_path': '/dbfs/tmp/industry.json',
    'topic_path': '/dbfs/tmp/topics.json',
    'punkt': '/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/punkt',
    'wordnet': '/dbfs/Users/antoine.amend@databricks.com/nlp/nltk/wordnet'
  }
  
  mlflow.pyfunc.log_model(
    'pipeline', 
    python_model=ESGBenchmarkAPI(pipeline), 
    conda_env=conda_env,
    artifacts=artifacts
    )
  
  api_run_id = mlflow.active_run().info.run_id
  print(api_run_id)

# COMMAND ----------

# DBTITLE 1,Register model
client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/pipeline".format(api_run_id)
model_name = "esg_lda_benchmark"
result = mlflow.register_model(model_uri, model_name)
version = result.version
print(version)

# COMMAND ----------

# MAGIC %md
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


