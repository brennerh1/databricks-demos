// Databricks notebook source
def get_kinesis_endpoint(): String = {
  "https://kinesis.us-west-2.amazonaws.com"
}

// COMMAND ----------

def get_partition_info(fullPath: String): Seq[com.databricks.backend.daemon.dbutils.FileInfo] = {
  // fullPath should be any DBFS mounted location 
  var input = fullPath
  // if fullPath ends with a "/" then we remove it to do some string adjustments 
  if ( fullPath.endsWith("/") ) { 
    input = fullPath.slice(0, fullPath.length - 1)
  }
  // Grab the last folder of the path 
  val endPath = input.split('/').last
  // Grab the prefix until the folder in question 
  val prefix = input.slice(0, input.length - endPath.length)
  // Filter and grab get the dbutils output for that given folder 
  val inputDir = dbutils.fs.ls(prefix).filter(x => x.name == endPath + "/")
  get_partition_helper(inputDir)
}

def get_partition_helper(paths: Seq[com.databricks.backend.daemon.dbutils.FileInfo]): Seq[com.databricks.backend.daemon.dbutils.FileInfo] = {
  val debug = false
  var ret = paths.filter(x => !x.name.startsWith("_"))
  var next = paths.flatMap(y => dbutils.fs.ls(y.path).filter(x => !x.name.startsWith("_")))
  if (debug) {
    ret.foreach(x => println(x.path))
  }
  if (next.filter(x => x.name contains ".parquet").length == 0) { 
    ret = get_partition_helper(paths.flatMap(y => dbutils.fs.ls(y.path).filter(x => !x.name.startsWith("_"))))
  } 
  ret 
}
