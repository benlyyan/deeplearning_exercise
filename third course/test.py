from pyspark import SparkContext
from pyspark import SparkContext as sc
from pyspark import SparkConf
import os  
os.environ['JAVA_HOME'] = "C:/Java/jre1.8.0_144/bin/java.exe"
os.environ['SPARK_HOME']="D:/bigdata/spark"
os.environ['PYSPARK_SUBMIT_ARGS'] = "--master mymaster --total-executor 2 --conf 'spark.driver.extraJavaOptions=-Dhttp.proxyHost=proxy.mycorp.com-Dhttp.proxyPort=1234 -Dhttp.nonProxyHosts=localhost|.mycorp.com|127.0.0.1 -Dhttps.proxyHost=proxy.mycorp.com -Dhttps.proxyPort=1234 -Dhttps.nonProxyHosts=localhost|.mycorp.com|127.0.0.1 pyspark-shell'"

conf=SparkConf().setAppName("miniProject1").setMaster("local[*]")
sc=SparkContext.getOrCreate(conf)
rdd = sc.parallelize([1,2,3,4,5])
print(rdd)
print(rdd.getNumPartitions())