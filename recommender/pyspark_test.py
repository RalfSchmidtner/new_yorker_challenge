from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType
from pyspark.sql import Row
from pyspark.sql.functions import *
import os

## SETUP
spark = SparkSession.builder.appName('session').getOrCreate()
schema = StructType([
    StructField("name", StringType(), False),
    StructField("age", LongType(), False),
])

rows = [
    Row('Ralf', 26),
    Row('Frank', 28),
    Row('Ute', 54),
    Row('Bernhard', 58)
]

df_schmidtner = spark.createDataFrame(rows, schema)
df_schmidtner.coalesce(1)

dir = os.getcwd()
df_schmidtner.write.json(os.path.join(dir, 'my_df'))
print("saved dataframe to {}".format(dir))
