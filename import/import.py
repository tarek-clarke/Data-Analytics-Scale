import sys
from awsglue.context import GlueContext
from pyspark.context import SparkContext

sc = SparkContext()
glueContext = GlueContext(sc)

# S3 URIs
s3_customer_path = "s3://tarekclarke/customer/landing/"
s3_step_path = "s3://tarekclarke/step_trainer/landing/"
s3_accelerometer_path = "s3://tarekclarke/accelerometer/landing/"

df_customer = glueContext.spark_session.read.json(s3_customer_path)
df_step = glueContext.spark_session.read.json(s3_step_path)
df_accelerometer = glueContext.spark_session.read.json(s3_accelerometer_path)

# print("All dataframes loaded successfully.")


# Show sample data
df_customer.show(5)
df_step.show(5)
df_accelerometer.show(5)
