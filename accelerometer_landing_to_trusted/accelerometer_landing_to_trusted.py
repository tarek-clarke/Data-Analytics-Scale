from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
import boto3

# Clean output S3 path before writing
bucket = "tarekclarke"
prefix = "accelerometer/trusted/"

s3 = boto3.resource('s3')
bucket_obj = s3.Bucket(bucket)
bucket_obj.objects.filter(Prefix=prefix).delete()

# Initialize Spark and Glue contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Read customer_trusted data from S3
customer_trusted = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/customer/trusted/"]},
    format="json"
)
customer_trusted_df = customer_trusted.toDF()

# Read accelerometer landing data from S3
accelerometer_landing = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/accelerometer/landing/"]},
    format="json"
)
accelerometer_df = accelerometer_landing.toDF()

# Normalize and trim join keys (case-insensitive)
customer_trusted_df = customer_trusted_df.withColumn("email", F.lower(F.trim(customer_trusted_df.email)))
accelerometer_df = accelerometer_df.withColumn("user", F.lower(F.trim(accelerometer_df.user)))

# Deduplicate customer_trusted on email
customer_trusted_df = customer_trusted_df.dropDuplicates(["email"])

# Debug: Print counts to verify deduplication
print("customer_trusted unique emails:", customer_trusted_df.select("email").distinct().count())
print("accelerometer_landing rows:", accelerometer_df.count())

# Join: Only keep accelerometer readings for customers in customer_trusted
accelerometer_trusted_df = accelerometer_df.join(
    customer_trusted_df,
    accelerometer_df.user == customer_trusted_df.email,
    "inner"
)

# Debug: Print result count
print("accelerometer_trusted rows:", accelerometer_trusted_df.count())

# Only keep accelerometer columns in the output
accelerometer_trusted_df = accelerometer_trusted_df.select(accelerometer_df.columns)

# Convert to DynamicFrame
accelerometer_trusted = DynamicFrame.fromDF(accelerometer_trusted_df, glueContext, "accelerometer_trusted")

# Write to Trusted Zone (overwrite mode)
glueContext.write_dynamic_frame.from_options(
    frame=accelerometer_trusted,
    connection_type="s3",
    connection_options={
        "path": "s3://tarekclarke/accelerometer/trusted/",
        "mode": "overwrite"
    },
    format="json"
)
