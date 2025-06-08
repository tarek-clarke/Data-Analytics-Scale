import boto3
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# --- Clean output S3 prefix before writing new data ---
bucket = "tarekclarke"
prefix = "customer/curated/"
s3 = boto3.resource('s3')
bucket_obj = s3.Bucket(bucket)
bucket_obj.objects.filter(Prefix=prefix).delete()

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# --- Read customer trusted data ---
customer_trusted = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/customer/trusted/"]},
    format="json"
)
customer_trusted_df = customer_trusted.toDF()

# --- Normalize email ---
customer_trusted_df = customer_trusted_df.withColumn(
    "email", F.lower(F.trim(customer_trusted_df.email))
)

# --- Filter customers who consented to research ---
customer_consented_df = customer_trusted_df.filter(
    (F.col("shareWithResearchAsOfDate").isNotNull()) &
    (F.col("shareWithResearchAsOfDate") > 0)
)

# --- Deduplicate before join (strict, all columns) ---
customer_consented_df = customer_consented_df.dropDuplicates()

# --- Read accelerometer data ---
accelerometer = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/accelerometer/landing/"]},
    format="json"
)
accelerometer_df = accelerometer.toDF()

# --- Normalize user column ---
accelerometer_df = accelerometer_df.withColumn(
    "user", F.lower(F.trim(accelerometer_df.user))
)

# --- Deduplicate accelerometer users (strict, all columns) ---
accelerometer_df = accelerometer_df.dropDuplicates()
accelerometer_users = accelerometer_df.select("user").distinct()

# --- Join customers who consented with accelerometer users ---
customer_curated_df = customer_consented_df.join(
    accelerometer_users,
    customer_consented_df.email == accelerometer_users.user,
    "inner"
).drop("user")

# --- Deduplicate after join ---

# Option 1: Deduplicate on email only (uncomment to use)
customer_curated_df = customer_curated_df.dropDuplicates(["email"])

# Option 2: Deduplicate on all columns (uncomment to use)
# customer_curated_df = customer_curated_df.dropDuplicates()

# --- Convert to DynamicFrame for Glue write ---
customer_curated = DynamicFrame.fromDF(customer_curated_df, glueContext, "customer_curated")

# --- Write to Curated Zone (overwrite mode) ---
glueContext.write_dynamic_frame.from_options(
    frame=customer_curated,
    connection_type="s3",
    connection_options={
        "path": "s3://tarekclarke/customer/curated/",
        "mode": "overwrite"
    },
    format="json"
)
