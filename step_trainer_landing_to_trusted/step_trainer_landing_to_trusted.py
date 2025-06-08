import boto3
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F

# Clean output S3 path before writing
bucket = "tarekclarke"
prefix = "step_trainer/trusted/"
s3 = boto3.resource('s3')
bucket_obj = s3.Bucket(bucket)
bucket_obj.objects.filter(Prefix=prefix).delete()

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Read curated customer data from S3
customer_curated = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/customer/curated/"]},
    format="json"
)
customer_curated_df = customer_curated.toDF()

# Read step trainer IoT data from S3
step_trainer_landing = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/step_trainer/landing/"]},
    format="json"
)
step_trainer_df = step_trainer_landing.toDF()

# Standardize all column names to lowercase and strip whitespace
customer_curated_df = customer_curated_df.toDF(*[c.lower().strip() for c in customer_curated_df.columns])
step_trainer_df = step_trainer_df.toDF(*[c.lower().strip() for c in step_trainer_df.columns])

# Print columns for debug
print("Customer Curated Columns:", customer_curated_df.columns)
print("Step Trainer Columns:", step_trainer_df.columns)

join_key = "serialnumber"

# Normalize and trim join keys (case-insensitive)
customer_curated_df = customer_curated_df.withColumn(join_key, F.lower(F.trim(customer_curated_df[join_key])))
step_trainer_df = step_trainer_df.withColumn(join_key, F.lower(F.trim(step_trainer_df[join_key])))

# Deduplicate customer_curated on join key
customer_curated_df = customer_curated_df.dropDuplicates([join_key])

# Alias DataFrames for join to avoid ambiguity
step_trainer_df_alias = step_trainer_df.alias("st")
customer_curated_df_alias = customer_curated_df.alias("cc")

# Join using aliases
step_trainer_trusted_df = step_trainer_df_alias.join(
    customer_curated_df_alias,
    step_trainer_df_alias[join_key] == customer_curated_df_alias[join_key],
    "inner"
)

# Only keep step trainer columns in the output
step_trainer_trusted_df = step_trainer_trusted_df.select([F.col(f"st.{col}") for col in step_trainer_df.columns])

# Convert to DynamicFrame
step_trainer_trusted = DynamicFrame.fromDF(step_trainer_trusted_df, glueContext, "step_trainer_trusted")

# Write to Trusted Zone (overwrite mode)
glueContext.write_dynamic_frame.from_options(
    frame=step_trainer_trusted,
    connection_type="s3",
    connection_options={
        "path": "s3://tarekclarke/step_trainer/trusted/",
        "mode": "overwrite"
    },
    format="json"
)
