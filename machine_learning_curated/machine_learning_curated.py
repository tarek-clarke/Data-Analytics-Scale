import boto3
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F

# Clean output S3 path before writing
bucket = "tarekclarke"
prefix = "machine_learning/curated/"
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

# Read step trainer trusted data from S3
step_trainer_trusted = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/step_trainer/trusted/"]},
    format="json"
)
step_trainer_df = step_trainer_trusted.toDF()

# Read accelerometer trusted data from S3
accelerometer_trusted = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/accelerometer/trusted/"]},
    format="json"
)
accelerometer_df = accelerometer_trusted.toDF()

# Standardize column names to lowercase and strip whitespace
customer_curated_df = customer_curated_df.toDF(*[c.lower().strip() for c in customer_curated_df.columns])
step_trainer_df = step_trainer_df.toDF(*[c.lower().strip() for c in step_trainer_df.columns])
accelerometer_df = accelerometer_df.toDF(*[c.lower().strip() for c in accelerometer_df.columns])

# Normalize join keys and time columns
customer_curated_df = customer_curated_df.withColumn("email", F.lower(F.trim(customer_curated_df["email"])))
step_trainer_df = step_trainer_df.withColumn("sensorreadingtime", F.trim(step_trainer_df["sensorreadingtime"]))
accelerometer_df = accelerometer_df.withColumn("user", F.lower(F.trim(accelerometer_df["user"])))
accelerometer_df = accelerometer_df.withColumn("timestamp", F.trim(accelerometer_df["timestamp"]))

# Prepare for join: create a common column for joining accelerometer and customer
accel_with_email = accelerometer_df.withColumnRenamed("user", "join_email")
customer_with_email = customer_curated_df.withColumnRenamed("email", "join_email")

# Join accelerometer to customer on email/user
accel_customer = accel_with_email.join(
    customer_with_email.select("join_email").dropDuplicates(["join_email"]),
    on="join_email",
    how="inner"
)

# Join with step trainer on timestamp/sensorreadingtime
result = accel_customer.join(
    step_trainer_df,
    accel_customer["timestamp"] == step_trainer_df["sensorreadingtime"],
    "inner"
)

# Optionally, prefix accelerometer columns to avoid name clashes
step_cols = step_trainer_df.columns
accel_cols = [c for c in accelerometer_df.columns if c not in ["user", "timestamp"]]
for c in accel_cols:
    result = result.withColumnRenamed(c, f"accel_{c}")

# Select output columns
output_cols = step_cols + ["join_email", "timestamp"] + [f"accel_{c}" for c in accel_cols]
result = result.select(*output_cols)

# Convert to DynamicFrame
machine_learning_curated = DynamicFrame.fromDF(result, glueContext, "machine_learning_curated")

# Write to Curated Zone (overwrite mode)
glueContext.write_dynamic_frame.from_options(
    frame=machine_learning_curated,
    connection_type="s3",
    connection_options={
        "path": "s3://tarekclarke/machine_learning/curated/",
        "mode": "overwrite"
    },
    format="json"
)
