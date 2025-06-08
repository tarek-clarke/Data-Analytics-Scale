from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
import boto3

# Clean output S3 path before writing
bucket = "tarekclarke"
prefix = "customer/trusted/"
s3 = boto3.resource('s3')
bucket_obj = s3.Bucket(bucket)
bucket_obj.objects.filter(Prefix=prefix).delete()

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

customer_landing = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://tarekclarke/customer/landing/"]},
    format="json"
)

customer_df = customer_landing.toDF()
print(f"Raw count: {customer_df.count()}")

customer_trusted_df = customer_df.filter(
    (customer_df.shareWithResearchAsOfDate.isNotNull()) & (customer_df.shareWithResearchAsOfDate > 0)
)
print(f"Filtered count: {customer_trusted_df.count()}")

customer_trusted_df = customer_trusted_df.dropDuplicates(["serialNumber"])
print(f"Deduplicated count: {customer_trusted_df.count()}")

customer_trusted = DynamicFrame.fromDF(customer_trusted_df, glueContext, "customer_trusted")

glueContext.write_dynamic_frame.from_options(
    frame=customer_trusted,
    connection_type="s3",
    connection_options={
        "path": "s3://tarekclarke/customer/trusted/",
        "mode": "overwrite"
    },
    format="json"
)
