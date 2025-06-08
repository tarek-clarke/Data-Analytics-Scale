CREATE EXTERNAL TABLE IF NOT EXISTS step_trainer_landing (
    sensorreadingtime string,
    serialnumber string,
    distancefromobject double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://tarekclarke/step_trainer/landing/'
TBLPROPERTIES ('has_encrypted_data'='false');
