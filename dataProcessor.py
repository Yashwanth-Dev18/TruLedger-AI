import pyspark
import warnings
from pyspark.sql import functions as F
from pyspark.sql.functions import radians, sin, cos, atan2, sqrt
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os

os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ["PYSPARK_PYTHON"] = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"

# Initializing Spark session  
spark = SparkSession.builder \
    .appName("TruLedger") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()


# Load dataset
df = spark.read.csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Raw/financeRecords.csv", header=True, inferSchema=True)


# # =============================================
# # 🔍 FEATURE ENGINEERING & DATA PREPROCESSING
# # =============================================

# DROPPED COLUMNS THAT ARE NOT USEFUL FOR ANOMALY DETECTION
df = df.drop("index", "merchant", "first", "last", "gender", "street", "city", "zip", 
              "city_pop", "trans_num", "unix_time")

# EXTRACTING TIME FROM DATE_TIME COLUMN
df = df.withColumn("trans_date_trans_time", F.date_format(F.to_timestamp("trans_date_trans_time"), "HH").cast("int"))
# rename the column from trans_date_trans_time to txn_time
df = df.withColumnRenamed("trans_date_trans_time", "txn_time")


# MERCH/TXN CATEGORIES
# Replace ',' with ' -' in category names
df = df.withColumn("category", F.regexp_replace(F.col("category"), ",", " -"))
Categories_list = [ctg['category'] for ctg in df.select("category").distinct().collect()]
# Creating columns for each category (ONE HOT ENCODING)
for category in Categories_list:
    df = df.withColumn(f"TXNctg_{category}", F.when(F.col("category") == category, 1).otherwise(0))
# Dropping the original category column
df = df.drop("category")



# STATE COLUMNS
States_list = [st['state'] for st in df.select("state").distinct().collect()]
# Creating columns for each state (ONE HOT ENCODING)
for state in States_list:
    df = df.withColumn(f"state_{state}", F.when(F.col("state") == state, 1).otherwise(0))
# Dropping the original state column
df = df.drop("state")



# JOB COLUMNS

# Reload both dataframes separately to avoid overwriting
jobcat_df = spark.read.csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/job_categories.csv", header=True, inferSchema=True)

# Cleaning 'job' column: replace ',' with ' -'
df = df.withColumn("job", F.regexp_replace(F.col("job"), ",", " -"))

# Creating (job, category) pairs from job_categories.csv
job_category_pairs = []
for category in jobcat_df.columns:
    jobs_in_cat = jobcat_df.select(category).where(F.col(category).isNotNull()).distinct().rdd.flatMap(lambda x: x).collect()
    job_category_pairs.extend([(job, category) for job in jobs_in_cat])

# Convert list of pairs to DataFrame
job_category_map_df = spark.createDataFrame(job_category_pairs, ["job", "job_category"])

# Joins finance records with job-category map
df = df.join(job_category_map_df, on="job", how="left")

# One-hot encoding job_category
unique_job_categories = [row["job_category"] for row in job_category_map_df.select("job_category").distinct().collect()]
for category in unique_job_categories:
    df = df.withColumn(
        f"JOBctg_{category.replace(' ', '_')}",
        F.when(F.col("job_category") == category, 1).otherwise(0)
    )
# Dropping the helper column
df = df.drop("job_category")
df = df.drop("job")



# DOB COLUMNS

# Extracts the year from dob
df = df.withColumn("dob_year", F.year(F.to_date("dob", "yyyy-MM-dd")))

# Creating decade-based columns dynamically
decades = list(range(1920, 2010, 10))  # 1920s, 1930s, ..., 2000s
for start_year in decades:
    col_name = f"dob_{str(start_year)[2:]}s"
    df = df.withColumn(
        col_name,
        F.when((F.col("dob_year") >= start_year) & (F.col("dob_year") < start_year + 10), 1).otherwise(0)
    )
# Dropping the helper year column
df = df.drop("dob_year")
df = df.drop("dob")



# AVERAGE DISTANCE BETWEEN CUSTOMER AND MERCHANT PER USER
df = df.withColumn("lat1_rad", radians("lat"))
df = df.withColumn("lon1_rad", radians("long"))  
df = df.withColumn("lat2_rad", radians("merch_lat"))
df = df.withColumn("lon2_rad", radians("merch_long"))

df = df.withColumn("dlat", F.col("lat2_rad") - F.col("lat1_rad"))
df = df.withColumn("dlon", F.col("lon2_rad") - F.col("lon1_rad"))

df = df.withColumn("a", 
    sin(F.col("dlat") / 2) ** 2 + 
    cos(F.col("lat1_rad")) * cos(F.col("lat2_rad")) * 
    sin(F.col("dlon") / 2) ** 2
)
df = df.withColumn("c", 2 * atan2(sqrt(F.col("a")), sqrt(1 - F.col("a"))))
df = df.withColumn("distance", F.col("c") * 6371)



# USER BEHAVIORAL METRICS
# SINGLE AGGREGATION FOR ALL METRICS (INCLUDING DISTANCE)
user_metrics_df = df.groupBy("cc_num").agg(
    F.round(F.avg("amt"), 2).alias("avg_txn_amt"),  # Average transaction amount per user
    F.round(F.stddev("amt"), 2).alias("stddev_txn_amt"),  # Standard deviation of transaction amount per user
    F.round(F.avg("txn_time"), 2).alias("avg_txn_time"),  # Average transaction time(hour) per user
    F.round(F.avg("distance"), 2).alias("avg_merchant_distance")  # Average distance b/w user and merchant per user
)
df = df.join(user_metrics_df, on="cc_num", how="left")



# Dropping temporary columns used for distance calculation
df = df.drop("lat1_rad", "lon1_rad", "lat2_rad", "lon2_rad", "dlat", "dlon", "a", "c", "distance")
# Dropping more columns that are not needed anymore
df = df.drop("cc_num", "lat", "long", "merch_lat", "merch_long")



# =======================================================
#  📌 WRITING PROCESSED DATA TO CSV USING APACHE HADOOP
# =======================================================

# Reordering columns to have 'is_fraud' as the first column
other_columns = [col for col in df.columns if col != "is_fraud"]
# Create new order with is_fraud first
new_column_order = ["is_fraud"] + other_columns
# Reorder the DataFrame
df = df.select(*new_column_order)


#Let's make an action to write the tranformed data to new csv file
# Function to clean up APACHE Spark CSV output
def clean_spark_output(temp_dir, final_path):
    import os
    import shutil
    
    # Find the CSV part file
    for file in os.listdir(temp_dir):
        if file.startswith("part-") and file.endswith(".csv"):
            # moved the file to final destination
            shutil.move(os.path.join(temp_dir, file), final_path)
            # rename the moved file to TrainingSet.csv
            os.rename(os.path.join(final_path, file), os.path.join(final_path, "TrainingSet.csv"))

            break
    
    # Remove temporary directory
    shutil.rmtree(temp_dir)

# Calling the function to write cleaned CSV, making it ready for ML model
temp_dir = "c:/Users/hp/LNU/TruLedger-AI/temp_output"
df.coalesce(1).write.csv(temp_dir, header=True, mode="overwrite")
clean_spark_output(temp_dir, "c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed")




# =====================
# 🔺 TESTING 🔺   
# =====================

# unique_count = df.select("lat").distinct().count()
# print(f"\nUnique customer latitudes: {unique_count}")

# unique_count = df.select("long").distinct().count()
# print(f"\nUnique customer longitudes: {unique_count}")

# unique_count = df.select("merch_lat").distinct().count()
# print(f"\nUnique merchant latitudes: {unique_count}")

# unique_count = df.select("merch_long").distinct().count()
# print(f"\nUnique merchant longitudes: {unique_count}")

# unique_cc_count = df.select("cc_num").distinct().count()
# print(f"\nUnique credit card numbers: {unique_cc_count}")

# unique_dob_count = df.select("dob").distinct().count()
# print(f"\nUnique date of birth values: {unique_dob_count}")

# unique_count = df.select("first").distinct().count()
# print(f"\nUnique first name values: {unique_count}")

# unique_count = df.select("state").distinct().count()
# print(f"\nUnique states: {unique_count}")

# Unique_names = df.select("first", "last").distinct().count()
# print(f"\nUnique full names: {Unique_names}")

# jobs = df.select("job").distinct()
# jobs.show(unique_count, truncate=False)
# fraud_count = df.filter(df.is_fraud == 1).count()
# print(f"\nFraudulent transactions: {fraud_count}")


spark.stop()
