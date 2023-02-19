#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F

MONGODB_CLOUD_URI = "mongodb://<username>:<password>@127.0.0.1:27017/stackoverflow?authSource=admin"
MONGODB_DATABASE = "stackoverflow"

if __name__ == "__main__":
    spark = SparkSession.builder \
                    .master("local") \
                    .appName("stackoverflow") \
                    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.1.1") \
                    .config("spark.mongodb.read.connection.uri", MONGODB_CLOUD_URI) \
                    .config("spark.mongodb.write.connection.uri", MONGODB_CLOUD_URI) \
                    .getOrCreate()

    sc = spark.sparkContext

    questions_df = spark.read \
                    .format("mongodb") \
                    .option("uri", MONGODB_CLOUD_URI) \
                    .option("collection", "questions") \
                    .load()

    answers_df = spark.read \
                    .format("mongodb") \
                    .option("uri", MONGODB_CLOUD_URI) \
                    .option("collection", "answers") \
                    .load()

    questions_df.printSchema()
    answers_df.printSchema()

    questions_df = questions_df.withColumn("OwnerUserId", F.col("OwnerUserId").cast("integer"))

    questions_df = questions_df.withColumn("CreationDate", F.to_date("CreationDate"))
    questions_df = questions_df.withColumn("ClosedDate", F.to_date("ClosedDate"))

    questions_df.printSchema()

    answers_df = answers_df.withColumn("OwnerUserId", F.col("OwnerUserId").cast("integer"))
    answers_df = answers_df.withColumn("ParentId", F.col("ParentId").cast("integer"))

    answers_df = answers_df.withColumn("CreationDate", F.to_date("CreationDate"))

    answers_df.printSchema()

    lang_pattern = r"(java|python|c\+\+|c\#|(?<![\w.])go(?![.\w])|ruby|javascript|php|(?<![\w.])html(?![.\w])|css|sql)"

    prog_lang_freq = questions_df.withColumnRenamed("Body", "Programing Language") \
                                .withColumn("Programing Language",
                                            F.regexp_extract(F.lower(F.col("Programing Language")),
                                                             lang_pattern,
                                                             0)) \
                                .groupBy("Programing Language") \
                                .count() \
                                .filter(F.col("Programing Language") != "") \
                                .show()

    domain_pattern = r"(?<=http:\/\/)(.*?\.(edu|com|net|org|info|coop|int|co\.uk|co\.us|co\.ca|org\.uk|ac\.uk|uk|us))"

    domain_freq = questions_df.withColumnRenamed("Body", "Domain") \
                            .withColumn("Domain",
                                        F.regexp_extract(F.col("Domain"),
                                                         domain_pattern,
                                                         0)) \
                            .where("trim(Domain) != ''") \
                            .groupBy("Domain") \
                            .count() \
                            .show()

    user_score_by_day = questions_df.select(F.col("OwnerUserId"), F.col("CreationDate"), F.col("Score")) \
                                    .groupBy("OwnerUserId", "CreationDate") \
                                    .agg(F.sum("Score").alias("Total Score")) \
                                    .sort(F.asc("OwnerUserId"), F.asc("CreationDate")) \
                                    .show()

    START = '2008-01-01'
    END = '2009-01-01'

    user_score_by_period = questions_df.select(F.col("OwnerUserId"), F.col("CreationDate"), F.col("Score")) \
                                    .filter(F.col("CreationDate") > START) \
                                    .filter(F.col("CreationDate") < END) \
                                    .groupBy("OwnerUserId", "CreationDate") \
                                    .agg(F.sum("Score").alias("Total Score")) \
                                    .sort(F.desc("Total Score")) \
                                    .show()

    spark.sql("CREATE DATABASE IF NOT EXISTS MY_DB")
    spark.sql("USE MY_DB")

    questions_df.write \
                .bucketBy(10, "Id") \
                .format("parquet") \
                .mode("overwrite") \
                .option("path", r"/content/tmp_questions") \
                .saveAsTable("MY_DB.questions")


    answers_df.write \
            .bucketBy(10, "ParentId") \
            .format("parquet") \
            .mode("overwrite") \
            .option("path", r"/content/tmp_answers") \
            .saveAsTable("MY_DB.answers")

    tmp_df_1 = spark.read.table("MY_DB.questions")
    tmp_df_2 = spark.read.table("MY_DB.answers")

    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    df_renamed = tmp_df_1.withColumnRenamed("Id", "QuestionID")
    df_renamed_1 = df_renamed.withColumnRenamed("OwnerUserId", "QuestionerID")
    df_renamed_2 = df_renamed_1.withColumnRenamed("Score", "QuestionScore")
    df_renamed_3 = df_renamed_2.withColumnRenamed("CreationDate", "QuestionCreationDate")

    join_expr = df_renamed_3.QuestionID == tmp_df_2.ParentId
    df_renamed_3.join(tmp_df_2, join_expr, "inner").show(10)

    result = df_renamed_3.join(tmp_df_2, join_expr, "inner") \
                        .select("QuestionID", "Id") \
                        .groupBy("QuestionID") \
                        .agg(F.count("Id").alias("Total Answers")) \
                        .filter(F.col("Total Answers") > 5) \
                        .sort(F.asc("QuestionID")) \
                        .show()

    # create a temp View on the joined dataframe to later query by sparkSQL
    # select some columns only
    df_renamed_3.join(tmp_df_2, join_expr, "inner") \
                .select("QuestionID", "Id", "OwnerUserId", "Score", "QuestionCreationDate", "CreationDate") \
                .createTempView("needed")

    # find all active users that meet the requirement
    active_users = spark.sql("""
        select OwnerUserId, 
               count(Id) as TotalAnswers, 
               sum(Score) as TotalScore
        from needed
        group by OwnerUserId
        having count(Id) > 50 or sum(Score) > 500
            or OwnerUserId in (select OwnerUserId
                               from needed
                               group by QuestionID, QuestionCreationDate, OwnerUserId, CreationDate
                               having count(Id) > 5 and QuestionCreationDate = CreationDate)

    """).show()

