import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg, setup_logging

log = setup_logging("RentIQ.Spark", cfg.LOGS_DIR)

SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, StringType, IntegerType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
    SPARK_AVAILABLE = True
except ImportError:
    log.warning("PySpark not installed. Spark pipeline is disabled.")


def build_spark_session() -> Optional[object]:
    if not SPARK_AVAILABLE:
        return None
    try:
        spark = (
            SparkSession.builder
            .appName(cfg.SPARK_APP_NAME)
            .master(cfg.SPARK_MASTER)
            .config("spark.executor.memory",  cfg.SPARK_EXECUTOR_MEMORY)
            .config("spark.driver.memory",    cfg.SPARK_DRIVER_MEMORY)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.local.dir",        "/tmp/spark_rentiq")
            .config("spark.ui.enabled",       "false")
            .config("spark.driver.maxResultSize", "512m")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        log.info(f"Spark session ready — version={spark.version}")
        return spark
    except Exception as e:
        log.error(f"Spark session failed: {e}")
        return None


def ingest_csv(spark, csv_path: Path = cfg.DATA_PATH):
    log.info(f"Ingesting CSV via Spark: {csv_path}")
    sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
    sdf = sdf.withColumn("price_per_sqft",
        (F.col("Rent") / F.greatest(F.col("Size"), F.lit(1))).cast(DoubleType())
    )
    sdf = sdf.filter(
        ~F.lower(F.col("Floor")).contains("basement")
    ).filter(
        F.col("price_per_sqft").between(2, 500)
    )
    log.info(f"Ingested {sdf.count()} rows across {sdf.rdd.getNumPartitions()} partitions")
    return sdf


def _encode_city(col_val):
    return F.when(col_val == "Mumbai",    0.0)\
            .when(col_val == "Delhi",     1.0)\
            .when(col_val == "Bangalore", 2.0)\
            .when(col_val == "Chennai",   3.0)\
            .when(col_val == "Hyderabad", 4.0)\
            .when(col_val == "Kolkata",   5.0)\
            .otherwise(6.0)


def _encode_furnish(col_val):
    return F.when(col_val == "Furnished",      0.0)\
            .when(col_val == "Semi-Furnished",  1.0)\
            .when(col_val == "Unfurnished",     2.0)\
            .otherwise(3.0)


def _encode_area_type(col_val):
    return F.when(col_val == "Super Area",  0.0)\
            .when(col_val == "Carpet Area", 1.0)\
            .when(col_val == "Built Area",  2.0)\
            .otherwise(3.0)


def _encode_tenant(col_val):
    return F.when(col_val == "Bachelors/Family", 0.0)\
            .when(col_val == "Bachelors",          1.0)\
            .when(col_val == "Family",             2.0)\
            .otherwise(3.0)


def _parse_floor_ratio(floor_col):
    return (
        F.when(F.lower(floor_col).contains("basement"), 0.0)
         .when(F.lower(floor_col).startswith("ground"), 0.0)
         .otherwise(
             F.coalesce(
                 F.regexp_extract(floor_col, r"(\d+)\s+out of\s+(\d+)", 1).cast(DoubleType())
                  / F.greatest(
                      F.regexp_extract(floor_col, r"(\d+)\s+out of\s+(\d+)", 2).cast(DoubleType()),
                      F.lit(1.0)
                  ),
                 F.lit(0.0)
             )
         )
    )


def build_features_spark(sdf):
    sdf = sdf.withColumn("city_enc",         _encode_city(F.col("City")).cast(DoubleType()))
    sdf = sdf.withColumn("bhk_d",            F.col("BHK").cast(DoubleType()))
    sdf = sdf.withColumn("size_d",           F.greatest(F.col("Size").cast(DoubleType()), F.lit(1.0)))
    sdf = sdf.withColumn("bathroom_d",       F.greatest(F.col("Bathroom").cast(DoubleType()), F.lit(1.0)))
    sdf = sdf.withColumn("furnishing_enc",   _encode_furnish(F.col("Furnishing Status")).cast(DoubleType()))
    sdf = sdf.withColumn("floor_ratio",      _parse_floor_ratio(F.col("Floor")))
    sdf = sdf.withColumn("size_per_bhk",     (F.col("size_d") / F.greatest(F.col("bhk_d"), F.lit(1.0))))
    sdf = sdf.withColumn("bath_per_bhk",     (F.col("bathroom_d") / F.greatest(F.col("bhk_d"), F.lit(1.0))))
    sdf = sdf.withColumn("area_type_enc",    _encode_area_type(F.col("Area Type")).cast(DoubleType()))
    sdf = sdf.withColumn("tenant_enc",       _encode_tenant(F.col("Tenant Preferred")).cast(DoubleType()))
    sdf = sdf.withColumn("log_size",         F.log1p(F.col("size_d")))
    sdf = sdf.withColumn("bhk_bath_inter",   F.col("bhk_d") * F.col("bathroom_d"))
    sdf = sdf.withColumn("log_rent",         F.log1p(F.col("Rent").cast(DoubleType())))
    thresh = 0.60
    sdf = sdf.withColumn("demand_risk", (F.col("price_per_sqft") >= F.lit(40.0)).cast(DoubleType()))
    return sdf


def run_spark_pipeline(csv_path: Path = cfg.DATA_PATH, out_path: Path = cfg.MODEL_PKL) -> dict:
    spark = build_spark_session()
    if spark is None:
        return {"engine": "unavailable", "note": "PySpark not installed"}

    try:
        sdf = ingest_csv(spark, csv_path)
        sdf = build_features_spark(sdf)

        sdf.cache()
        n_total = sdf.count()
        log.info(f"Total cached rows: {n_total}")

        feature_cols = [
            "city_enc", "bhk_d", "size_d", "bathroom_d",
            "furnishing_enc", "floor_ratio", "size_per_bhk",
            "bath_per_bhk", "area_type_enc", "tenant_enc",
            "log_size", "bhk_bath_inter",
        ]

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler    = StandardScaler(inputCol="raw_features", outputCol="features",
                                   withStd=True, withMean=True)

        train_sdf, test_sdf = sdf.randomSplit([0.85, 0.15], seed=42)
        train_sdf.cache()
        test_sdf.cache()

        log.info("Training GBTRegressor via Spark MLlib...")
        gbt_reg = GBTRegressor(
            featuresCol="features", labelCol="log_rent",
            maxIter=100, maxDepth=5, stepSize=0.05, subsamplingRate=0.8,
        )
        reg_pipeline = Pipeline(stages=[assembler, scaler, gbt_reg])
        reg_model    = reg_pipeline.fit(train_sdf)

        log.info("Training GBTClassifier via Spark MLlib...")
        gbt_clf = GBTClassifier(
            featuresCol="features", labelCol="demand_risk",
            maxIter=80, maxDepth=4, stepSize=0.06, subsamplingRate=0.8,
        )
        clf_pipeline = Pipeline(stages=[assembler, scaler, gbt_clf])
        clf_model    = clf_pipeline.fit(train_sdf)

        reg_preds = reg_model.transform(test_sdf)
        clf_preds = clf_model.transform(test_sdf)

        reg_eval = RegressionEvaluator(labelCol="log_rent", predictionCol="prediction", metricName="rmse")
        clf_eval = BinaryClassificationEvaluator(labelCol="demand_risk", metricName="areaUnderROC")
        rmse     = reg_eval.evaluate(reg_preds)
        auc      = clf_eval.evaluate(clf_preds)
        log.info(f"Spark GBT Regressor RMSE={rmse:.4f} | Classifier AUC={auc:.4f}")

        city_agg = (
            sdf.groupBy("City")
               .agg(
                   F.avg("Rent").alias("avg_rent"),
                   F.median("Rent").alias("median_rent"),
                   F.avg("price_per_sqft").alias("avg_psf"),
                   F.count("*").alias("n_listings"),
               )
               .toPandas()
               .set_index("City")
               .to_dict("index")
        )

        partitions_info = {
            "n_partitions": sdf.rdd.getNumPartitions(),
            "n_total_rows": n_total,
        }

        from sklearn.preprocessing import RobustScaler as SkScaler
        train_pd = train_sdf.select(feature_cols).toPandas()
        sk_scaler = SkScaler()
        sk_scaler.fit(train_pd.values)

        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
        X_tr = sk_scaler.transform(train_pd.values)
        y_tr = train_sdf.select("log_rent").toPandas().values.ravel()
        y_cr = train_sdf.select("demand_risk").toPandas().values.ravel().astype(int)

        sk_reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        sk_clf = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, random_state=42)
        sk_reg.fit(X_tr, y_tr)
        sk_clf.fit(X_tr, y_cr)

        X_te_pd  = test_sdf.select(feature_cols).toPandas()
        X_te_s   = sk_scaler.transform(X_te_pd.values)
        y_te_log = test_sdf.select("log_rent").toPandas().values.ravel()
        y_ce     = test_sdf.select("demand_risk").toPandas().values.ravel().astype(int)

        pred_log  = sk_reg.predict(X_te_s)
        pred_rent = np.expm1(pred_log)
        true_rent = np.expm1(y_te_log)
        mae  = mean_absolute_error(true_rent, pred_rent)
        r2   = r2_score(true_rent, pred_rent)
        acc  = accuracy_score(y_ce, sk_clf.predict(X_te_s))

        city_med = {row["City"]: row["median_rent"] for row in sdf.groupBy("City").agg(F.median("Rent").alias("median_rent")).collect()}

        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "mae": mae, "r2": r2, "clf_acc": acc,
            "spark_rmse": rmse, "spark_auc": auc,
            "n_train": len(X_tr), "n_test": len(X_te_pd),
            "n_partitions": partitions_info["n_partitions"],
            "engine": "spark-mllib",
        }
        artifact = {
            "regressor":     sk_reg,
            "classifier":    sk_clf,
            "scaler":        sk_scaler,
            "city_medians":  city_med,
            "city_agg":      city_agg,
            "partitions":    partitions_info,
            "feature_names": feature_cols,
            "metrics":       metrics,
        }
        with open(out_path, "wb") as f:
            pickle.dump(artifact, f)
        log.info(f"Spark pipeline complete — saved to {out_path}")

        sdf.unpersist()
        train_sdf.unpersist()
        test_sdf.unpersist()
        spark.stop()
        return metrics

    except Exception as e:
        log.error(f"Spark pipeline failed: {e}")
        try:
            spark.stop()
        except Exception:
            pass
        return {"engine": "spark-failed", "note": str(e)}


def get_spark_aggregations(csv_path: Path = cfg.DATA_PATH) -> dict:
    spark = build_spark_session()
    if spark is None:
        return {}
    try:
        sdf = ingest_csv(spark, csv_path)
        sdf = build_features_spark(sdf)
        sdf.cache()

        city_stats = (
            sdf.groupBy("City")
               .agg(
                   F.avg("Rent").alias("avg_rent"),
                   F.min("Rent").alias("min_rent"),
                   F.max("Rent").alias("max_rent"),
                   F.stddev("Rent").alias("std_rent"),
                   F.avg("price_per_sqft").alias("avg_psf"),
                   F.count("*").alias("n_listings"),
               )
               .toPandas()
        )

        bhk_stats = (
            sdf.groupBy("BHK", "City")
               .agg(F.avg("Rent").alias("avg_rent"), F.count("*").alias("n"))
               .filter(F.col("BHK").between(1, 4))
               .toPandas()
        )

        furnish_stats = (
            sdf.groupBy("Furnishing Status")
               .agg(F.avg("Rent").alias("avg_rent"), F.avg("price_per_sqft").alias("avg_psf"))
               .toPandas()
        )

        partition_sizes = (
            sdf.rdd
               .mapPartitionsWithIndex(lambda idx, rows: [(idx, sum(1 for _ in rows))])
               .collect()
        )

        sdf.unpersist()
        spark.stop()

        return {
            "city_stats":     city_stats,
            "bhk_stats":      bhk_stats,
            "furnish_stats":  furnish_stats,
            "partition_sizes": partition_sizes,
            "n_partitions":   len(partition_sizes),
            "total_rows":     int(city_stats["n_listings"].sum()),
        }
    except Exception as e:
        log.error(f"Spark aggregation failed: {e}")
        try:
            spark.stop()
        except Exception:
            pass
        return {}


if __name__ == "__main__":
    log.info("Running Spark training pipeline...")
    metrics = run_spark_pipeline()
    log.info(f"Done: {metrics}")
