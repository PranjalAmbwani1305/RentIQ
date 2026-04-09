import logging
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional

from config import cfg, setup_logging

log = setup_logging("RentIQ.Analytics", cfg.LOGS_DIR)


@lru_cache(maxsize=1)
def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(cfg.DATA_PATH)
    df = df[~df["Floor"].str.lower().str.contains("basement", na=False)].copy()
    df["price_per_sqft"] = (df["Rent"] / df["Size"].clip(lower=1)).round(2)
    df = df[df["price_per_sqft"].between(2, 500)].copy()
    return df


def get_dataset() -> pd.DataFrame:
    return _load_raw().copy()


def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("City")
          .agg(
              avg_rent    = ("Rent", "mean"),
              median_rent = ("Rent", "median"),
              min_rent    = ("Rent", "min"),
              max_rent    = ("Rent", "max"),
              std_rent    = ("Rent", "std"),
              avg_psf     = ("price_per_sqft", "mean"),
              n_listings  = ("Rent", "count"),
          )
          .round(0)
          .reset_index()
    )


def bhk_city_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["BHK"].between(1, 4)]
          .groupby(["BHK", "City"])["Rent"]
          .median()
          .reset_index()
    )


def furnish_comparison(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["City", "Furnishing Status"])
          .agg(avg_psf=("price_per_sqft", "median"), n=("Rent", "count"))
          .reset_index()
    )


def top_localities(df: pd.DataFrame, city: str, top_n: int = 10) -> pd.DataFrame:
    return (
        df[df["City"] == city]
          .groupby("Area Locality")
          .agg(avg_rent=("Rent", "mean"), n=("Rent", "count"))
          .sort_values("avg_rent", ascending=False)
          .head(top_n)
          .reset_index()
    )


def demand_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    thresh = df["price_per_sqft"].quantile(0.60)
    df     = df.copy()
    df["high_demand"] = (df["price_per_sqft"] >= thresh).astype(int)
    return (
        df.groupby(["City", "BHK"])["high_demand"]
          .mean()
          .reset_index()
          .rename(columns={"high_demand": "demand_rate"})
    )


def simulate_spark_partitions(df: pd.DataFrame, n_partitions: int = 4) -> list[dict]:
    rows_per   = len(df) // n_partitions
    partitions = []
    for i in range(n_partitions):
        chunk = df.iloc[i * rows_per:(i + 1) * rows_per]
        partitions.append({
            "partition": i,
            "rows":      len(chunk),
            "avg_rent":  round(chunk["Rent"].mean(), 0),
            "cities":    chunk["City"].nunique(),
        })
    return partitions


def get_full_analytics(df: Optional[pd.DataFrame] = None) -> dict:
    if df is None:
        df = get_dataset()
    return {
        "city_summary":      city_summary(df),
        "bhk_matrix":        bhk_city_matrix(df),
        "furnish_comparison": furnish_comparison(df),
        "demand_heatmap":    demand_heatmap(df),
        "partitions":        simulate_spark_partitions(df),
        "total_rows":        len(df),
        "cities":            sorted(df["City"].unique().tolist()),
    }
