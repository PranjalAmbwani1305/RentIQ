import re
import numpy as np
import pandas as pd
from config import cfg

FEATURE_NAMES = [
    "city_enc", "bhk", "size", "bathroom",
    "furnishing_enc", "floor_ratio", "size_per_bhk",
    "bath_per_bhk", "area_type_enc", "tenant_enc",
    "log_size", "bhk_bath_interaction",
]


def _cat(val: str, cats: list) -> int:
    try:
        return cats.index(str(val))
    except ValueError:
        return len(cats)


def _parse_floor(val: str) -> tuple[float, float]:
    val = str(val).lower().strip()
    if "basement" in val:
        return 0.0, 1.0
    if val.startswith("ground"):
        m = re.search(r"out of (\d+)", val)
        return 0.0, float(m.group(1)) if m else 1.0
    m = re.match(r"(\d+)\s+out of\s+(\d+)", val)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 1.0


def build_features(row: dict) -> np.ndarray:
    floor_n, total_f = _parse_floor(row.get("Floor", "1 out of 3"))
    size = max(float(row.get("Size", 800)), 1.0)
    bhk  = max(float(row.get("BHK", 2)), 1.0)
    bath = max(float(row.get("Bathroom", 2)), 1.0)

    return np.array([
        _cat(row.get("City", ""),              cfg.CITIES),
        bhk,
        size,
        bath,
        _cat(row.get("Furnishing Status", ""), cfg.FURNISH),
        floor_n / total_f if total_f > 0 else 0.0,
        size / bhk,
        bath / bhk,
        _cat(row.get("Area Type", ""),         cfg.AREA_T),
        _cat(row.get("Tenant Preferred", ""),  cfg.TENANTS),
        np.log1p(size),
        bhk * bath,
    ], dtype=np.float32)


def build_features_df(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([build_features(row) for row in df.to_dict("records")])


def listing_to_vector(row: dict) -> list[float]:
    floor_n, total_f = _parse_floor(row.get("Floor", "1 out of 3"))
    size = max(float(row.get("Size", 800)), 1.0)
    bhk  = max(float(row.get("BHK", 2)), 1.0)
    bath = max(float(row.get("Bathroom", 2)), 1.0)

    def norm_cat(val, cats):
        try:
            return cats.index(str(val)) / max(len(cats) - 1, 1)
        except ValueError:
            return 0.5

    v = np.array([
        norm_cat(row.get("City", ""),              cfg.CITIES),
        min(bhk  / 6.0, 1.0),
        min(size / 5000.0, 1.0),
        min(bath / 6.0, 1.0),
        norm_cat(row.get("Furnishing Status", ""), cfg.FURNISH),
        floor_n / total_f if total_f > 0 else 0.0,
        min((size / bhk) / 2000.0, 1.0),
        min(bath / bhk, 1.0),
        norm_cat(row.get("Area Type", ""),         cfg.AREA_T),
        norm_cat(row.get("Tenant Preferred", ""),  cfg.TENANTS),
        min(np.log1p(size) / 10.0, 1.0),
        min((bhk * bath) / 36.0, 1.0),
    ], dtype=np.float32)

    norm = np.linalg.norm(v)
    return (v / norm if norm > 0 else v).tolist()
