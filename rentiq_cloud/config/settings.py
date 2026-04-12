import os
from pathlib import Path

# Load .env for local dev; no-op on Streamlit Cloud
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Streamlit secrets bridge ──────────────────────────────────────────────────
# On Streamlit Cloud, secrets live in st.secrets (injected as env vars too).
# We prefer st.secrets when available so values show up in both paths.
def _secret(key: str, fallback: str = "") -> str:
    """Read from st.secrets first, then env, then fallback."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, fallback)

# ─────────────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent


class Settings:
    BASE_DIR        = BASE
    DATA_PATH       = BASE / "data" / "House_Rent_Dataset.csv"
    ARTIFACTS_DIR   = BASE / "artifacts"
    LOGS_DIR        = BASE / "logs"
    MODEL_PKL       = BASE / "artifacts" / "model.pkl"
    DL_MODEL_PATH   = BASE / "artifacts" / "dl_model.pt"
    SPARK_CHECKPT   = BASE / "artifacts" / "spark_checkpt"

    PINECONE_API_KEY        = _secret("PINECONE_API_KEY")
    PINECONE_ENV            = _secret("PINECONE_ENV", "us-east-1")
    PINECONE_CLOUD          = _secret("PINECONE_CLOUD", "aws")
    IDX_LISTINGS            = _secret("IDX_LISTINGS", "rentiq-listings")
    IDX_PREDICTIONS         = _secret("IDX_PREDICTIONS", "rentiq-predictions")
    VECTOR_DIM              = 12

    JWT_SECRET              = _secret("JWT_SECRET", "dev-secret-change-in-prod")
    JWT_EXPIRY_HOURS        = int(_secret("JWT_EXPIRY_HOURS", "8"))
    JWT_REFRESH_DAYS        = int(_secret("JWT_REFRESH_DAYS", "7"))
    MAX_LOGIN_ATTEMPTS      = int(_secret("MAX_LOGIN_ATTEMPTS", "5"))
    LOCK_DURATION_SECONDS   = int(_secret("LOCK_DURATION_SECONDS", "900"))

    SPARK_MASTER            = _secret("SPARK_MASTER", "local[*]")
    SPARK_APP_NAME          = "RentIQ-Production"
    SPARK_EXECUTOR_MEMORY   = "2g"
    SPARK_DRIVER_MEMORY     = "2g"

    DL_EMBEDDING_DIM        = 32
    DL_HIDDEN_DIMS          = [256, 128, 64]
    DL_DROPOUT              = 0.3
    DL_EPOCHS               = 80
    DL_BATCH_SIZE           = 256
    DL_LR                   = 1e-3

    GBT_N_ESTIMATORS        = 400
    GBT_MAX_DEPTH           = 5
    GBT_LEARNING_RATE       = 0.05
    GBT_SUBSAMPLE           = 0.8

    CITIES   = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]
    FURNISH  = ["Furnished", "Semi-Furnished", "Unfurnished"]
    AREA_T   = ["Super Area", "Carpet Area", "Built Area"]
    TENANTS  = ["Bachelors/Family", "Bachelors", "Family"]

    CITY_MEDIANS = {
        "Mumbai": 76500, "Delhi": 46500, "Bangalore": 45000,
        "Chennai": 25000, "Hyderabad": 26000, "Kolkata": 15000,
    }
    CITY_COLOR = {
        "Mumbai": "#4f8ef7", "Delhi": "#7c5cfa", "Bangalore": "#22d3ee",
        "Chennai": "#34d399", "Hyderabad": "#f59e0b", "Kolkata": "#f87171",
    }

    CACHE_TTL_SECONDS   = 3600
    DATA_CACHE_ROWS     = 50_000
    DEBUG               = _secret("DEBUG", "false").lower() == "true"


cfg = Settings()
