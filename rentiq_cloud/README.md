# RentIQ v3 — Production Rental Intelligence Platform

> AI-powered rental analytics combining GBT ensemble, deep learning (TabNet), Apache Spark, and Pinecone vector search.

---

## Architecture

```
rentiq_v3/
├── app.py                      # Streamlit UI (light theme)
├── config/
│   ├── settings.py             # Central config registry
│   └── logging_cfg.py          # Rotating file + console logging
├── core/
│   ├── features.py             # Shared feature engineering (12 features)
│   ├── inference.py            # GBT regressor + classifier ensemble
│   ├── database.py             # Pinecone vector DB + local-cache fallback
│   ├── explainability.py       # XAI — feature attribution + AI insights
│   └── security.py             # JWT, BCrypt, RBAC, rate limiting
├── deep_learning/
│   └── model.py                # RentTabNet: FeatureAttention + ResidualMLP
├── pipeline/
│   └── spark_pipeline.py       # PySpark distributed training + aggregations
├── analytics/
│   └── engine.py               # Pandas analytics engine (Spark-compatible)
├── scripts/
│   ├── train_all.py            # Train GBT + TabNet models
│   └── seed_pinecone.py        # Seed Pinecone with listing vectors
└── data/
    └── House_Rent_Dataset.csv
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill environment config
cp .env.example .env
# Edit .env — add PINECONE_API_KEY and JWT_SECRET

# 3. Train models (auto-trains on first run too)
python scripts/train_all.py

# 4. (Optional) Seed Pinecone
python scripts/seed_pinecone.py

# 5. Launch
streamlit run app.py
```

---

## ML Stack

| Component | Technology | Purpose |
|---|---|---|
| GBT Regressor | scikit-learn | Rent prediction (₹) |
| GBT Classifier | scikit-learn | Demand risk signal |
| RentTabNet | PyTorch | FeatureAttention + Residual MLP |
| Spark GBT | PySpark MLlib | Distributed training at scale |
| Pinecone | Serverless | 12-dim cosine similarity search |
| XAI Engine | Custom | Permutation importance + insights |

---

## Spark Pipeline

Run distributed training with:

```bash
# Install Spark (requires Java 11+)
pip install -r requirements-spark.txt

# Run full Spark pipeline
python pipeline/spark_pipeline.py
```

The Spark pipeline replaces scikit-learn artifacts with MLlib-trained equivalents — no app changes needed.

---

## Deep Learning Model

**RentTabNet** architecture:
- `FeatureAttention`: Multi-head self-attention over feature tokens (4 heads, embed_dim=32)
- Encoder: 3 progressive linear blocks (256→128→64) with LayerNorm + GELU
- 2× Residual blocks with skip connections
- Dual output heads: rent regression (HuberLoss) + risk classification (BCE)

---

## Pinecone Integration

Two serverless indexes (dim=12, cosine):

- **rentiq-listings** — all CSV listings as vectors for similarity search
- **rentiq-predictions** — every user prediction logged as a vector

Queries like *"similar areas with lower rent"* use cosine search with optional rent filters.

---

## Security

- JWT HS256 (8h access / 7d refresh)
- BCrypt password hashing (cost=12)
- Token bucket rate limiting
- RBAC permission bitfields (5 roles)
- Account lockout after 5 failed attempts

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | Optional | Enables vector search (falls back to local cache) |
| `JWT_SECRET` | Yes (prod) | 64-char hex secret for token signing |
| `SPARK_MASTER` | No | Default: `local[*]` |
| `DEBUG` | No | Enables verbose logging |
