import uuid
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg, setup_logging
from core.features import listing_to_vector

log = setup_logging("RentIQ.Database", cfg.LOGS_DIR)

_MEM_PREDICTIONS: list = []
_MEM_LISTINGS:    list = []


class RentIQDatabase:
    def __init__(self):
        self._pc        = None
        self._listings  = None
        self._preds     = None
        self.connected  = False
        self._init_pinecone()

    def _init_pinecone(self):
        if not cfg.PINECONE_API_KEY or cfg.PINECONE_API_KEY.startswith("your-"):
            log.warning("PINECONE_API_KEY not set — running in local-cache mode")
            self._seed_memory()
            return
        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
            existing = {i.name for i in pc.list_indexes()}

            for name in (cfg.IDX_LISTINGS, cfg.IDX_PREDICTIONS):
                if name not in existing:
                    log.info(f"Creating Pinecone index: {name}")
                    pc.create_index(
                        name=name,
                        dimension=cfg.VECTOR_DIM,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=cfg.PINECONE_CLOUD, region=cfg.PINECONE_ENV),
                    )
                    time.sleep(10)

            self._listings = pc.Index(cfg.IDX_LISTINGS)
            self._preds    = pc.Index(cfg.IDX_PREDICTIONS)
            self._pc       = pc
            self.connected = True
            log.info("Pinecone connected successfully")
        except Exception as e:
            log.error(f"Pinecone init failed: {e}")
            self._seed_memory()

    def _seed_memory(self):
        global _MEM_LISTINGS
        if _MEM_LISTINGS:
            return
        csv = cfg.DATA_PATH
        if not csv.exists():
            return
        try:
            df = pd.read_csv(csv, nrows=200)
            df = df[~df["Floor"].str.lower().str.contains("basement", na=False)].copy()
            df["price_per_sqft"] = df["Rent"] / df["Size"].clip(lower=1)
            df = df[df["price_per_sqft"].between(2, 500)].copy()
            for _, row in df.iterrows():
                vec = listing_to_vector(row.to_dict())
                _MEM_LISTINGS.append({
                    "id":       str(uuid.uuid4()),
                    "vector":   np.array(vec, dtype=np.float32),
                    "metadata": {
                        "City":          str(row.get("City", "")),
                        "BHK":           int(row.get("BHK", 2)),
                        "Size":          int(row.get("Size", 800)),
                        "Bathroom":      int(row.get("Bathroom", 1)),
                        "Furnishing":    str(row.get("Furnishing Status", "")),
                        "Rent":          int(row.get("Rent", 0)),
                        "Floor":         str(row.get("Floor", "")),
                        "Area_Locality": str(row.get("Area Locality", "")),
                        "Tenant":        str(row.get("Tenant Preferred", "")),
                        "price_per_sqft": round(float(row.get("price_per_sqft", 0)), 2),
                    },
                })
            log.info(f"Memory cache seeded with {len(_MEM_LISTINGS)} listings")
        except Exception as e:
            log.error(f"Memory seed failed: {e}")

    def seed_listings(self, csv_path: Path = cfg.DATA_PATH, batch_size: int = 100):
        if not self.connected:
            log.warning("Not connected to Pinecone; using memory cache")
            return
        try:
            df = pd.read_csv(csv_path)
            df = df[~df["Floor"].str.lower().str.contains("basement", na=False)].copy()
            df["price_per_sqft"] = df["Rent"] / df["Size"].clip(lower=1)
            df = df[df["price_per_sqft"].between(2, 500)].copy()
            log.info(f"Seeding {len(df)} listings into Pinecone...")

            vectors = []
            for _, row in df.iterrows():
                vec = listing_to_vector(row.to_dict())
                vectors.append({
                    "id":       str(uuid.uuid4()),
                    "values":   vec,
                    "metadata": {
                        "City":          str(row.get("City", "")),
                        "BHK":           int(row.get("BHK", 2)),
                        "Size":          int(row.get("Size", 800)),
                        "Bathroom":      int(row.get("Bathroom", 1)),
                        "Furnishing":    str(row.get("Furnishing Status", "")),
                        "Rent":          int(row.get("Rent", 0)),
                        "Floor":         str(row.get("Floor", "")),
                        "Area_Locality": str(row.get("Area Locality", "")),
                        "Tenant":        str(row.get("Tenant Preferred", "")),
                        "price_per_sqft": round(float(row.get("price_per_sqft", 0)), 2),
                    },
                })
                if len(vectors) >= batch_size:
                    self._listings.upsert(vectors=vectors)
                    vectors = []

            if vectors:
                self._listings.upsert(vectors=vectors)
            log.info("Pinecone listings seeded")
        except Exception as e:
            log.error(f"Seed failed: {e}")

    def find_similar(
        self,
        listing: dict,
        top_k: int = 8,
        same_city: bool = False,
        max_rent: Optional[int] = None,
        min_rent: Optional[int] = None,
    ) -> list[dict]:
        query_vec = listing_to_vector(listing)

        if self.connected:
            return self._pinecone_search(query_vec, top_k, same_city, listing, max_rent, min_rent)
        return self._memory_search(query_vec, top_k, same_city, listing, max_rent, min_rent)

    def _pinecone_search(self, query_vec, top_k, same_city, listing, max_rent, min_rent):
        try:
            kwargs = {"vector": query_vec, "top_k": top_k * 2, "include_metadata": True}
            if same_city:
                kwargs["filter"] = {"City": {"$eq": listing.get("City", "")}}
            resp = self._listings.query(**kwargs)
            hits = []
            for m in resp.get("matches", []):
                meta = m.get("metadata", {})
                rent = int(meta.get("Rent", 0))
                if max_rent and rent > max_rent:
                    continue
                if min_rent and rent < min_rent:
                    continue
                hits.append({"score": m["score"], "metadata": meta})
            return hits[:top_k]
        except Exception as e:
            log.error(f"Pinecone search failed: {e}")
            return []

    def _memory_search(self, query_vec, top_k, same_city, listing, max_rent, min_rent):
        global _MEM_LISTINGS
        if not _MEM_LISTINGS:
            return []
        qv   = np.array(query_vec, dtype=np.float32)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv /= norm

        scored = []
        for item in _MEM_LISTINGS:
            meta = item["metadata"]
            if same_city and meta.get("City") != listing.get("City"):
                continue
            rent = int(meta.get("Rent", 0))
            if max_rent and rent > max_rent:
                continue
            if min_rent and rent < min_rent:
                continue
            iv   = item["vector"]
            inn  = np.linalg.norm(iv)
            iv_n = iv / inn if inn > 0 else iv
            score = float(np.dot(qv, iv_n))
            scored.append({"score": score, "metadata": meta})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def log_prediction(self, data: dict):
        global _MEM_PREDICTIONS
        vec = listing_to_vector({
            "City": data.get("city", ""),
            "BHK": data.get("bhk", 2),
            "Size": data.get("size", 800),
            "Bathroom": max(1, int(data.get("bhk", 2)) - 1),
            "Furnishing Status": data.get("furnishing", ""),
            "Floor": "1 out of 3",
        })
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        rec_id = str(uuid.uuid4())

        if self.connected:
            try:
                self._preds.upsert(vectors=[{
                    "id": rec_id, "values": vec, "metadata": data,
                }])
            except Exception as e:
                log.warning(f"Prediction log failed: {e}")

        _MEM_PREDICTIONS.append({"id": rec_id, **data})

    def recent_predictions(self, limit: int = 100) -> list[dict]:
        return list(reversed(_MEM_PREDICTIONS))[:limit]

    def prediction_count(self) -> int:
        return len(_MEM_PREDICTIONS)

    def health(self) -> dict:
        if not self.connected:
            return {
                "status": "local-cache",
                "note": "Set PINECONE_API_KEY in .env to enable full vector search",
                "listings_vectors": len(_MEM_LISTINGS),
                "predictions_stored": len(_MEM_PREDICTIONS),
            }
        try:
            ls = self._listings.describe_index_stats()
            ps = self._preds.describe_index_stats()
            return {
                "status": "connected",
                "listings_vectors": ls.get("total_vector_count", 0),
                "predictions_stored": ps.get("total_vector_count", 0),
                "index_fullness": ls.get("index_fullness", 0),
            }
        except Exception as e:
            return {"status": "error", "note": str(e)}


_db: Optional[RentIQDatabase] = None

def get_db() -> RentIQDatabase:
    global _db
    if _db is None:
        _db = RentIQDatabase()
    return _db
