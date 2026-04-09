import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg, setup_logging
from core.database import get_db

log = setup_logging("RentIQ.Seed", cfg.LOGS_DIR)

if __name__ == "__main__":
    log.info("Seeding Pinecone with rental listings...")
    db = get_db()
    if db.connected:
        db.seed_listings()
        h = db.health()
        log.info(f"Pinecone seeded — {h.get('listings_vectors',0):,} vectors stored")
    else:
        log.warning("Pinecone not connected. Set PINECONE_API_KEY in .env")
