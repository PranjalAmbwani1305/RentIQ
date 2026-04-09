import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg, setup_logging
from core.inference import _train_and_save
from deep_learning.model import DeepLearningEngine

log = setup_logging("RentIQ.Train", cfg.LOGS_DIR)

if __name__ == "__main__":
    log.info("=== RentIQ Training Pipeline ===")

    log.info("Step 1/2 — Training GBT ensemble (scikit-learn)...")
    art = _train_and_save()
    log.info(f"  GBT done — R²={art['metrics']['r2']:.4f}  MAE=₹{art['metrics']['mae']:,.0f}")

    log.info("Step 2/2 — Training deep learning model (PyTorch TabNet)...")
    dl = DeepLearningEngine()
    m  = dl.train()
    if m:
        log.info(f"  TabNet done — R²={m.get('r2',0):.4f}  MAE=₹{m.get('mae',0):,.0f}")

    log.info("Training complete. Run: streamlit run app.py")
