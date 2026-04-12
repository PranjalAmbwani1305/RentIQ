import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg, setup_logging
from core.features import build_features_df, FEATURE_NAMES

log = setup_logging("RentIQ.DeepLearning", cfg.LOGS_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not installed. Deep learning inference will be disabled.")


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FeatureAttention(nn.Module):
    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, F = x.shape
        tokens = self.embed(x.unsqueeze(-1))
        out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(out + tokens)
        return out.reshape(B, -1)


class RentTabNet(nn.Module):
    def __init__(
        self,
        n_features: int = 12,
        embed_dim: int = cfg.DL_EMBEDDING_DIM,
        hidden_dims: list = cfg.DL_HIDDEN_DIMS,
        dropout: float = cfg.DL_DROPOUT,
    ):
        super().__init__()
        self.attn = FeatureAttention(n_features, embed_dim)
        attn_out  = n_features * embed_dim

        layers = []
        in_dim = attn_out + n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        self.encoder = nn.Sequential(*layers)
        self.res1 = ResidualBlock(hidden_dims[-1], dropout)
        self.res2 = ResidualBlock(hidden_dims[-1], dropout)

        self.rent_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attended  = self.attn(x)
        combined  = torch.cat([attended, x], dim=-1)
        encoded   = self.encoder(combined)
        encoded   = self.res2(self.res1(encoded))
        rent_pred = self.rent_head(encoded).squeeze(-1)
        risk_pred = self.risk_head(encoded).squeeze(-1)
        return rent_pred, risk_pred


class DeepLearningEngine:
    def __init__(self):
        self.model:  Optional[object] = None
        self.scaler: Optional[object] = None
        self.metrics: dict = {}
        self._device = "cpu"
        self._available = TORCH_AVAILABLE
        self._load()

    def _load(self):
        if not self._available:
            return
        dl_path = cfg.DL_MODEL_PATH
        if dl_path.exists():
            try:
                checkpoint = torch.load(dl_path, map_location="cpu", weights_only=False)
                self.model = RentTabNet()
                self.model.load_state_dict(checkpoint["model_state"])
                self.model.eval()
                self.scaler  = checkpoint.get("scaler")
                self.metrics = checkpoint.get("metrics", {})
                log.info(f"Deep learning model loaded from {dl_path}")
            except Exception as e:
                log.warning(f"DL model load failed: {e}")

    def train(self, csv_path: Path = cfg.DATA_PATH) -> dict:
        if not self._available:
            log.warning("PyTorch not available; skipping DL training.")
            return {}

        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        log.info("Starting deep learning training pipeline...")
        df = pd.read_csv(csv_path)
        df = df[~df["Floor"].str.lower().str.contains("basement", na=False)].copy()
        df["price_per_sqft"] = df["Rent"] / df["Size"].clip(lower=1)
        df = df[df["price_per_sqft"].between(2, 500)].copy()
        log.info(f"Training on {len(df)} samples")

        X = build_features_df(df)
        y_rent = np.log1p(df["Rent"].values).astype(np.float32)
        thresh  = df["price_per_sqft"].quantile(0.60)
        y_risk  = (df["price_per_sqft"] >= thresh).astype(np.float32).values

        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
            X, y_rent, y_risk, test_size=0.15, random_state=42
        )

        scaler  = RobustScaler()
        X_tr_s  = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_s  = scaler.transform(X_te).astype(np.float32)

        tr_ds = TensorDataset(
            torch.from_numpy(X_tr_s),
            torch.from_numpy(yr_tr),
            torch.from_numpy(yc_tr),
        )
        te_ds = TensorDataset(
            torch.from_numpy(X_te_s),
            torch.from_numpy(yr_te),
            torch.from_numpy(yc_te),
        )
        tr_dl = DataLoader(tr_ds, batch_size=cfg.DL_BATCH_SIZE, shuffle=True)

        model     = RentTabNet(n_features=X_tr_s.shape[1])
        optimizer = optim.AdamW(model.parameters(), lr=cfg.DL_LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.DL_EPOCHS)
        rent_loss = nn.HuberLoss(delta=0.5)
        risk_loss = nn.BCELoss()

        model.train()
        for epoch in range(cfg.DL_EPOCHS):
            epoch_loss = 0.0
            for xb, yr_b, yc_b in tr_dl:
                optimizer.zero_grad()
                r_pred, c_pred = model(xb)
                loss = rent_loss(r_pred, yr_b) + 0.3 * risk_loss(c_pred, yc_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if epoch % 20 == 0:
                log.info(f"  Epoch {epoch:3d} | loss={epoch_loss/len(tr_dl):.4f}")

        model.eval()
        with torch.no_grad():
            Xte_t       = torch.from_numpy(X_te_s)
            r_pred, c_p = model(Xte_t)
            pred_rent   = np.expm1(r_pred.numpy())
            true_rent   = np.expm1(yr_te)
            mae  = mean_absolute_error(true_rent, pred_rent)
            r2   = r2_score(true_rent, pred_rent)
            risk_acc = ((c_p.numpy() >= 0.5) == yc_te).mean()

        log.info(f"DL Model — MAE=₹{mae:,.0f}  R²={r2:.4f}  RiskAcc={risk_acc:.4f}")

        cfg.DL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        metrics = {"mae": mae, "r2": r2, "risk_acc": float(risk_acc),
                   "n_train": len(X_tr), "n_test": len(X_te), "engine": "pytorch-tabtransformer"}

        torch.save({
            "model_state": model.state_dict(),
            "scaler":      scaler,
            "metrics":     metrics,
        }, cfg.DL_MODEL_PATH)
        log.info(f"DL model saved to {cfg.DL_MODEL_PATH}")

        self.model   = model
        self.scaler  = scaler
        self.metrics = metrics
        return metrics

    def predict(self, features: np.ndarray) -> dict:
        if not self._available or self.model is None:
            return {}
        try:
            X_s  = self.scaler.transform(features.reshape(1, -1)).astype(np.float32)
            xt   = torch.from_numpy(X_s)
            with torch.no_grad():
                r_pred, c_pred = self.model(xt)
            rent = float(np.expm1(r_pred.numpy()[0]))
            risk = float(c_pred.numpy()[0])
            return {"dl_rent": round(rent, -2), "dl_risk": risk, "available": True}
        except Exception as e:
            log.error(f"DL prediction failed: {e}")
            return {"available": False}

    def get_attention_weights(self, features: np.ndarray) -> Optional[np.ndarray]:
        if not self._available or self.model is None:
            return None
        try:
            X_s = self.scaler.transform(features.reshape(1, -1)).astype(np.float32)
            xt  = torch.from_numpy(X_s)
            with torch.no_grad():
                tokens  = self.model.attn.embed(xt.unsqueeze(-1))
                _, w    = self.model.attn.attn(tokens, tokens, tokens)
            return w.squeeze(0).numpy()
        except Exception:
            return None


_dl_engine: Optional[DeepLearningEngine] = None

def get_dl_engine() -> DeepLearningEngine:
    global _dl_engine
    if _dl_engine is None:
        _dl_engine = DeepLearningEngine()
    return _dl_engine
