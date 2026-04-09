import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg, setup_logging
from core.features import build_features, build_features_df, FEATURE_NAMES

log = setup_logging("RentIQ.Inference", cfg.LOGS_DIR)


def _train_and_save(csv_path: Path = cfg.DATA_PATH, out_path: Path = cfg.MODEL_PKL) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

    log.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[~df["Floor"].str.lower().str.contains("basement", na=False)].copy()
    df["price_per_sqft"] = df["Rent"] / df["Size"].clip(lower=1)
    df = df[df["price_per_sqft"].between(2, 500)].copy()
    log.info(f"Clean dataset: {len(df)} rows")

    X      = build_features_df(df)
    y_rent = np.log1p(df["Rent"].values)
    thresh = df["price_per_sqft"].quantile(0.60)
    y_risk = (df["price_per_sqft"] >= thresh).astype(int).values

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_rent, y_risk, test_size=0.15, random_state=42
    )

    scaler  = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    log.info("Training GBT Regressor...")
    reg = GradientBoostingRegressor(
        n_estimators=cfg.GBT_N_ESTIMATORS,
        max_depth=cfg.GBT_MAX_DEPTH,
        learning_rate=cfg.GBT_LEARNING_RATE,
        subsample=cfg.GBT_SUBSAMPLE,
        min_samples_leaf=8,
        random_state=42,
    )
    reg.fit(X_tr_s, yr_tr)
    pred_rent = np.expm1(reg.predict(X_te_s))
    true_rent = np.expm1(yr_te)
    mae = mean_absolute_error(true_rent, pred_rent)
    r2  = r2_score(true_rent, pred_rent)
    log.info(f"  Regressor — MAE=₹{mae:,.0f}  R²={r2:.4f}")

    log.info("Training GBT Classifier...")
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.06, subsample=0.8, random_state=42,
    )
    clf.fit(X_tr_s, yc_tr)
    acc = accuracy_score(yc_te, clf.predict(X_te_s))
    log.info(f"  Classifier — Accuracy={acc:.4f}")

    city_med = df.groupby("City")["Rent"].median().to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "regressor":    reg,
        "classifier":   clf,
        "scaler":       scaler,
        "city_medians": city_med,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "mae": mae, "r2": r2, "clf_acc": acc,
            "n_train": len(X_tr), "n_test": len(X_te),
            "engine": "scikit-learn",
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    log.info(f"Artifacts saved: {out_path}")
    return artifact


class RentPredictor:
    def __init__(self):
        self._reg     = None
        self._clf     = None
        self._scaler  = None
        self._medians = dict(cfg.CITY_MEDIANS)
        self.metrics  = {}
        self._feature_names = FEATURE_NAMES
        self._load()

    def _load(self):
        art_path = cfg.MODEL_PKL
        if art_path.exists():
            try:
                with open(art_path, "rb") as f:
                    art = pickle.load(f)
                self._reg     = art["regressor"]
                self._clf     = art["classifier"]
                self._scaler  = art["scaler"]
                self._medians = art.get("city_medians", self._medians)
                self.metrics  = art.get("metrics", {})
                self._feature_names = art.get("feature_names", FEATURE_NAMES)
                log.info(f"GBT models loaded — engine={self.metrics.get('engine','unknown')}")
                return
            except Exception as e:
                log.warning(f"Model load failed ({e}), retraining...")

        if cfg.DATA_PATH.exists():
            art = _train_and_save()
            self._reg     = art["regressor"]
            self._clf     = art["classifier"]
            self._scaler  = art["scaler"]
            self._medians = art.get("city_medians", self._medians)
            self.metrics  = art.get("metrics", {})
        else:
            log.error(f"Dataset not found: {cfg.DATA_PATH}")

    def predict(
        self,
        city: str,
        bhk: int,
        size: int,
        furnishing: str,
        bathroom: int,
        floor: str,
        area_type: str = "Super Area",
        tenant: str = "Bachelors/Family",
    ) -> dict:
        if self._reg is None:
            return {
                "predicted_rent": 15000, "demand_risk": "Unknown",
                "risk_probability": 0.5, "price_per_sqft": 0,
                "vs_median_pct": 0, "rent_low": 12000, "rent_high": 18000,
                "features": None,
            }

        row = {
            "City": city, "BHK": bhk, "Size": size,
            "Furnishing Status": furnishing, "Bathroom": bathroom,
            "Floor": floor, "Area Type": area_type, "Tenant Preferred": tenant,
        }
        feat   = build_features(row)
        feat_s = self._scaler.transform(feat.reshape(1, -1))

        log_rent  = self._reg.predict(feat_s)[0]
        rent      = float(np.expm1(log_rent))

        rent_low  = rent * 0.88
        rent_high = rent * 1.12

        risk_prob = float(self._clf.predict_proba(feat_s)[0][1])
        risk      = "High" if risk_prob >= 0.5 else "Low"

        median    = self._medians.get(city, 25000)
        vs_median = (rent - median) / median * 100
        psf       = rent / max(size, 1)

        return {
            "predicted_rent":    round(rent, -2),
            "rent_low":          round(rent_low, -2),
            "rent_high":         round(rent_high, -2),
            "demand_risk":       risk,
            "risk_probability":  risk_prob,
            "price_per_sqft":    round(psf, 2),
            "vs_median_pct":     round(vs_median, 1),
            "city_median":       median,
            "features":          feat,
        }

    def get_feature_importances(self) -> dict:
        if self._reg is None:
            return {}
        imps = self._reg.feature_importances_
        return dict(zip(self._feature_names, imps.tolist()))


_predictor: Optional[RentPredictor] = None

def get_predictor() -> RentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RentPredictor()
    return _predictor
