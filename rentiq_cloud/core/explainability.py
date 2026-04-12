import numpy as np
from typing import Optional
from config import cfg, setup_logging
from core.features import FEATURE_NAMES

log = setup_logging("RentIQ.XAI", cfg.LOGS_DIR)

FEATURE_DESCRIPTIONS = {
    "city_enc":           "City location",
    "bhk":                "Number of bedrooms",
    "size":               "Property area (sqft)",
    "bathroom":           "Number of bathrooms",
    "furnishing_enc":     "Furnishing level",
    "floor_ratio":        "Floor height ratio",
    "size_per_bhk":       "Space per bedroom",
    "bath_per_bhk":       "Bathrooms per bedroom",
    "area_type_enc":      "Area measurement type",
    "tenant_enc":         "Tenant preference",
    "log_size":           "Log-scaled area",
    "bhk_bath_interaction": "BHK × Bathroom index",
}

INSIGHT_TEMPLATES = {
    "high_price_per_sqft": "This property commands a premium price per sqft — likely due to {city} demand and {furnishing} furnishing.",
    "low_price_per_sqft":  "Competitive pricing relative to {city} average — good value signal for tenants.",
    "high_floor":          "Higher floor placement typically adds 3-8% to rental premiums in Indian metros.",
    "premium_city":        "{city} ranks as one of India's most expensive rental markets, driving above-median estimates.",
    "furnished_premium":   "Fully furnished units command a 15-25% rent premium vs unfurnished equivalents.",
    "large_unit":          "Large unit size ({size} sqft) is the dominant rent driver — size_per_bhk ratio elevated.",
    "high_risk":           "High demand risk ({risk_pct}% confidence): supply constrained, rents likely to appreciate.",
    "low_risk":            "Low demand risk signal — market appears well-supplied in this segment.",
    "ensemble_agreement":  "GBT and deep learning models agree within ₹{diff:,.0f} — high confidence estimate.",
    "ensemble_diverge":    "Models diverge by ₹{diff:,.0f} — uncertainty is higher; treat range as wider.",
}


def compute_permutation_importance(
    model,
    scaler,
    X_base: np.ndarray,
    y_true: float,
    n_repeats: int = 20,
) -> dict:
    try:
        X_s   = scaler.transform(X_base.reshape(1, -1))
        base  = float(np.expm1(model.predict(X_s)[0]))
        imps  = {}
        for i, fname in enumerate(FEATURE_NAMES):
            diffs = []
            for _ in range(n_repeats):
                X_perm      = X_base.copy()
                X_perm[i]   = X_perm[i] * (0.5 + np.random.random())
                X_perm_s    = scaler.transform(X_perm.reshape(1, -1))
                perturbed   = float(np.expm1(model.predict(X_perm_s)[0]))
                diffs.append(abs(perturbed - base))
            imps[fname] = float(np.mean(diffs))
        total = sum(imps.values()) or 1.0
        return {k: v / total for k, v in imps.items()}
    except Exception as e:
        log.error(f"Permutation importance failed: {e}")
        return {name: 1.0 / len(FEATURE_NAMES) for name in FEATURE_NAMES}


def generate_prediction_insights(
    city: str,
    size: int,
    furnishing: str,
    floor_ratio: float,
    predicted_rent: float,
    risk: str,
    risk_prob: float,
    psf: float,
    vs_median: float,
    dl_result: dict = None,
    city_median: float = None,
) -> list[str]:
    insights = []
    city_median = city_median or cfg.CITY_MEDIANS.get(city, 35000)

    if psf > 80:
        insights.append(INSIGHT_TEMPLATES["high_price_per_sqft"].format(
            city=city, furnishing=furnishing.lower()
        ))
    elif psf < 25:
        insights.append(INSIGHT_TEMPLATES["low_price_per_sqft"].format(city=city))

    if floor_ratio > 0.6:
        insights.append(INSIGHT_TEMPLATES["high_floor"])

    if city in ("Mumbai", "Bangalore", "Delhi") and predicted_rent > city_median:
        insights.append(INSIGHT_TEMPLATES["premium_city"].format(city=city))

    if furnishing == "Furnished":
        insights.append(INSIGHT_TEMPLATES["furnished_premium"])

    if size > 1500:
        insights.append(INSIGHT_TEMPLATES["large_unit"].format(size=size))

    if risk == "High":
        insights.append(INSIGHT_TEMPLATES["high_risk"].format(
            risk_pct=int(risk_prob * 100)
        ))
    else:
        insights.append(INSIGHT_TEMPLATES["low_risk"])

    if dl_result and dl_result.get("available"):
        diff = abs(dl_result.get("dl_rent", predicted_rent) - predicted_rent)
        if diff < 0.08 * predicted_rent:
            insights.append(INSIGHT_TEMPLATES["ensemble_agreement"].format(diff=diff))
        else:
            insights.append(INSIGHT_TEMPLATES["ensemble_diverge"].format(diff=diff))

    return insights[:5]


def top_features_for_prediction(importances: dict, top_n: int = 5) -> list[dict]:
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    result = []
    for fname, importance in sorted_feats[:top_n]:
        result.append({
            "feature":     fname,
            "label":       FEATURE_DESCRIPTIONS.get(fname, fname),
            "importance":  round(importance, 4),
            "pct":         round(importance * 100, 1),
        })
    return result
