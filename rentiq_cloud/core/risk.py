"""
RentIQ Risk Engine — Multi-Factor Risk Scoring
===============================================
Produces a composite 0-100 risk score from 5 independent dimensions:

  1. Price Anomaly Risk     — how far rent deviates from city / segment norms
  2. Supply Pressure Risk   — demand/supply balance inferred from dataset density
  3. Affordability Risk     — rent-to-income stress (RBI affordability benchmarks)
  4. Property Risk          — structural signals (floor, furnishing, size/BHK ratio)
  5. Market Velocity Risk   — city-level appreciation trajectory

Each factor is scored 0-100. Composite = weighted average.
Level thresholds:  Low < 35 | Medium 35-60 | High 60-80 | Critical ≥ 80
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg, setup_logging

log = setup_logging("RentIQ.Risk", cfg.LOGS_DIR)

# ---------------------------------------------------------------------------
# City-level benchmark data (sourced from RBI / NHB reports)
# ---------------------------------------------------------------------------
CITY_INCOME_MEDIANS = {
    "Mumbai":    85_000,   # monthly household income (₹)
    "Delhi":     72_000,
    "Bangalore": 78_000,
    "Chennai":   58_000,
    "Hyderabad": 60_000,
    "Kolkata":   45_000,
}

# Annualised rent appreciation % (5-yr CAGR, Anarock 2024)
CITY_APPRECIATION = {
    "Mumbai":    8.2,
    "Delhi":     6.8,
    "Bangalore": 9.1,
    "Chennai":   5.4,
    "Hyderabad": 7.6,
    "Kolkata":   4.2,
}

# Supply pressure index (higher = tighter supply, more risk)
CITY_SUPPLY_PRESSURE = {
    "Mumbai":    0.82,
    "Delhi":     0.70,
    "Bangalore": 0.75,
    "Chennai":   0.55,
    "Hyderabad": 0.60,
    "Kolkata":   0.45,
}

FACTOR_WEIGHTS = {
    "price_anomaly":    0.30,
    "supply_pressure":  0.20,
    "affordability":    0.25,
    "property":         0.15,
    "market_velocity":  0.10,
}

LEVEL_THRESHOLDS = [
    (80, "Critical", "#dc2626", "🔴"),
    (60, "High",     "#f59e0b", "🟠"),
    (35, "Medium",   "#3b82f6", "🔵"),
    (0,  "Low",      "#16a34a", "🟢"),
]


# ---------------------------------------------------------------------------
# Dataset-derived benchmarks (lazy-loaded once)
# ---------------------------------------------------------------------------
_benchmarks: Optional[dict] = None

def _load_benchmarks() -> dict:
    global _benchmarks
    if _benchmarks is not None:
        return _benchmarks
    try:
        df = pd.read_csv(cfg.DATA_PATH)
        df = df[~df["Floor"].str.lower().str.contains("basement", na=False)].copy()
        df["price_per_sqft"] = df["Rent"] / df["Size"].clip(lower=1)
        df = df[df["price_per_sqft"].between(2, 500)].copy()

        city_stats = {}
        for city, grp in df.groupby("City"):
            city_stats[city] = {
                "rent_p25":  float(grp["Rent"].quantile(0.25)),
                "rent_p50":  float(grp["Rent"].quantile(0.50)),
                "rent_p75":  float(grp["Rent"].quantile(0.75)),
                "rent_p90":  float(grp["Rent"].quantile(0.90)),
                "psf_p50":   float(grp["price_per_sqft"].quantile(0.50)),
                "psf_p75":   float(grp["price_per_sqft"].quantile(0.75)),
                "count":     len(grp),
            }

        bhk_city_medians = {}
        for (city, bhk), grp in df.groupby(["City", "BHK"]):
            bhk_city_medians[(city, int(bhk))] = float(grp["Rent"].median())

        furnish_premiums = {}
        for city, grp in df.groupby("City"):
            base = grp[grp["Furnishing Status"] == "Unfurnished"]["Rent"].median()
            if base and base > 0:
                furnish_premiums[city] = {
                    "Furnished":      grp[grp["Furnishing Status"] == "Furnished"]["Rent"].median() / base,
                    "Semi-Furnished": grp[grp["Furnishing Status"] == "Semi-Furnished"]["Rent"].median() / base,
                    "Unfurnished":    1.0,
                }

        _benchmarks = {
            "city_stats":       city_stats,
            "bhk_city_medians": bhk_city_medians,
            "furnish_premiums": furnish_premiums,
        }
        log.info("Risk benchmarks loaded from dataset")
    except Exception as e:
        log.error(f"Benchmark load failed: {e}")
        _benchmarks = {"city_stats": {}, "bhk_city_medians": {}, "furnish_premiums": {}}
    return _benchmarks


# ---------------------------------------------------------------------------
# Dataclass for a scored factor
# ---------------------------------------------------------------------------
@dataclass
class RiskFactor:
    name:        str
    label:       str
    score:       float          # 0-100
    weight:      float
    description: str
    signals:     list[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    def level(self) -> tuple[str, str, str]:
        for threshold, lvl, color, icon in LEVEL_THRESHOLDS:
            if self.score >= threshold:
                return lvl, color, icon
        return "Low", "#16a34a", "🟢"


@dataclass
class RiskReport:
    composite:   float
    level:       str
    color:       str
    icon:        str
    factors:     list[RiskFactor]
    summary:     str
    actions:     list[str]
    # Backward-compat with old binary system
    demand_risk:      str   = ""
    risk_probability: float = 0.0

    def as_dict(self) -> dict:
        return {
            "composite":          round(self.composite, 1),
            "level":              self.level,
            "color":              self.color,
            "icon":               self.icon,
            "summary":            self.summary,
            "actions":            self.actions,
            "demand_risk":        self.demand_risk,
            "risk_probability":   self.risk_probability,
            "factors": [
                {
                    "name":        f.name,
                    "label":       f.label,
                    "score":       round(f.score, 1),
                    "weight":      f.weight,
                    "description": f.description,
                    "signals":     f.signals,
                    "level":       f.level()[0],
                    "color":       f.level()[1],
                    "icon":        f.level()[2],
                }
                for f in self.factors
            ],
        }


# ---------------------------------------------------------------------------
# Individual factor scorers
# ---------------------------------------------------------------------------

def _score_price_anomaly(
    city: str,
    bhk: int,
    predicted_rent: float,
    price_per_sqft: float,
    vs_median_pct: float,
    bmarks: dict,
) -> RiskFactor:
    """Score 0-100: how anomalous is this rent vs segment norms?"""
    signals = []
    score   = 0.0

    city_st = bmarks["city_stats"].get(city, {})
    if city_st:
        p25, p75, p90 = city_st["rent_p25"], city_st["rent_p75"], city_st["rent_p90"]
        iqr = max(p75 - p25, 1)

        # Z-score style deviation from median
        deviation_pct = abs(vs_median_pct)
        if deviation_pct > 80:
            score += 50; signals.append(f"Rent is {vs_median_pct:+.0f}% vs city median — extreme outlier")
        elif deviation_pct > 40:
            score += 30; signals.append(f"Rent is {vs_median_pct:+.0f}% vs city median — significant deviation")
        elif deviation_pct > 20:
            score += 15; signals.append(f"Rent is {vs_median_pct:+.0f}% vs city median — moderate deviation")

        # Above 90th percentile is inherently risky (overpriced or premium)
        if predicted_rent > p90:
            score += 25; signals.append("Rent exceeds 90th percentile — top-tier pricing")
        elif predicted_rent > p75:
            score += 12; signals.append("Rent above 75th percentile — above-average market")

        # PSF anomaly
        psf_p75 = city_st.get("psf_p75", 80)
        if price_per_sqft > psf_p75 * 1.5:
            score += 20; signals.append(f"Price/sqft ₹{price_per_sqft:.0f} is {price_per_sqft/psf_p75:.1f}× the 75th pct")
        elif price_per_sqft > psf_p75:
            score += 8

    # BHK-specific median check
    bhk_med = bmarks["bhk_city_medians"].get((city, bhk))
    if bhk_med and bhk_med > 0:
        bhk_dev = (predicted_rent - bhk_med) / bhk_med * 100
        if abs(bhk_dev) > 50:
            score += 10; signals.append(f"{bhk}BHK segment median: ₹{bhk_med:,.0f} ({bhk_dev:+.0f}% gap)")

    if not signals:
        signals.append("Rent is within normal city range")

    desc = "Measures how far the predicted rent deviates from city and segment benchmarks."
    return RiskFactor("price_anomaly", "Price Anomaly", min(score, 100), FACTOR_WEIGHTS["price_anomaly"], desc, signals)


def _score_supply_pressure(city: str, furnishing: str, bmarks: dict) -> RiskFactor:
    """Score 0-100: how constrained is supply in this segment?"""
    signals = []
    base    = CITY_SUPPLY_PRESSURE.get(city, 0.55) * 100   # 0-100

    score = base

    # Furnished units face higher demand competition
    if furnishing == "Furnished":
        score = min(score + 15, 100)
        signals.append("Fully furnished units face elevated tenant competition")
    elif furnishing == "Semi-Furnished":
        score = min(score + 8, 100)
        signals.append("Semi-furnished segment is in moderate demand")
    else:
        signals.append("Unfurnished segment has broader supply availability")

    city_count = bmarks["city_stats"].get(city, {}).get("count", 0)
    if city_count < 400:
        score = min(score + 10, 100)
        signals.append("Limited comparable listings — thinner market data")
    elif city_count > 900:
        score = max(score - 8, 0)
        signals.append("High listing density — more supply options available")

    signals.insert(0, f"{city} supply pressure index: {CITY_SUPPLY_PRESSURE.get(city, 0.55):.0%}")

    desc = "Estimates tenant competition intensity based on city supply dynamics and furnishing demand."
    return RiskFactor("supply_pressure", "Supply Pressure", score, FACTOR_WEIGHTS["supply_pressure"], desc, signals)


def _score_affordability(city: str, predicted_rent: float) -> RiskFactor:
    """Score 0-100: rent-to-income ratio stress (RBI: healthy < 30% of income)."""
    signals = []
    income  = CITY_INCOME_MEDIANS.get(city, 60_000)
    ratio   = predicted_rent / income  # fraction of monthly income

    # RBI benchmark: < 30% = safe, 30-40% = stressed, > 40% = high risk
    if ratio >= 0.60:
        score = 90; signals.append(f"Rent is {ratio:.0%} of median income — severe affordability stress")
    elif ratio >= 0.45:
        score = 70; signals.append(f"Rent is {ratio:.0%} of median income — high stress zone")
    elif ratio >= 0.30:
        score = 45; signals.append(f"Rent is {ratio:.0%} of median income — approaching RBI caution threshold")
    else:
        score = 15; signals.append(f"Rent is {ratio:.0%} of median income — within healthy range")

    signals.append(f"{city} median household income: ₹{income:,}/mo")

    desc = "Measures rent affordability stress using RBI income benchmarks (healthy threshold: < 30% of income)."
    return RiskFactor("affordability", "Affordability", score, FACTOR_WEIGHTS["affordability"], desc, signals)


def _score_property(
    bhk: int,
    size: int,
    furnishing: str,
    floor: str,
    bathroom: int,
) -> RiskFactor:
    """Score 0-100: property-level structural risk signals."""
    from core.features import _parse_floor
    signals = []
    score   = 0.0

    # Size-per-BHK check (< 250 sqft/BHK is cramped)
    spb = size / max(bhk, 1)
    if spb < 200:
        score += 25; signals.append(f"Very small space per BHK ({spb:.0f} sqft) — tenant retention risk")
    elif spb < 300:
        score += 12; signals.append(f"Compact layout ({spb:.0f} sqft/BHK)")
    else:
        signals.append(f"Spacious layout ({spb:.0f} sqft/BHK)")

    # Bathroom ratio
    bath_ratio = bathroom / max(bhk, 1)
    if bath_ratio < 0.5:
        score += 15; signals.append(f"Low bathroom ratio ({bath_ratio:.1f} per BHK)")
    elif bath_ratio >= 1.5:
        score += 5; signals.append(f"Premium bathroom ratio ({bath_ratio:.1f} per BHK)")

    # Floor risk
    floor_n, total_f = _parse_floor(floor)
    floor_ratio = floor_n / total_f if total_f > 0 else 0
    if floor_ratio == 0:
        score += 10; signals.append("Ground floor — lower desirability, security concerns")
    elif floor_ratio > 0.85:
        score += 8; signals.append("Top floor — potential heat/maintenance issues")
    else:
        signals.append(f"Floor position {floor_n:.0f}/{total_f:.0f} — mid-range desirability")

    # Furnishing depreciation risk
    if furnishing == "Furnished":
        score += 10; signals.append("Furnished inventory subject to wear and replacement cost")
    else:
        signals.append("Unfurnished — no furniture depreciation risk")

    desc = "Structural property signals: size efficiency, floor position, bathroom ratio, and furnishing wear."
    return RiskFactor("property", "Property Signals", min(score, 100), FACTOR_WEIGHTS["property"], desc, signals)


def _score_market_velocity(city: str, predicted_rent: float, bmarks: dict) -> RiskFactor:
    """Score 0-100: forward-looking market appreciation risk."""
    signals = []
    appreciation = CITY_APPRECIATION.get(city, 6.0)

    # High appreciation = higher risk of further rent hikes (risk for tenants)
    if appreciation >= 9.0:
        score = 80; signals.append(f"{city} 5-yr CAGR {appreciation}% — rapidly appreciating market")
    elif appreciation >= 7.5:
        score = 60; signals.append(f"{city} 5-yr CAGR {appreciation}% — strong growth trajectory")
    elif appreciation >= 6.0:
        score = 40; signals.append(f"{city} 5-yr CAGR {appreciation}% — moderate appreciation")
    else:
        score = 20; signals.append(f"{city} 5-yr CAGR {appreciation}% — stable, slower market")

    # Future rent burden estimate (1-year projection)
    future_rent = predicted_rent * (1 + appreciation / 100)
    signals.append(f"1-yr rent projection: ₹{future_rent:,.0f} at current trajectory")

    desc = "Forward-looking risk based on city rental appreciation trajectory (5-yr CAGR, Anarock 2024)."
    return RiskFactor("market_velocity", "Market Velocity", score, FACTOR_WEIGHTS["market_velocity"], desc, signals)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def _resolve_level(score: float) -> tuple[str, str, str]:
    for threshold, lvl, color, icon in LEVEL_THRESHOLDS:
        if score >= threshold:
            return lvl, color, icon
    return "Low", "#16a34a", "🟢"


def _build_summary(level: str, city: str, factors: list[RiskFactor]) -> str:
    top = sorted(factors, key=lambda f: f.score, reverse=True)[0]
    templates = {
        "Critical": f"Critical risk in {city} — driven by {top.label.lower()}. Immediate review recommended.",
        "High":     f"Elevated risk in {city}. Primary driver: {top.label.lower()}. Proceed with caution.",
        "Medium":   f"Moderate risk profile for this {city} listing. Watch {top.label.lower()}.",
        "Low":      f"Favourable risk profile. This {city} listing shows healthy market signals.",
    }
    return templates.get(level, "Risk assessment complete.")


def _build_actions(level: str, factors: list[RiskFactor], city: str) -> list[str]:
    actions = []
    high_factors = [f for f in factors if f.score >= 60]

    for f in high_factors:
        if f.name == "price_anomaly":
            actions.append("Negotiate rent or verify comparables — price is an outlier in this segment")
        elif f.name == "supply_pressure":
            actions.append("Act quickly — limited supply means high tenant competition in this market")
        elif f.name == "affordability":
            actions.append("Review budget: rent exceeds RBI's recommended income-to-rent threshold")
        elif f.name == "property":
            actions.append("Inspect property thoroughly — structural signals suggest potential issues")
        elif f.name == "market_velocity":
            actions.append(f"Lock in a long-term lease — {city} rents are appreciating rapidly")

    if not actions:
        actions.append("Standard due diligence applies — risk profile is within normal range")
        actions.append("Compare with 2-3 similar listings before committing")

    return actions[:4]


def score_risk(
    city:           str,
    bhk:            int,
    size:           int,
    furnishing:     str,
    floor:          str,
    bathroom:       int,
    predicted_rent: float,
    price_per_sqft: float,
    vs_median_pct:  float,
    # Optional: pass-through from classifier for backward compat
    clf_risk:       str   = "Low",
    clf_prob:       float = 0.0,
) -> RiskReport:
    """
    Main entry point. Returns a RiskReport with composite score,
    level, per-factor breakdown, summary, and action items.
    """
    bmarks = _load_benchmarks()

    factors = [
        _score_price_anomaly(city, bhk, predicted_rent, price_per_sqft, vs_median_pct, bmarks),
        _score_supply_pressure(city, furnishing, bmarks),
        _score_affordability(city, predicted_rent),
        _score_property(bhk, size, furnishing, floor, bathroom),
        _score_market_velocity(city, predicted_rent, bmarks),
    ]

    composite = sum(f.weighted_score for f in factors)
    level, color, icon = _resolve_level(composite)
    summary  = _build_summary(level, city, factors)
    actions  = _build_actions(level, factors, city)

    return RiskReport(
        composite=composite,
        level=level,
        color=color,
        icon=icon,
        factors=factors,
        summary=summary,
        actions=actions,
        demand_risk=clf_risk,
        risk_probability=clf_prob,
    )
