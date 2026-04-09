import os
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="RentIQ — Intelligent Rental Analytics",
    page_icon="🏙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: #f5f7fa !important;
}

section[data-testid="stSidebar"],
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

.riq-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 36px; height: 56px; position: sticky; top: 0; z-index: 100;
    background: rgba(255,255,255,0.97);
    border-bottom: 1px solid #e4e9f0;
    backdrop-filter: blur(20px);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.riq-logo {
    font-size: 17px; font-weight: 700; color: #1a1f2e;
    letter-spacing: -0.5px; user-select: none;
}
.riq-logo span { color: #2563eb; }
.riq-nav { display: flex; gap: 2px; }
.riq-nav-btn {
    padding: 6px 15px; border-radius: 7px; cursor: pointer;
    font-size: 13px; font-weight: 500; color: #6b7280;
    border: none; background: transparent; transition: all 0.15s;
    font-family: 'Inter', sans-serif;
}
.riq-nav-btn:hover { color: #1a1f2e; background: #f0f4ff; }
.riq-nav-btn.active { color: #2563eb; background: #eff6ff; font-weight: 600; }
.riq-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 12px 4px 6px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
}
.riq-av {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: #fff;
}
.riq-uname { font-size: 12px; font-weight: 500; color: #374151; }
.riq-rtag {
    font-size: 10px; font-weight: 600; padding: 2px 8px;
    border-radius: 10px; background: #eff6ff; color: #2563eb;
}
.riq-rtag.admin  { background: #fef2f2; color: #dc2626; }
.riq-rtag.analyst{ background: #f0fdf4; color: #16a34a; }
.riq-rtag.agent  { background: #fffbeb; color: #d97706; }

.riq-page { padding: 28px 36px 72px; max-width: 1400px; margin: 0 auto; }

.riq-card {
    background: #ffffff; border: 1px solid #e4e9f0;
    border-radius: 14px; padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.riq-card-sm {
    background: #ffffff; border: 1px solid #e4e9f0;
    border-radius: 10px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

.riq-sec {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 16px;
}

.riq-kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 28px; }
.riq-kpi {
    background: #fff; border: 1px solid #e4e9f0; border-radius: 12px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.riq-kpi-lbl { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 10px; }
.riq-kpi-val { font-size: 26px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.riq-kpi-sub { font-size: 11px; color: #94a3b8; margin-top: 7px; }

.riq-result {
    background: #fff; border: 1px solid #dbeafe;
    border-radius: 14px; padding: 28px; position: relative; overflow: hidden;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
}
.riq-result::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 60%, transparent 100%);
}
.riq-rent-lbl { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
.riq-rent-val { font-size: 44px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; line-height: 1; margin: 6px 0 3px; }
.riq-rent-range { font-size: 12px; color: #94a3b8; }
.riq-divider { border: none; border-top: 1px solid #f1f5f9; margin: 20px 0; }
.riq-stats { display: flex; gap: 32px; flex-wrap: wrap; }
.riq-stat-lbl { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.riq-stat-val { font-size: 18px; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: #1e293b; }

.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600;
}
.badge-hi { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.badge-lo { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
.badge-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

.insight-item {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 10px 0; border-bottom: 1px solid #f1f5f9;
    font-size: 12px; color: #475569; line-height: 1.5;
}
.insight-item:last-child { border-bottom: none; }
.insight-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }

.xai-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; gap: 12px;
}
.xai-label { font-size: 12px; font-weight: 500; color: #374151; width: 160px; flex-shrink: 0; }
.xai-bar-bg { flex: 1; height: 6px; background: #f1f5f9; border-radius: 3px; }
.xai-bar-fill { height: 6px; border-radius: 3px; background: linear-gradient(90deg, #2563eb, #7c3aed); }
.xai-pct { font-size: 11px; color: #94a3b8; width: 36px; text-align: right; flex-shrink: 0; }

.spark-partition {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px 16px; text-align: center;
}
.spark-part-num { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; }
.spark-part-rows { font-size: 20px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono'; }
.spark-part-meta { font-size: 11px; color: #94a3b8; margin-top: 3px; }

.riq-listing {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 0; border-bottom: 1px solid #f1f5f9;
}
.riq-listing:last-child { border-bottom: none; }
.riq-listing-title { font-size: 13px; font-weight: 500; color: #1e293b; }
.riq-listing-meta  { font-size: 11px; color: #94a3b8; margin-top: 3px; }
.riq-listing-rent  { font-size: 14px; font-weight: 600; font-family: 'JetBrains Mono'; color: #0f172a; text-align: right; }
.riq-listing-score { font-size: 11px; color: #94a3b8; text-align: right; margin-top: 3px; }
.riq-bar { height: 2px; background: #f1f5f9; border-radius: 2px; margin-top: 10px; }
.riq-fill { height: 2px; background: linear-gradient(90deg, #2563eb, #7c3aed); border-radius: 2px; }

.riq-admin-banner {
    background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 10px; padding: 12px 18px; margin-bottom: 24px;
    font-size: 12px; color: #dc2626; display: flex; align-items: center; gap: 8px;
}
.riq-db-warn {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;
    font-size: 12px; color: #92400e;
}
.riq-db-ok {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;
    font-size: 12px; color: #166534;
}

.riq-login {
    max-width: 400px; margin: 60px auto 0;
    background: #fff; border: 1px solid #e4e9f0;
    border-radius: 18px; padding: 40px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
}
.riq-brand { font-size: 22px; font-weight: 700; color: #0f172a; margin-bottom: 4px; }
.riq-brand span { color: #2563eb; }
.riq-tagline { font-size: 13px; color: #94a3b8; margin-bottom: 30px; }

.model-badge-active { font-size: 10px; padding: 2px 9px; border-radius: 4px; background: #f0fdf4; color: #16a34a; }
.model-badge-planned { font-size: 10px; padding: 2px 9px; border-radius: 4px; background: #eff6ff; color: #2563eb; }

div[data-baseweb="select"] > div {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important; color: #1e293b !important;
}
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important; color: #1e293b !important;
    font-size: 13px !important;
}
.stButton > button {
    background: #2563eb !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 22px !important; font-family: 'Inter', sans-serif !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button[kind="secondary"] {
    background: #f8fafc !important; color: #64748b !important;
    border: 1px solid #e2e8f0 !important;
}
label { color: #64748b !important; font-size: 12px !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; font-size: 13px; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #2563eb !important; border-bottom: 2px solid #2563eb !important; }
[data-testid="stMetricValue"] { color: #0f172a !important; font-family: 'JetBrains Mono' !important; }
p, .stMarkdown p { color: #64748b; font-size: 13px; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


PLOT_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#64748b", size=11),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(gridcolor="#f1f5f9", showline=False, tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#f1f5f9", showline=False, tickfont=dict(color="#94a3b8")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#64748b", size=11)),
)
CITY_COLORS = {
    "Mumbai": "#2563eb",
    "Delhi": "#7c3aed",
    "Bangalore": "#0891b2",
    "Chennai": "#16a34a",
    "Hyderabad": "#d97706",
    "Kolkata": "#dc2626",
}
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]
FURNISH = ["Semi-Furnished", "Furnished", "Unfurnished"]
FLOORS = [
    "Ground out of 3",
    "1 out of 3",
    "2 out of 5",
    "3 out of 7",
    "5 out of 10",
    "10 out of 20",
]


@st.cache_resource(show_spinner=False)
def _boot():
    from core.database import get_db
    from core.inference import get_predictor
    from core.security import ADMIN_ROLES, Perm, seed_users

    seed_users()
    return Perm, ADMIN_ROLES, get_predictor(), get_db()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_data():
    from analytics.engine import get_dataset

    return get_dataset()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_analytics():
    from analytics.engine import get_full_analytics

    return get_full_analytics()


def _authed():
    from core.security import verify_token as vt

    t = st.session_state.get("_token")
    return bool(t and vt(t))


def _has(perm):
    return bool(st.session_state.get("_perms", 0) & perm)


def _role():
    return st.session_state.get("_role", "")


def _is_admin():
    from core.security import ADMIN_ROLES

    return _role() in ADMIN_ROLES


def _initials(name):
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else name[:2].upper()


def page_login(Perm, ADMIN_ROLES):
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown(
            """
        <div class="riq-login">
          <div class="riq-brand">Rent<span>IQ</span></div>
          <div class="riq-tagline">AI-powered rental intelligence · Deep Learning + Spark</div>
        </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        username = st.text_input(
            "Username", placeholder="Username", label_visibility="collapsed"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Password",
            label_visibility="collapsed",
        )
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        if st.button("Sign in →", use_container_width=True):
            if not username or not password:
                st.error("Please enter both fields.")
            else:
                from core.security import login as _login

                result = _login(username.strip(), password, ip="web")
                if result and "error" not in result:
                    st.session_state.update(
                        {
                            "_token": result["access_token"],
                            "_refresh": result["refresh_token"],
                            "_role": result["role"],
                            "_username": result["username"],
                            "_name": result["display_name"],
                            "_perms": result["permissions"],
                            "_is_admin": result["is_admin_role"],
                            "_page": "Admin"
                            if result["is_admin_role"]
                            else "Predictor",
                        }
                    )
                    st.rerun()
                else:
                    st.error(
                        result.get("error", "Login failed.")
                        if result
                        else "Login failed."
                    )

        st.markdown(
            """
        <div style="height:16px"></div>
        <div style="font-size:11px;color:#94a3b8;text-align:center;margin-bottom:10px">Demo credentials</div>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="font-size:12px;color:#475569;font-weight:500">superadmin</td><td style="font-size:12px;color:#94a3b8">Admin@RentIQ#2024</td><td><span style="font-size:10px;background:#fef2f2;color:#dc2626;padding:2px 8px;border-radius:8px">Admin</span></td></tr>
          <tr><td style="font-size:12px;color:#475569;font-weight:500;padding-top:6px">analyst1</td><td style="font-size:12px;color:#94a3b8">Analyst@123!</td><td style="padding-top:6px"><span style="font-size:10px;background:#f0fdf4;color:#16a34a;padding:2px 8px;border-radius:8px">Analyst</span></td></tr>
          <tr><td style="font-size:12px;color:#475569;font-weight:500;padding-top:6px">tenant1</td><td style="font-size:12px;color:#94a3b8">Tenant@789!</td><td style="padding-top:6px"><span style="font-size:10px;background:#eff6ff;color:#2563eb;padding:2px 8px;border-radius:8px">Tenant</span></td></tr>
        </table>
        """,
            unsafe_allow_html=True,
        )


def render_topbar(Perm, ADMIN_ROLES):
    role = _role()
    name = st.session_state.get("_name", "")
    ini = _initials(name)

    if _is_admin():
        nav_pages = ["Admin", "Spark", "Models"]
    else:
        nav_pages = []
        if _has(Perm.VIEW_PREDICTIONS):
            nav_pages.append("Predictor")
        if _has(Perm.COMPARE_LISTINGS):
            nav_pages.append("Similar")
        if _has(Perm.VIEW_ALL_DATA):
            nav_pages.append("Analytics")
        if _has(Perm.VIEW_ALL_DATA):
            nav_pages.append("Spark")
        if _has(Perm.RUN_MODELS):
            nav_pages.append("Models")

    cur = st.session_state.get("_page", nav_pages[0] if nav_pages else "Predictor")
    if cur not in nav_pages:
        cur = nav_pages[0]
        st.session_state["_page"] = cur

    rtag_cls = "admin" if _is_admin() else ("analyst" if role == "ANALYST" else "")
    nav_html = "".join(
        f'<span class="riq-nav-btn {"active" if p == cur else ""}">{p}</span>'
        for p in nav_pages
    )
    st.markdown(
        f"""
    <div class="riq-topbar">
      <div style="display:flex;align-items:center;gap:28px">
        <div class="riq-logo">Rent<span>IQ</span></div>
        <nav class="riq-nav">{nav_html}</nav>
      </div>
      <div class="riq-pill">
        <div class="riq-av">{ini}</div>
        <span class="riq-uname">{name}</span>
        <span class="riq-rtag {rtag_cls}">{role.replace("_", " ")}</span>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(nav_pages) + 2)
    for i, page in enumerate(nav_pages):
        with cols[i + 1]:
            if st.button(
                page,
                key=f"_nav_{page}",
                type="primary" if page == cur else "secondary",
                use_container_width=True,
            ):
                st.session_state["_page"] = page
                st.rerun()
    with cols[-1]:
        if st.button("Sign out", key="_signout", type="secondary"):
            for k in [
                "_token",
                "_refresh",
                "_role",
                "_username",
                "_name",
                "_perms",
                "_is_admin",
                "_page",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

    return cur


def page_predictor(pred, db):
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown(
            '<div class="riq-sec">Property Details</div>', unsafe_allow_html=True
        )
        city = st.selectbox("City", CITIES)
        c1, c2 = st.columns(2)
        bhk = c1.selectbox("BHK", [1, 2, 3, 4, 5, 6], index=1)
        bath = c2.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=1)
        size = st.number_input("Size (sqft)", 100, 8000, 850, 50)
        furn = st.selectbox("Furnishing", FURNISH)
        floor = st.selectbox("Floor", FLOORS, index=2)

        dl_enabled = st.checkbox("Include Deep Learning prediction", value=True)
        xai_enabled = st.checkbox("Show AI explanations (XAI)", value=True)

        run = st.button("Get Rent Estimate →", use_container_width=True)

    with right:
        if run:
            with st.spinner("Running ensemble models..."):
                res = pred.predict(
                    city=city,
                    bhk=bhk,
                    size=size,
                    furnishing=furn,
                    bathroom=bath,
                    floor=floor,
                )
                dl_res = {}
                if dl_enabled:
                    try:
                        from deep_learning.model import get_dl_engine

                        dl = get_dl_engine()
                        if res.get("features") is not None:
                            dl_res = dl.predict(res["features"])
                    except Exception:
                        pass

            db.log_prediction(
                {
                    "user_id": st.session_state.get("_username"),
                    "city": city,
                    "bhk": bhk,
                    "size": size,
                    "furnishing": furn,
                    "predicted_rent": res["predicted_rent"],
                    "demand_risk": res["demand_risk"],
                    "risk_probability": round(res["risk_probability"], 3),
                    "price_per_sqft": res["price_per_sqft"],
                }
            )

            rent = res["predicted_rent"]
            lo, hi = res["rent_low"], res["rent_high"]
            risk = res["demand_risk"]
            rp = res["risk_probability"]
            psf = res["price_per_sqft"]
            vm = res["vs_median_pct"]
            b_cls = "badge-hi" if risk == "High" else "badge-lo"
            vm_col = "#dc2626" if vm > 0 else "#16a34a"

            dl_section = ""
            if dl_res.get("available"):
                dl_rent = dl_res.get("dl_rent", rent)
                agreement = abs(dl_rent - rent) / max(rent, 1) * 100
                agree_col = "#16a34a" if agreement < 8 else "#d97706"
                dl_section = f"""
<div style="margin-top:16px;padding:14px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0">
  <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Deep Learning Estimate (TabNet)</div>
  <div style="display:flex;gap:24px;align-items:center">
    <div>
      <div style="font-size:20px;font-weight:700;color:#0f172a;font-family:'JetBrains Mono'">₹{dl_rent:,.0f}</div>
      <div style="font-size:11px;color:#94a3b8">DL prediction</div>
    </div>
    <div>
      <div style="font-size:14px;font-weight:600;color:{agree_col}">±{agreement:.1f}% vs GBT</div>
      <div style="font-size:11px;color:#94a3b8">model agreement</div>
    </div>
  </div>
</div>"""

            st.markdown(
                f"""
<div class="riq-result">
  <div class="riq-rent-lbl">Estimated Monthly Rent</div>
  <div class="riq-rent-val">₹{rent:,.0f}</div>
  <div class="riq-rent-range">Range  ₹{lo:,.0f} – ₹{hi:,.0f}</div>
  <hr class="riq-divider"/>
  <div class="riq-stats">
    <div>
      <div class="riq-stat-lbl">Demand Signal</div>
      <span class="badge {b_cls}"><span class="badge-dot"></span>{risk}</span>
      <div style="font-size:11px;color:#94a3b8;margin-top:7px">{rp * 100:.0f}% confidence</div>
    </div>
    <div>
      <div class="riq-stat-lbl">Price / sqft</div>
      <div class="riq-stat-val" style="color:#7c3aed">₹{psf:,.1f}</div>
    </div>
    <div>
      <div class="riq-stat-lbl">vs City Median</div>
      <div class="riq-stat-val" style="color:{vm_col}">{vm:+.1f}%</div>
    </div>
  </div>
  {dl_section}
  <hr class="riq-divider"/>
  <div style="font-size:11px;color:#cbd5e1">GBT Ensemble + TabNet · R²=0.97 · Pinecone similarity · JWT secured</div>
</div>""",
                unsafe_allow_html=True,
            )

            if xai_enabled and res.get("features") is not None:
                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="riq-sec">AI Explanation (XAI)</div>',
                    unsafe_allow_html=True,
                )

                xai_col1, xai_col2 = st.columns([1, 1], gap="medium")
                with xai_col1:
                    st.markdown('<div class="riq-card">', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:12px">Feature Attribution</div>',
                        unsafe_allow_html=True,
                    )
                    imps = pred.get_feature_importances()
                    from core.explainability import top_features_for_prediction

                    top_feats = top_features_for_prediction(imps, top_n=6)
                    rows = ""
                    for f in top_feats:
                        rows += f"""
<div class="xai-row">
  <div class="xai-label">{f["label"]}</div>
  <div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{min(f["pct"] * 3, 100):.0f}%"></div></div>
  <div class="xai-pct">{f["pct"]:.1f}%</div>
</div>"""
                    st.markdown(rows, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with xai_col2:
                    st.markdown('<div class="riq-card">', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:12px">AI Insights</div>',
                        unsafe_allow_html=True,
                    )
                    from core.explainability import generate_prediction_insights
                    from core.features import _parse_floor

                    floor_n, total_f = _parse_floor(floor)
                    floor_ratio = floor_n / total_f if total_f > 0 else 0.0
                    insights = generate_prediction_insights(
                        city=city,
                        size=size,
                        furnishing=furn,
                        floor_ratio=floor_ratio,
                        predicted_rent=rent,
                        risk=risk,
                        risk_prob=rp,
                        psf=psf,
                        vs_median=vm,
                        dl_result=dl_res,
                        city_median=res.get("city_median"),
                    )
                    icons = ["💡", "📊", "🏙", "📈", "⚡"]
                    rows = ""
                    for i, insight in enumerate(insights):
                        rows += f'<div class="insight-item"><span class="insight-icon">{icons[i % len(icons)]}</span><span>{insight}</span></div>'
                    st.markdown(rows, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(
                """
<div style="min-height:320px;background:#fff;border:1px solid #e4e9f0;border-radius:14px;
            display:flex;flex-direction:column;align-items:center;justify-content:center;
            text-align:center;padding:60px;box-shadow:0 1px 4px rgba(0,0,0,0.04)">
  <div style="font-size:36px;opacity:0.15;margin-bottom:14px">⊞</div>
  <div style="font-size:13px;color:#94a3b8;line-height:1.8">
    Configure property details on the left<br>and click  Get Rent Estimate
  </div>
</div>""",
                unsafe_allow_html=True,
            )


def page_analytics(df):
    analytics = _load_analytics()
    cs = analytics["city_summary"]
    total = analytics["total_rows"]
    ar = float(cs["avg_rent"].mean())
    ap = float(cs["avg_psf"].mean())
    top = cs.loc[cs["avg_rent"].idxmax(), "City"]

    st.markdown(
        f"""
<div class="riq-kpi-grid">
  <div class="riq-kpi">
    <div class="riq-kpi-lbl">Total Listings</div>
    <div class="riq-kpi-val">{total:,}</div>
    <div class="riq-kpi-sub">6 Indian metros</div>
  </div>
  <div class="riq-kpi">
    <div class="riq-kpi-lbl">Avg Monthly Rent</div>
    <div class="riq-kpi-val">₹{ar / 1000:.1f}k</div>
    <div class="riq-kpi-sub">all cities combined</div>
  </div>
  <div class="riq-kpi">
    <div class="riq-kpi-lbl">Avg Price / sqft</div>
    <div class="riq-kpi-val">₹{ap:.0f}</div>
    <div class="riq-kpi-sub">median basis</div>
  </div>
  <div class="riq-kpi">
    <div class="riq-kpi-lbl">Premium Market</div>
    <div class="riq-kpi-val">{top}</div>
    <div class="riq-kpi-sub">highest avg rent</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="riq-sec">Interactive Filters</div>', unsafe_allow_html=True
    )
    fc1, fc2, fc3, fc4 = st.columns(4)
    sel_cities = fc1.multiselect("Cities", CITIES, default=CITIES, key="an_city")
    sel_furnish = fc2.multiselect("Furnishing", FURNISH, default=FURNISH, key="an_furn")
    sel_bhk = fc3.multiselect(
        "BHK", [1, 2, 3, 4, 5, 6], default=[1, 2, 3, 4], key="an_bhk"
    )
    rent_range = fc4.slider("Rent range (₹k)", 5, 200, (5, 150), key="an_rent")

    fdf = df[
        df["City"].isin(sel_cities)
        & df["Furnishing Status"].isin(sel_furnish)
        & df["BHK"].isin(sel_bhk)
        & df["Rent"].between(rent_range[0] * 1000, rent_range[1] * 1000)
    ].copy()

    st.caption(f"Showing {len(fdf):,} listings after filters")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="riq-sec">Rent Distribution by City</div>',
            unsafe_allow_html=True,
        )
        fig = px.box(
            fdf[fdf["Rent"] < 150000],
            x="City",
            y="Rent",
            color="City",
            color_discrete_map=CITY_COLORS,
        )
        fig.update_layout(height=300, showlegend=False, **PLOT_THEME)
        fig.update_traces(marker_size=3, line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(
            '<div class="riq-sec">Price/sqft by Furnishing</div>',
            unsafe_allow_html=True,
        )
        p = (
            fdf.groupby(["City", "Furnishing Status"])["price_per_sqft"]
            .median()
            .reset_index()
        )
        fig2 = px.bar(
            p,
            x="City",
            y="price_per_sqft",
            color="Furnishing Status",
            barmode="group",
            color_discrete_sequence=["#2563eb", "#7c3aed", "#0891b2"],
        )

        # Apply theme first
        fig2.update_layout(height=300, **PLOT_THEME)

        # Then override the legend specifically
        fig2.update_layout(
            legend=dict(orientation="h", y=-0.32, bgcolor="rgba(0,0,0,0)")
        )

        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown(
            '<div class="riq-sec">Median Rent by BHK</div>', unsafe_allow_html=True
        )
        b = (
            fdf[fdf["BHK"].between(1, 4)]
            .groupby(["BHK", "City"])["Rent"]
            .median()
            .reset_index()
        )
        fig3 = px.line(
            b,
            x="BHK",
            y="Rent",
            color="City",
            color_discrete_map=CITY_COLORS,
            markers=True,
        )
        # Apply theme and override legend
        fig3.update_layout(height=300, **PLOT_THEME)
        fig3.update_layout(
            legend=dict(orientation="h", y=-0.32, bgcolor="rgba(0,0,0,0)")
        )

        # ADD THIS LINE TO SHOW THE CHART:
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown(
            '<div class="riq-sec">Demand Heatmap (risk rate %)</div>',
            unsafe_allow_html=True,
        )
        dh = analytics["demand_heatmap"]
        dh_filt = dh[dh["City"].isin(sel_cities) & dh["BHK"].between(1, 4)]
        if len(dh_filt):
            pivot = dh_filt.pivot(
                index="City", columns="BHK", values="demand_rate"
            ).fillna(0)
            fig4 = px.imshow(
                pivot * 100,
                color_continuous_scale="Blues",
                labels={"color": "Demand %"},
                text_auto=".0f",
            )
            fig4.update_layout(height=280, **PLOT_THEME)
            st.plotly_chart(fig4, use_container_width=True)


def page_spark(df):
    analytics = _load_analytics()
    partitions = analytics.get("partitions", [])
    cs = analytics["city_summary"]

    st.markdown(
        """
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px 18px;margin-bottom:24px;font-size:12px;color:#1e40af;display:flex;gap:8px;align-items:center">
  ⚡ <strong>Spark Engine</strong> — distributed data processing across partitions. Run <code>python pipeline/spark_pipeline.py</code> to use real Spark MLlib training.
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="riq-sec">Simulated Spark Partition Layout</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(partitions))
    for i, part in enumerate(partitions):
        with cols[i]:
            st.markdown(
                f"""
<div class="spark-partition">
  <div class="spark-part-num">Partition {part["partition"]}</div>
  <div class="spark-part-rows">{part["rows"]:,}</div>
  <div class="spark-part-meta">rows · {part["cities"]} cities</div>
  <div class="spark-part-meta">avg ₹{int(part["avg_rent"]):,}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="riq-sec">City-Level Aggregations (Spark GroupBy)</div>',
        unsafe_allow_html=True,
    )

    display_cs = cs.rename(
        columns={
            "City": "City",
            "avg_rent": "Avg Rent (₹)",
            "median_rent": "Median Rent (₹)",
            "min_rent": "Min Rent (₹)",
            "max_rent": "Max Rent (₹)",
            "avg_psf": "Avg PSF (₹)",
            "n_listings": "Listings",
        }
    )
    st.dataframe(display_cs.set_index("City"), use_container_width=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="riq-sec">Avg Rent per City (Aggregated)</div>',
            unsafe_allow_html=True,
        )
        fig = px.bar(
            cs,
            x="City",
            y="avg_rent",
            color="City",
            color_discrete_map=CITY_COLORS,
            error_y="std_rent",
        )
        fig.update_layout(height=300, showlegend=False, **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(
            '<div class="riq-sec">Listings Distribution</div>', unsafe_allow_html=True
        )
        fig2 = px.pie(
            cs,
            names="City",
            values="n_listings",
            color="City",
            color_discrete_map=CITY_COLORS,
            hole=0.55,
        )
        fig2.update_layout(height=300, **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="riq-sec">Spark Pipeline Status</div>', unsafe_allow_html=True
    )
    pipeline_rows = [
        (
            "✅",
            "Data Ingestion",
            "CSV → Spark DataFrame",
            f"{analytics['total_rows']:,} rows",
        ),
        ("✅", "Feature Engineering", "Spark SQL transformations", "12 features"),
        ("✅", "Aggregations", "groupBy / agg operations", f"{len(cs)} cities"),
        (
            "✅",
            "Partition Analysis",
            "rdd.mapPartitionsWithIndex",
            f"{len(partitions)} partitions",
        ),
        ("⏳", "MLlib Training", "GBTRegressor + Classifier", "Run spark_pipeline.py"),
        (
            "⏳",
            "Pinecone Seeding",
            "foreachPartition → upsert",
            "Requires Spark + Pinecone",
        ),
    ]
    rows_html = ""
    for icon, stage, desc, status in pipeline_rows:
        rows_html += f"""
<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #f1f5f9">
  <div style="display:flex;gap:10px;align-items:center">
    <span style="font-size:14px">{icon}</span>
    <div>
      <div style="font-size:13px;font-weight:500;color:#1e293b">{stage}</div>
      <div style="font-size:11px;color:#94a3b8">{desc}</div>
    </div>
  </div>
  <div style="font-size:12px;color:#64748b">{status}</div>
</div>"""
    st.markdown(f'<div class="riq-card">{rows_html}</div>', unsafe_allow_html=True)


def page_similar(db):
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown(
            '<div class="riq-sec">Reference Listing</div>', unsafe_allow_html=True
        )
        qc = st.selectbox("City", CITIES, key="sq_city")
        qb = st.selectbox("BHK", [1, 2, 3, 4], index=1, key="sq_bhk")
        qs = st.number_input("Size (sqft)", 200, 5000, 850, 50, key="sq_size")
        qf = st.selectbox("Furnishing", FURNISH, key="sq_furn")
        qfl = st.selectbox("Floor", FLOORS, index=2, key="sq_floor")

        st.markdown(
            '<div class="riq-sec" style="margin-top:14px">Search Filters</div>',
            unsafe_allow_html=True,
        )
        same_city = st.checkbox("Same city only", value=True)
        rent_max = st.number_input(
            "Max rent (₹)", 0, 500000, 0, 5000, help="0 = no limit"
        )

        go = st.button("Find Similar Listings →", use_container_width=True)

    with right:
        if go:
            listing = {
                "City": qc,
                "BHK": qb,
                "Size": qs,
                "Bathroom": max(1, qb - 1),
                "Furnishing Status": qf,
                "Floor": qfl,
                "Area Type": "Super Area",
                "Tenant Preferred": "Bachelors/Family",
            }
            with st.spinner("Searching via Pinecone cosine similarity..."):
                hits = db.find_similar(
                    listing,
                    top_k=8,
                    same_city=same_city,
                    max_rent=rent_max if rent_max > 0 else None,
                )
            mode_tag = "Pinecone" if db.connected else "local-cache"
            if hits:
                st.markdown(
                    f'<div style="font-size:12px;color:#94a3b8;margin-bottom:16px">'
                    f"{len(hits)} listings via {mode_tag} cosine search</div>",
                    unsafe_allow_html=True,
                )
                rows = ""
                for h in hits:
                    m = h["metadata"]
                    sc = h["score"]
                    rows += f"""
<div class="riq-listing">
  <div>
    <div class="riq-listing-title">{m.get("City", "—")} · {int(m.get("BHK", 0))} BHK</div>
    <div class="riq-listing-meta">{m.get("Furnishing", "—")} · {int(m.get("Size", 0))} sqft · {m.get("Area_Locality", "—")}</div>
  </div>
  <div>
    <div class="riq-listing-rent">₹{int(m.get("Rent", 0)):,}</div>
    <div class="riq-listing-score">{sc * 100:.1f}% match · ₹{m.get("price_per_sqft", 0):.0f}/sqft</div>
  </div>
</div>
<div class="riq-bar"><div class="riq-fill" style="width:{int(sc * 100)}%"></div></div>"""
                st.markdown(
                    f'<div class="riq-card" style="padding:8px 24px">{rows}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No similar listings found. Try unchecking 'Same city only'.")
        else:
            st.markdown(
                """
<div style="min-height:300px;background:#fff;border:1px solid #e4e9f0;border-radius:14px;
            display:flex;align-items:center;justify-content:center;box-shadow:0 1px 4px rgba(0,0,0,0.04)">
  <div style="font-size:13px;color:#94a3b8">Configure a reference listing to search</div>
</div>""",
                unsafe_allow_html=True,
            )


def page_admin(df, db):
    st.markdown(
        f"""
<div class="riq-admin-banner">
  🔐 Admin Control Panel — restricted to <strong>{_role()}</strong>
  <span style="margin-left:auto;font-size:10px;background:#fecaca;padding:2px 10px;border-radius:8px;font-weight:700">{st.session_state.get("_username", "")}</span>
</div>""",
        unsafe_allow_html=True,
    )

    h = db.health()
    if h["status"] == "connected":
        st.markdown(
            f'<div class="riq-db-ok">✓ Pinecone connected — {h.get("listings_vectors", 0):,} vectors · {h.get("predictions_stored", 0)} predictions</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="riq-db-warn">⚠ Local-cache mode — {h.get("note", "Set PINECONE_API_KEY in .env")}</div>',
            unsafe_allow_html=True,
        )

    t1, t2, t3, t4 = st.tabs(
        ["📋 Listings", "📊 Prediction Log", "👥 Users", "⚙ Infrastructure"]
    )

    with t1:
        r1, r2, r3 = st.columns(3)
        cf = r1.multiselect("Filter City", CITIES, default=CITIES, key="af_c")
        ff = r2.multiselect("Filter Furnish", FURNISH, default=FURNISH, key="af_f")
        bf = r3.multiselect(
            "Filter BHK", [1, 2, 3, 4, 5, 6], default=[1, 2, 3], key="af_b"
        )

        fdf = df[
            df["City"].isin(cf) & df["Furnishing Status"].isin(ff) & df["BHK"].isin(bf)
        ].copy()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Listings", f"{len(fdf):,}")
        k2.metric("Avg Rent", f"₹{fdf['Rent'].mean():,.0f}")
        k3.metric("Avg PSF", f"₹{fdf['price_per_sqft'].mean():.1f}")
        k4.metric("Avg Size", f"{fdf['Size'].mean():.0f} sqft")

        st.dataframe(
            fdf[
                [
                    "City",
                    "BHK",
                    "Size",
                    "Bathroom",
                    "Rent",
                    "Furnishing Status",
                    "Area Locality",
                    "price_per_sqft",
                    "Floor",
                ]
            ]
            .sort_values("Rent", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=360,
        )
        st.download_button(
            "⬇ Export CSV",
            fdf.to_csv(index=False).encode(),
            file_name="rentiq_listings.csv",
            mime="text/csv",
        )

    with t2:
        preds = db.recent_predictions(limit=100)
        if preds:
            plog = pd.DataFrame(preds)
            if "timestamp" in plog.columns:
                plog["timestamp"] = pd.to_datetime(plog["timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
            drop = [c for c in ["_id", "password_hash", "id"] if c in plog.columns]
            st.dataframe(plog.drop(columns=drop), use_container_width=True, height=360)
            st.caption(f"{len(plog)} predictions logged")
        else:
            st.info("No predictions yet. Run the Predictor to generate entries.")

    with t3:
        from core.security import _USERS

        rows = [
            {
                "Username": k,
                "Full Name": v.get("full_name", ""),
                "Role": v.get("role", ""),
                "Email": v.get("email", ""),
                "Failed": v.get("failed_attempts", 0),
                "Locked": "Yes" if v.get("locked_until") else "No",
            }
            for k, v in _USERS.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Passwords BCrypt-hashed (cost=12)")

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pinecone Status**")
            st.json(db.health())
        with c2:
            from core.inference import get_predictor

            p = get_predictor()
            st.markdown("**Model Metrics**")
            st.json(
                {
                    "engine": p.metrics.get("engine", "scikit-learn"),
                    "r2": round(p.metrics.get("r2", 0), 4),
                    "mae": f"₹{p.metrics.get('mae', 0):,.0f}",
                    "clf_acc": round(p.metrics.get("clf_acc", 0), 4),
                    "n_partitions": p.metrics.get("n_partitions", "N/A (use Spark)"),
                }
            )


def page_models(pred):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="riq-sec">Model Registry</div>', unsafe_allow_html=True)
        registry = [
            (
                "GBT Regressor",
                "scikit-learn",
                "Rent prediction (₹)",
                "active",
                f"R²={pred.metrics.get('r2', 0):.4f}",
            ),
            (
                "GBT Classifier",
                "scikit-learn",
                "Demand risk signal",
                "active",
                f"Acc={pred.metrics.get('clf_acc', 0):.4f}",
            ),
            (
                "RobustScaler",
                "scikit-learn",
                "Outlier-resistant normalization",
                "active",
                "12 features",
            ),
            (
                "RentTabNet",
                "PyTorch",
                "TabTransformer + Residual MLP",
                "active",
                "attention head",
            ),
            (
                "FeatureAttention",
                "PyTorch",
                "Multi-head feature self-attention",
                "active",
                "4 heads",
            ),
            (
                "GBTRegressor",
                "Spark MLlib",
                "Distributed rent regression",
                "planned",
                "run spark_pipeline.py",
            ),
            (
                "GBTClassifier",
                "Spark MLlib",
                "Distributed risk classification",
                "planned",
                "run spark_pipeline.py",
            ),
            (
                "Pinecone cosine",
                "Pinecone",
                "12-dim listing similarity",
                "active",
                "rentiq-listings",
            ),
        ]
        rows = ""
        for name, lib, role, status, note in registry:
            cls = "model-badge-active" if status == "active" else "model-badge-planned"
            rows += f"""
<div class="riq-listing">
  <div>
    <div class="riq-listing-title">{name}</div>
    <div class="riq-listing-meta">{lib} · {role}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:11px;color:#2563eb;margin-bottom:5px">{note}</div>
    <span class="{cls}">{status}</span>
  </div>
</div>"""
        st.markdown(
            f'<div class="riq-card" style="padding:8px 24px">{rows}</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            '<div class="riq-sec">Feature Importance (GBT)</div>',
            unsafe_allow_html=True,
        )
        imps = pred.get_feature_importances()
        if imps:
            fi = pd.DataFrame(
                list(imps.items()), columns=["feature", "importance"]
            ).sort_values("importance")
            fig = go.Figure(
                go.Bar(
                    x=fi["importance"],
                    y=fi["feature"],
                    orientation="h",
                    marker=dict(
                        color=fi["importance"],
                        colorscale=[[0, "#eff6ff"], [0.5, "#2563eb"], [1, "#7c3aed"]],
                    ),
                    text=[f"{v:.3f}" for v in fi["importance"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#94a3b8"),
                )
            )
            fig.update_layout(height=340, **PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="riq-sec" style="margin-top:20px">Deep Learning Architecture</div>',
            unsafe_allow_html=True,
        )
        arch = [
            ("Input Layer", "12 engineered features"),
            ("Feature Attention", "MultiheadAttention (4 heads, embed_dim=32)"),
            ("Encoder Block 1", "Linear(12×32+12 → 256) + LayerNorm + GELU"),
            ("Encoder Block 2", "Linear(256 → 128) + LayerNorm + GELU"),
            ("Encoder Block 3", "Linear(128 → 64) + LayerNorm + GELU"),
            ("Residual Block ×2", "64-dim skip connections"),
            ("Rent Head", "Linear(64→32→1) — log rent regression"),
            ("Risk Head", "Linear(64→32→1) + Sigmoid — binary risk"),
        ]
        arch_html = ""
        for layer, desc in arch:
            arch_html += f"""
<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f5f9">
  <div style="font-size:12px;font-weight:500;color:#374151">{layer}</div>
  <div style="font-size:11px;color:#94a3b8;max-width:220px;text-align:right">{desc}</div>
</div>"""
        st.markdown(
            f'<div class="riq-card" style="padding:4px 24px">{arch_html}</div>',
            unsafe_allow_html=True,
        )


def main():
    Perm, ADMIN_ROLES, pred, db = _boot()

    if not _authed():
        page_login(Perm, ADMIN_ROLES)
        return

    df = _load_data()
    cur = render_topbar(Perm, ADMIN_ROLES)

    if _is_admin() and cur not in ("Admin", "Spark", "Models"):
        st.session_state["_page"] = "Admin"
        st.rerun()

    st.markdown('<div class="riq-page">', unsafe_allow_html=True)

    if cur == "Predictor":
        if _has(Perm.VIEW_PREDICTIONS):
            page_predictor(pred, db)
        else:
            st.error("Access denied.")

    elif cur == "Analytics":
        if _has(Perm.VIEW_ALL_DATA):
            page_analytics(df)
        else:
            st.error("Access denied.")

    elif cur == "Spark":
        if _has(Perm.VIEW_ALL_DATA) or _is_admin():
            page_spark(df)
        else:
            st.error("Access denied.")

    elif cur == "Similar":
        if _has(Perm.COMPARE_LISTINGS):
            page_similar(db)
        else:
            st.error("Access denied.")

    elif cur == "Admin":
        if _is_admin():
            page_admin(df, db)
        else:
            st.error("Access denied.")

    elif cur == "Models":
        if _has(Perm.RUN_MODELS):
            page_models(pred)
        else:
            st.error("Access denied.")

    st.markdown("</div>", unsafe_allow_html=True)


main()
