"""
PGA One and Done — Streamlit Dashboard

Run with:
    .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pga_oad.client import DataGolfClient
from pga_oad.kalshi import KalshiClient
from pga_oad.blend import SignalBlender, BlendedPlayer, _match_player
from pga_oad.models import Prediction

# Auth / DB / subscription imports
from sqlalchemy.orm import sessionmaker
from pga_oad.api.auth import hash_password, verify_password
from pga_oad.db.engine import init_db, get_engine
from pga_oad.db.crud import (
    create_user, get_user_by_email, get_user_by_username,
    get_subscription, update_subscription_tier,
)
from pga_oad.subscriptions import get_tier


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
HISTORY_YEARS = [2025, 2024, 2023, 2022, 2021]
CACHE_DIR = "data/raw"
_TIER_ORDER = {"free": 0, "pro": 1, "elite": 2}


# ──────────────────────────────────────────────
# Database init (once per process)
# ──────────────────────────────────────────────

@st.cache_resource
def _init_db():
    init_db()

_init_db()


def _db():
    """Return a new SQLAlchemy session. Caller must close it."""
    return sessionmaker(bind=get_engine())()


# ──────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────

def _do_login(email: str, password: str) -> str | None:
    """Validate credentials and populate session_state. Returns error string or None."""
    db = _db()
    try:
        user = get_user_by_email(db, email.strip().lower())
        if not user or not verify_password(password, user.password_hash):
            return "Invalid email or password."
        sub = get_subscription(db, user.id)
        tier = sub.tier if (sub and sub.is_active) else "free"
        st.session_state.user_id   = user.id
        st.session_state.username  = user.username
        st.session_state.email     = user.email
        st.session_state.tier_name = tier
        return None
    finally:
        db.close()


def _do_register(email: str, username: str, password: str) -> str | None:
    """Create account and populate session_state. Returns error string or None."""
    db = _db()
    try:
        if get_user_by_email(db, email.strip().lower()):
            return "Email already registered."
        if get_user_by_username(db, username.strip()):
            return "Username already taken."
        user = create_user(
            db, email.strip().lower(), username.strip(), hash_password(password)
        )
        db.commit()
        st.session_state.user_id   = user.id
        st.session_state.username  = user.username
        st.session_state.email     = user.email
        st.session_state.tier_name = "free"
        return None
    except Exception as exc:
        db.rollback()
        return str(exc)
    finally:
        db.close()


def _do_logout():
    for k in ("user_id", "username", "email", "tier_name"):
        st.session_state.pop(k, None)


def _upgrade_tier(new_tier: str):
    """Dev-mode tier upgrade: write to DB and refresh session_state."""
    uid = st.session_state.get("user_id")
    if not uid:
        return
    db = _db()
    try:
        update_subscription_tier(db, uid, new_tier)
        db.commit()
        st.session_state.tier_name = new_tier
    finally:
        db.close()


def _tier_gate(required: str, label: str) -> bool:
    """
    Return True if the current user's tier meets or exceeds `required`.
    Otherwise render a lock/upgrade prompt and return False.
    """
    current = st.session_state.get("tier_name", "free")
    if _TIER_ORDER.get(current, 0) >= _TIER_ORDER.get(required, 0):
        return True

    fi = get_tier(required)
    tier_emoji = {"pro": "⭐", "elite": "💎"}.get(required, "🔒")
    st.markdown(f"## {tier_emoji} {required.upper()} Feature")
    st.info(f"**{label}** requires a **{required.upper()}** subscription.")
    st.markdown(
        f"**{required.upper()} — ${fi.monthly_price:.2f}/mo** "
        f"or ${fi.yearly_price:.2f}/yr"
    )
    if st.session_state.get("user_id"):
        if st.button(
            f"Upgrade to {required.upper()}",
            type="primary",
            key=f"gate_upgrade_{required}",
        ):
            _upgrade_tier(required)
            st.rerun()
    else:
        st.warning("Sign in using the sidebar to upgrade your account.")
    return False


# ──────────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_predictions() -> dict:
    client = DataGolfClient(cache_dir=CACHE_DIR)
    return client.get_pre_tournament_predictions(tour="pga", odds_format="percent")


@st.cache_data(ttl=86400, show_spinner=False)
def load_schedule() -> list[dict]:
    client = DataGolfClient(cache_dir=CACHE_DIR)
    data = client.get_schedule(tour="pga", upcoming_only=True)
    return data.get("schedule", [])


@st.cache_data(ttl=86400, show_spinner=False)
def load_course_history(event_id: int) -> dict[int, dict[int, dict]]:
    """Returns {year: {dg_id: player_dict}} for HISTORY_YEARS."""
    client = DataGolfClient(cache_dir=CACHE_DIR)
    history: dict[int, dict[int, dict]] = {}
    for year in HISTORY_YEARS:
        try:
            arch = client.get_pre_tournament_archive(event_id=event_id, year=year)
            history[year] = {p["dg_id"]: p for p in arch.get("baseline", [])}
        except Exception:
            history[year] = {}
    return history


@st.cache_data(ttl=1800, show_spinner=False)
def load_kalshi_win(event_code: str) -> tuple[dict[str, float], set[str]]:
    """Returns (implied_probs, thin_player_names)."""
    client = KalshiClient(cache_dir=CACHE_DIR)
    probs = client.get_implied_probs(event_code, "win", normalize=True)
    thin = client.get_thin_players(event_code, "win")
    return probs, thin


@st.cache_data(ttl=1800, show_spinner=False)
def load_kalshi_raw(event_code: str, market: str) -> list[dict]:
    """Returns raw market data list for a given market type."""
    client = KalshiClient(cache_dir=CACHE_DIR)
    return client.get_market_data(event_code, market)


@st.cache_data(ttl=1800, show_spinner=False)
def detect_kalshi_event_code() -> str | None:
    client = KalshiClient(cache_dir=CACHE_DIR)
    return client.detect_current_event_code()


@st.cache_data(ttl=1800, show_spinner=False)
def load_proprietary_model(event_id: int, event_code: str) -> list:
    """Compute and cache the proprietary model output."""
    from pga_oad.proprietary import ProprietaryModel
    client = DataGolfClient(cache_dir=CACHE_DIR)
    kalshi_client = KalshiClient(cache_dir=CACHE_DIR)
    model = ProprietaryModel(client=client, kalshi=kalshi_client)
    return model.compute(event_id=event_id, event_code=event_code)


@st.cache_data(ttl=86400, show_spinner=False)
def load_schedule_winners(event_ids: tuple, years: tuple) -> dict:
    """
    Returns {event_id_str: {year: "First Last"}} for the given events and years.

    Fetches archive data for each (event_id, year) and extracts the winner
    (fin_text == "1"). Results are cached for 24 hours.
    """
    client = DataGolfClient(cache_dir=CACHE_DIR)
    result: dict = {}
    for event_id in event_ids:
        result[str(event_id)] = {}
        for year in years:
            try:
                arch = client.get_pre_tournament_archive(
                    event_id=int(event_id), year=int(year), odds_format="percent"
                )
                players = arch.get("baseline", [])
                winner = next(
                    (p["player_name"] for p in players if p.get("fin_text") == "1"),
                    None,
                )
                if winner:
                    # Convert "Last, First" → "First Last" for display
                    parts = winner.split(", ", 1)
                    display = f"{parts[1]} {parts[0]}" if len(parts) == 2 else winner
                    result[str(event_id)][year] = display
            except Exception:
                pass
    return result


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def finish_sort_key(fin: str) -> int:
    if not fin or fin in ("---", ""):
        return 9999
    clean = fin.lstrip("T").lstrip("=")
    try:
        return int(clean)
    except ValueError:
        return {"CUT": 200, "MDF": 199, "WD": 300, "DQ": 300}.get(fin, 9999)


def finish_badge(fin: str) -> str:
    """Return finish string unchanged — color applied via Styler."""
    return fin if fin else "—"


def dg_name_to_display(name: str) -> str:
    """'Gerard, Ryan' → 'Ryan Gerard'"""
    parts = name.split(", ", 1)
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else name


def parse_predictions(dg_data: dict) -> tuple[list[Prediction], dict[int, dict], dict[int, dict]]:
    baseline_raw = {p["dg_id"]: p for p in dg_data.get("baseline", [])}
    history_fit_raw = {p["dg_id"]: p for p in dg_data.get("baseline_history_fit", [])}
    preds = []
    for p in dg_data.get("baseline", []):
        try:
            preds.append(Prediction(
                player_name=p["player_name"],
                dg_id=p["dg_id"],
                win_prob=p.get("win", 0.0),
                top5_prob=p.get("top_5", 0.0),
                top10_prob=p.get("top_10", 0.0),
                top20_prob=p.get("top_20", 0.0),
                make_cut_prob=p.get("make_cut", 0.0),
            ))
        except Exception:
            pass
    return preds, baseline_raw, history_fit_raw


def build_rows(
    predictions: list[Prediction],
    baseline_raw: dict[int, dict],
    history_fit_raw: dict[int, dict],
    kalshi_win: dict[str, float],
    thin_players: set[str],
    history: dict[int, dict[int, dict]],
    use_history_model: bool,
) -> list[dict]:
    rows = []
    for pred in predictions:
        dg_id = pred.dg_id
        name = pred.player_name
        display_name = dg_name_to_display(name)

        base_win = baseline_raw.get(dg_id, {}).get("win", pred.win_prob)
        hf_win = history_fit_raw.get(dg_id, {}).get("win", base_win)
        fit_bump = hf_win - base_win
        dg_signal = hf_win if use_history_model else base_win

        # Kalshi match
        k_win_name = _match_player(name, kalshi_win)
        kal_val = k_win_name  # float or None
        # Check if thin
        if kal_val is not None:
            # Reverse-lookup name for thin check
            parts = name.split(", ", 1)
            normalized = f"{parts[1]} {parts[0]}".lower().strip() if len(parts) == 2 else name.lower()
            is_thin = any(normalized in tp.lower() for tp in thin_players)
            if is_thin:
                kal_val = None

        edge = (kal_val - dg_signal) if kal_val is not None else None

        # Course history
        year_fins: dict[int, str] = {}
        for year in HISTORY_YEARS:
            yr = history.get(year, {}).get(dg_id)
            year_fins[year] = yr.get("fin_text", "—") if yr else "—"

        best_fin = min(
            (finish_sort_key(f) for f in year_fins.values() if f not in ("—", "")),
            default=9999,
        )
        is_specialist = best_fin <= 10

        rows.append({
            "name": name,
            "display_name": display_name,
            "dg_id": dg_id,
            "base_win": base_win,
            "hf_win": hf_win,
            "dg_signal": dg_signal,
            "fit_bump": fit_bump,
            "top10": pred.top10_prob,
            "top20": pred.top20_prob,
            "make_cut": pred.make_cut_prob,
            "kal_win": kal_val,
            "edge": edge,
            "year_fins": year_fins,
            "best_fin": best_fin,
            "is_specialist": is_specialist,
        })

    # Sort by dg_signal descending and assign ranks
    rows.sort(key=lambda r: r["dg_signal"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["dg_rank"] = i

    kal_sorted = sorted(
        [r for r in rows if r["kal_win"] is not None],
        key=lambda r: r["kal_win"], reverse=True
    )
    for i, r in enumerate(kal_sorted, 1):
        r["kal_rank"] = i
    for r in rows:
        r.setdefault("kal_rank", None)

    return rows


def style_finish(val: str) -> str:
    k = finish_sort_key(val)
    if k == 1:
        return "background-color: #FFD700; color: black; font-weight: bold"
    elif k <= 5:
        return "background-color: #2ecc71; color: white; font-weight: bold"
    elif k <= 10:
        return "background-color: #a8e6cf; color: black"
    elif k <= 20:
        return "background-color: #ffeaa7; color: black"
    elif k in (200, 199):  # CUT / MDF
        return "background-color: #fab1a0; color: black"
    elif k >= 300:  # WD / DQ
        return "background-color: #d63031; color: white"
    return ""


def fmt_pct(v, decimals=1) -> str:
    if v is None:
        return "—"
    return f"{v*100:.{decimals}f}%"


def fmt_american(implied_prob: float | None) -> str:
    """Convert implied probability (with vig) to American odds string, e.g. +1800 or -150."""
    if implied_prob is None or implied_prob <= 0 or implied_prob >= 1:
        return "—"
    if implied_prob < 0.5:
        return f"+{round((1 / implied_prob - 1) * 100)}"
    return f"{round(-implied_prob / (1 - implied_prob) * 100)}"


def fmt_edge(v) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.1f}%"


def build_prop_tooltip(p) -> str:
    """Build the hover tooltip text showing how Prop Win% was calculated."""
    w = p.weights_used
    lines = []
    dg_pct  = p.dg_win_prob_history * 100
    dg_wt   = w.get("dg", 0) * 100
    dg_cont = p.dg_win_prob_history * w.get("dg", 0) * 100
    lines.append(f"{'DG Skill+Hist':14s} ({dg_wt:4.0f}%):  {dg_pct:5.2f}%  →  {dg_cont:5.2f}%")

    mkt = p.market_consensus_prob
    mkt_wt = w.get("market", 0) * 100
    if mkt is not None and mkt_wt > 0:
        mkt_cont = mkt * w["market"] * 100
        lines.append(f"{'DraftKings':14s} ({mkt_wt:4.0f}%):  {mkt*100:5.2f}%  →  {mkt_cont:5.2f}%")
    elif mkt_wt == 0:
        lines.append(f"{'DraftKings':14s}  (  0%):  n/a")

    kal = p.kalshi_win_prob
    kal_wt = w.get("kalshi", 0) * 100
    if kal is not None and kal_wt > 0:
        kal_cont = kal * w["kalshi"] * 100
        lines.append(f"{'Kalshi':14s} ({kal_wt:4.0f}%):  {kal*100:5.2f}%  →  {kal_cont:5.2f}%")
    elif kal_wt == 0:
        lines.append(f"{'Kalshi':14s}  (  0%):  n/a")

    frm = getattr(p, "recent_form_score", None)
    frm_wt = w.get("form", 0) * 100
    if frm is not None and frm_wt > 0:
        frm_cont = frm * w["form"] * 100
        lines.append(f"{'Form Score':14s} ({frm_wt:4.0f}%):  {frm:5.2f}   →  {frm_cont:5.2f}%")
    elif frm_wt == 0:
        lines.append(f"{'Form Score':14s}  (  0%):  n/a")

    crs = p.recency_course_score
    crs_wt = w.get("history", 0) * 100
    if crs is not None and crs_wt > 0:
        crs_cont = crs * w["history"] * 100
        lines.append(f"{'Crs Score':14s} ({crs_wt:4.0f}%):  {crs:5.2f}   →  {crs_cont:5.2f}%")
    elif crs_wt == 0:
        lines.append(f"{'Crs Score':14s}  (  0%):  n/a")

    lines.append("─" * 40)
    lines.append(f"{'Prop Win%':14s}         {p.proprietary_win_prob*100:5.2f}%")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PGA One and Done",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .pick-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 24px 32px;
        margin: 8px 0 20px 0;
        color: white;
    }
    .pick-name { font-size: 2rem; font-weight: 700; color: #e94560; margin: 4px 0; }
    .pick-meta { font-size: 0.9rem; color: #aaa; margin-top: 8px; }
    .pick-stat { font-size: 1.1rem; margin: 2px 0; }
    .stat-label { color: #888; font-size: 0.85rem; }
    .warning-box {
        background: #2d1b00; border-left: 4px solid #f39c12;
        padding: 10px 16px; border-radius: 4px; margin: 8px 0;
        color: #f9ca24;
    }
    /* Proprietary model HTML table */
    .prop-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .prop-table th {
        background: #1e1e2e; color: #bbb; padding: 6px 10px;
        text-align: left; border-bottom: 2px solid #444; white-space: nowrap;
    }
    .prop-table td { padding: 5px 10px; border-bottom: 1px solid #2a2a3a; white-space: nowrap; }
    .prop-table tr:hover td { background-color: rgba(255,255,255,0.04); }
    .prop-table .used td { opacity: 0.35; text-decoration: line-through; }
    /* Tooltip on Prop Win% cells */
    .prop-table .tip { position: relative; cursor: help; border-bottom: 1px dashed #888; }
    .prop-table .tip:hover::after {
        content: attr(data-tip);
        position: absolute; left: 0; top: 110%;
        white-space: pre; font-family: 'Courier New', monospace; font-size: 0.76rem;
        background: #12122a; color: #e8e8f0;
        border: 1px solid #555; border-radius: 6px;
        padding: 10px 14px; z-index: 9999; min-width: 340px;
        box-shadow: 3px 3px 12px rgba(0,0,0,0.7); pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data loading (with spinner)
# ──────────────────────────────────────────────
with st.spinner("Loading data..."):
    dg_data = load_predictions()
    schedule = load_schedule()
    event_code = detect_kalshi_event_code()

event_name = dg_data.get("event_name", "Current Event")
last_updated = dg_data.get("last_updated", "")
predictions, baseline_raw, history_fit_raw = parse_predictions(dg_data)

# Find event_id from schedule
event_id = None
for e in schedule:
    if event_name.lower() in e.get("event_name", "").lower() or \
       e.get("event_name", "").lower() in event_name.lower():
        event_id = e.get("event_id")
        break
if event_id is None and schedule:
    event_id = schedule[0].get("event_id")

with st.spinner("Loading course history..."):
    history = load_course_history(event_id) if event_id else {}

kalshi_win: dict[str, float] = {}
thin_players: set[str] = set()
if event_code:
    with st.spinner("Loading Kalshi markets..."):
        kalshi_win, thin_players = load_kalshi_win(event_code)

dg_ok = len(predictions) > 0
kalshi_ok = len(kalshi_win) > 0
hist_ok = any(len(v) > 0 for v in history.values())

all_player_display_names = sorted(dg_name_to_display(p.player_name) for p in predictions)
display_to_dg_name = {dg_name_to_display(p.player_name): p.player_name for p in predictions}
display_to_dg_id = {dg_name_to_display(p.player_name): p.dg_id for p in predictions}


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⛳ PGA One & Done")
    st.caption(f"**{event_name}**")

    # Data status
    st.markdown("**Data Status**")
    st.markdown(f"{'✅' if dg_ok else '❌'} DataGolf ({len(predictions)} players)")
    st.markdown(f"{'✅' if kalshi_ok else '❌'} Kalshi ({len(kalshi_win)} markets)")
    st.markdown(f"{'✅' if hist_ok else '❌'} Course history (event #{event_id})")
    if last_updated:
        st.caption(f"Updated: {last_updated[:16]} UTC")

    st.divider()

    # OAD Settings
    st.markdown("**One & Done Settings**")
    used_display = st.multiselect(
        "Players already used this season",
        options=all_player_display_names,
        help="These players will be excluded from recommendations and grayed out in the table.",
    )
    used_player_ids: set[int] = {display_to_dg_id[n] for n in used_display if n in display_to_dg_id}

    dg_weight = st.slider(
        "DataGolf model weight",
        min_value=0.0, max_value=1.0, value=0.65, step=0.05,
        format="%.0f%%",
        help="Kalshi weight is automatically 1 − DG weight.",
    )
    kalshi_weight = round(1.0 - dg_weight, 2)
    st.caption(f"Kalshi weight: {kalshi_weight:.0%}")

    use_history_model = st.radio(
        "DataGolf model",
        options=["Skill + Course History", "Skill only (baseline)"],
        index=0,
    ) == "Skill + Course History"

    st.divider()

    # Display filters
    st.markdown("**Display Filters**")
    min_win_pct = st.slider(
        "Min win% to show",
        min_value=0.0, max_value=5.0, value=0.0, step=0.1,
        format="%.1f%%",
    )
    specialists_only = st.checkbox("Course specialists only (top-10 history)")

    st.divider()
    if st.button("↺ Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── Account ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Account**")

    if not st.session_state.get("user_id"):
        signin_tab, register_tab = st.tabs(["Sign In", "Register"])

        with signin_tab:
            with st.form("login_form"):
                li_email    = st.text_input("Email", key="li_email")
                li_password = st.text_input("Password", type="password", key="li_pass")
                if st.form_submit_button("Sign In", use_container_width=True):
                    err = _do_login(li_email, li_password)
                    if err:
                        st.error(err)
                    else:
                        st.rerun()

        with register_tab:
            with st.form("register_form"):
                re_email    = st.text_input("Email", key="re_email")
                re_username = st.text_input("Username", key="re_user")
                re_password = st.text_input("Password", type="password", key="re_pass")
                if st.form_submit_button("Create Account", use_container_width=True):
                    err = _do_register(re_email, re_username, re_password)
                    if err:
                        st.error(err)
                    else:
                        st.rerun()
    else:
        _tier_name = st.session_state.tier_name
        _tier_badge = {"free": "🆓 FREE", "pro": "⭐ PRO", "elite": "💎 ELITE"}.get(
            _tier_name, _tier_name.upper()
        )
        st.markdown(f"**👤 {st.session_state.username}**")
        st.caption(st.session_state.email)
        st.markdown(f"Tier: **{_tier_badge}**")

        if _tier_name != "elite":
            st.divider()
            st.caption("Dev mode — upgrade without Stripe")
            _upgrades = [
                t for t in ["pro", "elite"]
                if _TIER_ORDER[t] > _TIER_ORDER.get(_tier_name, 0)
            ]
            _upgrade_to = st.selectbox("Select tier", ["—"] + _upgrades, key="sidebar_upgrade")
            if st.button("Apply Upgrade", use_container_width=True) and _upgrade_to != "—":
                _upgrade_tier(_upgrade_to)
                st.rerun()

        st.divider()
        if st.button("Sign Out", use_container_width=True):
            _do_logout()
            st.rerun()


# ──────────────────────────────────────────────
# Build data rows (recomputed when sidebar changes)
# ──────────────────────────────────────────────
rows = build_rows(
    predictions, baseline_raw, history_fit_raw,
    kalshi_win, thin_players, history, use_history_model,
)

# Apply display filters
dg_signal_label = "DG+Hist%" if use_history_model else "DG Base%"
display_rows = [
    r for r in rows
    if r["dg_signal"] * 100 >= min_win_pct
    and (not specialists_only or r["is_specialist"])
]


# ──────────────────────────────────────────────
# Main header
# ──────────────────────────────────────────────
st.header(f"⛳ {event_name}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Field size", len(predictions))
col2.metric("Kalshi markets", len(kalshi_win))
col3.metric("Course specialists", sum(1 for r in rows if r["is_specialist"]))
col4.metric("Players used", len(used_player_ids))

st.divider()


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Full Field",
    "🏌️ Course History",
    "📈 DG vs Kalshi",
    "🎯 OAD Pick",
    "💰 Kalshi Markets",
    "🔬 Proprietary Model",
])


# ══════════════════════════════════════════════
# TAB 1 — Full Field
# ══════════════════════════════════════════════
with tab1:
    st.subheader(f"Full Field — {len(display_rows)} players shown")

    # Build dataframe
    records = []
    for r in display_rows:
        is_used = r["dg_id"] in used_player_ids
        name_str = ("~~" + r["display_name"] + "~~") if is_used else r["display_name"]
        if r["is_specialist"]:
            name_str = ("★ " if not is_used else "~~★ ") + r["display_name"] + (" ~~" if is_used else "")

        kal_str = fmt_pct(r["kal_win"]) if r["kal_win"] is not None else "—"
        edge_str = fmt_edge(r["edge"]) if r["edge"] is not None else "—"
        kal_rank_str = str(r["kal_rank"]) if r["kal_rank"] else "—"

        row = {
            "DG#": r["dg_rank"],
            "Kal#": kal_rank_str,
            "Player": r["display_name"],
            "Used": "✓" if is_used else "",
            "Spec": "★" if r["is_specialist"] else "",
            dg_signal_label: fmt_pct(r["dg_signal"]),
            "DG Base%": fmt_pct(r["base_win"]),
            "Fit": fmt_edge(r["fit_bump"]),
            "Kal Win%": kal_str,
            "Edge": edge_str,
            "Top10%": fmt_pct(r["top10"]),
            "Cut%": fmt_pct(r["make_cut"]),
        }
        for year in HISTORY_YEARS:
            row[str(year)] = r["year_fins"].get(year, "—")
        records.append(row)

    df = pd.DataFrame(records)

    # Color styling
    def color_row(row):
        styles = [""] * len(row)
        idx = df.columns.tolist()
        player_col = idx.index("Player")
        used_col = idx.index("Used")
        spec_col = idx.index("Spec")

        is_used = row["Used"] == "✓"
        is_spec = row["Spec"] == "★"

        if is_used:
            styles = ["opacity: 0.4; text-decoration: line-through"] * len(row)
        elif is_spec:
            styles = ["background-color: #e8f5e9"] * len(row)
        return styles

    def color_edge(val):
        if val == "—" or not isinstance(val, str):
            return ""
        try:
            num = float(val.replace("%", "").replace("+", ""))
            if num >= 1.0:
                return "background-color: #ff7675; color: white; font-weight: bold"
            elif num >= 0.5:
                return "background-color: #fab1a0"
            elif num <= -1.0:
                return "background-color: #74b9ff; color: white; font-weight: bold"
            elif num <= -0.5:
                return "background-color: #a8d8ff"
        except Exception:
            pass
        return ""

    def color_finish_cell(val):
        return style_finish(str(val))

    year_cols = [str(y) for y in HISTORY_YEARS]
    styled = (
        df.style
        .apply(color_row, axis=1)
        .applymap(color_edge, subset=["Edge"])
        .applymap(color_finish_cell, subset=year_cols)
    )

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
    )

    st.caption("★ = top-10 at this course in last 3 years  |  Edge = Kal% minus DG%  |  Red edge = Kalshi more bullish  |  Blue edge = DataGolf more bullish")


# ══════════════════════════════════════════════
# TAB 2 — Course History
# ══════════════════════════════════════════════
with tab2:
    if _tier_gate("pro", "Course History"):
        st.subheader(f"Course History — Last 5 Years at {event_name}")

        show_top20_only = st.checkbox("Show only players with at least one top-20 finish here")

        hist_rows = []
        for r in rows:
            best = r["best_fin"]
            has_top20 = any(
                finish_sort_key(r["year_fins"].get(y, "—")) <= 20
                for y in HISTORY_YEARS
            )
            if show_top20_only and not has_top20:
                continue
            if best == 9999:
                continue  # Never played here

            record = {
                "Player": r["display_name"],
                "DG#": r["dg_rank"],
                dg_signal_label: fmt_pct(r["dg_signal"]),
                "Kal%": fmt_pct(r["kal_win"]) if r["kal_win"] is not None else "—",
                "Best": str(best) if best < 9999 else "—",
            }
            for year in HISTORY_YEARS:
                record[str(year)] = r["year_fins"].get(year, "—")
            record["_best_sort"] = best
            hist_rows.append(record)

        hist_rows.sort(key=lambda x: x["_best_sort"])
        hist_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_best_sort"} for r in hist_rows])

        if hist_df.empty:
            st.info("No course history found for current field. Try unchecking the filter.")
        else:
            year_cols = [str(y) for y in HISTORY_YEARS]
            styled_hist = hist_df.style.applymap(color_finish_cell, subset=year_cols)
            st.dataframe(styled_hist, use_container_width=True, height=500, hide_index=True)

            st.caption(
                "🥇 Gold = Win  🟢 Dark green = top-5  🟩 Light green = top-10  "
                "🟡 Yellow = top-20  🟠 Salmon = CUT  🔴 Red = WD/DQ  — = not in field"
            )


# ══════════════════════════════════════════════
# TAB 3 — DG vs Kalshi Divergences
# ══════════════════════════════════════════════
with tab3:
    if _tier_gate("pro", "DataGolf vs Kalshi Divergences"):
        st.subheader("DataGolf Model vs Kalshi Market — Divergences")

        if not kalshi_ok:
            st.warning("No Kalshi data available. Check that there are open golf markets.")
        else:
            edge_threshold = st.slider(
                "Minimum |edge| to show",
                min_value=0.0, max_value=3.0, value=0.5, step=0.1,
                format="%.1f%%",
            )

            # Filter: liquid markets only (exclude thin), edge above threshold
            liquid_rows = [
                r for r in rows
                if r["edge"] is not None and abs(r["edge"]) * 100 >= edge_threshold
            ]

            dg_higher = sorted([r for r in liquid_rows if r["edge"] < 0], key=lambda r: r["edge"])[:15]
            kal_higher = sorted([r for r in liquid_rows if r["edge"] > 0], key=lambda r: r["edge"], reverse=True)[:15]

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 📉 DataGolf more bullish than Kalshi")
                st.caption("Model assigns higher win% than the market price")
                if dg_higher:
                    fig = go.Figure(go.Bar(
                        x=[r["edge"] * 100 for r in dg_higher],
                        y=[r["display_name"] for r in dg_higher],
                        orientation="h",
                        marker_color="#74b9ff",
                        text=[f"{r['dg_signal']*100:.1f}% DG vs {r['kal_win']*100:.1f}% Kal" for r in dg_higher],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        xaxis_title="Edge (Kal - DG) %",
                        margin=dict(l=0, r=60, t=10, b=30),
                        height=max(300, len(dg_higher) * 28),
                        xaxis=dict(ticksuffix="%"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    dg_higher_df = pd.DataFrame([{
                        "Player": r["display_name"],
                        "DG#": r["dg_rank"],
                        dg_signal_label: fmt_pct(r["dg_signal"]),
                        "Kal%": fmt_pct(r["kal_win"]),
                        "Edge": fmt_edge(r["edge"]),
                        "Course": " / ".join(str(r["year_fins"].get(y, "—")) for y in HISTORY_YEARS[:3]),
                    } for r in dg_higher])
                    st.dataframe(dg_higher_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No players meet the edge threshold.")

            with col_b:
                st.markdown("#### 📈 Kalshi more bullish than DataGolf")
                st.caption("Market price implies higher win% than the model")
                if kal_higher:
                    fig2 = go.Figure(go.Bar(
                        x=[r["edge"] * 100 for r in kal_higher],
                        y=[r["display_name"] for r in kal_higher],
                        orientation="h",
                        marker_color="#ff7675",
                        text=[f"{r['dg_signal']*100:.1f}% DG vs {r['kal_win']*100:.1f}% Kal" for r in kal_higher],
                        textposition="outside",
                    ))
                    fig2.update_layout(
                        xaxis_title="Edge (Kal - DG) %",
                        margin=dict(l=0, r=60, t=10, b=30),
                        height=max(300, len(kal_higher) * 28),
                        xaxis=dict(ticksuffix="%"),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    kal_higher_df = pd.DataFrame([{
                        "Player": r["display_name"],
                        "DG#": r["dg_rank"],
                        dg_signal_label: fmt_pct(r["dg_signal"]),
                        "Kal%": fmt_pct(r["kal_win"]),
                        "Edge": fmt_edge(r["edge"]),
                        "Course": " / ".join(str(r["year_fins"].get(y, "—")) for y in HISTORY_YEARS[:3]),
                    } for r in kal_higher])
                    st.dataframe(kal_higher_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No players meet the edge threshold.")


# ══════════════════════════════════════════════
# TAB 4 — OAD Pick
# ══════════════════════════════════════════════
with tab4:
    st.subheader("One and Done — Weekly Recommendation")

    # Blend signals
    blender = SignalBlender(w_dg=dg_weight, w_kalshi=kalshi_weight)
    blended_players = blender.blend(
        predictions=predictions,
        history_fit_map=history_fit_raw if use_history_model else {},
        kalshi_win_probs=kalshi_win,
    )

    # Filter used players
    available = [p for p in blended_players if p.dg_id not in used_player_ids]

    if not available:
        st.error("All players have been used this season. No picks available.")
    else:
        pick = available[0]
        pick_history = history.get if history else None

        # Build 5-year history string for pick
        pick_hist_str = " / ".join(
            rows_by_id.get("year_fins", {}).get(y, "—")
            for y in HISTORY_YEARS
        ) if (rows_by_id := next((r for r in rows if r["dg_id"] == pick.dg_id), None)) else "—"

        pick_edge_str = ""
        if pick.kalshi_win_prob is not None:
            diff = pick.kalshi_win_prob - pick.blended_win_prob
            direction = "Kalshi more bullish" if pick.edge > 0 else "DataGolf more bullish"
            pick_edge_str = f"{direction} ({pick.edge*100:+.1f}%)"

        # Recommended pick card
        kal_display = f"{pick.kalshi_win_prob*100:.1f}%" if pick.kalshi_win_prob is not None else "N/A"
        pick_dg = pick.dg_win_prob_history_fit if use_history_model else pick.dg_win_prob

        st.markdown(f"""
        <div class="pick-card">
            <div class="stat-label">THIS WEEK'S RECOMMENDED PICK</div>
            <div class="pick-name">{dg_name_to_display(pick.player_name)}</div>
            <div style="display: flex; gap: 32px; margin-top: 12px;">
                <div>
                    <div class="stat-label">Blended Win%</div>
                    <div class="pick-stat"><b>{pick.blended_win_prob*100:.1f}%</b></div>
                </div>
                <div>
                    <div class="stat-label">{dg_signal_label}</div>
                    <div class="pick-stat">{pick_dg*100:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Kalshi Win%</div>
                    <div class="pick-stat">{kal_display}</div>
                </div>
            </div>
            <div class="pick-meta">
                Course history: {pick_hist_str}
                {"&nbsp;&nbsp;|&nbsp;&nbsp;" + pick_edge_str if pick_edge_str else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Low win% advisory
        if pick.blended_win_prob < 0.02:
            st.markdown(
                '<div class="warning-box">⚠️ This week\'s top pick has a win% below 2%. '
                'Consider saving your stronger players for higher-profile events '
                '(majors, elevated fields).</div>',
                unsafe_allow_html=True,
            )

        # Alternatives table
        st.markdown("#### Top 15 Alternatives")
        st.caption(f"Weights: DataGolf {dg_weight:.0%} / Kalshi {kalshi_weight:.0%}  |  Model: {'Skill + Course History' if use_history_model else 'Skill only'}")

        alt_records = []
        for i, p in enumerate(available[:15], 1):
            r_match = next((r for r in rows if r["dg_id"] == p.dg_id), None)
            hist_str = " / ".join(
                r_match["year_fins"].get(y, "—") for y in HISTORY_YEARS[:3]
            ) if r_match else "—"
            is_spec = r_match["is_specialist"] if r_match else False
            p_dg = p.dg_win_prob_history_fit if use_history_model else p.dg_win_prob

            alt_records.append({
                "#": i,
                "Player": ("★ " if is_spec else "") + dg_name_to_display(p.player_name),
                "Blended%": fmt_pct(p.blended_win_prob),
                dg_signal_label: fmt_pct(p_dg),
                "Kal%": fmt_pct(p.kalshi_win_prob) if p.kalshi_win_prob is not None else "—",
                "Edge": fmt_edge(p.edge) if p.edge is not None else "—",
                "Course (25/24/23)": hist_str,
            })

        alt_df = pd.DataFrame(alt_records)

        def color_alt_edge(val):
            return color_edge(val)

        styled_alt = alt_df.style.applymap(color_alt_edge, subset=["Edge"])
        st.dataframe(styled_alt, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("#### Remaining Season Schedule")
        if schedule:
            sched_events = schedule[:20]
            event_id_tuple = tuple(e.get("event_id", "") for e in sched_events)
            winner_years = (2024, 2023, 2022)
            with st.spinner("Loading past winners..."):
                past_winners = load_schedule_winners(event_id_tuple, winner_years)

            sched_rows = []
            for e in sched_events:
                eid = str(e.get("event_id", ""))
                ew = past_winners.get(eid, {})
                sched_rows.append({
                    "Event": e.get("event_name", ""),
                    "Course": e.get("course", "")[:40],
                    "Date": e.get("date", ""),
                    "2024 Winner": ew.get(2024, "—"),
                    "2023 Winner": ew.get(2023, "—"),
                    "2022 Winner": ew.get(2022, "—"),
                    "Event ID": e.get("event_id", ""),
                })
            sched_df = pd.DataFrame(sched_rows)
            st.dataframe(sched_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — Kalshi Markets
# ══════════════════════════════════════════════
with tab5:
    if _tier_gate("pro", "Kalshi Prediction Markets"):
        st.subheader("Kalshi Prediction Markets — Raw Prices")

        if not event_code:
            st.warning("No open Kalshi golf markets detected.")
        else:
            st.caption(f"Event code: **{event_code}**  |  Prices in cents (0–100).  "
                       "Mid = (Bid + Ask) / 2.  No = 100 − Yes.")

            MARKET_LABELS = {
                "win":      "🏆 Tournament Winner",
                "top5":     "Top 5 Finish",
                "top10":    "Top 10 Finish",
                "top20":    "Top 20 Finish",
                "make_cut": "Make the Cut",
            }

            selected_market = st.selectbox(
                "Market type",
                options=list(MARKET_LABELS.keys()),
                format_func=lambda k: MARKET_LABELS[k],
            )

            with st.spinner(f"Loading {MARKET_LABELS[selected_market]} markets..."):
                raw_data = load_kalshi_raw(event_code, selected_market)

            if not raw_data:
                st.info(f"No {MARKET_LABELS[selected_market]} markets found for event {event_code}.")
            else:
                # Build display rows
                mkt_records = []
                for m in raw_data:
                    yes_bid   = m["yes_bid"]    # cents
                    yes_ask   = m["yes_ask"]    # cents
                    last      = m["last_price"] # cents
                    mid       = round((yes_bid + yes_ask) / 2, 1) if yes_bid and yes_ask else None
                    no_bid    = (100 - yes_ask) if yes_ask else None
                    no_ask    = (100 - yes_bid) if yes_bid else None
                    implied   = m["raw_prob"]
                    thin      = m["thin"]

                    mkt_records.append({
                        "Player":       m["player_name"],
                        "Yes Bid":      yes_bid  if yes_bid  else "—",
                        "Yes Ask":      yes_ask  if yes_ask  else "—",
                        "Mid":          mid      if mid is not None else "—",
                        "No Bid":       no_bid   if no_bid is not None else "—",
                        "No Ask":       no_ask   if no_ask is not None else "—",
                        "Last":         last     if last    else "—",
                        "Implied%":     f"{implied*100:.1f}%" if implied else "—",
                        "Liquid":       "✅" if not thin else "⚠️ thin",
                        "_implied_raw": implied,
                        "_thin":        thin,
                    })

                # Sort by implied prob descending (liquid first, then thin)
                mkt_records.sort(key=lambda r: (r["_thin"], -r["_implied_raw"]))

                mkt_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in mkt_records])

                def color_liquid(val):
                    if val == "✅":
                        return "background-color: #e8f5e9; color: #2e7d32"
                    if "thin" in str(val):
                        return "background-color: #fff3e0; color: #e65100"
                    return ""

                def color_bid_ask(val):
                    if val == "—":
                        return "color: #aaa"
                    return ""

                styled_mkt = (
                    mkt_df.style
                    .applymap(color_liquid, subset=["Liquid"])
                    .applymap(color_bid_ask, subset=["Yes Bid", "Yes Ask", "No Bid", "No Ask", "Last"])
                )

                liquid_count = sum(1 for r in mkt_records if not r["_thin"])
                thin_count   = sum(1 for r in mkt_records if r["_thin"])

                col_l, col_t, col_i = st.columns(3)
                col_l.metric("Liquid markets", liquid_count)
                col_t.metric("Thin markets",   thin_count)
                col_i.metric("Total players",  len(mkt_records))

                st.dataframe(styled_mkt, use_container_width=True, height=550, hide_index=True)

                st.caption(
                    "**Yes Bid/Ask** = prices to buy/sell a YES contract (cents).  "
                    "**No Bid/Ask** = 100 − Yes Ask / 100 − Yes Bid.  "
                    "**Mid** = fair-value estimate.  "
                    "**⚠️ thin** = no active quotes (floor-price artifact, excluded from blended model)."
                )


# ══════════════════════════════════════════════
# TAB 6 — Proprietary Model
# ══════════════════════════════════════════════
with tab6:
    if _tier_gate("elite", "Proprietary Model"):
        st.subheader("Proprietary Win Prediction Model")

        with st.expander("📖 How this model works", expanded=False):
            st.markdown("""
**Overview**

A proprietary multi-signal ensemble that blends five independent data sources into a
single win probability. Differs from DataGolf by incorporating live DraftKings market
prices, recent tournament form, and a custom recency-weighted course history score.

---

**Signals & Base Weights**

| Signal | Weight | Source | What it captures |
|--------|--------|--------|-----------------|
| **DataGolf (Skill + History)** | 40% | DataGolf pre-tournament model | Player skill ratings + course history fit |
| **DraftKings Market** | 30% | DraftKings outright win odds, vig-removed | Market consensus / public money |
| **Kalshi** | 15% | Kalshi prediction markets (liquid only) | Crowd wisdom from prediction traders |
| **Recent Form** | 10% | Last 5 completed PGA Tour events | Hot/cold streaks going into this week |
| **Course History Score** | 5% | Our recency-weighted custom score | Performance at this specific course |

---

**DraftKings Vig Removal**

Raw DraftKings implied probs sum to >100% (the sportsbook's overround/vig).
We normalize the full field so probs sum to 1.0:
`vig_free = dk_implied / sum(all players' dk_implied)`

---

**Recent Form Scoring**

Finish positions from the last 5 completed PGA Tour events are scored and weighted
by recency (most recent = highest weight):

| Finish | Score | Event Recency | Weight |
|--------|-------|---------------|--------|
| Win | 100 | Most recent | 5× |
| T2–T5 | 65 | 2nd most recent | 4× |
| T6–T10 | 35 | 3rd most recent | 3× |
| T11–T20 | 15 | 4th most recent | 2× |
| T21–T30 | 5 | 5th most recent | 1× |
| CUT/WD/DQ | 0 | — | — |

Scores are normalized 0–1 against the hottest player in this week's field.
Players who missed all 5 recent events receive no form signal.

---

**Course History Scoring**

Same position scoring applied to past appearances at this specific course,
with exponential year weighting (2× decay per year):

| Year | Weight |
|------|--------|
| 2025 | 16× |
| 2024 | 8× |
| 2023 | 4× |
| 2022 | 2× |
| 2021 | 1× |

Normalized 0–1 against the best course performer in this week's field.

---

**Adaptive Weights**

When a signal is unavailable (thin Kalshi market, no DraftKings price, no recent events,
no course history), its weight is set to 0 and the remaining weights scale up
proportionally to always sum to 100%.

---

**Rank Delta (Δ vs DG)**

`Δ = DataGolf rank − Proprietary rank`
- **Green / positive** = our model ranks the player *higher* (more bullish than DG)
- **Red / negative** = our model ranks the player *lower* (more bearish than DG)
        """)

        st.caption(
            "Signals: **DataGolf (40%)** + **DraftKings odds (30%)** + "
            "**Kalshi (15%)** + **Recent Form (10%)** + **Course History (5%)**. "
            "Weights adapt when a signal is unavailable."
        )

        if not event_id:
            st.error("Cannot determine event ID from schedule. Cannot run proprietary model.")
        else:
            with st.spinner("Running proprietary model..."):
                try:
                    prop_players = load_proprietary_model(
                        event_id=int(event_id),
                        event_code=event_code or "",
                    )
                except Exception as _e:
                    st.error(f"Proprietary model error: {_e}")
                    prop_players = []

            if not prop_players:
                st.warning("No proprietary model output available.")
            else:
                # ── Summary metrics ──────────────────────────────────────────
                market_cov  = sum(1 for p in prop_players if p.market_consensus_prob is not None)
                kalshi_cov  = sum(1 for p in prop_players if p.kalshi_win_prob is not None)
                form_cov    = sum(1 for p in prop_players if getattr(p, "recent_form_score", None) is not None)
                history_cov = sum(1 for p in prop_players if p.recency_course_score is not None)
    
                pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                pm1.metric("Field size", len(prop_players))
                pm2.metric("Sportsbook coverage", f"{market_cov}/{len(prop_players)}")
                pm3.metric("Kalshi coverage", f"{kalshi_cov}/{len(prop_players)}")
                pm4.metric("Recent form", f"{form_cov}/{len(prop_players)}")
                pm5.metric("Course history", f"{history_cov}/{len(prop_players)}")
    
                st.divider()
    
                # ── Full field table ─────────────────────────────────────────
                st.markdown("#### Full Field Rankings")
    
                def _fmt_delta(v: int) -> str:
                    if v == 0:
                        return "—"
                    sign = "+" if v > 0 else ""
                    return f"{sign}{v}"
    
                def _delta_style(delta: int) -> str:
                    if delta >= 5:  return "background:#2ecc71;color:white;font-weight:bold"
                    if delta >= 2:  return "background:#a8e6cf;color:black"
                    if delta <= -5: return "background:#e74c3c;color:white;font-weight:bold"
                    if delta <= -2: return "background:#fab1a0;color:black"
                    return ""
    
                headers = ["Prop#", "Δ vs DG", "Player ⓘ", "Used", "Prop Win%",
                           "DG Win%", "DK Odds", "Market%", "Kalshi%", "Form", "Crs Score", "Weights"]
                th = "".join(f"<th>{h}</th>" for h in headers)
    
                rows_html = []
                for p in prop_players:
                    is_used = p.dg_id in used_player_ids
                    w = p.weights_used
                    wt_str = (f"DG{w.get('dg',0)*100:.0f}/"
                              f"Mkt{w.get('market',0)*100:.0f}/"
                              f"Kal{w.get('kalshi',0)*100:.0f}/"
                              f"Frm{w.get('form',0)*100:.0f}/"
                              f"Hist{w.get('history',0)*100:.0f}")
                    tip = build_prop_tooltip(p).replace('"', '&quot;')
                    delta_s = _delta_style(p.rank_delta)
                    frm = getattr(p, "recent_form_score", None)
                    rows_html.append(
                        f'<tr{"  class=\"used\"" if is_used else ""}>'
                        f"<td>{p.proprietary_rank}</td>"
                        f'<td style="{delta_s}">{_fmt_delta(p.rank_delta)}</td>'
                        f'<td class="tip" data-tip="{tip}">{dg_name_to_display(p.player_name)}</td>'
                        f'<td>{"✓" if is_used else ""}</td>'
                        f"<td>{fmt_pct(p.proprietary_win_prob)}</td>"
                        f"<td>{fmt_pct(p.dg_win_prob_history)}</td>"
                        f"<td>{fmt_american(getattr(p, 'dk_raw_prob', None))}</td>"
                        f'<td>{"—" if p.market_consensus_prob is None else fmt_pct(p.market_consensus_prob)}</td>'
                        f'<td>{"—" if p.kalshi_win_prob is None else fmt_pct(p.kalshi_win_prob)}</td>'
                        f'<td>{"—" if frm is None else f"{frm:.2f}"}</td>'
                        f'<td>{"—" if p.recency_course_score is None else f"{p.recency_course_score:.2f}"}</td>'
                        f'<td style="color:#888;font-size:0.75rem">{wt_str}</td>'
                        f"</tr>"
                    )
    
                prop_html = (
                    '<div style="overflow-y:auto;max-height:600px;overflow-x:auto">'
                    f'<table class="prop-table"><thead><tr>{th}</tr></thead>'
                    f'<tbody>{"".join(rows_html)}</tbody></table></div>'
                )
                st.markdown(prop_html, unsafe_allow_html=True)
                st.caption(
                    "**Player ⓘ** = hover for signal breakdown  |  "
                    "**Prop#** = our rank  |  "
                    "**Δ vs DG** = DG rank minus our rank "
                    "(green = we rank higher / more bullish, red = we rank lower / more bearish)  |  "
                    "**Form** = 0–1 recent form score (last 5 events)  |  "
                    "**Crs Score** = 0–1 recency-weighted course history  |  "
                    "**Weights** = adaptive signal weights applied for this player"
                )
    
                st.divider()
    
                # ── Key Divergences ──────────────────────────────────────────
                st.markdown("#### Key Divergences: Where We Differ Most from DataGolf")
                st.caption("Top 10 players where our proprietary rank most disagrees with DataGolf.")
    
                with_delta = [p for p in prop_players if abs(p.rank_delta) >= 2]
                top_div = sorted(with_delta, key=lambda p: abs(p.rank_delta), reverse=True)[:10]
    
                if not top_div:
                    st.info("No significant divergences (all players within 1 rank of DataGolf).")
                else:
                    div_records = []
                    for p in top_div:
                        direction = "We rank HIGHER" if p.rank_delta > 0 else "We rank LOWER"
                        hist_parts = []
                        for yr in [2025, 2024, 2023]:
                            fin = p.finish_history.get(yr, "—")
                            hist_parts.append(f"{yr}: {fin}")
    
                        _frm = getattr(p, "recent_form_score", None)
                        div_records.append({
                            "Player":    dg_name_to_display(p.player_name),
                            "DG#":       p.dg_rank,
                            "Prop#":     p.proprietary_rank,
                            "Δ":         _fmt_delta(p.rank_delta),
                            "Direction": direction,
                            "DG Win%":   fmt_pct(p.dg_win_prob_history),
                            "Prop Win%": fmt_pct(p.proprietary_win_prob),
                            "DK Odds":   fmt_american(getattr(p, "dk_raw_prob", None)),
                            "Market%":   fmt_pct(p.market_consensus_prob) if p.market_consensus_prob is not None else "—",
                            "Kalshi%":   fmt_pct(p.kalshi_win_prob) if p.kalshi_win_prob is not None else "—",
                            "Form":      f"{_frm:.2f}" if _frm is not None else "—",
                            "Crs Score": f"{p.recency_course_score:.2f}" if p.recency_course_score is not None else "—",
                            "History":   "  |  ".join(hist_parts),
                        })
    
                    div_df = pd.DataFrame(div_records)
    
                    def _color_direction(val: str) -> str:
                        if "HIGHER" in str(val):
                            return "background-color: #2ecc71; color: white; font-weight: bold"
                        if "LOWER" in str(val):
                            return "background-color: #e74c3c; color: white; font-weight: bold"
                        return ""
    
                    styled_div = div_df.style.applymap(_color_direction, subset=["Direction"])
                    st.dataframe(styled_div, use_container_width=True, hide_index=True)
