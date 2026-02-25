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

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
HISTORY_YEARS = [2025, 2024, 2023, 2022, 2021]
CACHE_DIR = "data/raw"


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
def detect_kalshi_event_code() -> str | None:
    client = KalshiClient(cache_dir=CACHE_DIR)
    return client.detect_current_event_code()


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


def fmt_edge(v) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.1f}%"


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
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Full Field", "🏌️ Course History", "📈 DG vs Kalshi", "🎯 OAD Pick"]
)


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
            sched_df = pd.DataFrame([{
                "Event": e.get("event_name", ""),
                "Course": e.get("course", "")[:40],
                "Date": e.get("date", ""),
                "Event ID": e.get("event_id", ""),
            } for e in schedule[:20]])
            st.dataframe(sched_df, hide_index=True, use_container_width=True)
