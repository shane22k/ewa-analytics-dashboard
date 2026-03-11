import math
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EWA Analytics Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(1200px 700px at 20% -10%, #1b2a41 0%, #0b1220 55%, #070c16 100%); }
    section[data-testid="stSidebar"] { background-color: #0b0f14; }
    .title-wrap { padding: 8px 2px 2px 2px; margin-bottom: 6px; }
    .title-main { font-size: 34px; font-weight: 800; letter-spacing: .5px; color: #f5f7ff; }
    .title-sub { font-size: 14px; color: #b6c2df; margin-top: 2px; }
    .card { border: 1px solid rgba(255,255,255,.1); border-radius: 14px; padding: 12px; background: rgba(10,18,32,.55); }
    .kpi-label { color: #a8b3d1; font-size: 12px; }
    .kpi-value { color: #f5f7ff; font-size: 26px; font-weight: 750; line-height: 1.1; }
    .kpi-good { box-shadow: inset 0 0 0 1px rgba(22,163,74,.55); }
    .kpi-bad { box-shadow: inset 0 0 0 1px rgba(220,38,38,.55); }
    .kpi-neutral { box-shadow: inset 0 0 0 1px rgba(148,163,184,.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title-wrap"><div class="title-main">EWA ANALYTICS DASHBOARD</div><div class="title-sub">Stable build for in-season use</div></div>',
    unsafe_allow_html=True,
)


def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def num(v):
    return pd.to_numeric(v, errors="coerce")


def to_float(v):
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n):
        return np.nan
    return float(n)


def safe_div(n, d):
    n = num(n)
    d = num(d)
    return n / d.replace(0, np.nan)


def as_pct_value(v):
    f = to_float(v)
    if pd.isna(f):
        return np.nan
    if f <= 1.5:
        f *= 100.0
    return f


def fmt3(x):
    f = to_float(x)
    return "—" if pd.isna(f) else f"{f:.3f}"


def fmt1pct(x):
    f = to_float(x)
    return "—" if pd.isna(f) else f"{f:.1f}%"


def metric_card(label: str, value: str, cls: str = "kpi-neutral"):
    st.markdown(
        f'<div class="card {cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def round_display(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            vals = pd.to_numeric(out[c], errors="coerce")
            def _fmt(x):
                if pd.isna(x):
                    return ""
                xv = float(x)
                if abs(xv - round(xv)) < 1e-9:
                    return str(int(round(xv)))
                return f"{xv:.{decimals}f}"
            out[c] = vals.map(_fmt)
    return out


def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("_", " ") for c in out.columns]
    return out


def show_df(df: pd.DataFrame, fit_rows: bool = False):
    show = prettify_columns(round_display(df))
    if fit_rows:
        # Row + header sizing so users don't need to scroll inside shorter tables.
        height = 42 + max(1, len(show)) * 36
        st.dataframe(show, width="stretch", hide_index=True, height=height)
    else:
        st.dataframe(show, width="stretch", hide_index=True)


def normalize_pct_col(df, col):
    if col not in df.columns:
        return
    vals = num(df[col])
    if vals.dropna().empty:
        df[col] = vals
        return
    if vals.dropna().max() <= 1.5:
        vals = vals * 100.0
    df[col] = vals


def build_two_strike_avg(pa_df):
    if pa_df.empty or "batter" not in pa_df.columns:
        return pd.DataFrame(columns=["Batter", "2K AB", "2K H", "2K AVG"])
    df = pa_df.copy()
    for c in ["two_strike_pa_flag", "ab_flag", "hit_flag"]:
        if c not in df.columns:
            df[c] = 0
    sub = df[df["two_strike_pa_flag"] == 1].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Batter", "2K AB", "2K H", "2K AVG"])
    g = sub.groupby("batter", dropna=True).agg(
        two_k_ab=("ab_flag", "sum"),
        two_k_h=("hit_flag", "sum"),
    ).reset_index()
    g["2K AVG"] = safe_div(g["two_k_h"], g["two_k_ab"])
    g = g.rename(columns={"batter": "Batter", "two_k_ab": "2K AB", "two_k_h": "2K H"})
    return g


def build_risp_qab(pa_df):
    cols = ["Batter", "RISP AVG"]
    if pa_df.empty or "batter" not in pa_df.columns:
        return pd.DataFrame(columns=cols)
    df = pa_df.copy()
    for c in ["risp_flag", "ab_flag", "hit_flag", "runner_adv_or_score_flag"]:
        if c not in df.columns:
            df[c] = 0
    sub = df[df["risp_flag"] == 1].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)
    g = sub.groupby("batter", dropna=True).agg(
        risp_ab=("ab_flag", "sum"),
        risp_h=("hit_flag", "sum"),
    ).reset_index()
    g["RISP AVG"] = safe_div(g["risp_h"], g["risp_ab"])
    g = g.rename(columns={"batter": "Batter"})
    g = g[["Batter", "RISP AVG"]]
    return g


def render_spray_svg(pa_player: pd.DataFrame):
    if pa_player.empty or "batted_ball_pos" not in pa_player.columns:
        return None

    df = pa_player.copy()
    df["batted_ball_pos"] = df["batted_ball_pos"].fillna("").astype(str).str.strip()
    df = df[df["batted_ball_pos"] != ""]
    if "pa_flag" in df.columns:
        df = df[df["pa_flag"] == 1]
    if df.empty:
        return None

    target = {
        "1B": (455, 315), "2B": (430, 260), "SS": (370, 260), "3B": (345, 315),
        "LF": (285, 165), "CF": (400, 120), "RF": (515, 165), "P": (400, 330), "C": (400, 385),
    }

    # Guardrail: keep rendering bounded even if file unexpectedly has a large volume.
    if len(df) > 500:
        df = df.tail(500).copy()

    rows = []
    counts = defaultdict(int)
    for _, r in df.iterrows():
        pos = str(r.get("batted_ball_pos", "")).upper()
        outcome = str(r.get("outcome", "")).strip().lower()
        is_hr = outcome == "home run"

        if is_hr:
            if pos == "LF":
                x2, y2 = (255, 130)
            elif pos == "RF":
                x2, y2 = (545, 130)
            else:
                x2, y2 = (400, 95)
            counts[f"HR_{pos or 'CF'}"] += 1
            n = counts[f"HR_{pos or 'CF'}"]
        else:
            if pos not in target:
                continue
            counts[pos] += 1
            n = counts[pos]
            x2, y2 = target[pos]
        # fan out repeated lines to same position so all balls are visible
        angle = (n - 1) * 0.12
        jitter = 8 * (n - 1)
        x2 = x2 + math.cos(angle) * jitter
        y2 = y2 - math.sin(angle) * jitter
        is_hit = int(r.get("hit_flag", 0)) == 1
        color = "#22c55e" if is_hit else "#9ca3af"
        rows.append((x2, y2, color))

    if not rows:
        return None

    lines = []
    for x2, y2, color in rows:
        lines.append(f"<line x1='400' y1='390' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='3' stroke-linecap='round' opacity='0.92' />")

    svg = f"""
    <svg width='800' height='420' viewBox='0 0 800 420' xmlns='http://www.w3.org/2000/svg'>
      <rect x='0' y='0' width='800' height='420' fill='#050b16'/>
      <path d='M 400 390 L 250 240' stroke='#6b7280' stroke-width='2' fill='none' />
      <path d='M 400 390 L 550 240' stroke='#6b7280' stroke-width='2' fill='none' />
      <path d='M 250 240 Q 400 60 550 240' stroke='#334155' stroke-width='2' fill='none' />
      <rect x='394' y='384' width='12' height='12' transform='rotate(45 400 390)' fill='#94a3b8' />
      {''.join(lines)}
    </svg>
    """
    return svg


pa = load_csv("plate_appearances_EWA.csv")
pa_all = load_csv("plate_appearances_ALL.csv")
bat = load_csv("batting_season_EWA.csv")
disc = load_csv("discipline_season_EWA.csv")
pit = load_csv("pitching_basic_EWA.csv")

if pa_all.empty:
    pa_all = pa.copy()

for df in [pa, pa_all, bat, disc, pit]:
    if not df.empty:
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

# normalize percent-like columns for display
for pct_col in ["K%", "Swing%", "Whiff%", "Contact%", "1stPitchSwing%", "1stPitchStrike%", "1stPitchInPlay%", "2K_Swing%", "2K_Whiff%", "2K_Contact%"]:
    normalize_pct_col(bat, pct_col)
    normalize_pct_col(disc, pct_col)

two_k_avg = build_two_strike_avg(pa)
risp_qab = build_risp_qab(pa)

tabs = st.tabs(["Hitting", "Discipline", "Player Profile", "Team Stats", "Pitching"])

with tabs[0]:
    st.subheader("Hitting")
    if bat.empty:
        st.warning("batting_season_EWA.csv is missing or empty.")
    else:
        sort_choice = st.selectbox("Sort by", [c for c in ["OBP", "AVG", "K%", "P/PA"] if c in bat.columns], index=0)
        show = bat.copy()
        if sort_choice in show.columns:
            asc = True if sort_choice == "K%" else False
            show = show.sort_values(sort_choice, ascending=asc)
        cols = [c for c in ["batter", "PA", "AB", "H", "XBH", "BB", "HBP", "K", "AVG", "OBP", "K%", "P/PA", "QAB%"] if c in show.columns]
        show = show[cols].rename(columns={"batter": "Batter"})
        show_df(show, fit_rows=True)

with tabs[1]:
    st.subheader("Discipline")
    if disc.empty:
        st.warning("discipline_season_EWA.csv is missing or empty.")
    else:
        d = disc.copy()
        d = d.rename(columns={"batter": "Batter"})
        d = d.merge(two_k_avg[["Batter", "2K AVG"]], on="Batter", how="left") if not two_k_avg.empty else d
        keep = [c for c in ["Batter", "1stPitchSwing%", "2K AVG", "2K_Swing%", "2K_Whiff%", "2K_Contact%", "Swing%", "Whiff%", "Contact%"] if c in d.columns]
        show_df(d[keep])

    st.markdown("### RISP AVG")
    if risp_qab.empty:
        st.info("No RISP rows yet.")
    else:
        show_df(risp_qab.sort_values("RISP AVG", ascending=False))

with tabs[2]:
    st.subheader("Player Profile")
    try:
        batters = []
        if not bat.empty and "batter" in bat.columns:
            batters = sorted([b for b in bat["batter"].dropna().unique().tolist() if str(b).lower() != "nan" and str(b).strip() != ""])

        if not batters:
            st.warning("No players found.")
            st.stop()

        player = st.selectbox("Player", batters)

        row_bat = bat[bat["batter"] == player].copy() if "batter" in bat.columns else pd.DataFrame()
        row_disc = disc[disc["batter"] == player].copy() if (not disc.empty and "batter" in disc.columns) else pd.DataFrame()

        c1, c2, c3, c4, c5 = st.columns(5)
        if not row_bat.empty:
            r = row_bat.iloc[0]
            pa_v = r.get("PA", np.nan)
            avg_v = r.get("AVG", np.nan)
            obp_v = r.get("OBP", np.nan)
            k_v = as_pct_value(r.get("K%", np.nan))
            ppa_v = r.get("P/PA", np.nan)
            avg_f = to_float(avg_v)
            obp_f = to_float(obp_v)
            ppa_f = to_float(ppa_v)
            pa_f = to_float(pa_v)

            with c1:
                metric_card("PA", "—" if pd.isna(pa_f) else str(int(pa_f)))
            with c2:
                metric_card("AVG", fmt3(avg_v), "kpi-good" if pd.notna(avg_f) and avg_f >= 0.333 else "kpi-bad")
            with c3:
                metric_card("OBP", fmt3(obp_v), "kpi-good" if pd.notna(obp_f) and obp_f >= 0.400 else "kpi-bad")
            with c4:
                metric_card("K%", fmt1pct(k_v), "kpi-good" if pd.notna(k_v) and float(k_v) <= 25 else "kpi-bad")
            with c5:
                metric_card("P/PA", fmt3(ppa_v), "kpi-good" if pd.notna(ppa_f) and ppa_f >= 4.5 else "kpi-bad")

        st.markdown("### Hitting")
        if not row_bat.empty:
            keep = [c for c in ["batter", "PA", "AB", "H", "XBH", "BB", "HBP", "K", "AVG", "OBP", "K%", "QAB%"] if c in row_bat.columns]
            show_df(row_bat[keep].rename(columns={"batter": "Batter"}))

        st.markdown("### Discipline + Situational")
        disc_row = pd.DataFrame()
        if not row_disc.empty:
            disc_row = row_disc.rename(columns={"batter": "Batter"}).copy()
            if "Batter" in disc_row.columns and not two_k_avg.empty:
                disc_row = disc_row.merge(two_k_avg[["Batter", "2K AVG"]], on="Batter", how="left")
            keep = [c for c in ["Batter", "1stPitchSwing%", "2K AVG", "2K_Swing%", "2K_Whiff%", "2K_Contact%", "Swing%", "Whiff%", "Contact%"] if c in disc_row.columns]
            disc_row = disc_row[keep]

        risp_row = pd.DataFrame()
        if not risp_qab.empty:
            risp_row = risp_qab[risp_qab["Batter"] == player].copy()

        if disc_row.empty and risp_row.empty:
            st.info("No discipline or RISP rows for this player yet.")
        else:
            if not disc_row.empty:
                show_df(disc_row)
            if not risp_row.empty:
                show_df(risp_row)

        st.markdown("### Spray Chart")
        pa_spray = pa_all[pa_all["batter"] == player].copy() if (not pa_all.empty and "batter" in pa_all.columns) else pd.DataFrame()
        spray_svg = render_spray_svg(pa_spray)
        if spray_svg is None:
            st.info("Spray chart data not ready yet. Re-run build_2026.py after using the latest fixed build script.")
        else:
            st.markdown(spray_svg, unsafe_allow_html=True)
            st.caption("Green lines = hits, gray lines = outs")
    except Exception as e:
        st.error(f"Player Profile error: {e}")
        st.exception(e)

with tabs[3]:
    st.subheader("Team Stats")
    if bat.empty:
        st.warning("No batting season file found.")
    else:
        PA = float(num(bat["PA"]).fillna(0).sum()) if "PA" in bat.columns else np.nan
        AB = float(num(bat["AB"]).fillna(0).sum()) if "AB" in bat.columns else np.nan
        H = float(num(bat["H"]).fillna(0).sum()) if "H" in bat.columns else np.nan
        BB = float(num(bat["BB"]).fillna(0).sum()) if "BB" in bat.columns else np.nan
        HBP = float(num(bat["HBP"]).fillna(0).sum()) if "HBP" in bat.columns else 0.0
        K = float(num(bat["K"]).fillna(0).sum()) if "K" in bat.columns else np.nan
        team = pd.DataFrame([{
            "PA": PA,
            "AB": AB,
            "H": H,
            "BB": BB,
            "HBP": HBP,
            "K": K,
            "AVG": (H / AB) if AB else np.nan,
            "OBP": ((H + BB + HBP) / PA) if PA else np.nan,
            "K%": ((K / PA) * 100.0) if PA else np.nan,
        }])
        st.markdown("### Hitting")
        show_df(team)

    st.markdown("### Discipline")
    if disc.empty:
        st.info("No discipline season file found.")
    else:
        d = disc.copy()
        team_disc = {}

        # Weighted team discipline rates when underlying counts exist.
        if all(c in d.columns for c in ["Swings", "Pitches"]):
            team_disc["Swing%"] = (num(d["Swings"]).sum() / num(d["Pitches"]).replace(0, np.nan).sum()) * 100.0
        elif "Swing%" in d.columns:
            team_disc["Swing%"] = num(d["Swing%"]).mean()

        if all(c in d.columns for c in ["Whiffs", "Swings"]):
            team_disc["Whiff%"] = (num(d["Whiffs"]).sum() / num(d["Swings"]).replace(0, np.nan).sum()) * 100.0
        elif "Whiff%" in d.columns:
            team_disc["Whiff%"] = num(d["Whiff%"]).mean()

        if all(c in d.columns for c in ["Contacts", "Swings"]):
            team_disc["Contact%"] = (num(d["Contacts"]).sum() / num(d["Swings"]).replace(0, np.nan).sum()) * 100.0
        elif "Contact%" in d.columns:
            team_disc["Contact%"] = num(d["Contact%"]).mean()

        if all(c in d.columns for c in ["FirstPitchSwings", "FirstPitches"]):
            team_disc["1stPitchSwing%"] = (num(d["FirstPitchSwings"]).sum() / num(d["FirstPitches"]).replace(0, np.nan).sum()) * 100.0
        elif "1stPitchSwing%" in d.columns:
            team_disc["1stPitchSwing%"] = num(d["1stPitchSwing%"]).mean()

        # 2K AVG from PA-derived two-strike table (team H / AB).
        if not two_k_avg.empty and all(c in two_k_avg.columns for c in ["2K AB", "2K H"]):
            ab2 = num(two_k_avg["2K AB"]).sum()
            h2 = num(two_k_avg["2K H"]).sum()
            team_disc["2K AVG"] = (h2 / ab2) if ab2 else np.nan
        elif not two_k_avg.empty and "2K AVG" in two_k_avg.columns:
            team_disc["2K AVG"] = num(two_k_avg["2K AVG"]).mean()

        for c in ["2K_Swing%", "2K_Whiff%", "2K_Contact%"]:
            if c in d.columns:
                team_disc[c] = num(d[c]).mean()

        order = ["1stPitchSwing%", "2K AVG", "2K_Swing%", "2K_Whiff%", "2K_Contact%", "Swing%", "Whiff%", "Contact%"]
        row = {}
        for k in order:
            row[k] = team_disc.get(k, np.nan)
        team_disc_df = pd.DataFrame([row])
        show_df(team_disc_df)

with tabs[4]:
    st.subheader("Pitching")
    if pit.empty:
        st.warning("pitching_basic_EWA.csv is missing or empty.")
    else:
        p = pit.copy()
        if "pitcher" in p.columns:
            p["pitcher"] = p["pitcher"].astype(str).str.strip()
        p = p.rename(columns={"pitcher": "Pitcher"})
        drop_cols = [c for c in ["WHIP", "BF", "AB", "R", "ER"] if c in p.columns]
        p = p.drop(columns=drop_cols) if drop_cols else p
        show_df(p)
    st.caption("CSW% = Called Strikes + Whiffs, divided by total pitches.")
