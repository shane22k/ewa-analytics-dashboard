import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

st.set_page_config(page_title="EWA Analytics Dashboard", layout="wide")
st.title("EWA ANALYTICS DASHBOARD")

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(name)

def safe_div(n, d):
    d = np.asarray(d)
    n = np.asarray(n)
    out = np.full_like(n, np.nan, dtype="float64")
    mask = d != 0
    out[mask] = n[mask] / d[mask]
    return out

def round_for_display(df: pd.DataFrame, pct_cols=None, decimals=3) -> pd.DataFrame:
    """Round numeric columns for display. pct_cols are assumed already in percent units (0-100)."""
    pct_cols = set(pct_cols or [])
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            if c in pct_cols:
                out[c] = out[c].round(1)
            else:
                out[c] = out[c].round(decimals)
    return out

def _color_scale_good(v, green_hi, red_lo):
    """Higher is better."""
    try:
        if pd.isna(v):
            return ""
        v = float(v)
    except Exception:
        return ""
    if v >= green_hi:
        return "background-color: rgba(0, 128, 0, 0.45);"
    if v <= red_lo:
        return "background-color: rgba(200, 0, 0, 0.45);"
    return "background-color: rgba(255, 215, 0, 0.20);"

def _color_scale_bad(v, green_lo, red_hi):
    """Lower is better."""
    try:
        if pd.isna(v):
            return ""
        v = float(v)
    except Exception:
        return ""
    if v <= green_lo:
        return "background-color: rgba(0, 128, 0, 0.45);"
    if v >= red_hi:
        return "background-color: rgba(200, 0, 0, 0.45);"
    return "background-color: rgba(255, 215, 0, 0.20);"

def style_player_hitting(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Stable styling without matplotlib gradients."""
    sty = df.style

    # thresholds requested
    if "P/PA" in df.columns:
        sty = sty.applymap(lambda v: _color_scale_good(v, green_hi=4.5, red_lo=4.499), subset=["P/PA"])
    if "AVG" in df.columns:
        sty = sty.applymap(lambda v: _color_scale_good(v, green_hi=0.333, red_lo=0.332999), subset=["AVG"])
    if "OBP" in df.columns:
        sty = sty.applymap(lambda v: _color_scale_good(v, green_hi=0.400, red_lo=0.399999), subset=["OBP"])

    return sty

def style_player_discipline(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # Keep discipline uncolored for now (stable / simple)
    return df.style


def metric_card(label: str, value: str, good: Optional[bool] = None):
    """Small colored value card for the Player Profile top row."""
    if good is True:
        color = "rgba(0, 128, 0, 0.65)"
    elif good is False:
        color = "rgba(200, 0, 0, 0.65)"
    else:
        color = "rgba(120, 120, 120, 0.35)"
    st.markdown(
        f"""<div style='padding:10px 12px;border-radius:12px;background:{color};text-align:center;'>
        <div style='font-size:12px;opacity:0.85'>{label}</div>
        <div style='font-size:22px;font-weight:700;line-height:1.1'>{value}</div>
        </div>""",
        unsafe_allow_html=True
    )


# -------------------- Load base outputs --------------------
pitch = load_csv("pitch_events_EWA.csv")
pa = load_csv("plate_appearances_EWA.csv")

bat_season_full = load_csv("batting_season_EWA.csv")
disc_season_full = load_csv("discipline_season_EWA.csv")

pitching_basic = load_csv("pitching_basic_EWA.csv")
fielding_basic = load_csv("fielding_basic_EWA.csv")

# -------------------- Helpers (player profile only) --------------------
def compute_player_batting(pa_df: pd.DataFrame) -> pd.DataFrame:
    if pa_df.empty:
        return pd.DataFrame(columns=["batter","PA","AB","H","BB","HBP","K","AVG","OBP","K%","P/PA"])

    df = pa_df.copy()
    # Defensive checks for expected columns
    for col in ["ab_flag","hit_flag","walk_flag","hbp_flag","k_flag","pitches_in_pa"]:
        if col not in df.columns:
            df[col] = 0

    out = df.groupby("batter", dropna=True).agg(
        PA=("pa_id","count") if "pa_id" in df.columns else ("batter","count"),
        AB=("ab_flag","sum"),
        H=("hit_flag","sum"),
        BB=("walk_flag","sum"),
        HBP=("hbp_flag","sum"),
        K=("k_flag","sum"),
        Pitches=("pitches_in_pa","sum"),
    ).reset_index()

    out["AVG"] = safe_div(out["H"], out["AB"])
    out["OBP"] = safe_div(out["H"] + out["BB"] + out["HBP"], out["PA"])
    out["K%"] = safe_div(out["K"], out["PA"]) * 100.0
    out["P/PA"] = safe_div(out["Pitches"], out["PA"])
    return out

def compute_player_discipline(pitch_df: pd.DataFrame) -> pd.DataFrame:
    if pitch_df.empty:
        return pd.DataFrame(columns=["batter","Pitches","Swings","Whiffs","Contacts","Swing%","Whiff%","Contact%","2K_Swing%","2K_Whiff%","2K_Contact%","1stPitchSwing%","1stPitchStrike%","1stPitchInPlay%"])

    df = pitch_df.copy()

    # expected flags
    for col in ["is_swing","is_whiff","is_contact","is_first_pitch","is_two_strike_pitch"]:
        if col not in df.columns:
            df[col] = 0

    if "pitch_id" not in df.columns:
        df["pitch_id"] = np.arange(len(df))

    base = df.groupby("batter", dropna=True).agg(
        Pitches=("pitch_id","count"),
        Swings=("is_swing","sum"),
        Whiffs=("is_whiff","sum"),
        Contacts=("is_contact","sum"),
    ).reset_index()

    base["Swing%"] = safe_div(base["Swings"], base["Pitches"]) * 100.0
    base["Whiff%"] = safe_div(base["Whiffs"], base["Swings"]) * 100.0
    base["Contact%"] = safe_div(base["Contacts"], base["Swings"]) * 100.0

    # 2K
    two = df[df["is_two_strike_pitch"] == 1].copy()
    if not two.empty:
        two_agg = two.groupby("batter", dropna=True).agg(
            TwoK_Pitches=("pitch_id","count"),
            TwoK_Swings=("is_swing","sum"),
            TwoK_Whiffs=("is_whiff","sum"),
            TwoK_Contacts=("is_contact","sum"),
        ).reset_index()
        two_agg["2K_Swing%"] = safe_div(two_agg["TwoK_Swings"], two_agg["TwoK_Pitches"]) * 100.0
        two_agg["2K_Whiff%"] = safe_div(two_agg["TwoK_Whiffs"], two_agg["TwoK_Swings"]) * 100.0
        two_agg["2K_Contact%"] = safe_div(two_agg["TwoK_Contacts"], two_agg["TwoK_Swings"]) * 100.0
        base = base.merge(two_agg[["batter","2K_Swing%","2K_Whiff%","2K_Contact%"]], on="batter", how="left")
    else:
        base["2K_Swing%"] = np.nan
        base["2K_Whiff%"] = np.nan
        base["2K_Contact%"] = np.nan

    # 1st pitch
    fp = df[df["is_first_pitch"] == 1].copy()
    if not fp.empty:
        if "pitch_result" in fp.columns:
            fp["fp_in_play"] = (fp["pitch_result"].astype(str).str.lower() == "in_play").astype(int)
        else:
            fp["fp_in_play"] = 0

        if "is_strike" in fp.columns:
            fp_strike = "is_strike"
        else:
            fp_strike = None

        fp_agg = fp.groupby("batter", dropna=True).agg(
            FirstPitches=("pitch_id","count"),
            FirstPitchSwings=("is_swing","sum"),
            FirstPitchInPlay=("fp_in_play","sum"),
        ).reset_index()

        if fp_strike:
            fp_strikes = fp.groupby("batter", dropna=True).agg(FirstPitchStrikes=(fp_strike,"sum")).reset_index()
            fp_agg = fp_agg.merge(fp_strikes, on="batter", how="left")
        else:
            fp_agg["FirstPitchStrikes"] = np.nan

        fp_agg["1stPitchSwing%"] = safe_div(fp_agg["FirstPitchSwings"], fp_agg["FirstPitches"]) * 100.0
        fp_agg["1stPitchInPlay%"] = safe_div(fp_agg["FirstPitchInPlay"], fp_agg["FirstPitches"]) * 100.0
        fp_agg["1stPitchStrike%"] = safe_div(fp_agg["FirstPitchStrikes"], fp_agg["FirstPitches"]) * 100.0

        base = base.merge(fp_agg[[
            "batter","FirstPitchSwings","FirstPitchStrikes","FirstPitchInPlay","1stPitchSwing%","1stPitchStrike%","1stPitchInPlay%"
        ]], on="batter", how="left")
    else:
        base["FirstPitchSwings"] = np.nan
        base["FirstPitchStrikes"] = np.nan
        base["FirstPitchInPlay"] = np.nan
        base["1stPitchSwing%"] = np.nan
        base["1stPitchStrike%"] = np.nan
        base["1stPitchInPlay%"] = np.nan

    return base



def compute_situational_tables(pa_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (risp_table, twoout_table) with Batter/AB/H/AVG."""
    if pa_df.empty:
        cols = ["Batter","AB","H","AVG"]
        empty = pd.DataFrame(columns=cols)
        return empty, empty

    df = pa_df.copy()

    # Ensure required cols exist
    for c in ["batter","ab_flag","hit_flag","risp_flag","two_out_flag"]:
        if c not in df.columns:
            df[c] = 0

    def _make(flag_col: str) -> pd.DataFrame:
        sub = df[df[flag_col] == 1].copy()
        if sub.empty:
            return pd.DataFrame(columns=["Batter","AB","H","AVG"])
        g = sub.groupby("batter", dropna=True).agg(
            AB=("ab_flag","sum"),
            H=("hit_flag","sum"),
        ).reset_index()
        g["AVG"] = safe_div(g["H"], g["AB"])
        g = g.rename(columns={"batter":"Batter"})
        return g

    return _make("risp_flag"), _make("two_out_flag")


def compute_team_avg_from_pa(pa_df: pd.DataFrame, flag_col: str) -> float:
    if pa_df.empty or flag_col not in pa_df.columns:
        return np.nan
    sub = pa_df[pa_df[flag_col] == 1].copy()
    if sub.empty:
        return np.nan
    ab = float(pd.to_numeric(sub.get("ab_flag", 0), errors="coerce").fillna(0).sum())
    h  = float(pd.to_numeric(sub.get("hit_flag", 0), errors="coerce").fillna(0).sum())
    return (h/ab) if ab != 0 else np.nan
# -------------------- Tabs --------------------
tabs = st.tabs(["Hitting", "Situational", "Player Profile", "Team Stats"])

# -------------------- Hitting (TEAM ONLY) --------------------
with tabs[0]:
    st.subheader("Hitting — Season")
    sort_choice = st.selectbox("Sort by", ["OBP", "AVG", "K%", "P/PA"], index=0)

    df = bat_season_full.copy()
    keep = [c for c in ["batter","PA","AB","H","XBH","BB","HBP","K","AVG","OBP","K%","P/PA"] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"batter":"Batter"})

    df = round_for_display(df, pct_cols=["K%"], decimals=3)

    asc = True if sort_choice in ["K%"] else False
    if sort_choice in df.columns:
        df = df.sort_values(sort_choice, ascending=asc)

    st.dataframe(df, use_container_width=True)

    st.subheader("Discipline — Season")
    d = disc_season_full.copy()
    disc_keep = [c for c in [
        "batter","Pitches","Swings","Whiffs","Contacts",
        "Swing%","Whiff%","Contact%","1stPitchSwing%","1stPitchStrike%","1stPitchInPlay%","2K_Swing%","2K_Whiff%","2K_Contact%"
    ] if c in d.columns]
    d = d[disc_keep].copy()
    d = d.rename(columns={"batter":"Batter"})
    pct_cols = [c for c in d.columns if c.endswith("%")]
    d = round_for_display(d, pct_cols=pct_cols, decimals=3)
    st.dataframe(d, use_container_width=True)

# -------------------- Situational (TEAM ONLY; keep format; capitalize Batter) --------------------
with tabs[1]:
    st.subheader("Situational")

    st.markdown("### 2-Strike Approach")
    two_cols = ["batter","2K_Swing%","2K_Whiff%","2K_Contact%"]
    d2 = disc_season_full[[c for c in two_cols if c in disc_season_full.columns]].copy()
    d2 = d2.rename(columns={"batter": "Batter"})
    pct_cols = [c for c in d2.columns if c.endswith("%")]
    d2 = round_for_display(d2, pct_cols=pct_cols, decimals=3)
    sort_col = "2K_Whiff%" if "2K_Whiff%" in d2.columns else None
    st.dataframe(d2.sort_values(sort_col, ascending=True) if sort_col else d2, use_container_width=True)

    st.markdown("### 1st Pitch Approach")
    fp_cols = ["batter","FirstPitchSwings","FirstPitchStrikes","FirstPitchInPlay","1stPitchSwing%","1stPitchInPlay%"]
    # Fall back to season columns if the detailed ones don't exist
    if any(c in disc_season_full.columns for c in ["FirstPitchSwings","FirstPitchStrikes","FirstPitchInPlay","1stPitchInPlay%"]):
        d3 = disc_season_full[[c for c in fp_cols if c in disc_season_full.columns]].copy()
    else:
        # keep prior simple version
        d3 = disc_season_full[[c for c in ["batter","1stPitchSwing%","1stPitchStrike%","1stPitchInPlay%"] if c in disc_season_full.columns]].copy()

    d3 = d3.rename(columns={"batter": "Batter"})
    pct_cols = [c for c in d3.columns if c.endswith("%")]
    d3 = round_for_display(d3, pct_cols=pct_cols, decimals=3)
    st.dataframe(d3, use_container_width=True)

    st.markdown("### AVG with RISP")
    risp_tbl, twoout_tbl = compute_situational_tables(pa)
    if not risp_tbl.empty:
        risp_tbl = round_for_display(risp_tbl, pct_cols=[], decimals=3)
        st.dataframe(risp_tbl.sort_values("AVG", ascending=False), use_container_width=True)
    else:
        st.write("No RISP plate appearances found.")

    st.markdown("### AVG with 2 Outs")
    if not twoout_tbl.empty:
        twoout_tbl = round_for_display(twoout_tbl, pct_cols=[], decimals=3)
        st.dataframe(twoout_tbl.sort_values("AVG", ascending=False), use_container_width=True)
    else:
        st.write("No 2-out plate appearances found.")


# -------------------- Player Profile (ONLY place with filters) --------------------
with tabs[2]:
    st.subheader("Player Profile")

    games = sorted(pd.Series(pa["game_id"].dropna().unique()).tolist()) if "game_id" in pa.columns else []
    quick = st.selectbox("Quick range", ["All games", "Last 3 games", "Last 5 games"], index=0)

    if games:
        if quick == "Last 3 games":
            selected_games = games[-3:]
        elif quick == "Last 5 games":
            selected_games = games[-5:]
        else:
            selected_games = games
    else:
        selected_games = []

    batters = sorted(pd.concat([
        bat_season_full.get("batter", pd.Series(dtype="object")),
        disc_season_full.get("batter", pd.Series(dtype="object")),
    ]).dropna().unique())

    player = st.selectbox("Player", batters)

    # Filter raw tables by games (if available)
    if selected_games and "game_id" in pa.columns:
        pa_p = pa[pa["game_id"].isin(selected_games)].copy()
    else:
        pa_p = pa.copy()

    if selected_games and "game_id" in pitch.columns:
        pitch_p = pitch[pitch["game_id"].isin(selected_games)].copy()
    else:
        pitch_p = pitch.copy()

    bat_prof = compute_player_batting(pa_p)
    disc_prof = compute_player_discipline(pitch_p)

    row_bat = bat_prof[bat_prof["batter"] == player].copy()
    row_disc = disc_prof[disc_prof["batter"] == player].copy()

    c1, c2, c3, c4, c5 = st.columns(5)

    if not row_bat.empty:
        r = row_bat.iloc[0]
        pa_val = int(r.get("PA", 0))

        avg_val = r.get("AVG", np.nan)
        obp_val = r.get("OBP", np.nan)
        k_val = r.get("K%", np.nan)
        ppa_val = r.get("P/PA", np.nan)

        avg_txt = f"{float(avg_val):.3f}" if pd.notna(avg_val) else "—"
        obp_txt = f"{float(obp_val):.3f}" if pd.notna(obp_val) else "—"
        k_txt = f"{float(k_val):.1f}%" if pd.notna(k_val) else "—"
        ppa_txt = f"{float(ppa_val):.3f}" if pd.notna(ppa_val) else "—"

        with c1:
            metric_card("PA", str(pa_val), None)
        with c2:
            metric_card("AVG", avg_txt, good=(pd.notna(avg_val) and float(avg_val) >= 0.333))
        with c3:
            metric_card("OBP", obp_txt, good=(pd.notna(obp_val) and float(obp_val) >= 0.400))
        with c4:
            # For K%, lower is better. Use a mild threshold.
            metric_card("K%", k_txt, good=(pd.notna(k_val) and float(k_val) <= 25.0))
        with c5:
            metric_card("P/PA", ppa_txt, good=(pd.notna(ppa_val) and float(ppa_val) >= 4.5))
    else:
        st.info("No plate appearance data found for this player in the selected range.")

    st.divider()

    st.markdown("### Hitting")
    if not row_bat.empty:
        show = row_bat.copy()
        keep = [c for c in ["batter","PA","AB","H","BB","HBP","K","AVG","OBP","K%","P/PA"] if c in show.columns]
        show = show[keep]
        show = round_for_display(show, pct_cols=["K%"], decimals=3)
        st.dataframe(style_player_hitting(show), use_container_width=True)

        # Situational AVG (player)
        st.markdown("### Situational AVG")
        pa_player = pa_p[pa_p.get("batter") == player].copy() if "batter" in pa_p.columns else pd.DataFrame()
        risp_avg = compute_team_avg_from_pa(pa_player, "risp_flag")
        two_avg = compute_team_avg_from_pa(pa_player, "two_out_flag")
        sit = pd.DataFrame([{
            "AVG_RISP": risp_avg,
            "AVG_2OUT": two_avg,
        }])
        sit = round_for_display(sit, pct_cols=[], decimals=3)
        st.dataframe(sit, use_container_width=True)


    st.markdown("### Discipline")
    if not row_disc.empty:
        keep = [c for c in [
            "batter","Pitches","Swings","Whiffs","Contacts",
            "Swing%","Whiff%","Contact%","2K_Swing%","2K_Whiff%","2K_Contact%",
            "FirstPitchSwings","FirstPitchStrikes","FirstPitchInPlay","1stPitchSwing%","1stPitchStrike%","1stPitchInPlay%"
        ] if c in row_disc.columns]
        show = row_disc[keep].copy()
        pct_cols = [c for c in show.columns if c.endswith("%")]
        show = round_for_display(show, pct_cols=pct_cols, decimals=3)
        st.dataframe(style_player_discipline(show), use_container_width=True)

# -------------------- Pitching --------------------

# -------------------- Team Stats (TEAM ONLY) --------------------
with tabs[3]:
    st.subheader("Team Stats")

    def _num(x):
        return pd.to_numeric(x, errors="coerce").fillna(0)

    def _safe_div(n, d):
        n = float(n) if pd.notna(n) else 0.0
        d = float(d) if pd.notna(d) else 0.0
        return (n / d) if d != 0 else np.nan

    def _sum_first_present(df: pd.DataFrame, options):
        for c in options:
            if c in df.columns:
                return float(_num(df[c]).sum())
        return np.nan

    # ---------------- TEAM HITTING (1 row) ----------------
    d = bat_season_full.copy()

    PA = float(_num(d["PA"]).sum()) if "PA" in d.columns else np.nan
    AB = float(_num(d["AB"]).sum()) if "AB" in d.columns else np.nan
    H  = float(_num(d["H"]).sum())  if "H" in d.columns else np.nan
    XBH = float(_num(d["XBH"]).sum()) if "XBH" in d.columns else np.nan
    BB = float(_num(d["BB"]).sum()) if "BB" in d.columns else np.nan
    K  = float(_num(d["K"]).sum())  if "K" in d.columns else np.nan
    HBP = float(_num(d["HBP"]).sum()) if "HBP" in d.columns else 0.0

    # Total pitches (prefer explicit Pitches; otherwise infer from P/PA * PA)
    pitches_total = np.nan
    if "Pitches" in d.columns:
        pitches_total = float(_num(d["Pitches"]).sum())
    elif all(c in d.columns for c in ["P/PA", "PA"]):
        pitches_total = float((_num(d["P/PA"]) * _num(d["PA"])).sum())
    # Team situational AVG from PA table (weighted)
    team_risp_sub = pa[pa.get("risp_flag", 0) == 1].copy() if (not pa.empty and "risp_flag" in pa.columns) else pd.DataFrame()
    team_2out_sub = pa[pa.get("two_out_flag", 0) == 1].copy() if (not pa.empty and "two_out_flag" in pa.columns) else pd.DataFrame()

    team_risp_ab = float(pd.to_numeric(team_risp_sub.get("ab_flag", 0), errors="coerce").fillna(0).sum()) if not team_risp_sub.empty else 0.0
    team_risp_h  = float(pd.to_numeric(team_risp_sub.get("hit_flag", 0), errors="coerce").fillna(0).sum()) if not team_risp_sub.empty else 0.0
    team_2out_ab = float(pd.to_numeric(team_2out_sub.get("ab_flag", 0), errors="coerce").fillna(0).sum()) if not team_2out_sub.empty else 0.0
    team_2out_h  = float(pd.to_numeric(team_2out_sub.get("hit_flag", 0), errors="coerce").fillna(0).sum()) if not team_2out_sub.empty else 0.0

    team_risp_avg = (team_risp_h / team_risp_ab) if team_risp_ab != 0 else np.nan
    team_2out_avg = (team_2out_h / team_2out_ab) if team_2out_ab != 0 else np.nan
    team_hit = pd.DataFrame([{
        "Batter": "TEAM",
        "PA": PA,
        "AB": AB,
        "H": H,
        "XBH": XBH,
        "BB": BB,
        "K": K,
        "AVG": _safe_div(H, AB),
        "OBP": _safe_div(H + BB + HBP, PA),

        "K%": _safe_div(K, PA) * 100.0,
        "P/PA": _safe_div(pitches_total, PA),
    }])

    # ---------------- TEAM DISCIPLINE (1 row) ----------------
    ds = disc_season_full.copy()

    pitches = _sum_first_present(ds, ["Pitches"])
    swings  = _sum_first_present(ds, ["Swings"])
    whiffs  = _sum_first_present(ds, ["Whiffs"])
    contacts= _sum_first_present(ds, ["Contacts"])

    fp_sw   = _sum_first_present(ds, ["FirstPitchSwings", "First Pitch Swings"])
    fp_str  = _sum_first_present(ds, ["FirstPitchStrikes", "First Pitch Strikes"])
    fp_in   = _sum_first_present(ds, ["FirstPitchInPlay", "First Pitch In Play"])

    first_total = (fp_sw if pd.notna(fp_sw) else 0.0) + (fp_str if pd.notna(fp_str) else 0.0) + (fp_in if pd.notna(fp_in) else 0.0)

    # 2K counts (try several common column names)
    tk_p = _sum_first_present(ds, ["TwoK_Pitches", "2K_Pitches", "TwoStrikePitches", "TwoStrikePitchesFaced"])
    tk_s = _sum_first_present(ds, ["TwoK_Swings", "2K_Swings", "TwoStrikeSwings"])
    tk_w = _sum_first_present(ds, ["TwoK_Whiffs", "2K_Whiffs", "TwoStrikeWhiffs"])
    tk_c = _sum_first_present(ds, ["TwoK_Contacts", "2K_Contacts", "TwoStrikeContacts"])

    team_disc = {
        "Batter": "TEAM",
        "Pitches": pitches,
        "Swings": swings,
        "Whiffs": whiffs,
        "Contacts": contacts,
        "Swing%": _safe_div(swings, pitches) * 100.0,
        "Whiff%": _safe_div(whiffs, swings) * 100.0,
        "Contact%": _safe_div(contacts, swings) * 100.0,
        "First Pitch Swings": fp_sw,
        "First Pitch Strikes": fp_str,
        "First Pitch In Play": fp_in,
        "1stPitchSwing%": (fp_sw / first_total * 100.0) if first_total > 0 else np.nan,
        "1stPitchStrike%": (fp_str / first_total * 100.0) if first_total > 0 else np.nan,
        "1stPitchInPlay%": (fp_in / first_total * 100.0) if first_total > 0 else np.nan,
    }

    # 2K team averages
    if pd.notna(tk_p) and tk_p > 0 and pd.notna(tk_s):
        team_disc["2K_Swing%"] = _safe_div(tk_s, tk_p) * 100.0
    if pd.notna(tk_s) and tk_s > 0 and pd.notna(tk_w):
        team_disc["2K_Whiff%"] = _safe_div(tk_w, tk_s) * 100.0
    if pd.notna(tk_s) and tk_s > 0 and pd.notna(tk_c):
        team_disc["2K_Contact%"] = _safe_div(tk_c, tk_s) * 100.0

    # Fallback: if only % columns exist in the file (avoid double *100)
    # Detect if values look like 0-100 already vs 0-1 fractions.
    def _pct_series(name_options, weight_options=None):
        col = next((c for c in name_options if c in ds.columns), None)
        if col is None:
            return None
        s = pd.to_numeric(ds[col], errors="coerce")
        w = None
        if weight_options:
            wcol = next((c for c in weight_options if c in ds.columns), None)
            if wcol is not None:
                w = _num(ds[wcol])
        val = None
        try:
            if w is not None and w.sum() != 0:
                val = float(np.average(s, weights=w))
            else:
                val = float(s.mean())
        except Exception:
            return None
        # normalize
        if pd.notna(val) and val <= 1.5:
            val = val * 100.0
        return val

    if "2K_Swing%" not in team_disc or pd.isna(team_disc.get("2K_Swing%")):
        v = _pct_series(["2K_Swing%", "TwoK_Swing%", "2K Swing%"], weight_options=["TwoStrikePitches", "TwoK_Pitches", "2K_Pitches", "TwoStrikeSwings", "TwoK_Swings", "2K_Swings"])
        if v is not None:
            team_disc["2K_Swing%"] = v
    if "2K_Whiff%" not in team_disc or pd.isna(team_disc.get("2K_Whiff%")):
        v = _pct_series(["2K_Whiff%", "TwoK_Whiff%", "2K Whiff%"], weight_options=["TwoStrikeSwings", "TwoK_Swings", "2K_Swings"])
        if v is not None:
            team_disc["2K_Whiff%"] = v
    if "2K_Contact%" not in team_disc or pd.isna(team_disc.get("2K_Contact%")):
        v = _pct_series(["2K_Contact%", "TwoK_Contact%", "2K Contact%"], weight_options=["TwoStrikeSwings", "TwoK_Swings", "2K_Swings"])
        if v is not None:
            team_disc["2K_Contact%"] = v

    team_disc = pd.DataFrame([team_disc])

    # Display (match your style: 3 decimals for non-%; % keep as %)
    for col in ["AVG", "OBP", "P/PA"]:
        if col in team_hit.columns:
            team_hit[col] = pd.to_numeric(team_hit[col], errors="coerce").round(3)

    if "K%" in team_hit.columns:
        team_hit["K%"] = pd.to_numeric(team_hit["K%"], errors="coerce").round(1)

    # Discipline rounding
    for col in team_disc.columns:
        if col.endswith("%"):
            team_disc[col] = pd.to_numeric(team_disc[col], errors="coerce").round(1)
        else:
            if team_disc[col].dtype.kind in "fc":
                team_disc[col] = pd.to_numeric(team_disc[col], errors="coerce").round(3)

    st.markdown("#### Team Hitting")
    st.dataframe(team_hit, use_container_width=True)

    st.markdown("#### Team Discipline")
    st.dataframe(team_disc, use_container_width=True)

    st.markdown("#### Team Situational (AVG)")
    team_sit = pd.DataFrame([{
        "Context": "RISP",
        "AB": int(team_risp_ab),
        "H": int(team_risp_h),
        "AVG": team_risp_avg,
    },{
        "Context": "2 Outs",
        "AB": int(team_2out_ab),
        "H": int(team_2out_h),
        "AVG": team_2out_avg,
    }])
    team_sit = round_for_display(team_sit, pct_cols=[], decimals=3)
    st.dataframe(team_sit, use_container_width=True)