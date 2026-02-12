import os, re, pathlib
import pandas as pd
import numpy as np

TEAM_MATCH = "East Wake Academy"

INNING_RE = re.compile(r'^(Top|Bottom)\s+(\d+)(?:st|nd|rd|th)\s+-\s+(.*)$', re.IGNORECASE)

PA_TYPES = [
    # Common GameChanger PA outcomes (header lines)
    "Walk","Intentional Walk","Hit By Pitch",
    "Strikeout","Dropped 3rd Strike",
    "Single","Double","Triple","Home Run",
    "Ground Out","Fly Out","Line Out","Pop Out","Field Out",
    "Double Play","Grounded Into Double Play",
    "Error","Reached on Error",
    "Fielder's Choice","Reached on Fielder's Choice",
    "Sac Fly","Sacrifice Fly","Sac Bunt","Sacrifice Bunt","Bunt",
    "Catcher's Interference","Interference",
    "Runner Out",
]
PA_HEADER_RE = re.compile(r'^(' + "|".join([re.escape(x) for x in PA_TYPES]) + r')(?:\|.*)?$', re.IGNORECASE)

PITCH_TOKEN_RE = re.compile(
    r'\b(Ball\s+\d+|Ball|Strike\s+\d+\s+looking|Strike\s+\d+\s+swinging|Strike\s+\d+|'
    r'Strike\s+3\s+looking|Strike\s+3\s+swinging|Foul tip|Foul|In play)\b',
    re.IGNORECASE
)

BATTER_PITCHER_RE = re.compile(r'^([A-Z]\s+[A-Za-z\.\'-]+)\s+.*?,\s+([A-Z]\s+[A-Za-z\.\'-]+)\s+pitching\.$')

SENTENCE_OUTCOME_RE = re.compile(
    r'^([A-Z]\s+[A-Za-z\.\'-]+)\s+'
    r'(singles|doubles|triples|homers|walks|strikes out|reaches on error|reached on error|hit by pitch)\b',
    re.IGNORECASE
)
# Runner movement parsing (for RISP / outs tracking)
RUNNER_ADV_RE = re.compile(r'([A-Z]\s+[A-Za-z\.\'-]+)\s+(?:advances|steals)\s+to\s+(1st|2nd|3rd|home)', re.IGNORECASE)
RUNNER_REMAINS_RE = re.compile(r'([A-Z]\s+[A-Za-z\.\'-]+)\s+remains\s+at\s+(1st|2nd|3rd)', re.IGNORECASE)
RUNNER_SCORES_RE = re.compile(r'([A-Z]\s+[A-Za-z\.\'-]+)\s+scores', re.IGNORECASE)
RUNNER_OUT_RE = re.compile(r'([A-Z]\s+[A-Za-z\.\'-]+)\s+(?:is\s+out|out)\b', re.IGNORECASE)

BASE_MAP = {"1st": 1, "2nd": 2, "3rd": 3, "home": 0}

def _remove_runner(bases, name):
    for b in (1,2,3):
        if bases.get(b) == name:
            bases.pop(b, None)

def _set_base(bases, name, base_num):
    _remove_runner(bases, name)
    if base_num in (1,2,3):
        bases[base_num] = name

def apply_desc_to_bases_and_outs(desc_lines, bases, outs):
    # Apply explicit runner moves from GameChanger sentences.
    for ln in desc_lines:
        for m in RUNNER_SCORES_RE.finditer(ln):
            _remove_runner(bases, m.group(1).strip())
        for m in RUNNER_ADV_RE.finditer(ln):
            name = m.group(1).strip()
            base = BASE_MAP.get(m.group(2).lower())
            if base == 0:
                _remove_runner(bases, name)
            else:
                _set_base(bases, name, base)
        for m in RUNNER_REMAINS_RE.finditer(ln):
            name = m.group(1).strip()
            base = BASE_MAP.get(m.group(2).lower())
            if base:
                _set_base(bases, name, base)
        # crude runner out detection (e.g., "X out at second")
        # If line includes "out at" or "picked off" we count an out and clear that runner.
        if "out at" in ln.lower() or "picked off" in ln.lower() or "caught stealing" in ln.lower():
            m = RUNNER_OUT_RE.search(ln)
            if m:
                _remove_runner(bases, m.group(1).strip())
            outs += 1
    return bases, outs

def apply_pa_outcome(pa_outcome, batter, desc_lines, bases, outs):
    out_l = (pa_outcome or "").lower()

    # Apply batter result first (default assumptions)
    batter_reaches = None  # base number or 0 for scores
    if out_l in {"single","double","triple","home run"}:
        batter_reaches = {"single":1,"double":2,"triple":3,"home run":0}[out_l]
    elif out_l in {"walk","intentional walk","hit by pitch","error","reached on error","catcher's interference","interference","reached on fielder's choice"}:
        batter_reaches = 1
    elif "fielder's choice" in out_l:
        batter_reaches = 1
        outs += 1
    elif "double play" in out_l:
        outs += 2
    elif "sac" in out_l:
        outs += 1
    elif out_l in {"strikeout","dropped 3rd strike","ground out","fly out","line out","pop out","field out","runner out"}:
        outs += 1

    # Apply explicit runner movement text (overrides)
    bases, outs = apply_desc_to_bases_and_outs(desc_lines, bases, outs)

    # Place batter if they reached and weren't explicitly scored out
    if batter and batter_reaches is not None:
        if batter_reaches == 0:
            _remove_runner(bases, batter)
        else:
            _set_base(bases, batter, batter_reaches)

    # Cap outs at 3; clear inning if over
    if outs >= 3:
        outs = 3
    return bases, outs



DEF_TO_RE = re.compile(
    r'\bto\s+(shortstop|third baseman|second baseman|first baseman|left fielder|center fielder|right fielder|catcher|pitcher)\s+([A-Z]\s+[A-Za-z\.\'-]+)\b',
    re.IGNORECASE
)
ERR_BY_RE = re.compile(
    r'error by\s+(shortstop|third baseman|second baseman|first baseman|left fielder|center fielder|right fielder|catcher|pitcher)\s+([A-Z]\s+[A-Za-z\.\'-]+)',
    re.IGNORECASE
)

def position_code(pos: str) -> str:
    p = pos.lower()
    return {
        "pitcher":"P","catcher":"C","first baseman":"1B","second baseman":"2B","third baseman":"3B",
        "shortstop":"SS","left fielder":"LF","center fielder":"CF","right fielder":"RF"
    }.get(p, pos)

def normalize_pitch_token(tok: str) -> str:
    t = tok.strip().lower()
    if t.startswith("ball"): return "ball"
    if "looking" in t: return "called_strike"
    if "swinging" in t: return "swinging_strike"
    if t == "foul": return "foul"
    if t == "foul tip": return "foul_tip"
    if t == "in play": return "in_play"
    if t.startswith("strike"): return "strike_other"
    return "other"

def is_swing(pr): return pr in {"swinging_strike","foul","foul_tip","in_play"}
def is_whiff(pr): return pr == "swinging_strike"
def is_contact(pr): return pr in {"foul","foul_tip","in_play"}

def safe_div(n, d):
    return np.where(d==0, np.nan, n/d)

def main():
    base = pathlib.Path(__file__).resolve().parent
    raw_dir = base / "raw_games"
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        print("No .txt files found in raw_games/ — clearing outputs...")

        empty_files = [
            "pitch_events_ALL.csv",
            "plate_appearances_ALL.csv",
            "pitch_events_EWA.csv",
            "plate_appearances_EWA.csv",
            "batting_season_EWA.csv",
            "discipline_season_EWA.csv",
            "pitching_basic_EWA.csv",
            "fielding_basic_EWA.csv",
            "batting_count_splits_EWA.csv",
        ]

        for fn in empty_files:
            p = base / fn
            if p.exists():
                try:
                    df0 = pd.read_csv(p)
                    df0.head(0).to_csv(p, index=False)
                except Exception:
                    p.write_text("")
        print("Outputs cleared ✅")
        return

    pitch_rows, pa_rows, def_rows = [], [], []
    counters = {"pa_id": 1, "pitch_id": 1}

    def flush_pa(state):
        nonlocal bases, outs
        if state["pa"] is None:
            return
        pa = state["pa"]
        tokens = state["tokens"]
        desc_lines = state.get("desc", [])

        balls, strikes = 0, 0
        first_pitch_result = None
        first_pitch_swing = 0

        for i, tok in enumerate(tokens, start=1):
            pr = normalize_pitch_token(tok)
            pitch_rows.append({
                "pitch_id": counters["pitch_id"],
                "pa_id": pa["pa_id"],
                "game_id": pa["game_id"],
                "inning": pa["inning"],
                "half": pa["half"],
                "offense_team": pa["offense_team"],
                "batter": pa.get("batter"),
                "pitcher": pa.get("pitcher"),
                "pitch_number_in_pa": i,
                "balls_before": balls,
                "strikes_before": strikes,
                "pitch_result": pr,
                "is_first_pitch": 1 if i==1 else 0,
                "is_two_strike_pitch": 1 if strikes==2 else 0,
                "is_swing": 1 if is_swing(pr) else 0,
                "is_whiff": 1 if is_whiff(pr) else 0,
                "is_contact": 1 if is_contact(pr) else 0,
            })
            counters["pitch_id"] += 1

            if i == 1:
                first_pitch_result = pr
                first_pitch_swing = 1 if is_swing(pr) else 0

            if pr == "ball":
                balls = min(4, balls+1)
            elif pr in {"called_strike","swinging_strike"}:
                strikes = min(3, strikes+1)
            elif pr in {"foul","foul_tip"} and strikes < 2:
                strikes += 1

        outcome = pa["outcome"]
        out_l = outcome.lower()
        is_hit = out_l in {"single","double","triple","home run"}
        hit_type = outcome if is_hit else None
        is_walk = out_l in {"walk","intentional walk"}
        is_hbp = out_l == "hit by pitch"
        is_k = out_l in {"strikeout","dropped 3rd strike"}
        ab_flag = 0 if (is_walk or is_hbp or ("sac" in out_l) or ("interference" in out_l)) else 1

        swings = sum(is_swing(normalize_pitch_token(t)) for t in tokens)
        whiffs = sum(is_whiff(normalize_pitch_token(t)) for t in tokens)
        contacts = sum(is_contact(normalize_pitch_token(t)) for t in tokens)

        pa_rows.append({
            "pa_id": pa["pa_id"],
            "game_id": pa["game_id"],
            "inning": pa["inning"],
            "half": pa["half"],
            "offense_team": pa["offense_team"],
            "batter": pa.get("batter"),
            "outs_before": pa.get("outs_before", np.nan),
            "risp_flag": pa.get("risp_flag", 0),
            "two_out_flag": pa.get("two_out_flag", 0),
            "pitcher": pa.get("pitcher"),
            "outcome": outcome,
            "hit_type": hit_type,
            "pitches_in_pa": len(tokens),
            "first_pitch_result": first_pitch_result,
            "first_pitch_swing": first_pitch_swing,
            "swings": int(swings),
            "whiffs": int(whiffs),
            "contacts": int(contacts),
            "walk_flag": int(is_walk),
            "hbp_flag": int(is_hbp),
            "k_flag": int(is_k),
            "ab_flag": int(ab_flag),
            "hit_flag": int(is_hit),
        })

        bases, outs = apply_pa_outcome(pa.get("outcome"), pa.get("batter"), desc_lines, bases, outs)

        state["pa"] = None
        state["tokens"] = []
        state["desc"] = []

    for f in files:
        game_id = f.stem
        raw = f.read_text(errors="ignore")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        inning_half, inning_num, offense_team = None, None, None
        bases = {}
        outs = 0
        state = {"pa": None, "tokens": [], "desc": []}

        last_pitcher = None

        for ln in lines:
            m_in = INNING_RE.match(ln)
            if m_in:
                flush_pa(state)
                inning_half = m_in.group(1).capitalize()
                inning_num = int(m_in.group(2))
                offense_team = m_in.group(3).strip()
                bases = {}
                outs = 0
                continue

            # Standalone PA outcome header lines (GameChanger variant)
            first = ln.split("|")[0].strip()
            if PA_HEADER_RE.match(first):
                flush_pa(state)
                state["pa"] = {
                    "pa_id": counters["pa_id"],
                    "game_id": game_id,
                    "inning": inning_num,
                    "half": inning_half,
                    "offense_team": offense_team,
                    "outs_before": outs,
                    "risp_flag": 1 if (bases.get(2) or bases.get(3)) else 0,
                    "two_out_flag": 1 if outs==2 else 0,
                    "outcome": first,
                    "batter": None,
                    "pitcher": last_pitcher,
                }
                counters["pa_id"] += 1
                continue

            # Pitch tokens can appear before any explicit outcome header.
            toks = PITCH_TOKEN_RE.findall(ln)

            if state["pa"] is not None:
                state["desc"].append(ln)

            # Sentence-style outcomes like "J Fuentes doubles on a fly ball ..."
            m_sent = SENTENCE_OUTCOME_RE.match(ln)
            if m_sent:
                batter_name = m_sent.group(1).strip()
                verb = m_sent.group(2).lower()

                verb_map = {
                    "singles": "Single",
                    "doubles": "Double",
                    "triples": "Triple",
                    "homers": "Home Run",
                    "walks": "Walk",
                    "strikes out": "Strikeout",
                    "reaches on error": "Reached on Error",
                    "reached on error": "Reached on Error",
                    "hit by pitch": "Hit By Pitch",
                }
                outcome = verb_map.get(verb, verb.title())

                # If we don't have an active PA yet, start one now.
                if state["pa"] is None:
                    state["pa"] = {
                        "pa_id": counters["pa_id"],
                        "game_id": game_id,
                        "inning": inning_num,
                        "half": inning_half,
                        "offense_team": offense_team,
                        "outs_before": outs,
                        "risp_flag": 1 if (bases.get(2) or bases.get(3)) else 0,
                        "two_out_flag": 1 if outs==2 else 0,
                        "outcome": outcome,
                        "batter": batter_name,
                        "pitcher": last_pitcher,
                    }
                    counters["pa_id"] += 1
                else:
                    state["pa"]["outcome"] = outcome
                    state["pa"]["batter"] = state["pa"].get("batter") or batter_name
                    state["pa"]["pitcher"] = state["pa"].get("pitcher") or last_pitcher

                # Add any tokens found on the same line, then PA is complete.
                if toks:
                    state["tokens"].extend(toks)
                flush_pa(state)
                continue

            # If we have tokens but no active PA, start a placeholder PA that we can finalize later.
            if toks and state["pa"] is None:
                state["pa"] = {
                    "pa_id": counters["pa_id"],
                    "game_id": game_id,
                    "inning": inning_num,
                    "half": inning_half,
                    "offense_team": offense_team,
                    "outs_before": outs,
                    "risp_flag": 1 if (bases.get(2) or bases.get(3)) else 0,
                    "two_out_flag": 1 if outs==2 else 0,
                    "outcome": "Unknown",
                    "batter": None,
                    "pitcher": last_pitcher,
                }
                counters["pa_id"] += 1

            if state["pa"] is not None:
                if toks:
                    state["tokens"].extend(toks)

                m_bp = BATTER_PITCHER_RE.match(ln)
                if m_bp:
                    state["pa"]["batter"] = m_bp.group(1).strip()
                    state["pa"]["pitcher"] = m_bp.group(2).strip()
                    last_pitcher = m_bp.group(2).strip()

                for m in DEF_TO_RE.finditer(ln):
                    def_rows.append({
                        "game_id": game_id,
                        "inning": inning_num,
                        "half": inning_half,
                        "offense_team": offense_team,
                        "batter": state["pa"].get("batter"),
                        "event_type": "ball_to",
                        "position": position_code(m.group(1)),
                        "fielder": m.group(2).strip(),
                        "event_line": ln,
                    })
                for m in ERR_BY_RE.finditer(ln):
                    def_rows.append({
                        "game_id": game_id,
                        "inning": inning_num,
                        "half": inning_half,
                        "offense_team": offense_team,
                        "batter": state["pa"].get("batter"),
                        "event_type": "error",
                        "position": position_code(m.group(1)),
                        "fielder": m.group(2).strip(),
                        "event_line": ln,
                    })

        flush_pa(state)


    pitch = pd.DataFrame(pitch_rows)
    pa = pd.DataFrame(pa_rows)
    defense = pd.DataFrame(def_rows)

    # --- TEAM VIEWS ---
    pa_ewa = pa[pa["offense_team"].fillna("").str.contains(TEAM_MATCH, case=False, regex=False)].copy()
    pitch_ewa = pitch[pitch["offense_team"].fillna("").str.contains(TEAM_MATCH, case=False, regex=False)].copy()

    # pitching view: opponent offense, pitcher is our guy (we'll roster-map later)
    pa_opp = pa[~pa["offense_team"].fillna("").str.contains(TEAM_MATCH, case=False, regex=False)].copy()
    pitch_opp = pitch[~pitch["offense_team"].fillna("").str.contains(TEAM_MATCH, case=False, regex=False)].copy()

    # --- BASIC BATTING ---
    bat = pa_ewa.copy()
    bat["TB"] = bat["hit_type"].map({"Single":1,"Double":2,"Triple":3,"Home Run":4}).fillna(0).astype(int)

    bat_season = bat.groupby("batter", dropna=True).agg(
        PA=("pa_id","count"),
        AB=("ab_flag","sum"),
        H=("hit_flag","sum"),
        XBH=("hit_type", lambda s: ((s=="Double") | (s=="Triple") | (s=="Home Run")).sum()),
        BB=("walk_flag","sum"),
        HBP=("hbp_flag","sum"),
        K=("k_flag","sum"),
        TB=("TB","sum"),
        Pitches=("pitches_in_pa","sum"),
    ).reset_index()

    bat_season["AVG"] = safe_div(bat_season["H"], bat_season["AB"])
    bat_season["OBP"] = safe_div(bat_season["H"]+bat_season["BB"]+bat_season["HBP"], bat_season["PA"])
    bat_season["SLG"] = safe_div(bat_season["TB"], bat_season["AB"])
    bat_season["OPS"] = bat_season["OBP"] + bat_season["SLG"]
    bat_season["K%"] = safe_div(bat_season["K"], bat_season["PA"])
    bat_season["P/PA"] = safe_div(bat_season["Pitches"], bat_season["PA"])

    # --- DISCIPLINE ---
    disc = pitch_ewa.copy()
    fp = disc[disc["is_first_pitch"]==1].groupby("batter").agg(
        FirstPitchSwings=("is_swing","sum"),
        FirstPitches=("pitch_id","count"),
        FirstPitchStrikes=("pitch_result", lambda s: s.isin(["called_strike","swinging_strike","foul","foul_tip","in_play","strike_other"]).sum())
    ).reset_index()

    # Derived count on each pitch (balls_before-strikes_before), e.g., "1-2"
    disc["count"] = disc["balls_before"].astype(str) + "-" + disc["strikes_before"].astype(str)

    # First pitch in-play flag (for 1st pitch in-play%)
    disc["first_pitch_in_play"] = ((disc["is_first_pitch"] == 1) & (disc["pitch_result"] == "in_play")).astype(int)

    disc_season = disc.groupby("batter", dropna=True).agg(
        Pitches=("pitch_id","count"),
        Swings=("is_swing","sum"),
        Whiffs=("is_whiff","sum"),
        Contacts=("is_contact","sum"),
        TwoStrikeSwings=("is_swing", lambda s: 0),  # placeholder (we compute below)
    ).reset_index().merge(fp, on="batter", how="left").fillna(0)

    disc_season["Swing%"] = safe_div(disc_season["Swings"], disc_season["Pitches"])
    disc_season["Whiff%"] = safe_div(disc_season["Whiffs"], disc_season["Swings"])
    disc_season["Contact%"] = safe_div(disc_season["Contacts"], disc_season["Swings"])
    disc_season["1stPitchSwing%"] = safe_div(disc_season["FirstPitchSwings"], disc_season["FirstPitches"])
    disc_season["1stPitchStrike%"] = safe_div(disc_season["FirstPitchStrikes"], disc_season["FirstPitches"])

    # 1st pitch in-play%
    fp_inplay = disc.groupby("batter", dropna=True).agg(
        FirstPitchInPlay=("first_pitch_in_play", "sum"),
        FirstPitches2=("is_first_pitch", "sum")
).reset_index()
    disc_season = disc_season.merge(fp_inplay, on="batter", how="left").fillna(0)
    disc_season["1stPitchInPlay%"] = safe_div(disc_season["FirstPitchInPlay"], disc_season["FirstPitches2"])

    # 2-strike swing/whiff/contact rates (pitches where strikes_before == 2)
    two = disc[disc["is_two_strike_pitch"] == 1].copy()
    two_agg = two.groupby("batter", dropna=True).agg(
        TwoK_Pitches=("pitch_id", "count"),
        TwoK_Swings=("is_swing", "sum"),
        TwoK_Whiffs=("is_whiff", "sum"),
        TwoK_Contacts=("is_contact", "sum"),
).reset_index()

    two_agg["2K_Swing%"] = safe_div(two_agg["TwoK_Swings"], two_agg["TwoK_Pitches"])
    two_agg["2K_Whiff%"] = safe_div(two_agg["TwoK_Whiffs"], two_agg["TwoK_Swings"])
    two_agg["2K_Contact%"] = safe_div(two_agg["TwoK_Contacts"], two_agg["TwoK_Swings"])

    disc_season = disc_season.merge(two_agg[["batter","2K_Swing%","2K_Whiff%","2K_Contact%"]], on="batter", how="left")

    # --- QAB (simple, editable definition) ---
    # QAB if: reach base OR see 6+ pitches (grind) OR put ball in play with 2 strikes (not K)
    bat["QAB"] = (
        (bat["hit_flag"]==1) |
        (bat["walk_flag"]==1) |
        (bat["hbp_flag"]==1) |
        (bat["pitches_in_pa"]>=6) |
        ((bat["k_flag"]==0) & (bat["pitches_in_pa"]>=4))
    ).astype(int)

    qab = bat.groupby("batter", dropna=True).agg(QAB=("QAB","sum"), PA=("pa_id","count")).reset_index()
    qab["QAB%"] = safe_div(qab["QAB"], qab["PA"])
    bat_season = bat_season.merge(qab[["batter","QAB%"]], on="batter", how="left")

    # --- PITCHING (basic, from opponent offense PAs) ---
    # We count outcomes while our pitcher is listed. (Later we roster-map to know which pitchers are ours.)
    pit = pa_opp.copy()
    pit_basic = pit.groupby("pitcher", dropna=True).agg(
        BF=("pa_id","count"),
        H=("hit_flag","sum"),
        BB=("walk_flag","sum"),
        HBP=("hbp_flag","sum"),
        K=("k_flag","sum"),
        XBH=("hit_type", lambda s: ((s=="Double") | (s=="Triple") | (s=="Home Run")).sum()),
        HR=("hit_type", lambda s: (s=="Home Run").sum()),
    ).reset_index()

    # --- FIELDING (basic from parsed defensive events; beta) ---
    def_basic = defense[~defense["offense_team"].fillna("").str.contains(TEAM_MATCH, case=False, regex=False)].copy()
    field_sum = def_basic.groupby(["position","fielder"], dropna=True).agg(
        Chances=("event_type","count"),
        Errors=("event_type", lambda s: (s=="error").sum()),
    ).reset_index()
    field_sum["Fld%"] = safe_div((field_sum["Chances"]-field_sum["Errors"]), field_sum["Chances"])

    # --- COUNT SPLITS (simple buckets using final pitch count in PA as proxy) ---
    last_pitch = pitch_ewa.sort_values(["pa_id","pitch_number_in_pa"]).groupby("pa_id").tail(1)[
        ["pa_id","balls_before","strikes_before"]
    ].copy()
    last_pitch["finish_count"] = last_pitch["balls_before"].astype(str) + "-" + last_pitch["strikes_before"].astype(str)

    bat2 = pa_ewa.merge(last_pitch[["pa_id","finish_count"]], on="pa_id", how="left")

    def bucket(c):
        if pd.isna(c): return "Unknown"
        if c == "0-0": return "0-0"
        if c in {"1-0","2-0","2-1","3-1","3-0"}: return "Hitter"
        if c in {"0-1","0-2","1-2","2-2"}: return "Pitcher"
        if c.endswith("-2"): return "2K"
        return "Other"

    bat2["count_bucket"] = bat2["finish_count"].apply(bucket)

    count_split = bat2.groupby(["batter","count_bucket"], dropna=True).agg(
        PA=("pa_id","count"),
        AB=("ab_flag","sum"),
        H=("hit_flag","sum"),
        K=("k_flag","sum"),
    ).reset_index()
    count_split["AVG"] = safe_div(count_split["H"], count_split["AB"])
    count_split["K%"] = safe_div(count_split["K"], count_split["PA"])

    # Save outputs
    pitch.to_csv(base/"pitch_events_ALL.csv", index=False)
    pa.to_csv(base/"plate_appearances_ALL.csv", index=False)

    pitch_ewa.to_csv(base/"pitch_events_EWA.csv", index=False)
    pa_ewa.to_csv(base/"plate_appearances_EWA.csv", index=False)

    bat_season.to_csv(base/"batting_season_EWA.csv", index=False)
    disc_season.to_csv(base/"discipline_season_EWA.csv", index=False)

    pit_basic.to_csv(base/"pitching_basic_EWA.csv", index=False)
    field_sum.to_csv(base/"fielding_basic_EWA.csv", index=False)
    count_split.to_csv(base/"batting_count_splits_EWA.csv", index=False)

    print("Build complete ✅")

if __name__ == "__main__":
    main()
