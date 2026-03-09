import os, re, pathlib
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.3f}'.format
def round_df(df):
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].round(3)
    return df

TEAM_MATCH = "East Wake"

INNING_RE = re.compile(r'^(Top|Bottom)\s+(\d+)(?:st|nd|rd|th)\s+-\s+(.*)$', re.IGNORECASE)

PA_TYPES = [
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

BATTER_PITCHER_RE = re.compile(
    r'^([A-Z]\s+[A-Za-z\.\'-]+)\s+.*?,\s+([A-Z]\s+[A-Za-z\.\'-]+)\s+pitching\.$'
)
PITCHER_SUB_RE = re.compile(r'^([A-Z]\s+[A-Za-z\.\'-]+)\s+in\s+for\s+pitcher\s+[A-Z]\s+[A-Za-z\.\'-]+', re.IGNORECASE)

SENTENCE_OUTCOME_RE = re.compile(
    r'^([A-Z]\s+[A-Za-z\.\'-]+)\s+'
    r'(singles|doubles|triples|homers|walks|strikes out|reaches on error|reached on error|hit by pitch)\b',
    re.IGNORECASE
)

BATTER_EVENT_RE = re.compile(
    r'^([A-Z]\s+[A-Za-z\.\'-]+)\s+'
    r'(singles|doubles|triples|homers|walks|strikes out|reaches|reached|grounds|flies|lines|pops|bunts|is hit|out on)\b',
    re.IGNORECASE
)

RUNNER_ADV_RE = re.compile(
    r'([A-Z]\s+[A-Za-z\.\'-]+)\s+(?:advances|steals)\s+to\s+(1st|2nd|3rd|home)', re.IGNORECASE
)
RUNNER_REMAINS_RE = re.compile(
    r'([A-Z]\s+[A-Za-z\.\'-]+)\s+remains\s+at\s+(1st|2nd|3rd)', re.IGNORECASE
)
RUNNER_SCORES_RE = re.compile(r'([A-Z]\s+[A-Za-z\.\'-]+)\s+scores', re.IGNORECASE)
RUNNER_OUT_RE = re.compile(r'([A-Z]\s+[A-Za-z\.\'-]+)\s+(?:is\s+out|out)\b', re.IGNORECASE)

BASE_MAP = {"1st": 1, "2nd": 2, "3rd": 3, "home": 0}

DEF_TO_RE = re.compile(
    r'\bto\s+(shortstop|third baseman|second baseman|first baseman|left fielder|center fielder|right fielder|catcher|pitcher)\s+([A-Z]\s+[A-Za-z\.\'-]+)\b',
    re.IGNORECASE
)
ERR_BY_RE = re.compile(
    r'error by\s+(shortstop|third baseman|second baseman|first baseman|left fielder|center fielder|right fielder|catcher|pitcher)\s+([A-Z]\s+[A-Za-z\.\'-]+)',
    re.IGNORECASE
)
COURTESY_RUNNER_RE = re.compile(r'^Courtesy runner\s+([A-Z]\s+[A-Za-z\.\'-]+)\s+in\s+for\b', re.IGNORECASE)


def _remove_runner(bases, name):
    for b in (1, 2, 3):
        if bases.get(b) == name:
            bases.pop(b, None)


def _set_base(bases, name, base_num):
    _remove_runner(bases, name)
    if base_num in (1, 2, 3):
        bases[base_num] = name


def apply_desc_to_bases_and_outs(desc_lines, bases, outs):
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
        if "out at" in ln.lower() or "out advancing" in ln.lower() or "picked off" in ln.lower() or "caught stealing" in ln.lower():
            m = RUNNER_OUT_RE.search(ln)
            if m:
                _remove_runner(bases, m.group(1).strip())
            outs += 1
    return bases, outs


def apply_pa_outcome(pa_outcome, batter, desc_lines, bases, outs):
    out_l = (pa_outcome or "").lower()
    batter_reaches = None

    if out_l in {"single", "double", "triple", "home run"}:
        batter_reaches = {"single": 1, "double": 2, "triple": 3, "home run": 0}[out_l]
    elif out_l in {"walk", "intentional walk", "hit by pitch", "error", "reached on error",
                   "catcher's interference", "interference", "reached on fielder's choice"}:
        batter_reaches = 1
    elif "fielder's choice" in out_l:
        batter_reaches = 1
    elif "double play" in out_l:
        outs += 2
    elif "sac" in out_l:
        outs += 1
    elif out_l in {"strikeout", "dropped 3rd strike", "ground out", "fly out",
                   "line out", "pop out", "field out", "runner out"}:
        outs += 1

    bases, outs = apply_desc_to_bases_and_outs(desc_lines, bases, outs)

    if batter and batter_reaches is not None:
        if batter_reaches == 0:
            _remove_runner(bases, batter)
        else:
            _set_base(bases, batter, batter_reaches)

    if outs >= 3:
        outs = 3
    return bases, outs


def position_code(pos: str) -> str:
    p = pos.lower()
    return {
        "pitcher": "P", "catcher": "C", "first baseman": "1B", "second baseman": "2B",
        "third baseman": "3B", "shortstop": "SS", "left fielder": "LF",
        "center fielder": "CF", "right fielder": "RF"
    }.get(p, pos)


SPRAY_COORDS = {
    "P": (0, 60), "C": (0, 20),
    "1B": (70, 100), "2B": (15, 135), "SS": (-15, 135), "3B": (-70, 100),
    "LF": (-140, 235), "CF": (0, 265), "RF": (140, 235),
}


def infer_batted_ball_position(desc_lines):
    for ln in desc_lines:
        m = DEF_TO_RE.search(ln)
        if m:
            return position_code(m.group(1))
    return None


def infer_runner_advance_or_score(desc_lines):
    for ln in desc_lines:
        if RUNNER_ADV_RE.search(ln) or RUNNER_SCORES_RE.search(ln):
            return 1
    return 0


def infer_rbi_flag(desc_lines):
    # Approximation: if a run scores and the line does not attribute it to an error,
    # count as RBI for custom QAB logic.
    for ln in desc_lines:
        low = ln.lower()
        if RUNNER_SCORES_RE.search(ln) and "error by" not in low and "on error" not in low:
            return 1
    return 0


def normalize_pitch_token(tok: str) -> str:
    t = tok.strip().lower()
    if t.startswith("ball"):      return "ball"
    if "looking" in t:            return "called_strike"
    if "swinging" in t:           return "swinging_strike"
    if t == "foul tip":           return "foul_tip"
    if t == "foul":               return "foul"
    if t == "in play":            return "in_play"
    if t.startswith("strike"):    return "strike_other"
    return "other"


def is_swing(pr):    return pr in {"swinging_strike", "foul", "foul_tip", "in_play"}
def is_whiff(pr):    return pr == "swinging_strike"
def is_contact(pr):  return pr in {"foul", "foul_tip", "in_play"}
def is_strike_result(pr):
    return pr in {"called_strike", "swinging_strike", "foul", "foul_tip", "in_play", "strike_other"}


def safe_div(n, d):
    return np.where(d == 0, np.nan, n / d)


def _parse_num_series(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"-": np.nan, "": np.nan}), errors="coerce")


def find_gamechanger_stats_csv(base: pathlib.Path):
    csv_dir = base / "2026 CSV"
    local = []
    if csv_dir.exists() and csv_dir.is_dir():
        local.extend(list(csv_dir.glob("*Stats.csv")))
        local.extend(list(csv_dir.glob("*stats.csv")))
    local.extend(list(base.glob("*Stats.csv")))
    local.extend(list(base.glob("*stats.csv")))
    local = sorted(local)
    if local:
        return max(local, key=lambda p: p.stat().st_mtime)
    dl = pathlib.Path.home() / "Downloads"
    if dl.exists():
        cands = sorted(list(dl.glob("*Stats.csv")) + list(dl.glob("*stats.csv")))
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    return None


def load_gamechanger_overrides(base: pathlib.Path):
    csv_path = find_gamechanger_stats_csv(base)
    if not csv_path:
        return pd.DataFrame(), pd.DataFrame(), None
    try:
        gc = pd.read_csv(csv_path, header=1)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), None

    required = {"Number", "Last", "First"}
    if not required.issubset(set(gc.columns)):
        return pd.DataFrame(), pd.DataFrame(), None

    player_rows = gc[gc["Number"].astype(str).str.fullmatch(r"\d+")].copy()
    if player_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), str(csv_path)

    player_rows["batter"] = (
        player_rows["First"].astype(str).str[:1].str.upper() + " " + player_rows["Last"].astype(str)
    ).str.strip()

    # Batting override (keeps discipline from pitch events pipeline untouched).
    bat = pd.DataFrame({"batter": player_rows["batter"]})
    bat["PA"] = _parse_num_series(player_rows.get("PA", pd.Series(dtype="object"))).fillna(0)
    bat["AB"] = _parse_num_series(player_rows.get("AB", pd.Series(dtype="object"))).fillna(0)
    bat["H"] = _parse_num_series(player_rows.get("H", pd.Series(dtype="object"))).fillna(0)
    bat["1B"] = _parse_num_series(player_rows.get("1B", pd.Series(dtype="object"))).fillna(0)
    bat["2B"] = _parse_num_series(player_rows.get("2B", pd.Series(dtype="object"))).fillna(0)
    bat["3B"] = _parse_num_series(player_rows.get("3B", pd.Series(dtype="object"))).fillna(0)
    bat["HR"] = _parse_num_series(player_rows.get("HR", pd.Series(dtype="object"))).fillna(0)
    bat["BB"] = _parse_num_series(player_rows.get("BB", pd.Series(dtype="object"))).fillna(0)
    bat["HBP"] = _parse_num_series(player_rows.get("HBP", pd.Series(dtype="object"))).fillna(0)
    bat["K"] = _parse_num_series(player_rows.get("SO", pd.Series(dtype="object"))).fillna(0)
    bat["TB"] = bat["1B"] + 2 * bat["2B"] + 3 * bat["3B"] + 4 * bat["HR"]
    bat["XBH"] = bat["2B"] + bat["3B"] + bat["HR"]
    bat["Pitches"] = _parse_num_series(player_rows.get("PS", pd.Series(dtype="object"))).fillna(0)
    bat["AVG"] = _parse_num_series(player_rows.get("AVG", pd.Series(dtype="object")))
    bat["OBP"] = _parse_num_series(player_rows.get("OBP", pd.Series(dtype="object")))
    bat["SLG"] = _parse_num_series(player_rows.get("SLG", pd.Series(dtype="object")))
    bat["OPS"] = _parse_num_series(player_rows.get("OPS", pd.Series(dtype="object")))
    bat["K%"] = safe_div(bat["K"], bat["PA"])
    bat["P/PA"] = _parse_num_series(player_rows.get("PS/PA", pd.Series(dtype="object")))
    qab = _parse_num_series(player_rows.get("QAB%", pd.Series(dtype="object")))
    bat["QAB%"] = np.where(qab > 1.5, qab / 100.0, qab)
    bat = bat[[
        "batter", "PA", "AB", "H", "XBH", "BB", "HBP", "K", "TB", "Pitches",
        "AVG", "OBP", "SLG", "OPS", "K%", "P/PA", "QAB%"
    ]].copy()

    # Pitching override (only on overlapping trusted box-score fields).
    pit_rows = player_rows[_parse_num_series(player_rows.get("IP", pd.Series(dtype="object"))).fillna(0) > 0].copy()
    pit = pd.DataFrame()
    if not pit_rows.empty:
        pit["pitcher"] = (
            pit_rows["First"].astype(str).str[:1].str.upper() + " " + pit_rows["Last"].astype(str)
        ).str.strip()
        pit["IP"] = _parse_num_series(pit_rows.get("IP", pd.Series(dtype="object")))
        pit["H"] = _parse_num_series(pit_rows.get("H.1", pd.Series(dtype="object")))
        pit["BB"] = _parse_num_series(pit_rows.get("BB.1", pd.Series(dtype="object")))
        pit["K"] = _parse_num_series(pit_rows.get("SO.1", pd.Series(dtype="object")))
        pit["R"] = _parse_num_series(pit_rows.get("R.1", pd.Series(dtype="object")))
        pit["ER"] = _parse_num_series(pit_rows.get("ER", pd.Series(dtype="object")))

    return bat, pit, str(csv_path)


def extract_batter_from_desc(desc_lines):
    skip_prefixes = (
        "lineup changed:",
        "courtesy runner",
        "half-inning ended",
        "unknown player",
    )
    for d in desc_lines:
        dl = d.strip().lower()
        if dl.startswith(skip_prefixes):
            continue
        m = BATTER_EVENT_RE.match(d.strip())
        if m:
            return m.group(1).strip()
    return None


def make_pa_dict(pa_id_counter, game_id, inning_num, inning_half,
                 offense_team, outs, bases, outcome, batter, last_pitcher):
    return {
        "pa_id":        pa_id_counter,
        "game_id":      game_id,
        "inning":       inning_num,
        "inning_half":  inning_half,
        "offense_team": offense_team,
        "outs_before":  outs,
        "risp_flag":    1 if (bases.get(2) or bases.get(3)) else 0,
        "two_out_flag": 1 if outs == 2 else 0,
        "outcome":      outcome,
        "batter":       batter,
        "pitcher":      last_pitcher,
        "pa_flag":      1,
    }


def main():
    base    = pathlib.Path(__file__).resolve().parent
    raw_dir = base / "raw_games"
    files   = sorted(raw_dir.glob("*.txt"))

    if not files:
        print("No .txt files found in raw_games/ — clearing outputs...")
        empty_files = [
            "pitch_events_ALL.csv", "plate_appearances_ALL.csv",
            "pitch_events_EWA.csv", "plate_appearances_EWA.csv",
            "batting_season_EWA.csv", "discipline_season_EWA.csv",
            "pitching_basic_EWA.csv", "fielding_basic_EWA.csv",
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
    ewa_extra_batters = set()
    pa_id_counter    = 1
    pitch_id_counter = 1

    # ── flush_pa ────────────────────────────────────────────────────────────────
    def flush_pa(state):
        nonlocal pa_id_counter, pitch_id_counter

        pa = state.get("pa")
        if not pa:
            return

        tokens     = state.get("tokens", [])
        desc_lines = state.get("desc_lines", [])   # FIX: was "desc" in original
        bases      = state.get("bases", {})
        outs       = state.get("outs", 0)

        balls   = 0
        strikes = 0
        pitch_num = 0

        outcome = pa.get("outcome", "") or ""
        out_l   = str(outcome).lower()

        for tok in tokens:
            pr = normalize_pitch_token(tok)
            if not pr or pr == "other":
                continue

            pitch_num       += 1
            is_first         = 1 if pitch_num == 1 else 0
            is_two_strike    = 1 if strikes == 2 else 0
            balls_before     = balls
            strikes_before   = strikes

            # Advance counts AFTER capturing before-state
            if pr == "ball":
                balls = min(4, balls + 1)
            elif pr in {"called_strike", "swinging_strike"}:
                strikes = min(3, strikes + 1)
            elif pr in {"foul", "foul_tip"} and strikes < 2:
                strikes += 1

            pitch_rows.append({
                "pitch_id":           pitch_id_counter,
                "pa_id":              pa["pa_id"],
                "game_id":            pa["game_id"],
                "inning_half":        pa["inning_half"],
                "inning":             pa["inning"],
                "offense_team":       pa["offense_team"],
                "batter":             pa["batter"],
                "pitcher":            pa.get("pitcher") or state.get("last_pitcher"),
                "pitch_number_in_pa": pitch_num,
                "pitch_result":       pr,
                "balls_before":       balls_before,
                "strikes_before":     strikes_before,
                "is_swing":           1 if is_swing(pr)   else 0,
                "is_whiff":           1 if is_whiff(pr)   else 0,
                "is_contact":         1 if is_contact(pr) else 0,
                "is_first_pitch":     is_first,
                "is_two_strike_pitch": is_two_strike,
            })
            pitch_id_counter += 1

        # Recover missing batter names from true batting-action lines only.
        if not pa.get("batter"):
            recovered = extract_batter_from_desc(desc_lines)
            if recovered:
                pa["batter"] = recovered

        # ── PA-level flags ────────────────────────────────────────────────────
        is_hit          = out_l in {"single", "double", "triple", "home run"}
        hit_type        = outcome if is_hit else None
        is_walk         = out_l in {"walk", "intentional walk", "base on balls", "bb"}
        is_hbp          = ("hit by pitch" in out_l) or (out_l == "hbp")
        is_sac          = ("sac" in out_l) or ("sacrifice" in out_l) or (out_l in {"sf", "sh"})
        is_interference = "interference" in out_l
        is_k            = out_l in {"strikeout", "dropped 3rd strike"}

        ab_flag   = 0 if (is_walk or is_hbp or is_sac or is_interference) else 1
        if out_l == "runner out":
            pa["pa_flag"] = 0
            ab_flag = 0
        hit_flag  = 1 if is_hit   else 0
        walk_flag = 1 if is_walk  else 0
        hbp_flag  = 1 if is_hbp   else 0
        k_flag    = 1 if is_k     else 0

        valid_prs = [normalize_pitch_token(t) for t in tokens]
        valid_prs = [p for p in valid_prs if p and p != "other"]

        swings   = sum(1 for p in valid_prs if is_swing(p))
        whiffs   = sum(1 for p in valid_prs if is_whiff(p))
        contacts = sum(1 for p in valid_prs if is_contact(p))

        first_pitch            = valid_prs[0] if valid_prs else None
        is_fp_swing            = 1 if (first_pitch and is_swing(first_pitch))        else 0
        is_fp_strike           = 1 if (first_pitch and is_strike_result(first_pitch)) else 0
        is_fp_in_play          = 1 if (first_pitch == "in_play")                     else 0

        two_strike_swings = two_strike_whiffs = two_strike_contacts = 0
        s2 = 0
        for tok in tokens:
            pr = normalize_pitch_token(tok)
            if not pr or pr == "other":
                continue
            if s2 >= 2:
                if is_swing(pr):   two_strike_swings  += 1
                if is_whiff(pr):   two_strike_whiffs  += 1
                if is_contact(pr): two_strike_contacts += 1
            if pr == "ball":
                pass   # s2 unchanged
            elif pr in {"called_strike", "swinging_strike"}:
                s2 = min(3, s2 + 1)
            elif pr in {"foul", "foul_tip"} and s2 < 2:
                s2 += 1

        try:
            bases, outs = apply_pa_outcome(pa.get("outcome"), pa.get("batter"), desc_lines, bases, outs)
        except Exception:
            pass

        pitcher_name = pa.get("pitcher")

        for dl in reversed(desc_lines):
            m_pitch = re.search(r'(?:,\s*|\b)(?:pitcher\s+)?([A-Z]\s+[A-Za-z\.\'-]+)\s+pitching', dl)
            if m_pitch:
                pitcher_name = m_pitch.group(1).strip()
                break

        if not pitcher_name:
            pitcher_name = "Unknown"

        two_strike_pa_flag = 1 if two_strike_swings > 0 or two_strike_whiffs > 0 or two_strike_contacts > 0 else 0
        runner_adv_or_score_flag = infer_runner_advance_or_score(desc_lines)
        rbi_flag = infer_rbi_flag(desc_lines)
        batted_ball_pos = infer_batted_ball_position(desc_lines)
        spray_x = np.nan
        spray_y = np.nan
        if batted_ball_pos in SPRAY_COORDS:
            spray_x, spray_y = SPRAY_COORDS[batted_ball_pos]

        pa_rows.append({
            "pa_id":               pa["pa_id"],
            "game_id":             pa["game_id"],
            "inning_half":         pa["inning_half"],
            "inning":              pa["inning"],
            "offense_team":        pa["offense_team"],
            "batter":              pa["batter"],
            "pitcher":             pitcher_name,
            "outcome":             outcome,
            "hit_type":            hit_type,
            "pa_flag":             pa.get("pa_flag", 1),
            "ab_flag":             ab_flag,
            "hit_flag":            hit_flag,
            "walk_flag":           walk_flag,
            "hbp_flag":            hbp_flag,
            "k_flag":              k_flag,
            "pitches_in_pa":       len(valid_prs),
            "swings_in_pa":        swings,
            "whiffs_in_pa":        whiffs,
            "contacts_in_pa":      contacts,
            "is_first_pitch_swing":    is_fp_swing,
            "is_first_pitch_strike":   is_fp_strike,
            "is_first_pitch_in_play":  is_fp_in_play,
            "two_strike_swings":       two_strike_swings,
            "two_strike_whiffs":       two_strike_whiffs,
            "two_strike_contacts":     two_strike_contacts,
            "two_strike_pa_flag":      two_strike_pa_flag,
            "outs_before":         pa["outs_before"],
            "outs_on_pa":          max(0, int(outs) - int(pa["outs_before"])),
            "risp_flag":           pa["risp_flag"],
            "two_out_flag":        pa["two_out_flag"],
            "runner_adv_or_score_flag": runner_adv_or_score_flag,
            "rbi_flag":             rbi_flag,
            "batted_ball_pos":     batted_ball_pos,
            "spray_x":             spray_x,
            "spray_y":             spray_y,
        })

        # Reset for next PA
        state["pa"]         = None
        state["tokens"]     = []
        state["desc_lines"] = []
        state["bases"]      = bases
        state["outs"]       = outs

    # ── Main parsing loop ───────────────────────────────────────────────────────
    for f in files:
        game_id = f.stem
        raw     = f.read_text(errors="ignore")
        lines   = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        inning_half = inning_num = offense_team = None
        bases        = {}
        outs         = 0
        last_pitcher = None
        last_pitcher_ewa = None
        last_pitcher_opp = None

        state = {"pa": None, "tokens": [], "desc_lines": [], "bases": bases, "outs": outs}

        for ln in lines:
            # ── Inning header ──────────────────────────────────────────────
            m_in = INNING_RE.match(ln)
            if m_in:
                flush_pa(state)
                inning_half   = m_in.group(1).capitalize()
                inning_num    = int(m_in.group(2))
                offense_team  = m_in.group(3).strip()
                bases         = {}
                outs          = 0
                state["bases"] = bases
                state["outs"]  = outs
                continue

            # Capture courtesy-runner substitutions for EWA so zero-PA subs can still appear in season table.
            if offense_team and TEAM_MATCH.lower() in offense_team.lower():
                m_cr = COURTESY_RUNNER_RE.match(ln)
                if m_cr:
                    ewa_extra_batters.add(m_cr.group(1).strip())

            # ── PA outcome header (e.g. "Single|...") ─────────────────────
            first = ln.split("|")[0].strip()
            if PA_HEADER_RE.match(first):
                flush_pa(state)
                cur_outs = state.get("outs", outs)
                cur_bases = state.get("bases", bases)
                state["pa"] = make_pa_dict(
                    pa_id_counter, game_id, inning_num, inning_half,
                    offense_team, cur_outs, cur_bases, first, None,
                    last_pitcher_ewa if "East Wake" not in offense_team else last_pitcher_opp
                )
                pa_id_counter += 1
                continue

            toks = PITCH_TOKEN_RE.findall(ln)

            # Accumulate description lines for active PA
            if state["pa"] is not None:
                state["desc_lines"].append(ln)

            # update current pitcher immediately if the line says "X pitching"
            m_pitch_any = re.search(r'([A-Z]\s+[A-Za-z\.\'-]+)\s+pitching\.?', ln)
            if m_pitch_any:
                last_pitcher = m_pitch_any.group(1).strip()
                if offense_team and "East Wake" in offense_team:
                    last_pitcher_opp = last_pitcher
                else:
                    last_pitcher_ewa = last_pitcher
                if state["pa"] is not None:
                    state["pa"]["pitcher"] = last_pitcher

            m_sub = PITCHER_SUB_RE.match(ln)
            if m_sub:
                sub_pitcher = m_sub.group(1).strip()
                if offense_team and "East Wake" in offense_team:
                    last_pitcher_opp = sub_pitcher
                else:
                    last_pitcher_ewa = sub_pitcher
                last_pitcher = sub_pitcher
                if state["pa"] is not None and (not state["pa"].get("pitcher") or state["pa"].get("pitcher") == "Unknown"):
                    state["pa"]["pitcher"] = sub_pitcher

            # ── Sentence-style outcome ("J Fuentes doubles on…") ──────────
            m_sent = SENTENCE_OUTCOME_RE.match(ln)
            if m_sent:
                batter_name = m_sent.group(1).strip()
                verb        = m_sent.group(2).lower()
                verb_map = {
                    "singles":          "Single",
                    "doubles":          "Double",
                    "triples":          "Triple",
                    "homers":           "Home Run",
                    "walks":            "Walk",
                    "strikes out":      "Strikeout",
                    "reaches on error": "Reached on Error",
                    "reached on error": "Reached on Error",
                    "hit by pitch":     "Hit By Pitch",
                }
                outcome = verb_map.get(verb, verb.title())

                if state["pa"] is None:
                    cur_outs = state.get("outs", outs)
                    cur_bases = state.get("bases", bases)
                    state["pa"] = make_pa_dict(
                        pa_id_counter, game_id, inning_num, inning_half,
                        offense_team, cur_outs, cur_bases, outcome, batter_name, last_pitcher
                    )
                    pa_id_counter += 1
                else:
                    state["pa"]["outcome"] = outcome
                    state["pa"]["batter"]  = state["pa"].get("batter") or batter_name
                    state["pa"]["pitcher"] = state["pa"].get("pitcher") or last_pitcher

                if toks:
                    state["tokens"].extend(toks)
                flush_pa(state)
                continue

            # ── Pitch tokens with no active PA → start placeholder ────────
            if toks and state["pa"] is None:
                seed_pitcher = last_pitcher_opp if (offense_team and "East Wake" in offense_team) else last_pitcher_ewa
                cur_outs = state.get("outs", outs)
                cur_bases = state.get("bases", bases)
                state["pa"] = make_pa_dict(
                    pa_id_counter, game_id, inning_num, inning_half,
                    offense_team, cur_outs, cur_bases, "Unknown", None, seed_pitcher
                )
                pa_id_counter += 1

            # ── Accumulate into active PA ─────────────────────────────────
            if state["pa"] is not None:
                if toks:
                    state["tokens"].extend(toks)

                m_bp = BATTER_PITCHER_RE.match(ln)
                if m_bp:
                    state["pa"]["batter"]  = m_bp.group(1).strip()
                    state["pa"]["pitcher"] = m_bp.group(2).strip()
                    last_pitcher           = m_bp.group(2).strip()

                for m in DEF_TO_RE.finditer(ln):
                    def_rows.append({
                        "game_id":     game_id,
                        "inning":      inning_num,
                        "inning_half": inning_half,
                        "offense_team": offense_team,
                        "batter":      state["pa"].get("batter"),
                        "event_type":  "ball_to",
                        "position":    position_code(m.group(1)),
                        "fielder":     m.group(2).strip(),
                        "event_line":  ln,
                    })
                for m in ERR_BY_RE.finditer(ln):
                    def_rows.append({
                        "game_id":     game_id,
                        "inning":      inning_num,
                        "inning_half": inning_half,
                        "offense_team": offense_team,
                        "batter":      state["pa"].get("batter"),
                        "event_type":  "error",
                        "position":    position_code(m.group(1)),
                        "fielder":     m.group(2).strip(),
                        "event_line":  ln,
                    })

        flush_pa(state)

    # ── Build DataFrames ────────────────────────────────────────────────────────
    pitch   = pd.DataFrame(pitch_rows)   if pitch_rows  else pd.DataFrame()
    pa      = pd.DataFrame(pa_rows)      if pa_rows     else pd.DataFrame()
    defense = pd.DataFrame(def_rows)     if def_rows    else pd.DataFrame()

    # ── Team splits ─────────────────────────────────────────────────────────────

    pa_ewa = pd.DataFrame()
    pa_opp = pd.DataFrame()    

    if not pa.empty:
        pa_ewa = pa[pa["offense_team"].str.contains(TEAM_MATCH, case=False, na=False)].copy()
        pa_opp = pa[~pa["offense_team"].str.contains(TEAM_MATCH, case=False, na=False)].copy()
    else:
        pa_ewa  = pd.DataFrame()
        pa_opp  = pd.DataFrame()

    # Infer unknown pitcher rows in the latest inning as the likely final reliever.
    if not pa_opp.empty and "pitcher" in pa_opp.columns:
        pa_opp = pa_opp.sort_values(["game_id", "inning", "inning_half", "pa_id"]).copy()
        pa_opp["pitcher"] = pa_opp["pitcher"].fillna("").astype(str).str.strip()
        team_batters = set(pa_ewa["batter"].dropna().astype(str).str.strip()) if not pa_ewa.empty and "batter" in pa_ewa.columns else set()
        for game_id, grp in pa_opp.groupby("game_id", dropna=False):
            gidx = grp.index
            if grp.empty:
                continue
            max_inning = grp["inning"].max()
            unknown_last = gidx[(grp["inning"] == max_inning) & (grp["pitcher"].str.lower().isin(["", "unknown"]))]
            if len(unknown_last) == 0:
                continue
            # If J Bailey is on the team and not already identified as a pitcher, treat latest-inning unknown rows as his relief inning.
            known_pitchers = set(grp.loc[~grp["pitcher"].str.lower().isin(["", "unknown"]), "pitcher"].tolist())
            fallback = "J Bailey" if ("J Bailey" in team_batters and "J Bailey" not in known_pitchers) else "Unknown"
            pa_opp.loc[unknown_last, "pitcher"] = fallback

    if not pitch.empty and not pa_ewa.empty:
        ewa_ids  = set(pa_ewa["pa_id"])
        opp_ids  = set(pa_opp["pa_id"]) if not pa_opp.empty else set()
        pitch_ewa = pitch[pitch["pa_id"].isin(ewa_ids)].copy()
        pitch_opp = pitch[pitch["pa_id"].isin(opp_ids)].copy()
    else:
        pitch_ewa = pd.DataFrame()
        pitch_opp = pd.DataFrame()

    # ── Batting season ──────────────────────────────────────────────────────────
    if not pa_ewa.empty:
        bat = pa_ewa.copy()
        bat["TB"] = bat["hit_type"].map(
            {"Single": 1, "Double": 2, "Triple": 3, "Home Run": 4}
        ).fillna(0).astype(int)

        bat = bat[bat["batter"].notna() & (bat["batter"] != "") & (bat["batter"] != "Unknown")].copy()

        bat_season = bat.groupby("batter", dropna=True).agg(
            PA      = ("pa_flag",  "sum"),
            AB      = ("ab_flag",  "sum"),
            H       = ("hit_flag", "sum"),
            XBH     = ("hit_type", lambda s: ((s=="Double")|(s=="Triple")|(s=="Home Run")).sum()),
            BB      = ("walk_flag","sum"),
            HBP     = ("hbp_flag", "sum"),
            K       = ("k_flag",   "sum"),
            TB      = ("TB",       "sum"),
            Pitches = ("pitches_in_pa","sum"),
        ).reset_index()

        bat_season["AVG"]  = safe_div(bat_season["H"],  bat_season["AB"])
        bat_season["OBP"]  = safe_div(bat_season["H"] + bat_season["BB"] + bat_season["HBP"], bat_season["PA"])
        bat_season["SLG"]  = safe_div(bat_season["TB"], bat_season["AB"])
        bat_season["OPS"]  = bat_season["OBP"] + bat_season["SLG"]
        bat_season["K%"]   = safe_div(bat_season["K"],       bat_season["PA"])
        bat_season["P/PA"] = safe_div(bat_season["Pitches"], bat_season["PA"])

        outcome_l = bat["outcome"].fillna("").astype(str).str.lower()
        reaches_non_error = (
            (bat["hit_flag"] == 1) |
            (bat["walk_flag"] == 1) |
            (bat["hbp_flag"] == 1) |
            outcome_l.isin(["fielder's choice", "reached on fielder's choice", "dropped 3rd strike", "catcher's interference", "interference"])
        ) & (~outcome_l.str.contains("error", na=False))

        bat["QAB"] = (
            (bat["hit_flag"] == 1) |
            reaches_non_error |
            (bat.get("runner_adv_or_score_flag", 0) == 1) |
            (bat.get("rbi_flag", 0) == 1) |
            ((bat["ab_flag"] == 1) & (bat["pitches_in_pa"] > 7))
        ).astype(int)

        qab = bat.groupby("batter", dropna=True).agg(
            QAB=("QAB","sum"), PA=("pa_flag","sum")
        ).reset_index()
        qab["QAB%"] = safe_div(qab["QAB"], qab["PA"])
        bat_season = bat_season.merge(qab[["batter","QAB%"]], on="batter", how="left")

        # Add courtesy-runner subs (0 PA / 0 AB rows) so team table aligns with GameChanger roster usage.
        missing = sorted([nm for nm in ewa_extra_batters if nm and nm not in set(bat_season["batter"].astype(str))])
        if missing:
            zero_rows = []
            for nm in missing:
                zero_rows.append({
                    "batter": nm,
                    "PA": 0, "AB": 0, "H": 0, "XBH": 0, "BB": 0, "HBP": 0, "K": 0, "TB": 0, "Pitches": 0,
                    "AVG": np.nan, "OBP": np.nan, "SLG": np.nan, "OPS": 0.0, "K%": np.nan, "P/PA": np.nan, "QAB%": np.nan
                })
            bat_season = pd.concat([bat_season, pd.DataFrame(zero_rows)], ignore_index=True, sort=False)
            bat_season = bat_season.sort_values("batter").reset_index(drop=True)
    else:
        bat_season = pd.DataFrame()

    # ── Discipline season ───────────────────────────────────────────────────────
    if not pitch_ewa.empty:
        disc = pitch_ewa.copy()

        # First-pitch sub-frame
        fp_df = disc[disc["is_first_pitch"] == 1].copy()

        fp_agg = fp_df.groupby("batter", dropna=True).agg(
            FirstPitchSwings  = ("is_swing",    "sum"),
            FirstPitches      = ("pitch_id",    "count"),
            FirstPitchStrikes = ("pitch_result", lambda s:
                s.isin(["called_strike","swinging_strike","foul","foul_tip","in_play","strike_other"]).sum()),
            FirstPitchInPlay  = ("pitch_result", lambda s: (s == "in_play").sum()),
        ).reset_index()

        disc_season = disc.groupby("batter", dropna=True).agg(
            Pitches  = ("pitch_id",  "count"),
            Swings   = ("is_swing",  "sum"),
            Whiffs   = ("is_whiff",  "sum"),
            Contacts = ("is_contact","sum"),
        ).reset_index()

        disc_season = disc_season.merge(fp_agg, on="batter", how="left").fillna(0)

        disc_season["Swing%"]           = safe_div(disc_season["Swings"],           disc_season["Pitches"])
        disc_season["Whiff%"]           = safe_div(disc_season["Whiffs"],            disc_season["Swings"])
        disc_season["Contact%"]         = safe_div(disc_season["Contacts"],          disc_season["Swings"])
        disc_season["1stPitchSwing%"]   = safe_div(disc_season["FirstPitchSwings"],  disc_season["FirstPitches"])
        disc_season["1stPitchStrike%"]  = safe_div(disc_season["FirstPitchStrikes"], disc_season["FirstPitches"])
        disc_season["1stPitchInPlay%"]  = safe_div(disc_season["FirstPitchInPlay"],  disc_season["FirstPitches"])

        # 2-strike splits
        two = disc[disc["is_two_strike_pitch"] == 1].copy()
        if not two.empty:
            two_agg = two.groupby("batter", dropna=True).agg(
                TwoK_Pitches  = ("pitch_id",  "count"),
                TwoK_Swings   = ("is_swing",  "sum"),
                TwoK_Whiffs   = ("is_whiff",  "sum"),
                TwoK_Contacts = ("is_contact","sum"),
            ).reset_index()
            two_agg["2K_Swing%"]   = safe_div(two_agg["TwoK_Swings"],   two_agg["TwoK_Pitches"])
            two_agg["2K_Whiff%"]   = safe_div(two_agg["TwoK_Whiffs"],   two_agg["TwoK_Swings"])
            two_agg["2K_Contact%"] = safe_div(two_agg["TwoK_Contacts"], two_agg["TwoK_Swings"])
            disc_season = disc_season.merge(
                two_agg[["batter","2K_Swing%","2K_Whiff%","2K_Contact%"]], on="batter", how="left"
            )
        else:
            disc_season["2K_Swing%"]   = np.nan
            disc_season["2K_Whiff%"]   = np.nan
            disc_season["2K_Contact%"] = np.nan
    else:
        disc_season = pd.DataFrame()

    # ── Pitching basic ──────────────────────────────────────────────────────────

    # ── Pitching basic ──────────────────────────────────────────────────────────

    pit = pa_opp.copy()
    pit = pit.drop_duplicates(subset=["pa_id"])
    if "pa_flag" in pit.columns:
        pit = pit[pit["pa_flag"] == 1].copy()

    pit["pitcher"] = pit["pitcher"].fillna("").astype(str).str.strip()
    pit = pit[pit["pitcher"] != ""]

    if "outs_on_pa" not in pit.columns:
        pit["outs_on_pa"] = 0

    pit_basic = pit.groupby("pitcher", dropna=True).agg(
        BF=("pa_id","nunique"),
        AB=("ab_flag","sum"),
        H=("hit_flag","sum"),
        BB=("walk_flag","sum"),
        HBP=("hbp_flag","sum"),
        K=("k_flag","sum"),
        HR=("hit_type", lambda s: (s=="Home Run").sum()),
        XBH=("hit_type", lambda s: ((s=="Double") | (s=="Triple") | (s=="Home Run")).sum()),
        FP_STRIKE=("is_first_pitch_strike","sum"),
        OUTS=("outs_on_pa", "sum"),
        SWINGS=("swings_in_pa","sum"),
        WHIFFS=("whiffs_in_pa","sum"),
        PITCHES=("pitches_in_pa","sum"),
    ).reset_index()

    pit_basic["Whiff%"] = pit_basic["WHIFFS"] / pit_basic["SWINGS"].replace(0,1)

    pit_basic["CSW%"] = (pit_basic["WHIFFS"] + pit_basic["K"]) / pit_basic["PITCHES"].replace(0,1)

    pit_basic["IP"] = pit_basic["OUTS"] / 3

    pit_basic["WHIP"] = (pit_basic["BB"] + pit_basic["H"]) / pit_basic["IP"].replace(0,1)

    pit_basic["K%"] = pit_basic["K"] / pit_basic["BF"]

    pit_basic["BB%"] = pit_basic["BB"] / pit_basic["BF"]

    pit_basic["K-BB%"] = pit_basic["K%"] - pit_basic["BB%"]

    pit_basic["H%"] = pit_basic["H"] / pit_basic["BF"]

    pit_basic["F-Strike%"] = pit_basic["FP_STRIKE"] / pit_basic["BF"]

    pit_basic["Opp BA"] = pit_basic["H"] / pit_basic["AB"].replace(0,1)

    pit_basic["ER"] = pit_basic["HR"]

    pit_basic = pit_basic[
    [
    "pitcher",
    "IP",
    "BF",
    "AB",
    "H",
    "BB",
    "HBP",
    "K",
    "HR",
    "WHIP",
    "Opp BA",
    "F-Strike%",
    "K%",
    "BB%",
    "K-BB%",
    "Whiff%",
    "CSW%"
    ]
    ]
    pit_basic = round_df(pit_basic)

    # Optional GameChanger CSV override for box-score batting/pitching (discipline still from pitch events).
    gc_bat, gc_pit, gc_path = load_gamechanger_overrides(base)
    if not gc_bat.empty:
        bat_season = gc_bat.copy()
        # Preserve custom QAB definition from parsed PA events, even when box-score overrides batting totals.
        if not pa_ewa.empty and "batter" in pa_ewa.columns:
            qdf = pa_ewa.copy()
            q_out = qdf["outcome"].fillna("").astype(str).str.lower()
            q_reach_non_error = (
                (qdf.get("hit_flag", 0) == 1) |
                (qdf.get("walk_flag", 0) == 1) |
                (qdf.get("hbp_flag", 0) == 1) |
                q_out.isin(["fielder's choice", "reached on fielder's choice", "dropped 3rd strike", "catcher's interference", "interference"])
            ) & (~q_out.str.contains("error", na=False))
            qdf["QAB"] = (
                (qdf.get("hit_flag", 0) == 1) |
                q_reach_non_error |
                (qdf.get("runner_adv_or_score_flag", 0) == 1) |
                (qdf.get("rbi_flag", 0) == 1) |
                ((qdf.get("ab_flag", 0) == 1) & (qdf.get("pitches_in_pa", 0) > 7))
            ).astype(int)
            qmerge = qdf.groupby("batter", dropna=True).agg(
                qab=("QAB", "sum"),
                pa=("pa_flag", "sum")
            ).reset_index()
            qmerge["QAB%"] = safe_div(qmerge["qab"], qmerge["pa"])
            bat_season = bat_season.drop(columns=[c for c in ["QAB%"] if c in bat_season.columns], errors="ignore")
            bat_season = bat_season.merge(qmerge[["batter", "QAB%"]], on="batter", how="left")

    if not gc_pit.empty:
        if pit_basic.empty:
            pit_basic = gc_pit.copy()
        else:
            base_cols = pit_basic.columns.tolist()
            for col in gc_pit.columns:
                if col not in base_cols:
                    pit_basic[col] = np.nan
            pit_basic = pit_basic.set_index("pitcher", drop=False)
            for _, r in gc_pit.iterrows():
                name = str(r["pitcher"]).strip()
                if not name:
                    continue
                if name not in pit_basic.index:
                    new_row = {c: np.nan for c in pit_basic.columns}
                    new_row["pitcher"] = name
                    pit_basic.loc[name] = new_row
                for c in ["IP", "H", "BB", "K", "R", "ER"]:
                    if c in gc_pit.columns and c in pit_basic.columns:
                        pit_basic.at[name, c] = r.get(c, np.nan)
            pit_basic = pit_basic.reset_index(drop=True)
        pit_basic = round_df(pit_basic)

    # ── Fielding basic ──────────────────────────────────────────────────────────
    if not defense.empty:
        field_sum = defense.groupby(["position","fielder"], dropna=True).agg(
            Chances = ("event_type", "count"),
            Errors  = ("event_type", lambda s: (s=="error").sum()),
        ).reset_index()
        field_sum["Fld%"] = safe_div(
            field_sum["Chances"] - field_sum["Errors"], field_sum["Chances"]
        )
    else:
        field_sum = pd.DataFrame()

    # ── Count splits ────────────────────────────────────────────────────────────
    if not pitch_ewa.empty and not pa_ewa.empty:
        last_pitch = (
            pitch_ewa
            .sort_values(["pa_id","pitch_number_in_pa"])
            .groupby("pa_id")
            .tail(1)[["pa_id","balls_before","strikes_before"]]
            .copy()
        )
        last_pitch["finish_count"] = (
            last_pitch["balls_before"].astype(str) + "-" +
            last_pitch["strikes_before"].astype(str)
        )

        bat2 = pa_ewa.merge(last_pitch[["pa_id","finish_count"]], on="pa_id", how="left")

        def bucket(c):
            if pd.isna(c):                               return "Unknown"
            if c == "0-0":                               return "0-0"
            if c in {"1-0","2-0","2-1","3-1","3-0"}:    return "Hitter"
            if c in {"0-1","0-2","1-2","2-2"}:           return "Pitcher"
            if str(c).endswith("-2"):                    return "2K"
            return "Other"

        bat2["count_bucket"] = bat2["finish_count"].apply(bucket)

        count_split = bat2.groupby(["batter","count_bucket"], dropna=True).agg(
            PA = ("pa_flag",  "sum"),
            AB = ("ab_flag",  "sum"),
            H  = ("hit_flag", "sum"),
            K  = ("k_flag",   "sum"),
        ).reset_index()
        count_split["AVG"] = safe_div(count_split["H"], count_split["AB"])
        count_split["K%"]  = safe_div(count_split["K"], count_split["PA"])
    else:
        count_split = pd.DataFrame()

    # ── Save outputs ────────────────────────────────────────────────────────────

    pitch.to_csv(base / "pitch_events_ALL.csv",         index=False, float_format="%.3f")
    pa.to_csv(base / "plate_appearances_ALL.csv",       index=False, float_format="%.3f")
    pitch_ewa.to_csv(base / "pitch_events_EWA.csv",     index=False, float_format="%.3f")
    pa_ewa.to_csv(base / "plate_appearances_EWA.csv",   index=False, float_format="%.3f")
    bat_season.to_csv(base / "batting_season_EWA.csv",  index=False, float_format="%.3f")
    disc_season.to_csv(base / "discipline_season_EWA.csv", index=False, float_format="%.3f")
    pit_basic.to_csv(base / "pitching_basic_EWA.csv",   index=False, float_format="%.3f")
    field_sum.to_csv(base / "fielding_basic_EWA.csv",   index=False, float_format="%.3f")
    count_split.to_csv(base / "batting_count_splits_EWA.csv", index=False, float_format="%.3f")    

    msg = f"Build complete ✅  ({len(pa)} total PAs | {len(pa_ewa)} EWA PAs | {len(pitch)} pitches)"
    if gc_path:
        msg += f" | GC CSV: {pathlib.Path(gc_path).name}"
    print(msg)


if __name__ == "__main__":
    main()
