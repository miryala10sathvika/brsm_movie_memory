"""
Shared data loading for BRSM hypothesis plots.
Import and call load_data() to get the trials DataFrame and per-participant aggregates.
"""

import os, re, glob, warnings, ast
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def parse_rt(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0: return float(parsed[0])
        return float(parsed)
    except: pass
    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else np.nan

def load_data():
    """Returns (trials, pp, pp_tt) DataFrames."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "BRSM data csv")
    all_trials = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        fname = os.path.basename(f)
        if "sub42_nb" in fname.lower(): continue
        try: df = pd.read_csv(f, on_bad_lines="skip")
        except: continue
        if len(df) < 5: continue
        p = df["participant"].dropna().iloc[0] if "participant" in df.columns else ""
        p_str = str(p)
        p_upper, fname_upper = p_str.upper(), fname.upper()
        if "_AB" in p_upper or "_AB" in fname_upper: cond = "AB"
        elif "_NB" in p_upper or "_NB" in fname_upper: cond = "NB"
        else: continue
        sid = None
        for src in [p_str, fname]:
            m = re.search(r"[Ss][Uu]*[Bb][Hh]?(\d+)", src)
            if m: sid = int(m.group(1)); break
        if sid is None: continue
        r = df[df["movie_id"].notna() & df["resp.corr"].notna()].copy()
        if r.empty: continue
        r["condition"], r["sub_id"] = cond, sid
        r["target_type"] = r["target_img"].apply(
            lambda s: "BB" if "_BB_" in str(s) else ("EM" if "_EM_" in str(s) else np.nan))
        r["acc"] = pd.to_numeric(r["resp.corr"], errors="coerce")
        r["rt"] = r["resp.rt"].apply(parse_rt)
        r["conf"] = pd.to_numeric(r["conf_radio.response"], errors="coerce")
        r["trial_n"] = pd.to_numeric(r["recogloop.thisN"], errors="coerce")
        r["movie_id_int"] = pd.to_numeric(r["movie_id"], errors="coerce").astype("Int64")
        def parse_key(v):
            s = str(v).strip()
            if "'l'" in s: return "left"
            if "'r'" in s: return "right"
            return np.nan
        r["resp_side"] = r["resp.keys"].apply(parse_key)
        r["_file"] = fname
        all_trials.append(r)
    trials = pd.concat(all_trials, ignore_index=True)
    first_file = trials.groupby(["sub_id","condition"])["_file"].first().reset_index()
    ff_set = set(zip(first_file["sub_id"], first_file["condition"], first_file["_file"]))
    trials = trials[trials.apply(lambda r: (r["sub_id"], r["condition"], r["_file"]) in ff_set, axis=1)].drop(columns=["_file"])
    pp = trials.groupby(["sub_id","condition"]).agg(
        acc=("acc","mean"), rt=("rt","mean"), conf=("conf","mean")).reset_index()
    pp_tt = trials.dropna(subset=["target_type"]).groupby(
        ["sub_id","condition","target_type"]).agg(
        acc=("acc","mean"), rt=("rt","mean"), conf=("conf","mean")).reset_index()
    return trials, pp, pp_tt
