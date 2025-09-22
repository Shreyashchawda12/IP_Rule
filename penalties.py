import pandas as pd
import numpy as np

# ======================
# Helpers (shared)
# ======================
def parse_excel_or_datetime(s: pd.Series) -> pd.Series:
    """
    Parse timestamps that may be normal strings or Excel/ODS/Excel-serials.
    Returns pandas datetime (NaT for invalid).
    """
    s = s.copy()
    nums = pd.to_numeric(s, errors="coerce")
    if nums.notna().any() and nums.dropna().median() > 10000:
        return pd.to_datetime(nums, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


# ======================
# Rule 1 + Rule 2
# ======================
def apply_penalties_two_rules_with_reason(
    df: pd.DataFrame,
    vf_col: str = "VF ID",
    start_col: str = "APPALARMTIME",
    end_col: str   = "APPCANCELTIME",
    dur_col: str   = "OUTAGEDURATION",
    penalty_col: str = "Penalty",
    reason_col: str  = "PenaltyRule",
    end_window_min: float = 2.0,   # end near 23:59 (<=2 min)
    start_window_min: float = 2.0, # start near 00:00 (<=2 min)
    max_gap_min: float = 0.0,      # STRICT: no gap
    threshold: float = 1440.0,
) -> pd.DataFrame:
    """
    Rule 1: If any outage ≈1440 (1439–1441) for a site → Penalty=1440 (exclude site from Rule 2).
    Rule 2: For remaining sites, find midnight pairs (last on D + first on D+1) contiguous to midnight
            with pair_sum ≥ threshold. Set Penalty = pair_sum (rounded).
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # timestamps & duration
    out["_start_ts"] = parse_excel_or_datetime(out[start_col])
    out["_end_ts"]   = parse_excel_or_datetime(out[end_col])
    out["_dur_min"]  = (out["_end_ts"] - out["_start_ts"]).dt.total_seconds() / 60.0
    out[dur_col]     = out["_dur_min"]

    # init outputs
    out[penalty_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out[reason_col]  = pd.Series(pd.NA, index=out.index, dtype="string")

    # drop unusable
    out = out.dropna(subset=[vf_col, "_start_ts", "_end_ts", dur_col])

    # --- Rule 1 (tolerance around 1440) ---
    dur_min = out[dur_col]
    sites_rule1 = out.loc[dur_min.between(1439, 1441), vf_col].drop_duplicates()

    if not sites_rule1.empty:
        m1 = out[vf_col].isin(sites_rule1)
        out.loc[m1, penalty_col] = 1440
        out.loc[m1, reason_col]  = "Rule1"

    # --- Rule 2 ---
    remaining = out.loc[~out[vf_col].isin(sites_rule1), [vf_col, "_start_ts", "_end_ts", dur_col]].drop_duplicates()
    if not remaining.empty:
        remaining["_day"] = remaining["_start_ts"].dt.date
        remaining = remaining.sort_values([vf_col, "_day", "_start_ts", "_end_ts"])

        last_on_day = (
            remaining.loc[remaining.groupby([vf_col, "_day"])["_end_ts"].idxmax()]
            [[vf_col, "_day", "_end_ts", dur_col]]
            .rename(columns={"_end_ts": "end_ts_D", dur_col: "dur_D"})
        )
        first_on_day = (
            remaining.loc[remaining.groupby([vf_col, "_day"])["_start_ts"].idxmin()]
            [[vf_col, "_day", "_start_ts", dur_col]]
            .rename(columns={"_start_ts": "start_ts_D1", dur_col: "dur_D1"})
        )

        pairs = last_on_day.merge(first_on_day.rename(columns={"_day": "_day_plus1"}), on=[vf_col], how="inner")
        if not pairs.empty:
            day_gap = (pd.to_datetime(pairs["_day_plus1"]) - pd.to_datetime(pairs["_day"])).dt.days
            pairs = pairs.loc[day_gap.eq(1)].copy()

            if not pairs.empty:
                pairs["end_tod_min"]   = pairs["end_ts_D"].dt.hour*60 + pairs["end_ts_D"].dt.minute + pairs["end_ts_D"].dt.second/60.0
                pairs["start_tod_min"] = pairs["start_ts_D1"].dt.hour*60 + pairs["start_ts_D1"].dt.minute + pairs["start_ts_D1"].dt.second/60.0
                gap_min = (pairs["start_ts_D1"] - pairs["end_ts_D"]).dt.total_seconds() / 60.0

                end_close   = pairs["end_tod_min"]   >= (1440 - end_window_min)
                start_close = pairs["start_tod_min"] <=  start_window_min
                small_gap   = gap_min <= max_gap_min

                pairs = pairs.loc[end_close & start_close & small_gap].copy()
                if not pairs.empty:
                    pairs["pair_sum"] = (pairs["dur_D"] + pairs["dur_D1"])
                    pairs = pairs.loc[pairs["pair_sum"] >= threshold].copy()

                    if not pairs.empty:
                        pairs["pair_sum"] = pairs["pair_sum"].round().astype("Int64")

                        # map to both days
                        left  = pairs[[vf_col, "_day", "pair_sum"]].rename(columns={"_day": "PenDay"})
                        right = pairs[[vf_col, "_day_plus1", "pair_sum"]].rename(columns={"_day_plus1": "PenDay"})
                        site_day_pen = pd.concat([left, right], ignore_index=True)
                        site_day_pen = (
                            site_day_pen.groupby([vf_col, "PenDay"], as_index=False)["pair_sum"]
                                        .max()
                                        .rename(columns={"pair_sum": "Penalty_val"})
                        )

                        out["_day"] = out["_start_ts"].dt.date
                        out = out.merge(site_day_pen, left_on=[vf_col, "_day"], right_on=[vf_col, "PenDay"], how="left")

                        fill_mask = out[penalty_col].isna() & out["Penalty_val"].notna()
                        out.loc[fill_mask, penalty_col] = out.loc[fill_mask, "Penalty_val"].astype("Int64")
                        out.loc[fill_mask, reason_col]  = "Rule2"
                        out.drop(columns=["PenDay","Penalty_val"], inplace=True, errors="ignore")

    # cleanup
    out.drop(columns=["_start_ts","_end_ts","_dur_min","_day"], inplace=True, errors="ignore")
    out[penalty_col] = out[penalty_col].astype("Int64")
    out[reason_col]  = out[reason_col].astype("string")
    return out


# ======================
# Rule 3 (Non DHQ, still blank)
# ======================
def apply_rule3_non_dhq(
    df_out: pd.DataFrame,
    vf_col: str = "VF ID",
    dhq_col: str = "DHQ/Non DHQ",
    start_col: str = "APPALARMTIME",
    end_col: str   = "APPCANCELTIME",
    dur_col: str   = "OUTAGEDURATION",
    penalty_col: str = "Penalty",
    reason_col: str  = "PenaltyRule",
    end_window_min: float = 2.0,
    start_window_min: float = 2.0,
    max_gap_min: float = 0.0,   # STRICT: no gap
) -> pd.DataFrame:
    """
    Only for rows where Penalty is blank & DHQ/Non DHQ == 'Non DHQ'.
    Penalty = max( max single outage in [900,1440), strict midnight pair-sum in [900,1440) ).
    """
    out = df_out.copy()
    out.columns = [str(c).strip() for c in out.columns]

    start_ts = parse_excel_or_datetime(out[start_col])
    end_ts   = parse_excel_or_datetime(out[end_col])
    out[dur_col] = (end_ts - start_ts).dt.total_seconds() / 60.0

    non_dhq_mask = out[dhq_col].astype(str).str.strip().str.lower().eq("non dhq")
    r3_mask = out[penalty_col].isna() & non_dhq_mask
    if not r3_mask.any():
        return out

    slim = out.loc[r3_mask, [vf_col, start_col, end_col, dur_col]].drop_duplicates()
    slim[start_col] = parse_excel_or_datetime(slim[start_col])
    slim[end_col]   = parse_excel_or_datetime(slim[end_col])
    slim[dur_col]   = pd.to_numeric(slim[dur_col], errors="coerce")

    # A) max single in [900,1440) with tolerance
    single_mask = (slim[dur_col] >= 900) & (slim[dur_col] < 1441)
    max_single = (
        slim.loc[single_mask]
            .groupby(vf_col)[dur_col]
            .max()
            .round()
            .astype("Int64")
            .rename("R3_single_max")
    )

    # B) strict midnight pair-sum
    slim["_day"] = slim[start_col].dt.date
    slim = slim.sort_values([vf_col, "_day", start_col, end_col])

    last_on_day = (
        slim.loc[slim.groupby([vf_col, "_day"])[end_col].idxmax()]
        [[vf_col, "_day", end_col, dur_col]]
        .rename(columns={end_col: "end_ts_D", dur_col: "dur_D"})
    )
    first_on_day = (
        slim.loc[slim.groupby([vf_col, "_day"])[start_col].idxmin()]
        [[vf_col, "_day", start_col, dur_col]]
        .rename(columns={start_col: "start_ts_D1", dur_col: "dur_D1"})
    )

    pairs = last_on_day.merge(first_on_day.rename(columns={"_day": "_day_plus1"}), on=[vf_col], how="inner")

    if not pairs.empty:
        day_gap = (pd.to_datetime(pairs["_day_plus1"]) - pd.to_datetime(pairs["_day"])).dt.days
        pairs = pairs.loc[day_gap.eq(1)].copy()

        if not pairs.empty:
            pairs["end_tod_min"]   = pairs["end_ts_D"].dt.hour*60 + pairs["end_ts_D"].dt.minute + pairs["end_ts_D"].dt.second/60.0
            pairs["start_tod_min"] = pairs["start_ts_D1"].dt.hour*60 + pairs["start_ts_D1"].dt.minute + pairs["start_ts_D1"].dt.second/60.0
            gap_min = (pairs["start_ts_D1"] - pairs["end_ts_D"]).dt.total_seconds() / 60.0

            end_close   = pairs["end_tod_min"]   >= (1440 - end_window_min)
            start_close = pairs["start_tod_min"] <=  start_window_min
            small_gap   = gap_min <= max_gap_min

            pairs = pairs.loc[end_close & start_close & small_gap].copy()
            if not pairs.empty:
                r3_pair = (
                    (pairs["dur_D"] + pairs["dur_D1"]).round()
                    .groupby(pairs[vf_col]).max()
                    .astype("Int64")
                    .rename("R3_pair_sum")
                )
            else:
                r3_pair = pd.Series(dtype="Int64", name="R3_pair_sum")
        else:
            r3_pair = pd.Series(dtype="Int64", name="R3_pair_sum")
    else:
        r3_pair = pd.Series(dtype="Int64", name="R3_pair_sum")

    r3 = pd.concat([max_single, r3_pair], axis=1)

    # enforce tolerance [900,1440)
    if "R3_pair_sum" in r3.columns:
        r3.loc[(r3["R3_pair_sum"] < 900) | (r3["R3_pair_sum"] >= 1441), "R3_pair_sum"] = pd.NA
    if "R3_single_max" in r3.columns:
        r3.loc[(r3["R3_single_max"] < 900) | (r3["R3_single_max"] >= 1441), "R3_single_max"] = pd.NA

    r3["Penalty_R3"] = r3[["R3_single_max", "R3_pair_sum"]].max(axis=1)
    r3["Penalty_R3"] = r3["Penalty_R3"].astype("Int64")

    # merge back
    r3.index.name = vf_col
    r3_reset = r3.reset_index()
    if vf_col not in r3_reset.columns:
        r3_reset = r3_reset.rename(columns={"index": vf_col})

    out = out.merge(r3_reset[[vf_col, "Penalty_R3"]], on=vf_col, how="left")
    fill_mask = out[penalty_col].isna() & out["Penalty_R3"].notna()
    out.loc[fill_mask, penalty_col] = out.loc[fill_mask, "Penalty_R3"]
    out.loc[fill_mask, reason_col]  = "Rule3"

    out.drop(columns=["Penalty_R3"], inplace=True)
    return out


# ======================
# Rule 4 (DHQ, still blank; 420..1440 + midnight pair)
# ======================
def apply_rule4_dhq(
    df_out: pd.DataFrame,
    vf_col: str = "VF ID",
    dhq_col: str = "DHQ/Non DHQ",
    start_col: str = "APPALARMTIME",
    end_col: str   = "APPCANCELTIME",
    dur_col: str   = "OUTAGEDURATION",
    penalty_col: str = "Penalty",
    reason_col: str  = "PenaltyRule",
    end_window_min: float = 2.0,
    start_window_min: float = 2.0,
    max_gap_min: float = 0.0,   # STRICT: no gap
) -> pd.DataFrame:
    """
    Only for rows where Penalty is blank & DHQ/Non DHQ == 'DHQ'.
    Penalty = max( max single outage in [420,1440), strict midnight pair-sum in [420,1440) ).
    """
    out = df_out.copy()
    out.columns = [str(c).strip() for c in out.columns]

    start_ts = parse_excel_or_datetime(out[start_col])
    end_ts   = parse_excel_or_datetime(out[end_col])
    out[dur_col] = (end_ts - start_ts).dt.total_seconds() / 60.0

    dhq_mask = out[dhq_col].astype(str).str.strip().str.lower().eq("dhq")
    r4_mask = out[penalty_col].isna() & dhq_mask
    if not r4_mask.any():
        return out

    slim = out.loc[r4_mask, [vf_col, start_col, end_col, dur_col]].drop_duplicates()
    slim[start_col] = parse_excel_or_datetime(slim[start_col])
    slim[end_col]   = parse_excel_or_datetime(slim[end_col])
    slim[dur_col]   = pd.to_numeric(slim[dur_col], errors="coerce")

    # A) max single in [420,1440) with tolerance
    single_mask = (slim[dur_col] >= 420) & (slim[dur_col] < 1441)
    max_single = (
        slim.loc[single_mask]
            .groupby(vf_col)[dur_col]
            .max()
            .round()
            .astype("Int64")
            .rename("R4_single_max")
    )

    # B) strict midnight pair-sum
    slim["_day"] = slim[start_col].dt.date
    slim = slim.sort_values([vf_col, "_day", start_col, end_col])

    last_on_day = (
        slim.loc[slim.groupby([vf_col, "_day"])[end_col].idxmax()]
        [[vf_col, "_day", end_col, dur_col]]
        .rename(columns={end_col: "end_ts_D", dur_col: "dur_D"})
    )
    first_on_day = (
        slim.loc[slim.groupby([vf_col, "_day"])[start_col].idxmin()]
        [[vf_col, "_day", start_col, dur_col]]
        .rename(columns={start_col: "start_ts_D1", dur_col: "dur_D1"})
    )

    pairs = last_on_day.merge(first_on_day.rename(columns={"_day": "_day_plus1"}), on=[vf_col], how="inner")

    if not pairs.empty:
        day_gap = (pd.to_datetime(pairs["_day_plus1"]) - pd.to_datetime(pairs["_day"])).dt.days
        pairs = pairs.loc[day_gap.eq(1)].copy()

        if not pairs.empty:
            pairs["end_tod_min"]   = pairs["end_ts_D"].dt.hour*60 + pairs["end_ts_D"].dt.minute + pairs["end_ts_D"].dt.second/60.0
            pairs["start_tod_min"] = pairs["start_ts_D1"].dt.hour*60 + pairs["start_ts_D1"].dt.minute + pairs["start_ts_D1"].dt.second/60.0
            gap_min = (pairs["start_ts_D1"] - pairs["end_ts_D"]).dt.total_seconds() / 60.0

            end_close   = pairs["end_tod_min"]   >= (1440 - end_window_min)
            start_close = pairs["start_tod_min"] <=  start_window_min
            small_gap   = gap_min <= max_gap_min

            pairs = pairs.loc[end_close & start_close & small_gap].copy()
            if not pairs.empty:
                r4_pair = (
                    (pairs["dur_D"] + pairs["dur_D1"]).round()
                    .groupby(pairs[vf_col]).max()
                    .astype("Int64")
                    .rename("R4_pair_sum")
                )
            else:
                r4_pair = pd.Series(dtype="Int64", name="R4_pair_sum")
        else:
            r4_pair = pd.Series(dtype="Int64", name="R4_pair_sum")
    else:
        r4_pair = pd.Series(dtype="Int64", name="R4_pair_sum")

    r4 = pd.concat([max_single, r4_pair], axis=1)

    # enforce tolerance [420,1440)
    if "R4_pair_sum" in r4.columns:
        r4.loc[(r4["R4_pair_sum"] < 420) | (r4["R4_pair_sum"] >= 1441), "R4_pair_sum"] = pd.NA
    if "R4_single_max" in r4.columns:
        r4.loc[(r4["R4_single_max"] < 420) | (r4["R4_single_max"] >= 1441), "R4_single_max"] = pd.NA

    r4["Penalty_R4"] = r4[["R4_single_max", "R4_pair_sum"]].max(axis=1)
    r4["Penalty_R4"] = r4["Penalty_R4"].astype("Int64")

    # merge back
    r4.index.name = vf_col
    r4_reset = r4.reset_index()
    if vf_col not in r4_reset.columns:
        r4_reset = r4_reset.rename(columns={"index": vf_col})

    out = out.merge(r4_reset[[vf_col, "Penalty_R4"]], on=vf_col, how="left")
    fill_mask = out[penalty_col].isna() & out["Penalty_R4"].notna()
    out.loc[fill_mask, penalty_col] = out.loc[fill_mask, "Penalty_R4"]
    out.loc[fill_mask, reason_col]  = "Rule4"

    out.drop(columns=["Penalty_R4"], inplace=True)
    return out
