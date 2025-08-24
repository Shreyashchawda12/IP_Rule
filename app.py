from typing import List
import streamlit as st
import pandas as pd

from io_utils import read_any_table, normalize_headers, to_excel_bytes
from penalties import (
    apply_penalties_two_rules_with_reason,
    apply_rule3_non_dhq,
    apply_rule4_dhq,
)

st.set_page_config(page_title="Penalty Builder (Rules 1–4)", layout="wide")
st.title("Penalty Builder — Rules 1 to 4")

# ---- Fixed column names (no optional renaming) ----
VF_COL       = "VF ID"
DHQ_COL      = "DHQ/Non DHQ"
START_COL    = "APPALARMTIME"
END_COL      = "APPCANCELTIME"
DUR_COL      = "OUTAGEDURATION"
PENALTY_COL  = "Penalty"
REASON_COL   = "PenaltyRule"

# ---- Midnight settings (strict) ----
END_WINDOW_MIN   = 2.0   # end must be within 2 min of 23:59
START_WINDOW_MIN = 2.0   # start must be within 2 min of 00:00
MAX_GAP_MIN      = 0.0   # absolutely no gap allowed

# ---- Upload ----
uploaded = st.file_uploader(
    "Upload data file (.xlsx, .xls, .ods, .xlsb, .csv)",
    type=["xlsx", "xls", "ods", "xlsb", "csv"],
)

REQUIRED_COLS = [VF_COL, DHQ_COL, START_COL, END_COL]

def validate_columns(df: pd.DataFrame) -> List[str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    # Rule 1 + Rule 2
    df12 = apply_penalties_two_rules_with_reason(
        df,
        vf_col=VF_COL,
        start_col=START_COL,
        end_col=END_COL,
        dur_col=DUR_COL,
        penalty_col=PENALTY_COL,
        reason_col=REASON_COL,
        end_window_min=END_WINDOW_MIN,
        start_window_min=START_WINDOW_MIN,
        max_gap_min=MAX_GAP_MIN,
        threshold=1440.0,
    )
    # Rule 3 (Non DHQ)
    df123 = apply_rule3_non_dhq(
        df12,
        vf_col=VF_COL,
        dhq_col=DHQ_COL,
        start_col=START_COL,
        end_col=END_COL,
        dur_col=DUR_COL,
        penalty_col=PENALTY_COL,
        reason_col=REASON_COL,
        end_window_min=END_WINDOW_MIN,
        start_window_min=START_WINDOW_MIN,
        max_gap_min=MAX_GAP_MIN,
    )
    # Rule 4 (DHQ)
    df_final = apply_rule4_dhq(
        df123,
        vf_col=VF_COL,
        dhq_col=DHQ_COL,
        start_col=START_COL,
        end_col=END_COL,
        dur_col=DUR_COL,
        penalty_col=PENALTY_COL,
        reason_col=REASON_COL,
        end_window_min=END_WINDOW_MIN,
        start_window_min=START_WINDOW_MIN,
        max_gap_min=MAX_GAP_MIN,
    )
    return df_final

if uploaded is None:
    st.info("⬆️ Upload a file to begin.")
    st.stop()

# Read & preview
try:
    df = read_any_table(uploaded)
except Exception as e:
    st.error("Could not read the uploaded file.")
    st.exception(e)
    st.stop()

df = normalize_headers(df)

st.subheader("Preview (first 200 rows)")
st.dataframe(df.head(200), use_container_width=True)
st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

# Validate columns before enabling run
missing = validate_columns(df)
if missing:
    st.error(f"Missing required columns: {missing}\n\n"
             f"Your file must contain: {REQUIRED_COLS}")
    st.stop()

# The button that executes the pipeline
run_clicked = st.button("Run Rules 1–4")

if run_clicked:
    try:
        with st.spinner("Applying Rules 1–4..."):
            df_final = run_pipeline(df)

        st.success("Done! Showing results.")
        st.subheader("Penalties applied (sample)")
        show_cols = [c for c in [VF_COL, DHQ_COL, START_COL, END_COL, DUR_COL, PENALTY_COL, REASON_COL] if c in df_final.columns]
        st.dataframe(df_final.loc[df_final[PENALTY_COL].notna(), show_cols].head(200), use_container_width=True)

        st.subheader("Download results")
        xbytes = to_excel_bytes(df_final, sheet_name="Penalties")
        st.download_button(
            "📥 Download Excel (.xlsx)",
            data=xbytes,
            file_name="penalties_rules_1_to_4.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        csv_bytes = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV (.csv)",
            data=csv_bytes,
            file_name="penalties_rules_1_to_4.csv",
            mime="text/csv",
        )

        st.subheader("Result (first 200 rows)")
        st.dataframe(df_final.head(200), use_container_width=True)

    except Exception as e:
        st.error("An error occurred while running the rules.")
        st.exception(e)
