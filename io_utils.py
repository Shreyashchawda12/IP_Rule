import io
import pandas as pd

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def read_any_table(uploaded_file) -> pd.DataFrame:
    """Read CSV/XLSX/XLS/ODS/XLSB with safe fallbacks for encodings and engines."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")

    if name.endswith(".ods"):
        return pd.read_excel(uploaded_file, engine="odf")

    if name.endswith(".xlsb"):
        return pd.read_excel(uploaded_file, engine="pyxlsb")

    # default: xlsx / xls
    try:
        return pd.read_excel(uploaded_file)  # openpyxl/xlrd auto
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, engine="odf")

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Penalties") -> bytes:
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buff.seek(0)
    return buff.getvalue()
