# trimmed_app.py
import os
import base64
import mimetypes
import io
import re
import tempfile
import shutil
import time
import glob
import traceback
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
import unicodedata

# ---------------- CONFIG ----------------
REGIONS = ['us', 'emu', 'uk', 'jap', 'pac', 'gem']
PILLARS = {
    'Valuations': ['PE', 'PBV', 'PS', 'PEBTDA'],
    'Profitability': ['ROE', 'ROA', 'ROC', 'PM', 'OM'],
    'Leverage': ['NDE', 'TDTA', 'TDTE']
}

VARIABLE_FULLNAMES = {
    'PE': 'Price-to-earnings',
    'PBV': 'Price-to-book',
    'PS': 'Price-to-sales',
    'PEBTDA': 'Price-to-EBITDA',
    'ROE': 'Return on equity',
    'ROA': 'Return on assets',
    'ROC': 'Return on capital',
    'PM': 'Profitability margin',
    'OM': 'Operating margin',
    'DY': 'Dividend yield',
    'EM': 'EBITDA margin',         
    'NDE': 'Net-debt-to-EBITDA',
    'TDTA': 'Total-debt-to-assets',
    'TDTE': 'Total-debt-to-equity'
}
_region_regex = re.compile(r'\b(' + '|'.join(REGIONS) + r')\b', flags=re.IGNORECASE)

# Defaults (files expected in same folder as app.py)
APP_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = APP_DIR / 'Template_HP_filter_INPUT_31122025.xlsx'
DEFAULT_LOGO = APP_DIR / 'Logo_Amundi_investment_solutions_4c.jpg'

def _template_date_for_sort(fname):
    base = Path(fname).stem

    # Prefer a trailing 8-digit date (DDMMYYYY)
    m = re.search(r'(\d{8})(?=[^\d]*$)', base) or re.search(r'(\d{8})', base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d%m%Y")
        except Exception:
            pass

    # Then trailing 6-digit date (DDMMYY)
    m = re.search(r'(\d{6})(?=[^\d]*$)', base) or re.search(r'(\d{6})', base)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d%m%y")
        except Exception:
            pass

    return None


def template_sort_key(fname):
    dt = _template_date_for_sort(fname)
    base = Path(fname).name.lower()

    # dated files first, newest first; undated files last
    if dt is None:
        return (1, 0, base)
    return (0, -dt.timestamp(), base)


def find_templates():
    base_dir = APP_DIR
    patterns = ["*template*.xlsx", "Template*.xlsx", "input_template*.xlsx", "*.xlsx"]
    seen = {}

    for p in patterns:
        for fp in base_dir.rglob(p):
            if fp.is_file():
                rel_path = str(fp.relative_to(base_dir))
                if rel_path not in seen:
                    seen[rel_path] = str(fp.resolve())

    if DEFAULT_XLSX.exists():
        rel_default = str(DEFAULT_XLSX.relative_to(base_dir))
        seen.setdefault(rel_default, str(DEFAULT_XLSX.resolve()))

    # recent first
    files = sorted(seen.keys(), key=template_sort_key)
    return files, seen


TEMPLATE_NAMES, TEMPLATE_MAP = find_templates()

# ---------------- LOGO HELPER ----------------
def logo_html(path: Path, width: int = 220, alt: str = "Logo"):
    if path is None:
        return f"<div style='width:{width}px;height:100px;display:flex;align-items:center;justify-content:center;color:#999;background:#f7f7f7;border-radius:4px'>{alt}</div>"
    p = Path(path)
    if not p.exists():
        return f"<div style='width:{width}px;height:100px;display:flex;align-items:center;justify-content:center;color:#999;background:#f7f7f7;border-radius:4px'>{alt}</div>"
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    html = f"<div style='display:flex;align-items:center;'><img src='data:{mime};base64,{b64}' alt='{alt}' style='max-width:100%;width:{width}px;height:auto;' /></div>"
    return html

# ---------------- PROCESSING HELPERS ----------------
def mrollhp_py_trend(x, w, f):
    x = np.asarray(x, dtype=float)
    orig_1d = False
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        orig_1d = True
    n, m = x.shape
    datihp = np.zeros((n, m), dtype=float)
    for i in range(m):
        p = x[:, i]
        hp1 = np.zeros(n, dtype=float)
        for j in range(1, n - w + 1):
            length = w + j
            window = p[:length]
            try:
                cycle, trend = hpfilter(window, lamb=f)
                hp1[length - 1] = float(trend[-1])
            except Exception:
                hp1[length - 1] = 0.0
        datihp[:, i] = hp1
    if orig_1d:
        return datihp[:, 0]
    return datihp

def extract_region_from_col(colname: str):
    c = str(colname)
    m = _region_regex.search(c)
    if m:
        return m.group(1).lower()
    low = c.lower().strip()
    for r in REGIONS:
        if low.endswith('_' + r) or low.endswith('-' + r) or low.endswith('.' + r):
            return r
    tokens = re.split(r'[\s_\-\.\/\(\)]+', low)
    if tokens and tokens[-1] in REGIONS:
        return tokens[-1]
    if tokens and tokens[0] in REGIONS:
        return tokens[0]
    return None

def sheet_belongs_to_pillar(sheet_name: str):
    s = sheet_name.upper()
    for p, tokens in PILLARS.items():
        for tok in tokens:
            if tok.upper() in s:
                return p
    return None

def get_series_from_sheet_name(sheet_name: str):
    if not sheet_name:
        return None
    s = sheet_name.upper()
    for tokens in PILLARS.values():
        for tok in tokens:
            if tok.upper() in s:
                return tok
    return None

def _label_to_key(lbl, st):
    if not st or 'label_to_sheet' not in st or lbl is None:
        return lbl
    return st['label_to_sheet'].get(lbl, lbl)

def _plot_for_label(selected_label, regs, st, mode):
    import plotly.graph_objects as go
    if not st or 'processed' not in st:
        return go.Figure()
    real_key = _label_to_key(selected_label, st)
    if real_key not in st['processed']:
        return go.Figure()
    return build_custom_series_plot(st['processed'], real_key, regs or [], mode)

def expand_variable_names(s: str):
    """
    Replace known short tokens (PE, PBV, ROE, ...) with full descriptions using VARIABLE_FULLNAMES.
    Leaves the rest of the string (regions, punctuation) unchanged.
    """
    if not s:
        return s
    out = str(s)
    for tok, fullname in VARIABLE_FULLNAMES.items():
        out = re.sub(r'\b' + re.escape(tok) + r'\b', fullname, out, flags=re.IGNORECASE)
    # tidy spaces
    out = re.sub(r'\s+', ' ', out).strip()
    return out

# ---------------- CORE PROCESS ----------------
# Replace/insert this full function
def process_file_core(file_obj):
    """
    Read Excel (file-like or path), process each sheet:
      - detect date column and raw series columns
      - detect region token from column names
      - name output columns as 'var <REG>' and 'trend <REG>'
      - compute HP trend and compute cyclical = raw - trend BUT do NOT write cyc to Excel.
      - store computed cycles in processed[sheet_key]['computed_cycles'] (dict region -> pd.Series aligned with results rows)
      - clip results to `start_date` if data exists after that date (default 2007-01-01)
    Returns dict with processed sheets, excel bytes, sheet names, vars_per_sheet, regions, mapping_report.
    """
    def mrollhp_py_trend_inner(x, w, f):
        x = np.asarray(x, dtype=float)
        orig_1d = False
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            orig_1d = True
        n, m = x.shape
        datihp = np.zeros((n, m), dtype=float)
        for i in range(m):
            p = x[:, i]
            hp1 = np.zeros(n, dtype=float)
            for j in range(1, n - w + 1):
                length = w + j
                window = p[:length]
                try:
                    cycle, trend = hpfilter(window, lamb=f)
                    hp1[length - 1] = float(trend[-1])
                except Exception:
                    hp1[length - 1] = 0.0
            datihp[:, i] = hp1
        if orig_1d:
            return datihp[:, 0]
        return datihp

    # default start date for plotting custom chart
    start_date_str = "2007-01-01"
    start_date = pd.to_datetime(start_date_str)

    try:
        if file_obj is not None and hasattr(file_obj, "read"):
            file_bytes = file_obj.read()
            all_sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine='openpyxl')
        else:
            file_path = None
            if isinstance(file_obj, str) and file_obj.strip():
                file_path = Path(file_obj)
            elif file_obj is not None and hasattr(file_obj, "name") and isinstance(file_obj.name, str):
                file_path = Path(file_obj.name)
            if file_path is None:
                file_path = DEFAULT_XLSX
            if not file_path.exists():
                return {"error": f"Excel file not found at {file_path}."}
            all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    except Exception as e:
        return {"error": f"Could not read Excel: {e}"}

    processed = {}
    mapping_report = {}
    used_regions = set()

    # Desired ordering (region tokens lowercased)
    desired_regions = ['us', 'emu', 'uk', 'jap', 'pac', 'gem']
    # Display form for region (we use upper for consistency)
    desired_display = {r: r.upper() for r in desired_regions}

    for raw_sheet_name, raw_df in all_sheets.items():
        sheet_name = raw_sheet_name
        sheet_key = unicodedata.normalize("NFKC", str(sheet_name))
        sheet_key = sheet_key.replace("\u00A0", " ")
        sheet_key = re.sub(r'\s+', ' ', sheet_key).strip()
        if not sheet_key:
            sheet_key = str(sheet_name)

        df = raw_df.copy()

        # find date column
        date_col = None
        for c in df.columns:
            if str(c).strip().lower() == "date":
                date_col = c
                break
        if date_col is None:
            poss = [c for c in df.columns if 'date' in str(c).lower()]
            if not poss:
                return {"error": f"No 'date' column found in sheet {sheet_name}."}
            date_col = poss[0]

        # ensure datetime and sorted
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)

        # Start building results with date
        results_df = pd.DataFrame({'date': df[date_col]})

        # ---- FILTER: keep candidate raw series columns (exclude any column with 'trend' or 'cyclical' or 'cycle') ----
        cols = [
            c for c in df.columns
            if c != date_col and not re.search(r'\b(trend|cyclical|cyc|cycle)\b', str(c), flags=re.IGNORECASE)
        ]

        mapping_lines = []
        region_src_map = {}  # maps region token -> source column name
        region_out_map = {}  # maps region token -> output (var) column name

        # detect sheet-level series token and sheet-level region
        sheet_series = get_series_from_sheet_name(sheet_name)
        sheet_region = extract_region_from_col(sheet_name)

        for c in cols:
            region = extract_region_from_col(c)
            if region is None:
                lowc = str(c).strip().lower()
                if lowc in REGIONS:
                    region = lowc
            if region is None and sheet_region is not None:
                region = sheet_region

            if not region:
                mapping_lines.append(f"SKIP sheet='{sheet_name}' col='{c}' (no region detected in col or sheet)")
                continue

            # normalize region token to lower
            region_token = str(region).strip().lower()

            # derive out col name
            if sheet_series and sheet_series.lower() in str(c).lower():
                out_col_name = str(c)
                mapping_lines.append(f"MAP sheet='{sheet_name}' col='{c}' -> region='{region_token}' (kept original name)")
            else:
                if sheet_series:
                    out_col_name = f"{sheet_series} {region_token}"
                    mapping_lines.append(f"MAP sheet='{sheet_name}' col='{c}' -> region='{region_token}' (renamed to '{out_col_name}')")
                else:
                    out_col_name = str(c)
                    mapping_lines.append(f"MAP sheet='{sheet_name}' col='{c}' -> region='{region_token}' (no series token found; kept original name)")

            region_src_map[region_token] = c
            region_out_map[region_token] = out_col_name
            used_regions.add(region_token)

        mapping_report[sheet_key] = mapping_lines
        mapped_regions = list(region_out_map.keys())

        # initialize computed cycles container for this sheet
        computed_cycles = {}

        # compute trend for desired regions only (and only output var & trend columns)
        for region in desired_regions:
            if region not in region_src_map:
                continue
            src_col = region_src_map[region]
            price = df[src_col].copy()

            # output names formatted as requested
            disp = desired_display.get(region, region.upper())
            raw_col_name = f"var {disp}"
            trend_col_name = f"trend {disp}"

            first_valid = price.first_valid_index()
            if first_valid is None:
                results_df[raw_col_name] = price.values
                results_df[trend_col_name] = 0.0
                # store computed cycle (raw - trend) aligned to results_df
                try:
                    raw_numeric = pd.to_numeric(price, errors='coerce').fillna(0.0).astype(float)
                    cyc_full = raw_numeric.values - 0.0
                    computed_cycles[region] = pd.Series(cyc_full, index=results_df.index, dtype=float)
                except Exception:
                    computed_cycles[region] = pd.Series(0.0, index=results_df.index, dtype=float)
                continue

            # compute trend on the valid numeric data
            try:
                valid_price_series = price.loc[first_valid:].astype(float)
            except Exception:
                valid_price_series = pd.to_numeric(price.loc[first_valid:], errors='coerce').fillna(0.0).astype(float)

            valid_price_values = valid_price_series.values
            trend_valid = mrollhp_py_trend_inner(valid_price_values, 60, 14400.0)

            trend_full = pd.Series(0.0, index=price.index, dtype=float)
            try:
                trend_full.loc[first_valid:] = trend_valid
            except Exception:
                L = len(trend_valid)
                trend_full.iloc[first_valid:first_valid + L] = trend_valid[:min(L, len(trend_full) - first_valid)]

            # compute cyclical = raw_numeric - trend (numeric-safe), but DO NOT write cyc to Excel.
            try:
                raw_numeric = pd.to_numeric(price, errors='coerce').fillna(0.0).astype(float)
                cyc_full_vals = raw_numeric.values - trend_full.values
            except Exception:
                cyc_full_vals = np.zeros(len(price), dtype=float)

            # add raw and trend to results_df (these are written to Excel)
            results_df[raw_col_name] = price.values
            results_df[trend_col_name] = trend_full.values

            # store computed cycles aligned to results_df rows; we'll trim later to start_date
            computed_cycles[region] = pd.Series(cyc_full_vals, index=results_df.index, dtype=float)

        # clip results to start_date if there are rows >= start_date
        if 'date' in results_df.columns:
            try:
                mask = pd.to_datetime(results_df['date']) >= start_date
                if mask.any():
                    results_trim = results_df.loc[mask].reset_index(drop=True)
                    # trim computed_cycles to the same mask (positional)
                    trimmed_computed = {}
                    for r, s in computed_cycles.items():
                        try:
                            trimmed_s = s.loc[mask].reset_index(drop=True)
                        except Exception:
                            # fallback: positional selection
                            try:
                                trimmed_s = pd.Series(np.asarray(s)[np.where(mask)[0]], dtype=float).reset_index(drop=True)
                            except Exception:
                                trimmed_s = pd.Series(dtype=float)
                        trimmed_computed[r] = trimmed_s
                    computed_cycles = trimmed_computed
                else:
                    results_trim = results_df.copy().reset_index(drop=True)
                    # reset indices for computed cycles
                    computed_cycles = {r: (s.reset_index(drop=True) if isinstance(s, pd.Series) else pd.Series(dtype=float)) for r, s in computed_cycles.items()}
            except Exception:
                results_trim = results_df.copy().reset_index(drop=True)
                computed_cycles = {r: (s.reset_index(drop=True) if isinstance(s, pd.Series) else pd.Series(dtype=float)) for r, s in computed_cycles.items()}
        else:
            results_trim = results_df.copy().reset_index(drop=True)
            computed_cycles = {r: (s.reset_index(drop=True) if isinstance(s, pd.Series) else pd.Series(dtype=float)) for r, s in computed_cycles.items()}

        desired_var_cols = [f"var {desired_display[r]}" for r in desired_regions]
        desired_trend_cols = [f"trend {desired_display[r]}" for r in desired_regions]

        # Pick the present var columns (in the desired order) then the present trend columns
        present_var_cols = [c for c in desired_var_cols if c in results_trim.columns]
        present_trend_cols = [c for c in desired_trend_cols if c in results_trim.columns]

        final_cols = ['date'] + present_var_cols + present_trend_cols
        results_trim = results_trim[final_cols]

        # store processed under normalized sheet_key
        processed[sheet_key] = {
            "original": df,
            "results": results_trim,
            "cols": [c for c in final_cols if c != 'date'],
            "region_col_map": {r: f"var {desired_display[r]}" for r in desired_regions if f"var {desired_display[r]}" in results_trim.columns},
            "computed_cycles": computed_cycles
        }

    # Build Excel export
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            for key, obj in processed.items():
                safe_name = (key or "sheet")[:31]
                obj['results'].to_excel(writer, sheet_name=safe_name, index=False)
        buf.seek(0)
        excel_bytes = buf.read()
    except Exception as e:
        return {"error": f"Could not build export Excel: {e}"}

    vars_per_sheet = {sheet: processed[sheet]['cols'] for sheet in processed}
    regions_list = sorted(list(used_regions), key=lambda x: REGIONS.index(x) if x in REGIONS else 999)

    return {
        "processed": processed,
        "excel_bytes": excel_bytes,
        "sheet_names": list(processed.keys()),
        "vars_per_sheet": vars_per_sheet,
        "regions": regions_list,
        "mapping_report": mapping_report
    }


def build_custom_series_plot(processed, sheet_name, selected_regions, display_mode='trend+cycle', debug=False):
    """
    Custom plot clipped to >= 2007-01-01 (local only). Uses Plotly qualitative palette,
    distinct colors for trend vs trend+cycle, short-dash for cycle lines.
    - processed: dict from process_file_core
    - sheet_name: chosen sheet key
    - selected_regions: list of region codes (lowercase)
    - display_mode: 'trend+cycle' | 'trend' | 'cycle'
    """
    import plotly.express as px
    import plotly.graph_objects as go
    import re

    def _to_rgb_tuple(color):
        s = str(color).strip()
        if s.startswith("#"):
            h = s.lstrip("#")
            if len(h) == 3:
                h = ''.join(ch*2 for ch in h)
            try:
                return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
            except Exception:
                return (0,0,0)
        if s.startswith("rgb"):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            try:
                return tuple(int(float(n)) for n in nums[:3])
            except Exception:
                return (0,0,0)
        return (0,0,0)

    def _rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*[max(0, min(255, int(x))) for x in rgb])

    def _rgba(col, a=1.0):
        r,g,b = _to_rgb_tuple(col); return f"rgba({r},{g},{b},{a})"

    def lighten(col, amount=0.25):
        r,g,b = _to_rgb_tuple(col)
        r = r + (255-r)*amount; g = g + (255-g)*amount; b = b + (255-b)*amount
        return _rgb_to_hex((r,g,b))

    def darken(col, amount=0.25):
        r,g,b = _to_rgb_tuple(col)
        r = r*(1-amount); g = g*(1-amount); b = b*(1-amount)
        return f"rgb({int(r)},{int(g)},{int(b)})"

    if not sheet_name or sheet_name not in processed:
        if debug: print("[custom_plot] missing sheet:", sheet_name)
        return go.Figure()

    res_full = processed[sheet_name].get('results')
    if res_full is None or res_full.empty or 'date' not in res_full.columns:
        if debug: print("[custom_plot] no results for sheet:", sheet_name)
        return go.Figure()

    # local copy + datetime
    res_full = res_full.copy()
    res_full['date'] = pd.to_datetime(res_full['date'], errors='coerce')

    # clip for custom chart only
    start_date = pd.Timestamp("2007-01-01")
    plot_df = res_full.loc[res_full['date'] >= start_date].reset_index(drop=True)
    if plot_df.empty:
        if debug: print(f"[custom_plot] no data >= {start_date.date()} for sheet {sheet_name}")
        return go.Figure()

    sel_regs = [str(r).lower() for r in (selected_regions or []) if r]
    if not sel_regs:
        if debug: print("[custom_plot] no regions selected")
        return go.Figure()

    palette = px.colors.qualitative.Plotly
    base_map = {r: palette[i % len(palette)] for i, r in enumerate(REGIONS)}

    # region_col_map (original mapping may reference full res)
    region_map = processed[sheet_name].get('region_col_map', {}) or {}
    region_map_lc = {k.lower(): v for k,v in region_map.items() if k is not None}

    def find_first(columns):
        for c in columns:
            if c is None: continue
            key = str(c).lower()
            for col in res_full.columns:
                if key == col.lower(): return col
        return None

    def find_contains(tokens):
        toks = [t.lower() for t in tokens if t]
        for col in res_full.columns:
            cl = col.lower()
            if all(tok in cl for tok in toks): return col
        return None

    def get_series(col):
        if not col or col not in plot_df.columns:
            return None
        try:
            return plot_df[col].astype(float)
        except Exception:
            return pd.to_numeric(plot_df[col], errors='coerce')

    fig = go.Figure()
    dash_try = "3px 2px"
    dash_fallback = "dash"

    for r in sel_regs:
        token = r.lower()
        raw_col = None
        if token in region_map_lc:
            cand = region_map_lc[token]
            if cand in res_full.columns: raw_col = cand
        if raw_col is None:
            raw_col = find_first([f"equity {token}", f"equity_{token}", token])

        trend_col = find_contains(['trend', token]) or find_first([f"{raw_col} trend" if raw_col else None])
        cyc_col = find_contains(['cyc', token]) or find_first([f"{raw_col} cyclical" if raw_col else None])

        if debug:
            print(f"[custom_plot] region={r}, raw={raw_col}, trend={trend_col}, cyc={cyc_col}")

        y_raw = get_series(raw_col)
        y_trend = get_series(trend_col)
        y_cyc = get_series(cyc_col)

        # derive missing
        if y_raw is not None and y_trend is None and y_cyc is not None:
            try: y_trend = y_raw - y_cyc
            except Exception: y_trend = None
        if y_raw is not None and y_cyc is None and y_trend is not None:
            try: y_cyc = y_raw - y_trend
            except Exception: y_cyc = None

        name_label = r.upper()
        base_col = base_map.get(token, palette[0])
        trend_color = darken(base_col, 0.25)
        tc_line = base_col
        tc_fill = _rgba(base_col, 0.35)
        x = plot_df['date']

        if display_mode == 'trend':
            if y_trend is not None:
                fig.add_trace(go.Scatter(x=x, y=y_trend, mode='lines', name=f"{name_label} - trend",
                                         line=dict(color=trend_color, width=2)))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(x=x, y=y_raw, mode='lines', name=f"{name_label} - raw",
                                         line=dict(color=trend_color, width=2)))
        elif display_mode == 'cycle':
            if y_cyc is not None:
                try:
                    fig.add_trace(go.Scatter(x=x, y=y_cyc, mode='lines', name=f"{name_label} - cycle",
                                             line=dict(color=tc_line, width=1.5, dash=dash_try)))
                except Exception:
                    fig.add_trace(go.Scatter(x=x, y=y_cyc, mode='lines', name=f"{name_label} - cycle",
                                             line=dict(color=tc_line, width=1.5, dash=dash_fallback)))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(x=x, y=y_raw, mode='lines', name=f"{name_label} - raw",
                                         line=dict(color=tc_line, width=1.5)))
        else:  # trend+cycle
            if y_trend is not None and y_cyc is not None:
                base = y_trend
                top = y_trend + y_cyc
                fig.add_trace(go.Scatter(x=x, y=base, mode='lines', name=f"{name_label} - trend",
                                         line=dict(color=trend_color, width=1),
                                         fill='tozeroy', fillcolor=_rgba(trend_color, 0.22)))
                fig.add_trace(go.Scatter(x=x, y=top, mode='lines', name=f"{name_label} - trend+cycle",
                                         line=dict(color=tc_line, width=1),
                                         fill='tonexty', fillcolor=tc_fill))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(x=x, y=y_raw, mode='lines', name=f"{name_label} - raw",
                                         line=dict(color=trend_color, width=2)))

    title_name = expand_variable_names(sheet_name)
    fig.update_layout(title=f"{title_name} for selected regions", xaxis_title='Date', yaxis_title='Value', showlegend=True)
    return fig

def _label_from_filename(fname):
    """
    Extract trailing date token from filename and return label 'DD Mon YYYY' (e.g. '31 Dec 2025').
    Handles 8-digit (DDMMYYYY) and 6-digit (DDMMYY -> assumes 20YY).
    Falls back to file stem if no date found.
    """
    base = Path(fname).stem
    # try 8-digit then 6-digit sequences near the end of the name
    m = re.search(r'(\d{8})(?=[^\d]*$)', base)  # eight digits at end
    if m:
        s = m.group(1)
        try:
            dt = datetime.strptime(s, "%d%m%Y")
            return dt.strftime("%d %b %Y")
        except Exception:
            pass
    m2 = re.search(r'(\d{6})(?=[^\d]*$)', base)  # six digits at end
    if m2:
        s = m2.group(1)
        try:
            # assume 20YY for two-digit years
            dt = datetime.strptime(s, "%d%m%y")
            return dt.strftime("%d %b %Y")
        except Exception:
            pass
    # fallback: try to find any 8/6-digit group anywhere
    m = re.search(r'(\d{8})', base)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%d%m%Y")
            return dt.strftime("%d %b %Y")
        except Exception:
            pass
    m2 = re.search(r'(\d{6})', base)
    if m2:
        try:
            dt = datetime.strptime(m2.group(1), "%d%m%y")
            return dt.strftime("%d %b %Y")
        except Exception:
            pass
    # final fallback: return short stem
    return base

TEMPLATE_DISPLAY_MAP = {}
TEMPLATE_DISPLAY_NAMES = []

for fname in TEMPLATE_NAMES:
    label = _label_from_filename(fname)

    # If two files have the same display label, add a suffix
    base_label = label
    i = 1
    while label in TEMPLATE_DISPLAY_MAP:
        label = f"{base_label} ({i})"
        i += 1

    TEMPLATE_DISPLAY_MAP[label] = fname   # label -> relative path
    TEMPLATE_DISPLAY_NAMES.append(label)
most_recent_label = None
try:
    label_dates = []
    for lbl, fname in TEMPLATE_DISPLAY_MAP.items():
        # parse a date token (8 or 6 digits) from the filename stem
        base = Path(fname).stem
        m = re.search(r'(\d{8})', base) or re.search(r'(\d{6})', base)
        if m:
            s = m.group(1)
            parsed = None
            try:
                parsed = datetime.strptime(s, "%d%m%Y")
            except Exception:
                try:
                    parsed = datetime.strptime(s, "%d%m%y")
                except Exception:
                    parsed = None
            if parsed:
                label_dates.append((lbl, parsed))
    if label_dates:
        # pick the label with the newest parsed date
        most_recent_label = max(label_dates, key=lambda t: t[1])[0]
except Exception:
    most_recent_label = None

# fallback to first display label if nothing parsed
if not most_recent_label and TEMPLATE_DISPLAY_NAMES:
    most_recent_label = TEMPLATE_DISPLAY_NAMES[1]

# ---------------- PLOTTING ----------------
def build_pillar_sheetmap(processed):
    """
    Return dict mapping each pillar name -> list of sheet keys (from processed)
    """
    pillar_map = {p: [] for p in PILLARS}
    for sheet in (processed or {}).keys():
        p = sheet_belongs_to_pillar(sheet)
        if p:
            pillar_map[p].append(sheet)
    return pillar_map


def build_pillar_sheetmap(processed):
    pillar_map = {p: [] for p in PILLARS}
    for sh in (processed or {}).keys():
        p = sheet_belongs_to_pillar(sh)
        if p:
            pillar_map[p].append(sh)
    return pillar_map


def build_radar_for_snapshots(processed, regions, snapshot_dates, pad_rel=0.02, abs_pad=0.01, debug=False):
    """
    Build 3 polar subplots (Valuations / Profitability / Leverage) using cyclical values
    from processed (per-sheet results). snapshot_dates is a list of datetimes (or strings).
    Prefers in-memory computed cycles (processed[sheet]['computed_cycles']) and falls back
    to explicit cyc columns or raw-trend computation.
    """
    # prepare up to 3 unique snapshot datetimes
    uniq_dates = []
    for d in (snapshot_dates or []):
        if d is None:
            continue
        try:
            dt = pd.to_datetime(d)
        except Exception:
            continue
        if dt not in uniq_dates:
            uniq_dates.append(dt)
        if len(uniq_dates) >= 3:
            break

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    pillar_sheetmap = build_pillar_sheetmap(processed)
    pillars = list(pillar_sheetmap.keys())

    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]])

    rotation_deg = 90
    domain_starts = [0.03, 0.345, 0.67]
    domain_ends   = [0.33, 0.665, 0.97]

    # Precompute per-sheet helpers: res, dates_index (pd.DatetimeIndex), cyc_series_map (region -> pd.Series or None)
    sheet_helpers = {}
    for sh, obj in processed.items():
        res = obj.get('results')
        if res is None or 'date' not in res.columns:
            continue
        try:
            dates_index = pd.to_datetime(res['date'])
        except Exception:
            try:
                dates_index = pd.DatetimeIndex(np.array(res['date'], dtype='datetime64[ns]'))
            except Exception:
                dates_index = pd.DatetimeIndex([])

        declared_map = obj.get('region_col_map', {}) or {}
        computed_cycles = obj.get('computed_cycles', {}) or {}
        cyc_series_map = {}
        cols = list(res.columns)
        cols_lower_map = {c.lower(): c for c in cols}

        for r in regions:
            r_lc = r.lower()
            # 1) in-memory computed cycles
            if r_lc in computed_cycles and isinstance(computed_cycles[r_lc], pd.Series) and not computed_cycles[r_lc].empty:
                # Ensure it's reset_index'd (positional alignment)
                cyc_series_map[r] = computed_cycles[r_lc].reset_index(drop=True)
                continue

            # 2) try to find explicit cyc column variants based on declared raw name
            raw_col = declared_map.get(r_lc)
            found = None
            if raw_col and raw_col in res.columns:
                candidates = [
                    f"{raw_col} cyclical", f"{raw_col} cycle",
                    f"{raw_col}_cyclical", f"{raw_col}_cycle",
                    f"{raw_col}.cyclical", f"{raw_col}.cycle"
                ]
                for cand in candidates:
                    if cand.lower() in cols_lower_map:
                        found = cols_lower_map[cand.lower()]; break
            if found:
                cyc_series_map[r] = res[found].reset_index(drop=True)
                continue

            # 3) fallback: compute from raw - trend if both columns can be found
            found_raw = None
            found_trend = None
            # candidate raw names
            raw_candidates = []
            if raw_col:
                raw_candidates.append(raw_col)
            raw_candidates += [f"equity {r}", f"equity_{r}", r, r.upper(), r.lower()]
            for cand in raw_candidates:
                if cand and cand.lower() in cols_lower_map:
                    found_raw = cols_lower_map[cand.lower()]; break
            # trend candidate search
            for c in cols:
                cl = c.lower()
                if 'trend' in cl and r.lower() in cl:
                    found_trend = c
                    break
            if not found_trend and raw_col:
                t1 = f"{raw_col} trend"
                if t1.lower() in cols_lower_map:
                    found_trend = cols_lower_map[t1.lower()]

            if found_raw and found_trend:
                try:
                    s_raw = pd.to_numeric(res[found_raw], errors='coerce').fillna(0.0).astype(float).reset_index(drop=True)
                    s_trend = pd.to_numeric(res[found_trend], errors='coerce').fillna(0.0).astype(float).reset_index(drop=True)
                    cyc_series_map[r] = (s_raw - s_trend)
                except Exception:
                    cyc_series_map[r] = None
                continue

            # 4) last resort: find any column containing both token and cyc/cycle
            token = r.lower()
            found_col = None
            for c in cols:
                cl = c.lower()
                if (('cyc' in cl) or ('cycle' in cl)) and (token in cl):
                    found_col = c; break
            if found_col:
                cyc_series_map[r] = res[found_col].reset_index(drop=True)
            else:
                cyc_series_map[r] = None

        sheet_helpers[sh] = {'res': res, 'dates_index': dates_index, 'cyc_series_map': cyc_series_map}

    # Build per-pillar aggregated values for each snapshot
    for col_idx, pillar in enumerate(pillars, start=1):
        sheets = pillar_sheetmap.get(pillar, [])
        region_codes = regions if regions else ['none']
        categories = region_codes
        display_labels = [rc.upper() for rc in region_codes]

        per_snapshot_vals = []
        debug_mappings = []

        for dt in uniq_dates:
            vals = []
            for r in categories:
                collected = []
                for sh in sheets:
                    helper = sheet_helpers.get(sh)
                    if not helper:
                        continue
                    res = helper['res']
                    dates_index = helper['dates_index']
                    cyc_series = helper['cyc_series_map'].get(r)

                    if cyc_series is None:
                        # no cycle source for this sheet-region
                        if debug and len(debug_mappings) < 200:
                            debug_mappings.append((sh, r, 'no_cycle_source'))
                        continue

                    # robust nearest-date lookup using pandas
                    try:
                        if len(dates_index) == 0:
                            continue
                        pos_idx = dates_index.get_indexer([pd.to_datetime(dt)], method='nearest')[0]
                        if pos_idx == -1:
                            pos_idx = 0
                        pos_idx = max(0, min(int(pos_idx), len(dates_index) - 1))
                    except Exception:
                        # fallback: numpy searchsorted
                        try:
                            dates_np = np.array(dates_index.values, dtype='datetime64[ns]')
                            pos = int(np.searchsorted(dates_np, np.datetime64(pd.to_datetime(dt))))
                            pos = max(0, min(pos, len(dates_np) - 1))
                            pos_idx = pos
                        except Exception:
                            pos_idx = None

                    if pos_idx is None:
                        v = np.nan
                    else:
                        try:
                            v = cyc_series.iloc[pos_idx]
                        except Exception:
                            try:
                                row_label = cyc_series.index[pos_idx] if pos_idx < len(cyc_series.index) else cyc_series.index[-1]
                                v = cyc_series.loc[row_label]
                            except Exception:
                                v = np.nan

                    # append if numeric
                    if pd.notna(v):
                        try:
                            collected.append(float(v))
                        except Exception:
                            try:
                                collected.append(float(np.asarray(v, dtype=float)))
                            except Exception:
                                pass

                    if debug and len(debug_mappings) < 200:
                        try:
                            debug_mappings.append((sh, r, None, pos_idx, None if pd.isna(v) else float(v)))
                        except Exception:
                            debug_mappings.append((sh, r, None, pos_idx, None))

                vals.append(float(np.nanmean(collected)) if collected else np.nan)
            per_snapshot_vals.append(vals)

        if debug:
            print(f"[RADAR DEBUG] pillar={pillar} sample mappings (first 60):", debug_mappings[:60])

        # flatten and compute tick range
        flat_vals = [v for row in per_snapshot_vals for v in row if (v is not None and not np.isnan(v))]
        flat_vals = np.array(flat_vals, dtype=float) if flat_vals else np.array([])

        if flat_vals.size == 0:
            tick_min, tick_max = -1.0, 1.0
        else:
            vmin = float(np.nanmin(flat_vals)); vmax = float(np.nanmax(flat_vals))
            if np.isclose(vmin, vmax):
                center = vmax; pad = max(abs(center) * abs_pad, 0.02); tick_min = center - pad; tick_max = center + pad
            else:
                span = vmax - vmin; pad = max(pad_rel * span, abs_pad * max(abs(vmax), abs(vmin), 1.0)); tick_min = vmin - pad; tick_max = vmax + pad

        if np.isclose(tick_min, tick_max):
            tick_min -= 0.02; tick_max += 0.02

        ticks = np.linspace(tick_min, tick_max, 5)
        mag = max(abs(tick_min), abs(tick_max))
        fmt = "{:.3f}" if mag < 1 else "{:.2f}"
        tick_text = [fmt.format(t) for t in ticks]

        theta = display_labels + [display_labels[0]]

        # add traces for each snapshot (even if NaN -> will be replaced with 0.0 for plotting)
        for date_idx, dt_vals in enumerate(per_snapshot_vals):
            label = uniq_dates[date_idx].strftime('%d/%m/%Y') if hasattr(uniq_dates[date_idx], "strftime") else str(uniq_dates[date_idx])
            color = colors[date_idx % len(colors)]
            plot_vals = [v if (v is not None and not np.isnan(v)) else 0.0 for v in dt_vals]
            rvals = list(plot_vals) + [plot_vals[0] if plot_vals else 0.0]
            fig.add_trace(
                go.Scatterpolar(
                    r=rvals, theta=theta, mode='lines+markers', name=label,
                    line=dict(color=color, width=3), marker=dict(color=color, size=7),
                    legendgroup=label, showlegend=False, fill='none', hovertemplate='%{theta}: %{r:.3f}<extra></extra>'
                ),
                row=1, col=col_idx
            )

        y_domain = [0.12, 0.94]

        fig.update_polars(
            angularaxis=dict(rotation=rotation_deg, direction='counterclockwise', tickfont=dict(size=11)),
            radialaxis=dict(visible=True, tickvals=ticks.tolist(), ticktext=tick_text, range=[ticks[0], ticks[-1]],
                            tickfont=dict(size=11), ticklen=6),
            domain=dict(x=[domain_starts[col_idx-1], domain_ends[col_idx-1]], y=y_domain),
            row=1, col=col_idx)

    # Title annotations for the three pillars
    title_y = 0.985
    for idx, pillar in enumerate(pillars):
        x_mid = (domain_starts[idx] + domain_ends[idx]) / 2.0
        fig.add_annotation(
            x=x_mid, y=title_y, xref='paper', yref='paper',
            text=pillar,
            showarrow=False,
            font=dict(size=16, family='Arial', color='#2e4053'),
            xanchor='center',
            yanchor='bottom'
        )

    # Legend annotations (date bullets)
    x_positions = [(domain_starts[i] + domain_ends[i]) / 2.0 for i in range(len(domain_starts))]
    if not uniq_dates:
        legend_y = -0.01
        for xp in x_positions:
            fig.add_annotation(
                x=xp, y=legend_y, xref='paper', yref='paper',
                text="No snapshots", showarrow=False,
                font=dict(size=11, color="#666"), align='center', xanchor='center'
            )
    else:
        spacing = 0.055
        n_dates = len(uniq_dates)
        mid = (n_dates - 1) / 2.0
        legend_y = -0.01
        for xp in x_positions:
            for j, dt in enumerate(uniq_dates):
                x_ann = xp + (j - mid) * spacing
                txt = dt.strftime('%d/%m/%Y') if hasattr(dt, "strftime") else str(dt)
                fig.add_annotation(
                    x=x_ann, y=legend_y, xref='paper', yref='paper',
                    text=f"● {txt}",
                    showarrow=False,
                    font=dict(size=11, color=colors[j % len(colors)]),
                    align='center',
                    xanchor='center'
                )

    fig.update_layout(
        autosize=True,
        height=700,
        showlegend=False,
        margin=dict(l=8, r=8, t=70, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig


def build_pillar_agg_timeseries(processed, regions):
    pillar_sheetmap = build_pillar_sheetmap(processed)
    figs = {}
    for pillar, sheets in pillar_sheetmap.items():
        fig = go.Figure()
        if not sheets or not regions:
            fig.update_layout(title=f"{pillar} - Aggregate Raw Series (no data)")
            figs[pillar] = fig; continue
        base_sheet = sheets[0]; base_dates = processed[base_sheet]['results']['date']
        for r in regions:
            series_list = []
            for sh in sheets:
                res = processed[sh]['results']; col = processed[sh]['region_col_map'].get(r)
                if col and col in res.columns:
                    series_list.append(res[col])
            if not series_list: continue
            stacked = pd.concat(series_list, axis=1)
            agg = stacked.mean(axis=1)
            fig.add_trace(go.Scatter(x=base_dates, y=agg, mode='lines', name=r.upper()))
            fig.update_layout(title=f"{pillar} - Aggregate Raw Series", xaxis_title='Date', yaxis_title='Aggregate')
            figs[pillar] = fig
    return figs

def build_custom_series_plot(processed, sheet_name, selected_regions, display_mode='trend+cycle', debug=False):
    """
    Custom plot clipped to >= 2007-01-01, preferring in-memory computed cycles.
    - processed: dict from process_file_core
    - sheet_name: chosen sheet key
    - selected_regions: list of region codes (lowercase)
    - display_mode: 'trend+cycle' | 'trend' | 'cycle'
    """
    import plotly.express as px
    import plotly.graph_objects as go

    if not sheet_name or sheet_name not in processed:
        if debug: print("[custom_plot] missing sheet:", sheet_name)
        return go.Figure()

    res_full = processed[sheet_name].get('results')
    if res_full is None or res_full.empty or 'date' not in res_full.columns:
        if debug: print("[custom_plot] no results for sheet:", sheet_name)
        return go.Figure()

    # local copy + datetime
    res_full = res_full.copy()
    res_full['date'] = pd.to_datetime(res_full['date'], errors='coerce')

    # clip for custom chart only
    start_date = pd.Timestamp("2007-01-01")
    plot_df = res_full.loc[res_full['date'] >= start_date].reset_index(drop=True)
    if plot_df.empty:
        if debug: print(f"[custom_plot] no data >= {start_date.date()} for sheet {sheet_name}")
        return go.Figure()

    sel_regs = [str(r).lower() for r in (selected_regions or []) if r]
    if not sel_regs:
        if debug: print("[custom_plot] no regions selected")
        return go.Figure()

    palette = px.colors.qualitative.Plotly
    base_map = {r: palette[i % len(palette)] for i, r in enumerate(REGIONS)}

    # region_col_map (original mapping may reference full res)
    region_map = processed[sheet_name].get('region_col_map', {}) or {}
    region_map_lc = {k.lower(): v for k, v in region_map.items() if k is not None}

    # prepared computed cycles (alignment: processed[...] results and computed_cycles were reset to 0..N-1)
    comp_cyc_map = processed.get(sheet_name, {}).get('computed_cycles', {}) or {}

    cols_lower_map = {c.lower(): c for c in res_full.columns}

    def first_match_from_candidates(candidates):
        for cand in candidates:
            if cand is None:
                continue
            key = cand.lower()
            if key in cols_lower_map:
                return cols_lower_map[key]
        return None

    def find_col_by_tokens(tokens):
        tokens = [t.lower() for t in tokens if t]
        for c in res_full.columns:
            cl = c.lower()
            if all(tok in cl for tok in tokens):
                return c
        return None

    fig = go.Figure()

    if debug:
        print(f"[custom_plot] building plot for sheet='{sheet_name}', regions={sel_regs}, display_mode={display_mode}")
        print("[custom_plot] available columns:", list(res_full.columns))
        print("[custom_plot] region_map (orig):", region_map)
        print("[custom_plot] computed_cycles keys:", list(comp_cyc_map.keys()))

    for i_r, r in enumerate(sel_regs):
        token = r.lower()
        color = base_map.get(token, palette[0])

        # 1) find raw column
        raw_col = None
        if token in region_map_lc:
            cand = region_map_lc[token]
            if cand in res_full.columns:
                raw_col = cand
        if raw_col is None:
            candidates = [f"equity {token}", f"equity_{token}", f"{token} equity", token]
            raw_col = first_match_from_candidates(candidates)
        if raw_col is None:
            for c in res_full.columns:
                if token in c.lower():
                    raw_col = c
                    break

        # 2) find trend and cyc columns (if present)
        trend_col = find_col_by_tokens(['trend', token]) or find_col_by_tokens([token, 'trend'])
        if not trend_col and raw_col:
            trend_col = first_match_from_candidates([f"{raw_col} trend", f"{raw_col}_trend", f"{raw_col}.trend"])

        cyc_col = find_col_by_tokens(['cyc', token]) or find_col_by_tokens(['cycle', token])
        if not cyc_col and raw_col:
            cyc_col = first_match_from_candidates([f"{raw_col} cyclical", f"{raw_col} cycle", f"{raw_col}_cyclical", f"{raw_col}_cycle"])

        if debug:
            print(f"[custom_plot] region={r} -> raw_col={raw_col}, trend_col={trend_col}, cyc_col={cyc_col}")

        # fetch series safely
        def safe_get(col, df=res_full):
            if col and col in df.columns:
                try:
                    return pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    return df[col]
            return None

        y_raw = safe_get(raw_col)
        y_trend = safe_get(trend_col)

        # prefer in-memory computed cyc if available
        y_cyc = None
        if token in comp_cyc_map and isinstance(comp_cyc_map[token], pd.Series) and not comp_cyc_map[token].empty:
            try:
                # comp_cyc_map[token] has been aligned to processed[sheet]['results'] rows and reset_index
                comp_series = comp_cyc_map[token].reset_index(drop=True)
                # plot_df was created from res_full filtered and reset_index; if processed['results'] already trimmed to start_date,
                # comp_series length should match plot_df length. Try direct alignment by position.
                if len(comp_series) == len(plot_df):
                    y_cyc = comp_series
                else:
                    # If lengths mismatch, try to align by matching dates (best-effort)
                    try:
                        # build series indexed by res_full['date'] and then reindex to plot_df['date']
                        idxed = pd.Series(comp_series.values, index=pd.to_datetime(res_full['date'].reset_index(drop=True)))
                        y_cyc = idxed.reindex(pd.to_datetime(plot_df['date']), method='nearest').reset_index(drop=True)
                    except Exception:
                        # fallback to nearest positional slice if possible
                        y_cyc = pd.Series(comp_series.values[:len(plot_df)]).reset_index(drop=True)
            except Exception:
                y_cyc = None

        # if still no cycs from in-memory map, try cyc_col (explicit cycle column in DataFrame)
        if y_cyc is None and cyc_col:
            y_cyc = safe_get(cyc_col)

        # derive missing components where possible
        if y_raw is not None and y_trend is None and y_cyc is not None:
            try:
                y_trend = y_raw - y_cyc
                if debug: print(f"[custom_plot] derived trend for {r}")
            except Exception:
                y_trend = None
        if y_raw is not None and y_cyc is None and y_trend is not None:
            try:
                y_cyc = y_raw - y_trend
                if debug: print(f"[custom_plot] derived cyc for {r}")
            except Exception:
                y_cyc = None

        # If safe_get returned full res_full-length series, reindex to plot_df rows (positional alignment)
        def align_to_plot_df(s):
            if s is None:
                return None
            try:
                s2 = pd.Series(s.values, index=range(len(s)))
                # pick first len(plot_df)
                return s2.iloc[:len(plot_df)].reset_index(drop=True)
            except Exception:
                try:
                    return pd.Series(s.values[:len(plot_df)]).reset_index(drop=True)
                except Exception:
                    return None

        y_raw = align_to_plot_df(y_raw)
        y_trend = align_to_plot_df(y_trend)
        y_cyc = align_to_plot_df(y_cyc)

        name_label = r.upper()
        legendgroup = name_label  # group traces by region in legend

        # plot traces using the region color; make fill/opacity style vary for trend+cycle
        if display_mode == 'trend':
            if y_trend is not None:
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=y_trend, mode='lines', name=f"{name_label} - trend",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=y_raw, mode='lines', name=f"{name_label} - raw (no trend)",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
        elif display_mode == 'cycle':
            if y_cyc is not None:
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=y_cyc, mode='lines', name=f"{name_label} - cycle",
                    line=dict(color=color, width=2, dash='dash'), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=y_raw, mode='lines', name=f"{name_label} - raw (no cycle)",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
        else:  # 'trend+cycle'
            if y_trend is not None and y_cyc is not None:
                base = y_trend
                top = (y_trend + y_cyc)
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=base, mode='lines', name=f"{name_label} - trend",
                    line=dict(color=color, width=1), fill='tozeroy', opacity=0.35,
                    legendgroup=legendgroup
                ))
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=top, mode='lines', name=f"{name_label} - trend+cycle",
                    line=dict(color=color, width=1), fill='tonexty', opacity=0.6,
                    legendgroup=legendgroup
                ))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=y_raw, mode='lines', name=f"{name_label} - raw (fallback)",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
            else:
                if debug:
                    print(f"[custom_plot] nothing available to plot for region {r}")

    try:
        fig.update_layout(colorway=px.colors.qualitative.Plotly)
    except Exception:
        pass

    fig.update_layout(title=f"Selected regions from sheet {sheet_name}", xaxis_title='Date', yaxis_title='Value', showlegend=True)
    return fig


# ================== UI ==================
CUSTOM_CSS = """
:root {
  --dropbox-height: 90px;
  --dropbox-pill-height: 36px;
  --dropbox-pill-font: 11px;
  --dropbox-pill-icon: 8px;
  --dropbox-center-font: 14px;
  --dropbox-left-reserve: 84px;
}

#load_template_btn .gr-button { background:#e9e9e9; color:#222; height:48px; font-weight:600; }
#download_template_btn .gr-button { background:#f5821f; color:white; height:48px; font-weight:600; }
#process_uploaded_btn .gr-button { background:#1f77b4; color:white; height:44px; font-weight:600; }

.app_header { display:flex; align-items:center; }
.template_col { padding-left:18px; padding-right:18px; }
.logo_img img { max-width:220px; }

#drop_file_box,
#drop_file_box .file-dropzone,
.drop_area,
#drop_file_box .file-dropzone .file-card {
  height: var(--dropbox-height) !important;
  min-height: var(--dropbox-height) !important;
  max-height: var(--dropbox-height) !important;
  box-sizing: border-box;
  padding: 6px !important;
  border: 2px dashed #ddd;
  border-radius: 6px;
  background: transparent;
  color: #666;
  overflow: hidden !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

#drop_file_box .file-dropzone { position: relative !important; }

#drop_file_box .file-dropzone .file-dropzone-button,
#drop_file_box .file-dropzone .file-dropzone-label,
#drop_file_box .file-dropzone .file-upload,
#drop_file_box .file-dropzone .file-upload button,
#drop_file_box .file-dropzone .file-card .file-dropzone-button {
  position: absolute !important;
  left: 8px !important;
  top: 8px !important;
  z-index: 5 !important;
  height: var(--dropbox-pill-height) !important;
  line-height: var(--dropbox-pill-height) !important;
  font-size: var(--dropbox-pill-font) !important;
  padding: 6px 10px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-sizing: border-box;
}

#drop_file_box .file-dropzone .file-dropzone-button svg,
#drop_file_box .file-dropzone .file-dropzone-label svg,
#drop_file_box .file-dropzone .file-upload svg {
  width: var(--dropbox-pill-icon) !important;
  height: var(--dropbox-pill-icon) !important;
}

.file-drop-text,
.file-dropzone .file-drop-text,
.drop_area .file-drop-text,
#drop_file_box .file-dropzone .file-drop-text,
#drop_file_box .file-dropzone .file-dropzone-label {
  font-size: 12px !important;
  line-height: 1.2 !important;
  color: #666 !important;
  white-space: normal !important;
}

#download_processed_btn .gr-button,
#process_uploaded_btn .gr-button,
#download_template_btn .gr-button,
#load_template_btn .gr-button {
  width: 100% !important;
  box-sizing: border-box;
  margin-top: 12px !important;
}

#template_file_box .file-dropzone,
#template_file_box .file-list,
#template_file_box .file-upload,
#template_file_box .file-card,
#template_file_box .gr-file {
  max-height: 140px !important;
  overflow: auto !important;
  padding: 8px !important;
}

.gradio-container .gr-button { border-radius:6px; }
.gradio-container .gr-dropdown { min-width:260px; }

#drop_file_box, #drop_file_box *,
.drop_area, .drop_area * {
  font-size: 12px !important;
  line-height: 1.2 !important;
}
"""


with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.HTML("""
    <div style="width:100%; text-align:center;">
      <h1 style="margin:0; font-size:36px; font-weight:800; line-height:1.05;">Cross-Asset Research - Regional Equity Cycle Dashboard</h1>
    """)

    with gr.Row(elem_classes="app_header"):
        with gr.Column(scale=1, elem_classes="logo_img"):
            logo_html_str = logo_html(DEFAULT_LOGO, width=320)
            logo = gr.HTML(value=logo_html_str, elem_id="app_logo_html")

        with gr.Column(scale=1, elem_classes="template_col"):
            gr.Markdown("### Choose data from")
            template_dropdown = gr.Dropdown(
                choices=TEMPLATE_DISPLAY_NAMES or [_label_from_filename(DEFAULT_XLSX.name)],
                value=(most_recent_label if most_recent_label is not None else (_label_from_filename(DEFAULT_XLSX.name))),
                label=None)
            load_template_btn = gr.Button("Load data", elem_id="load_template_btn")
            # removed download_template_btn and upload column entirely

    # remaining UI: status, selectors, plots, etc.
    status_box = gr.Textbox(label="Status", interactive=False, value="Ready")
    snapshot_selector = gr.CheckboxGroup(label="Choose snapshot dates (for radar) (up to 3)", choices=[], visible=False)
    radar_output = gr.Plot(label="Radar (Valuations / Profitability / Leverage)", visible=False)

    DEFAULT_REGION_SELECTION = ['us', 'emu']
    with gr.Column():
        sheet_selector = gr.Dropdown(label="Select sheet (series)", choices=[], visible=False)
        region_check = gr.CheckboxGroup(choices=REGIONS, value=DEFAULT_REGION_SELECTION, label="Filter by region(s) (for custom plot)", visible=False)
        plot_mode = gr.Radio(choices=['trend+cycle', 'trend', 'cycle'], value='trend+cycle', label="Show", visible=True)

    custom_plot = gr.Plot(label="Custom selected variables", visible=False, elem_id="custom_plot")

    agg_valuations = gr.Plot(label="Aggregate - Valuations", visible=False, elem_id="agg_valuations")
    agg_profitability = gr.Plot(label="Aggregate - Profitability", visible=False, elem_id="agg_profitability")
    agg_leverage = gr.Plot(label="Aggregate - Leverage", visible=False, elem_id="agg_leverage")

    state = gr.State({})

    def safe_handle_process(input_file_or_path):
        """
        Process a template path (or file-like). Build snapshot checkbox choices using:
        - older Decembers (one per year) for dates older than latest-11months
        - last 11 months (one per month)
        Return the 10 outputs expected by the UI.
        """
        try:
            print("[safe_handle_process] called with:", repr(input_file_or_path))
            res = process_file_core(input_file_or_path)
        except Exception as e:
            print("[safe_handle_process] processing exception:", traceback.format_exc())
            return (gr.update(value=f"Exception processing file: {e}"),) + tuple([gr.update(visible=False)] * 8) + (None,)

        if not res or 'error' in (res or {}):
            msg = f"Error: {res.get('error')}" if res else "Unknown error from processing"
            print("[safe_handle_process]", msg)
            return (gr.update(value=msg),) + tuple([gr.update(visible=False)] * 8) + (None,)

        processed = res.get('processed', {})
        excel_bytes = res.get('excel_bytes')
        sheet_names = res.get('sheet_names', []) or []
        vars_per_sheet = res.get('vars_per_sheet', {}) or {}
        regions = res.get('regions', []) or []
        mapping_report = res.get('mapping_report', {}) or {}

        print(f"[safe_handle_process] sheets={len(sheet_names)}, regions={regions}")

        # default real sheet key (unchanged)
        default_sheet = sheet_names[1] if len(sheet_names) > 1 else (sheet_names[0] if sheet_names else None)

        # Build friendly display labels for the dropdown (label -> real sheet key)
        label_to_sheet = {}
        sheet_labels = []
        for sh in sheet_names:
            label = str(sh)
            # replace known short tokens with full phrases (word-boundary, case-insensitive)
            for tok, fullname in VARIABLE_FULLNAMES.items():
                label = re.sub(r'\b' + re.escape(tok) + r'\b', fullname, label, flags=re.IGNORECASE)
            label = re.sub(r'\s+', ' ', label).strip()

            # ensure unique labels (add suffix if needed)
            base = label; i = 1
            while label in label_to_sheet:
                label = f"{base} ({i})"; i += 1

            label_to_sheet[label] = sh
            sheet_labels.append(label)

        # find display label for the default sheet
        default_sheet_label = None
        if default_sheet is not None:
            for lbl, key in label_to_sheet.items():
                if key == default_sheet:
                    default_sheet_label = lbl
                    break
        if default_sheet_label is None and sheet_labels:
            default_sheet_label = sheet_labels[0]

        # collect all dates across sheets
        all_dates = []
        for sh, obj in (processed or {}).items():
            res_sh = obj.get('results') if obj else None
            if res_sh is None or 'date' not in res_sh.columns:
                continue
            try:
                ds = pd.to_datetime(res_sh['date'], errors='coerce').dropna().unique()
                all_dates.extend(list(pd.to_datetime(ds)))
            except Exception:
                continue

        # build choices according to rule: older Decembers (one per year) + recent last-11-months (one per month)
        uniq_dates = []
        if all_dates:
            all_dates_dt = sorted(list(set(all_dates)))
            latest = all_dates_dt[-1]
            recent_start = latest - relativedelta(months=11)

            # older_decembers: for dates < recent_start, keep at most one per year (latest dec)
            older = [d for d in all_dates_dt if d < recent_start and d.month == 12]
            older_by_year = {}
            for d in older:
                y = d.year
                if y not in older_by_year or d > older_by_year[y]:
                    older_by_year[y] = d
            older_decembers = sorted(older_by_year.values())

            # recent_months: for dates >= recent_start, keep latest date per (year,month)
            recent = [d for d in all_dates_dt if d >= recent_start]
            recent_by_ym = {}
            for d in recent:
                ym = (d.year, d.month)
                if ym not in recent_by_ym or d > recent_by_ym[ym]:
                    recent_by_ym[ym] = d
            recent_months = sorted(recent_by_ym.values())

            combined = sorted(older_decembers) + sorted(recent_months)
            uniq_dates = [d.strftime('%Y-%m-%d') for d in combined]

        # default snapshots: last 3 of combined (chronologically most recent)
        default_snapshots = uniq_dates[-3:] if uniq_dates else []

        # build radar and pillar aggs defensively
        try:
            radar_fig = build_radar_for_snapshots(processed, regions, [pd.to_datetime(d) for d in default_snapshots])
        except Exception:
            print("[safe_handle_process] radar build failed:", traceback.format_exc())
            radar_fig = go.Figure()

        try:
            pillar_aggs = build_pillar_agg_timeseries(processed, regions)
        except Exception:
            print("[safe_handle_process] pillar_agg build failed:", traceback.format_exc())
            pillar_aggs = {}

        # default selected regions (only if available)
        desired_defaults = ['us', 'emu']
        default_selected = [r for r in desired_defaults if r in regions]

        # initial custom plot (use default_selected or regions fallback)
        try:
            init_regs = default_selected if default_selected else regions
            initial_plot = build_custom_series_plot(processed, default_sheet, init_regs, 'trend+cycle')
        except Exception:
            print("[safe_handle_process] initial custom plot failed:", traceback.format_exc())
            initial_plot = go.Figure()

        # derive a friendly default name for the status bar
        default_display = None

        # if we built a display label earlier, use it (sheet_labels / default_sheet_label)
        if 'default_sheet_label' in locals() and default_sheet_label:
            default_display = default_sheet_label
        else:
            # try to expand known variable tokens inside the real sheet key
            if default_sheet:
                label_try = str(default_sheet)
                for tok, fullname in VARIABLE_FULLNAMES.items():
                    label_try = re.sub(r'\b' + re.escape(tok) + r'\b', fullname, label_try, flags=re.IGNORECASE)
                label_try = re.sub(r'\s+', ' ', label_try).strip()
                default_display = label_try or str(default_sheet)

        msg = f"Loaded {len(sheet_names)} sheets. Default: {default_display}"

        return (
            gr.update(value=msg),
            gr.update(visible=True, choices=uniq_dates, value=default_snapshots[:3]),
            gr.update(visible=True, value=radar_fig),
            gr.update(visible=True, value=pillar_aggs.get('Valuations', go.Figure())),
            gr.update(visible=True, value=pillar_aggs.get('Profitability', go.Figure())),
            gr.update(visible=True, value=pillar_aggs.get('Leverage', go.Figure())),
            gr.update(visible=True, choices=sheet_labels, value=default_sheet_label),
            gr.update(visible=True, choices=regions, value=default_selected),
            gr.update(visible=True, value=initial_plot),
            {
                "processed": processed,
                "excel_bytes": excel_bytes,
                "sheet_names": sheet_names,
                "vars_per_sheet": vars_per_sheet,
                "regions": regions,
                "mapping_report": mapping_report,
                "label_to_sheet": label_to_sheet
            }
        )


    # wire up load_template -> safe_handle_process (same as before)
    def _path_for_template_label(label):
        # label -> filename -> absolute path; fall back to DEFAULT_XLSX
        try:
            # try label -> filename (display map), then filename -> path (TEMPLATE_MAP)
            fname = TEMPLATE_DISPLAY_MAP.get(label) if 'TEMPLATE_DISPLAY_MAP' in globals() else None
            path = TEMPLATE_MAP.get(fname) if fname and fname in TEMPLATE_MAP else None
            # also accept the (unlikely) case where label is already a filename/key in TEMPLATE_MAP
            if path is None and label in TEMPLATE_MAP:
                path = TEMPLATE_MAP[label]
            return path if path is not None else str(DEFAULT_XLSX)
        except Exception:
            return str(DEFAULT_XLSX)

    load_template_btn.click(
        lambda label: safe_handle_process(_path_for_template_label(label)),
        inputs=[template_dropdown],
        outputs=[status_box, snapshot_selector, radar_output, agg_valuations, agg_profitability, agg_leverage,
                sheet_selector, region_check, custom_plot, state]
    )


    # existing interactions for snapshot_selector, sheet_selector, region_check, plot_mode...
    snapshot_selector.change(
        lambda snaps, st: on_snapshot_change(snaps, st),
        inputs=[snapshot_selector, state],
        outputs=[radar_output, agg_valuations, agg_profitability, agg_leverage]
    )
    
    def on_snapshot_change(snapshot_dates, st):
        try:
            if not st or 'processed' not in st:
                return go.Figure(), go.Figure(), go.Figure(), go.Figure()
            try:
                snaps = [pd.to_datetime(d) for d in (snapshot_dates or [])][:3]
            except Exception:
                snaps = []
            try:
                radar_fig = build_radar_for_snapshots(st['processed'], st.get('regions', []), snaps) or go.Figure()
            except Exception:
                print("[on_snapshot_change] radar builder failed:", traceback.format_exc())
                radar_fig = go.Figure()
            try:
                pillar_aggs = build_pillar_agg_timeseries(st['processed'], st.get('regions', []))
                return radar_fig, pillar_aggs.get('Valuations', go.Figure()), pillar_aggs.get('Profitability', go.Figure()), pillar_aggs.get('Leverage', go.Figure())
            except Exception:
                print("[on_snapshot_change] pillar agg builder failed:", traceback.format_exc())
                return radar_fig, go.Figure(), go.Figure(), go.Figure()
        except Exception:
            print("[on_snapshot_change] unexpected error:", traceback.format_exc())
            return go.Figure(), go.Figure(), go.Figure(), go.Figure()

    sheet_selector.change(
        _plot_for_label,
        inputs=[sheet_selector, region_check, state, plot_mode],
        outputs=[custom_plot]
    )

    region_check.change(
        _plot_for_label,
        inputs=[sheet_selector, region_check, state, plot_mode],
        outputs=[custom_plot]
    )

    plot_mode.change(
        _plot_for_label,
        inputs=[sheet_selector, region_check, state, plot_mode],
        outputs=[custom_plot]
    )

# Launch
port = 8115
demo.launch(debug=True, share=False, server_port=port, server_name='0.0.0.0',
            root_path=f'/studio/workspace/vscode/proxy/{os.getenv("ALTO_STUDIO_USERNAME")}/{port}')
