import os
import base64
import mimetypes
import io
import re
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from dateutil.relativedelta import relativedelta
import unicodedata

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_FOLDER = APP_DIR
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------- CONFIG ----------------
REGIONS = ['us', 'emu', 'uk', 'jap', 'pac', 'gem']
PILLARS = {
    'Valuations': ['PE', 'PBV', 'PS', 'PEBTDA'],
    'Profitability': ['ROE', 'ROA', 'ROC', 'OM', 'PM', 'DY', 'EM'],
    'Leverage': ['NDE', 'TDTA', 'TDTE']
}

_region_regex = re.compile(r'\b(' + '|'.join(REGIONS) + r')\b', flags=re.IGNORECASE)

# Defaults
DEFAULT_XLSX = APP_DIR / 'Template_HP_filter_INPUT_31122025.xlsx'
DEFAULT_LOGO = APP_DIR / 'Logo_Amundi_investment_solutions_4c.jpg'


def extract_date_from_filename(filename: str):
    """
    Try to extract a date from filename patterns like:
    - 31012026
    - 31_01_2026
    - 31-01-2026
    - 2026-01-31
    Returns pd.Timestamp or None.
    """
    name = Path(filename).stem

    patterns = [
        r'(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)',           # ddmmyyyy
        r'(?<!\d)(\d{2})[._-](\d{2})[._-](\d{4})(?!\d)', # dd_mm_yyyy / dd-mm-yyyy / dd.mm.yyyy
        r'(?<!\d)(\d{4})[._-](\d{2})[._-](\d{2})(?!\d)', # yyyy_mm_dd / yyyy-mm-dd / yyyy.mm.dd
    ]

    for i, pattern in enumerate(patterns):
        m = re.search(pattern, name)
        if not m:
            continue
        try:
            if i == 0:
                return pd.to_datetime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", dayfirst=True)
            elif i == 1:
                return pd.to_datetime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", dayfirst=True)
            else:
                return pd.to_datetime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        except Exception:
            pass

    return None


def find_templates():
    base_dir = APP_DIR
    patterns = ["*template*.xlsx", "Template*.xlsx", "input_template*.xlsx", "*.xlsx"]
    seen = {}

    for pattern in patterns:
        for fp in base_dir.glob(pattern):
            if fp.name not in seen:
                seen[fp.name] = str(fp.resolve())

    # keep the default if present
    if DEFAULT_XLSX.exists():
        seen.setdefault(DEFAULT_XLSX.name, str(DEFAULT_XLSX.resolve()))

    def sort_key(name):
        dt = extract_date_from_filename(name)
        if dt is not None:
            return (0, dt)  # dated files first
        try:
            return (1, pd.Timestamp(Path(seen[name]).stat().st_mtime, unit="s"))
        except Exception:
            return (2, pd.Timestamp.min)

    sorted_names = sorted(seen.keys(), key=sort_key, reverse=True)
    return sorted_names, seen


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

# ---------------- HELPERS / PROCESSING ----------------
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
    """
    Try to extract a series token from the sheet name using the PILLARS tokens.
    Returns the matched token (as-is, e.g. 'PE') or None.
    """
    if not sheet_name:
        return None
    s = sheet_name.upper()
    for tokens in PILLARS.values():
        for tok in tokens:
            if tok.upper() in s:
                return tok  # keep original token case from PILLARS
    return None


def process_file_core(file_obj):
    """
    Read Excel (file-like or path), process each sheet:
      - detect date column and raw series columns
      - detect region token from column names
      - name output columns <SERIES> <region>
      - compute HP trend (using inner mrollhp_py_trend_inner) and cyclical = raw - trend
      - clip results to `start_date` if data exists after that date (default 2007-01-01) for the custom chart
    Returns dict with processed sheets,excel infos, sheet names, vars_per_sheet, regions, mapping_report.
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

        # Start building results
        results_df = pd.DataFrame({'date': df[date_col]})

        # ---- FILTER: keep candidate raw series columns ----
        cols = [
            c for c in df.columns
            if c != date_col and not re.search(r'\b(trend|cyclical|cyc|cycle)\b', str(c), flags=re.IGNORECASE)
        ]

        mapping_lines = []
        region_src_map = {}  
        region_out_map = {}   

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

            # output column name
            out_col_name = None
            col_contains_series = False
            if sheet_series and sheet_series.lower() in str(c).lower():
                col_contains_series = True

            if col_contains_series:
                out_col_name = str(c)  # keep original column name
                mapping_lines.append(f"MAP sheet='{sheet_name}' col='{c}' -> region='{region}' (kept original name)")
            else:
                if sheet_series:
                    out_col_name = f"{sheet_series} {region}"
                    mapping_lines.append(f"MAP sheet='{sheet_name}' col='{c}' -> region='{region}' (renamed to '{out_col_name}')")
                else:
                    out_col_name = str(c)
                    mapping_lines.append(f"MAP sheet='{sheet_name}' col='{c}' -> region='{region}' (no series token found; kept original name)")

            region_src_map[region.lower()] = c
            region_out_map[region.lower()] = out_col_name
            used_regions.add(region.lower())

        mapping_report[sheet_key] = mapping_lines
        mapped_regions = list(region_out_map.keys())

        # compute trend & cyclical for each region 
        for region in mapped_regions:
            src_col = region_src_map.get(region)
            out_name = region_out_map.get(region)
            if src_col is None:
                continue

            price = df[src_col].copy()
            trend_col = f"{out_name} trend"

            first_valid = price.first_valid_index()
            if first_valid is None:
                results_df[out_name] = price.values
                results_df[trend_col] = 0.0
                results_df[f"{out_name} cyclical"] = price.values - 0.0
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

            cyclical_full = price.values - trend_full.values

            results_df[out_name] = price.values
            results_df[trend_col] = trend_full.values
            results_df[f"{out_name} cyclical"] = cyclical_full

        # clip results to start_date if there are rows >= start_date
        if 'date' in results_df.columns:
            try:
                mask = pd.to_datetime(results_df['date']) >= start_date
                if mask.any():
                    results_trim = results_df.loc[mask].reset_index(drop=True)
                else:
                    results_trim = results_df.copy()
            except Exception:
                results_trim = results_df.copy()
        else:
            results_trim = results_df.copy()

        # store processed under normalized sheet_key
        processed[sheet_key] = {
            "original": df,
            "results": results_trim,
            "cols": list(region_out_map.values()),
            "region_col_map": region_out_map
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

def _sanitize_filename(name: str) -> str:
        name = os.path.basename(name or "")
        # keep only safe chars
        return re.sub(r'[^A-Za-z0-9._-]', '_', name) or f"uploaded_{int(time.time())}.xlsx"

def save_uploaded_to_root(file_obj, dest_dir: Path = None) -> Path:
    """
    Save various Gradio upload in the same folder.
    Returns Path to saved file or raises Exception on failure.
    """
    dest_dir = Path(__file__).parent if dest_dir is None else Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _target_for(name_hint):
        fname = _sanitize_filename(name_hint)
        target = dest_dir / fname
        if target.exists():
            target = dest_dir / f"{target.stem}_{int(time.time())}{target.suffix}"
        return target

    # Case: Gradio returns a dict (common)
    if isinstance(file_obj, dict):
        # try tmp_path-like entries (copy if path exists)
        for key in ("tmp_path", "tempfile", "tempfile_path", "file"):
            val = file_obj.get(key)
            if val:
                p = Path(val)
                if p.exists():
                    target = _target_for(file_obj.get("name") or p.name)
                    shutil.copy(p, target)
                    print(f"[save_uploaded_to_root] copied {key} ({p}) -> {target}")
                    return target
        # try bytes fields inside the dict
        for bkey in ("data", "bytes", "content", "file"):
            if bkey in file_obj and isinstance(file_obj[bkey], (bytes, bytearray)):
                target = _target_for(file_obj.get("name") or f"uploaded_{int(time.time())}.xlsx")
                with open(target, "wb") as out:
                    out.write(bytes(file_obj[bkey]))
                print(f"[save_uploaded_to_root] wrote bytes from dict key '{bkey}' -> {target}")
                return target
        # some Gradio versions put a file-like object under 'file' or 'tempfile'
        candidate = file_obj.get("file") or file_obj.get("tempfile") or file_obj.get("tmp")
        if candidate and hasattr(candidate, "read"):
            fname_hint = file_obj.get("name") or getattr(candidate, "name", None) or f"uploaded_{int(time.time())}.xlsx"
            target = _target_for(fname_hint)
            try:
                candidate.seek(0)
            except Exception:
                pass
            with open(target, "wb") as out:
                chunk = candidate.read()
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                out.write(chunk)
            print(f"[save_uploaded_to_root] saved file-like from dict -> {target}")
            return target

    # Case: file-like object passed directly
    if hasattr(file_obj, "read"):
        fname_hint = getattr(file_obj, "name", None) or f"uploaded_{int(time.time())}.xlsx"
        target = _target_for(fname_hint)
        try:
            try:
                file_obj.seek(0)
            except Exception:
                pass
            with open(target, "wb") as out:
                data = file_obj.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                out.write(data)
            print(f"[save_uploaded_to_root] saved file-like -> {target}")
            return target
        except Exception as e:
            raise Exception(f"Failed to write file-like object: {e}")

    # Case: path string
    if isinstance(file_obj, str):
        p = Path(file_obj)
        if p.exists():
            target = _target_for(p.name)
            shutil.copy(p, target)
            print(f"[save_uploaded_to_root] copied path string {p} -> {target}")
            return target
        else:
            raise Exception(f"String provided but path does not exist: {file_obj}")

    # Case: raw bytes
    if isinstance(file_obj, (bytes, bytearray)):
        target = _target_for(f"uploaded_{int(time.time())}.xlsx")
        with open(target, "wb") as out:
            out.write(bytes(file_obj))
        print(f"[save_uploaded_to_root] wrote raw bytes -> {target}")
        return target

    # Unknown shape
    raise Exception(f"Unsupported uploaded object shape: {type(file_obj)}. See server log for details.")


# ---------------- PLOTTING ----------------
def build_pillar_sheetmap(processed):
    pillar_map = {p: [] for p in PILLARS}
    for sheet in processed.keys():
        p = sheet_belongs_to_pillar(sheet)
        if p:
            pillar_map[p].append(sheet)
    return pillar_map

def build_radar_for_snapshots(processed, regions, snapshot_dates, pad_rel=0.02, abs_pad=0.01, debug=False):
    """
    Build 3 polar subplots (Valuations / Profitability / Leverage) using cyclical values
    from processed (per-sheet results). snapshot_dates is a list of datetimes (or strings).
    Use nearest-index lookup row for each requested snapshot date.
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

    # Precompute per-sheet helpers: res, dates_index (pd.DatetimeIndex), cyc_map (region -> cyc_col or None)
    sheet_helpers = {}
    for sh, obj in processed.items():
        res = obj.get('results')
        if res is None or 'date' not in res.columns:
            continue
        try:
            dates_index = pd.to_datetime(res['date'])
        except Exception:
            # fallback to numpy datetime64 array converted to DatetimeIndex
            try:
                dates_index = pd.DatetimeIndex(np.array(res['date'], dtype='datetime64[ns]'))
            except Exception:
                dates_index = pd.DatetimeIndex([])
        declared_map = obj.get('region_col_map', {}) or {}
        cyc_map = {}
        cols = list(res.columns)
        cols_lower_map = {c.lower(): c for c in cols}

        for r in regions:
            raw_col = declared_map.get(r)
            if raw_col and raw_col in res.columns:
                # look for cyclical/cycle variants
                candidates = [
                    f"{raw_col} cyclical", f"{raw_col} cycle",
                    f"{raw_col}_cyclical", f"{raw_col}_cycle",
                    f"{raw_col}.cyclical", f"{raw_col}.cycle"
                ]
                found = None
                for cand in candidates:
                    if cand.lower() in cols_lower_map:
                        found = cols_lower_map[cand.lower()]; break
                if found:
                    cyc_map[r] = found
                    continue

            # try common raw names to then detect cyc column variants
            found_raw = None
            for cand in (f"equity {r}", f"equity_{r}", r, r.upper(), r.lower()):
                lc = str(cand).lower()
                if lc in cols_lower_map:
                    found_raw = cols_lower_map[lc]; break
            if found_raw:
                candidates = [
                    f"{found_raw} cyclical", f"{found_raw} cycle",
                    f"{found_raw}_cyclical", f"{found_raw}_cycle"
                ]
                found = None
                for cand in candidates:
                    if cand.lower() in cols_lower_map:
                        found = cols_lower_map[cand.lower()]; break
                if found:
                    cyc_map[r] = found
                    continue

            # in last resort search any column containing both token and cyc/cycle
            token = r.lower()
            found = None
            for c in cols:
                cl = c.lower()
                if (('cyc' in cl) or ('cycle' in cl)) and (token in cl):
                    found = c; break
            cyc_map[r] = found

        sheet_helpers[sh] = {'res': res, 'dates_index': dates_index, 'cyc_map': cyc_map}

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
                    cyc_col = helper['cyc_map'].get(r)

                    if not cyc_col or cyc_col not in res.columns:
                        continue

                    # robust nearest-date lookup using pandas
                    try:
                        if len(dates_index) == 0:
                            continue
                        # get nearest positional index 
                        pos_idx = dates_index.get_indexer([pd.to_datetime(dt)], method='nearest')[0]
                        if pos_idx == -1:
                            pos_idx = 0
                        pos_idx = max(0, min(int(pos_idx), len(dates_index) - 1))
                        try:
                            v = res.iloc[pos_idx].get(cyc_col, np.nan)
                        except Exception:
                            try:
                                row_label = res.index[pos_idx] if pos_idx < len(res.index) else res.index[-1]
                                v = res.loc[row_label].get(cyc_col, np.nan)
                            except Exception:
                                v = np.nan
                    except Exception:
                        try:
                            dates_np = np.array(dates_index.values, dtype='datetime64[ns]')
                            pos = int(np.searchsorted(dates_np, np.datetime64(pd.to_datetime(dt))))
                            pos = max(0, min(pos, len(dates_np) - 1))
                            try:
                                v = res.iloc[pos].get(cyc_col, np.nan)
                            except Exception:
                                row_label = res.index[pos] if pos < len(res.index) else res.index[-1]
                                v = res.loc[row_label].get(cyc_col, np.nan)
                            pos_idx = pos
                        except Exception:
                            v = np.nan
                            pos_idx = None

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
                            debug_mappings.append((sh, r, cyc_col, pos_idx, None if pd.isna(v) else float(v)))
                        except Exception:
                            debug_mappings.append((sh, r, cyc_col, pos_idx if 'pos_idx' in locals() else None, None))

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
    Custom series plot with stable per-region colors.
    - processed: dict from process_file_core
    - sheet_name: sheet key to plot
    - selected_regions: list of region codes (e.g. ['us','emu'])
    - display_mode: 'trend+cycle', 'trend', or 'cycle'
    - debug: prints mapping decisions when True
    """
    import plotly.express as px
    import plotly.graph_objects as go

    if not sheet_name or sheet_name not in processed:
        if debug: print("[custom_plot] no sheet or sheet not in processed:", sheet_name)
        return go.Figure()

    res = processed[sheet_name].get('results')
    if res is None or res.empty or 'date' not in res.columns:
        if debug: print("[custom_plot] results missing or empty for sheet:", sheet_name)
        return go.Figure()

    # normalize selected regions (lowercase)
    sel_regs = [r for r in (selected_regions or []) if r is not None]
    sel_regs = [str(r).lower() for r in sel_regs]
    if not sel_regs:
        if debug: print("[custom_plot] no regions selected")
        return go.Figure()

    # choose palette (feel free to change to Tableau20, Plotly, D3, Category10, etc.)
    palette = px.colors.qualitative.Dark24
    # stable mapping region -> color (keeps order of sel_regs)
    region_colors = {r: palette[i % len(palette)] for i, r in enumerate(sel_regs)}

    cols_lower_map = {c.lower(): c for c in res.columns}
    region_map = processed[sheet_name].get('region_col_map', {}) or {}
    region_map_lc = {k.lower(): v for k, v in region_map.items() if k is not None}

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
        for c in res.columns:
            cl = c.lower()
            if all(tok in cl for tok in tokens):
                return c
        return None

    fig = go.Figure()

    if debug:
        print(f"[custom_plot] building plot for sheet='{sheet_name}', regions={sel_regs}, display_mode={display_mode}")
        print("[custom_plot] available columns:", list(res.columns))
        print("[custom_plot] region_map (orig):", region_map)
        print("[custom_plot] region_map_lc:", region_map_lc)
        print("[custom_plot] region_colors:", region_colors)

    for r in sel_regs:
        token = r.lower()
        color = region_colors.get(r, None)

        # 1) find raw column
        raw_col = None
        if token in region_map_lc:
            cand = region_map_lc[token]
            if cand in res.columns:
                raw_col = cand
        if raw_col is None:
            candidates = [f"equity {token}", f"equity_{token}", f"{token} equity", token]
            raw_col = first_match_from_candidates(candidates)
        if raw_col is None:
            for c in res.columns:
                if token in c.lower():
                    raw_col = c
                    break

        # 2) find trend and cyc columns
        trend_col = find_col_by_tokens(['trend', token]) or find_col_by_tokens([token, 'trend'])
        if not trend_col and raw_col:
            trend_col = first_match_from_candidates([f"{raw_col} trend", f"{raw_col}_trend", f"{raw_col}.trend"])

        cyc_col = find_col_by_tokens(['cyc', token]) or find_col_by_tokens(['cycle', token])
        if not cyc_col and raw_col:
            cyc_col = first_match_from_candidates([f"{raw_col} cyclical", f"{raw_col} cycle", f"{raw_col}_cyclical", f"{raw_col}_cycle"])

        if debug:
            print(f"[custom_plot] region={r} -> raw_col={raw_col}, trend_col={trend_col}, cyc_col={cyc_col}, color={color}")

        # fetch series safely
        def safe_get(col):
            if col and col in res.columns:
                try:
                    return res[col].astype(float)
                except Exception:
                    return res[col]
            return None

        y_raw = safe_get(raw_col)
        y_trend = safe_get(trend_col)
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

        name_label = r.upper()
        legendgroup = name_label  # group traces by region in legend

        # plot traces using the region color; make fill/opacity style vary for trend+cycle
        if display_mode == 'trend':
            if y_trend is not None:
                fig.add_trace(go.Scatter(
                    x=res['date'], y=y_trend, mode='lines', name=f"{name_label} - trend",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(
                    x=res['date'], y=y_raw, mode='lines', name=f"{name_label} - raw (no trend)",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
        elif display_mode == 'cycle':
            if y_cyc is not None:
                fig.add_trace(go.Scatter(
                    x=res['date'], y=y_cyc, mode='lines', name=f"{name_label} - cycle",
                    line=dict(color=color, width=2, dash='dash'), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(
                    x=res['date'], y=y_raw, mode='lines', name=f"{name_label} - raw (no cycle)",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
        else:  # 'trend+cycle'
            if y_trend is not None and y_cyc is not None:
                base = y_trend
                top = y_trend + y_cyc
                # trend (lower area)
                fig.add_trace(go.Scatter(
                    x=res['date'], y=base, mode='lines', name=f"{name_label} - trend",
                    line=dict(color=color, width=1), fill='tozeroy', opacity=0.35,
                    legendgroup=legendgroup
                ))
                # trend+cycle (upper area)
                fig.add_trace(go.Scatter(
                    x=res['date'], y=top, mode='lines', name=f"{name_label} - trend+cycle",
                    line=dict(color=color, width=1), fill='tonexty', opacity=0.6,
                    legendgroup=legendgroup
                ))
            elif y_raw is not None:
                fig.add_trace(go.Scatter(
                    x=res['date'], y=y_raw, mode='lines', name=f"{name_label} - raw (fallback)",
                    line=dict(color=color, width=2), marker=dict(color=color),
                    legendgroup=legendgroup
                ))
            else:
                if debug:
                    print(f"[custom_plot] nothing available to plot for region {r}")

    # Ensure legend items use the trace colors we set; optionally set a global colorway
    try:
        fig.update_layout(colorway=px.colors.qualitative.Dark24)
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

/* Buttons look */
#load_template_btn .gr-button { background:#e9e9e9; color:#222; height:48px; font-weight:600; }
#download_template_btn .gr-button { background:#f5821f; color:white; height:48px; font-weight:600; }
#process_uploaded_btn .gr-button { background:#1f77b4; color:white; height:44px; font-weight:600; }

/* Layout helpers */
.app_header { display:flex; align-items:center; }
.template_col { padding-left:18px; padding-right:18px; }
.logo_img img { max-width:220px; }

/* Drop box outer (keeps height, border, centering) */
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

/* make dropzone positioned for absolute children */
#drop_file_box .file-dropzone { position: relative !important; }

/* small top-left pill/button inside dropzone */
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

/* pill icon sizing */
#drop_file_box .file-dropzone .file-dropzone-button svg,
#drop_file_box .file-dropzone .file-dropzone-label svg,
#drop_file_box .file-dropzone .file-upload svg {
  width: var(--dropbox-pill-icon) !important;
  height: var(--dropbox-pill-icon) !important;
}

/* central dropzone text: prefer targeting the real text node(s) */
.file-drop-text,
.file-dropzone .file-drop-text,
.drop_area .file-drop-text,
#drop_file_box .file-dropzone .file-drop-text,
#drop_file_box .file-dropzone .file-dropzone-label {
  font-size: 12px !important;  /* change to 10px/11px/14px as you prefer */
  line-height: 1.2 !important;
  color: #666 !important;
  white-space: normal !important;
}

/* Buttons consistent spacing */
#download_processed_btn .gr-button,
#process_uploaded_btn .gr-button,
#download_template_btn .gr-button,
#load_template_btn .gr-button {
  width: 100% !important;
  box-sizing: border-box;
  margin-top: 12px !important;
}

/* left preview area constraints */
#template_file_box .file-dropzone,
#template_file_box .file-list,
#template_file_box .file-upload,
#template_file_box .file-card,
#template_file_box .gr-file {
  max-height: 140px !important;
  overflow: auto !important;
  padding: 8px !important;
}

/* small tweaks */
.gradio-container .gr-button { border-radius:6px; }
.gradio-container .gr-dropdown { min-width:260px; }

/* ===== Fallbacks ===== */

#drop_file_box, #drop_file_box *,
.drop_area, .drop_area * {
  font-size: 12px !important;
  line-height: 1.2 !important;
}

"""


with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.HTML("""
    <div style="width:100%; text-align:center;">
      <h1 style="margin:0; font-size:36px; font-weight:800; line-height:1.05;">Regional Equities Cycle Visualisation</h1>
      <div style="color:#666; margin-top:6px; font-size:14px;">HP filter calculation and time-series analysis </div>
    """)

    

    with gr.Row(elem_classes="app_header"):
        # Three equal columns: left=logo, middle=template chooser, right=dropzone + process/download
        with gr.Column(scale=1, elem_classes="logo_img"):
            # larger logo in left column
            logo_html_str = logo_html(DEFAULT_LOGO, width=320)
            logo = gr.HTML(value=logo_html_str, elem_id="app_logo_html")

        with gr.Column(scale=1, elem_classes="template_col"):
            gr.Markdown("### Choose template")
            template_dropdown = gr.Dropdown(
                choices=TEMPLATE_NAMES or [DEFAULT_XLSX.name],
                value=(TEMPLATE_NAMES[0] if TEMPLATE_NAMES else DEFAULT_XLSX.name),
                label=None
            )
            load_template_btn = gr.Button("2) Load Template", elem_id="load_template_btn")
            download_template_btn = gr.DownloadButton(
                "Download current template",
                elem_id="download_template_btn"
            )

        with gr.Column(scale=1, elem_classes="upload_col"):
            gr.Markdown("### Upload Template (drag & drop .xlsx)")
            # drag & drop area (big dashed box)
            file_in = gr.File(label=None, file_types=['.xlsx'],
                  elem_classes="drop_area", elem_id="drop_file_box")
            # Process uploaded file (uses file_in)
            process_uploaded_btn = gr.Button("1) Process uploaded file", elem_id="process_uploaded_btn")
            # Download processed data (below process button)
            download_processed_btn = gr.Button("Download processed data", elem_id="download_processed_btn")
            # HTML box where processed download link / filename will be shown
            download_processed_html = gr.HTML("", visible=True)


    # Remaining UI components
    status_box = gr.Textbox(label="Status", interactive=False, value="Ready")
    snapshot_selector = gr.CheckboxGroup(label="Choose snapshot dates (for radar) (up to 3)", choices=[], visible=False)
    radar_output = gr.Plot(label="Radar (Valuations / Profitability / Leverage)", visible=False)
    agg_valuations = gr.Plot(label="Aggregate - Valuations", visible=False, elem_id="agg_valuations")
    agg_profitability = gr.Plot(label="Aggregate - Profitability", visible=False, elem_id="agg_profitability")
    agg_leverage = gr.Plot(label="Aggregate - Leverage", visible=False, elem_id="agg_leverage")
    sheet_selector = gr.Dropdown(label="Select sheet (series)", choices=[], visible=False)
    region_check = gr.CheckboxGroup(choices=REGIONS, label="Filter by region(s) (for custom plot)", visible=False)
    custom_plot = gr.Plot(label="Custom selected variables", visible=False, elem_id="custom_plot")
    plot_mode = gr.Radio(
    choices=['trend+cycle', 'trend', 'cycle'],
    value='trend+cycle',
    label="Show",
    visible=True
    )

    state = gr.State({})

    # safe wrapper that returns status + outputs
    def safe_handle_process(input_file_or_path):
        try:
            res = process_file_core(input_file_or_path)

            # --- ADDED: handle the case where process_file_core returns None ---
            if res is None:
                msg = f"Error: processing returned no result (process_file_core returned None). Input: {repr(input_file_or_path)}"
                print(msg)
                # return status + 8 invisible placeholders + None state
                return (gr.update(value=msg),) + tuple([gr.update(visible=False)] * 8) + (None,)
            # ------------------------------------------------------------------

            if 'error' in res:
                msg = f"Error: {res['error']}"
                print(msg)
                # return status + 8 invisible placeholders + None state
                return (gr.update(value=msg),) + tuple([gr.update(visible=False)] * 8) + (None,)
            st = {
                "processed": res['processed'],
                "excel_bytes": res['excel_bytes'],
                "sheet_names": res['sheet_names'],
                "vars_per_sheet": res['vars_per_sheet'],
                "regions": res['regions'],
                "mapping_report": res['mapping_report']
            }

            default_sheet = st['sheet_names'][1]
            results_df = st['processed'][default_sheet]['results']

            # all distinct dates from the results, as datetimes
            all_dates_dt = sorted(pd.to_datetime(results_df['date'].unique()))

            if all_dates_dt:
                latest = all_dates_dt[-1]
                recent_start = latest - relativedelta(months=11)
                recent_dates = [d for d in all_dates_dt if d >= recent_start]
                older_decembers = [d for d in all_dates_dt if d < recent_start and d.month == 12]
                combined = sorted(older_decembers) + sorted(recent_dates)
                uniq_dates = [d.strftime('%Y-%m-%d') for d in combined]
            else:
                uniq_dates = []

            # Default selection: keep last 3 chronologically (most recent)
            default_snapshots = uniq_dates[-3:] if uniq_dates else []

            radar_fig = build_radar_for_snapshots(st['processed'], st['regions'], [pd.to_datetime(d) for d in default_snapshots])
            pillar_aggs = build_pillar_agg_timeseries(st['processed'], st['regions'])
            pre_regions = st['regions']

            msg = f"Loaded {len(st['sheet_names'])} sheets from template. Default sheet: {default_sheet}"
            return (
                gr.update(value=msg),
                gr.update(visible=True, choices=uniq_dates, value=default_snapshots[:3]),
                gr.update(visible=True, value=radar_fig),
                gr.update(visible=True, value=pillar_aggs.get('Valuations', go.Figure())),
                gr.update(visible=True, value=pillar_aggs.get('Profitability', go.Figure())),
                gr.update(visible=True, value=pillar_aggs.get('Leverage', go.Figure())),
                gr.update(visible=True, choices=st['sheet_names'], value=default_sheet),
                gr.update(visible=True, choices=st['regions'], value=pre_regions),
                gr.update(visible=True, value=build_custom_series_plot(st['processed'], default_sheet, pre_regions)),
                st
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            return (gr.update(value=f"Exception: {e}. See server log."),) + tuple([gr.update(visible=False)] * 8) + (None,)

    # Load template button -> uses selected template from dropdown
    def on_load_template(name):
        path = TEMPLATE_MAP.get(name) if name in TEMPLATE_MAP else (str(DEFAULT_XLSX) if DEFAULT_XLSX.exists() else None)
        if path is None:
            return (gr.update(value="Template file not found."),) + tuple([gr.update(visible=False)] * 8) + (None,)
        return safe_handle_process(path)

    load_template_btn.click(
        on_load_template,
        inputs=[template_dropdown],
        outputs=[status_box, snapshot_selector, radar_output, agg_valuations, agg_profitability, agg_leverage,
                sheet_selector, region_check, custom_plot, state]
    )

    # Download current template -> returns the file path to download_template_out
    def on_download_template(name):
        p = TEMPLATE_MAP.get(name) if name in TEMPLATE_MAP else (
            str(DEFAULT_XLSX) if DEFAULT_XLSX.exists() else None
        )
        if not p or not os.path.exists(p):
            raise gr.Error("Template file not found.")
        return p

    download_template_btn.click(
        fn=on_download_template,
        inputs=[template_dropdown],
        outputs=[download_template_btn]
    )


    # Download processed data (results per sheet) -> returns a path for download_processed_out
    def on_download_processed(state_dict):
        """
        Build an in-memory Excel workbook containing each processed sheet's 'results'
        with columns ordered as:
        date, <raw US>, <raw EMU>, <raw UK>, <raw JAP>, <raw PAC>, <raw GEM>,
            <trend US>, <trend EMU>, <trend UK>, <trend JAP>, <trend PAC>, <trend GEM>.
        Cycle/cyclical columns are intentionally NOT written to the export file
        (they remain in-memory for plotting).
        Returns an HTML data-uri anchor (gr.update) so the browser can download the file.
        """
        if not state_dict or 'processed' not in state_dict:
            return gr.update(value="<div style='color:red'>No processed data available</div>", visible=True)

        processed = state_dict['processed']
        if not processed:
            return gr.update(value="<div style='color:red'>Processed dict empty</div>", visible=True)

        suffixes = ['us', 'emu', 'uk', 'jap', 'pac', 'gem']

        def find_first_col(candidates, columns_index):
            lower_to_col = {col.lower(): col for col in columns_index}
            for cand in candidates:
                if cand is None:
                    continue
                key = cand.lower()
                if key in lower_to_col:
                    return lower_to_col[key]
            return None

        out_buf = io.BytesIO()
        try:
            with pd.ExcelWriter(out_buf, engine='openpyxl') as writer:
                for sheet_name, obj in processed.items():
                    results = obj.get('results')
                    if results is None or results.empty:
                        # write empty sheet
                        pd.DataFrame().to_excel(writer, sheet_name=(sheet_name[:31] if sheet_name else "sheet"), index=False)
                        continue

                    columns_index = list(results.columns)
                    ordered_cols = []

                    # date first if present
                    date_col = find_first_col(['date', 'Date', 'DATE'], columns_index)
                    if date_col:
                        ordered_cols.append(date_col)

                    # Use the sheet's region_col_map (region -> raw column name) as primary source
                    region_map = obj.get('region_col_map', {}) or {}
                    region_map_lc = {k.lower(): v for k, v in region_map.items() if k is not None}

                    # 1) append all raw columns in desired suffix order, using region_map when possible
                    found_raw = {}
                    for s in suffixes:
                        raw_name = None
                        # prefer explicit mapping from region_col_map
                        if s in region_map_lc:
                            candidate = region_map_lc[s]
                            if candidate in results.columns:
                                raw_name = candidate
                            else:
                                # sometimes mapping stored but column slightly differs in case -> try case-insensitive match
                                lower_to_col = {col.lower(): col for col in columns_index}
                                if candidate.lower() in lower_to_col:
                                    raw_name = lower_to_col[candidate.lower()]
                        # fallback: try to find a column that contains the region token (common patterns)
                        if raw_name is None:
                            for col in columns_index:
                                cl = col.lower()
                                # match "... us", "..._us", "...-us", or ending with the suffix
                                if (cl.endswith(f" {s}") or cl.endswith(f"_{s}") or cl.endswith(f"-{s}") or cl.endswith(f".{s}") or cl.split()[-1] == s):
                                    raw_name = col
                                    break
                            # last resort: any column where token occurs and it's not a 'trend'/'cyc' name
                            if raw_name is None:
                                for col in columns_index:
                                    cl = col.lower()
                                    if s in cl and ('trend' not in cl) and ('cyc' not in cl) and ('cycle' not in cl):
                                        raw_name = col
                                        break

                        if raw_name and raw_name not in ordered_cols:
                            ordered_cols.append(raw_name)
                            found_raw[s] = raw_name

                    # 2) then append trend columns in the same region order
                    # trend variants to try (prefer "<raw> trend")
                    for s in suffixes:
                        raw_name = found_raw.get(s)
                        cand = None
                        if raw_name:
                            candidates = [
                                f"{raw_name} trend",
                                f"{raw_name}_trend",
                                f"{raw_name}.trend",
                                f"trend {s}",
                                f"{s} trend",
                                f"trend_{s}",
                                f"{raw_name} Trend"
                            ]
                            cand = find_first_col(candidates, columns_index)
                        else:
                            # raw not found: try any trend columns that include the suffix token
                            candidates = [f"trend {s}", f"{s} trend", f"trend_{s}", f"trend-{s}"]
                            cand = find_first_col(candidates, columns_index)
                            if not cand:
                                # fallback: any column containing both 'trend' and the region token
                                for col in columns_index:
                                    cl = col.lower()
                                    if 'trend' in cl and s in cl:
                                        cand = col
                                        break

                        if cand and cand not in ordered_cols:
                            ordered_cols.append(cand)

                    # Filter out cycle/cyclical columns from export
                    export_cols = [c for c in ordered_cols if c in results.columns]
                    export_cols = [c for c in export_cols if not re.search(r'\b(cyc|cycle|cyclical)\b', c, flags=re.IGNORECASE)]

                    # If nothing to reorder (only date or nothing), export sheet as-is
                    min_keep = 1 if date_col else 0
                    if len(export_cols) <= min_keep:
                        results.to_excel(writer, sheet_name=(sheet_name[:31] if sheet_name else "sheet"), index=False)
                    else:
                        export_df = results[export_cols].copy()
                        export_df.to_excel(writer, sheet_name=(sheet_name[:31] if sheet_name else "sheet"), index=False)

            out_buf.seek(0)
            data_bytes = out_buf.read()
        except Exception as e:
            print("on_download_processed build error:", e)
            return gr.update(value=f"<div style='color:red'>Unable to prepare file: {e}</div>", visible=True)

        try:
            b64 = base64.b64encode(data_bytes).decode("ascii")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_export_{timestamp}.xlsx"
            href = f'<a download="{filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">Download processed Excel ({filename})</a>'
            return gr.update(value=href, visible=True)
        except Exception as e:
            print("on_download_processed encoding error:", e)
            return gr.update(value=f"<div style='color:red'>Unable to prepare download link: {e}</div>", visible=True)

    
    
    # Process uploaded file

    def on_process_uploaded(file_obj):
        if not file_obj:
            return (gr.update(value="No file uploaded."),) + tuple([gr.update(visible=False)] * 8) + (None, gr.update(visible=True, choices=TEMPLATE_NAMES))

        try:
            saved_path = save_uploaded_to_root(file_obj)
            print(f"[on_process_uploaded] saved uploaded file to: {saved_path}")
        except Exception as e:
            tb = traceback.format_exc()
            print("[on_process_uploaded] save error:", tb)
            return (gr.update(value=f"Failed to save uploaded file: {e}. See server log for details."),) + tuple([gr.update(visible=False)] * 8) + (None, gr.update(visible=True, choices=TEMPLATE_NAMES))

        saved_name = saved_path.name
        TEMPLATE_MAP[saved_name] = str(saved_path.resolve())
        if saved_name not in TEMPLATE_NAMES:
            TEMPLATE_NAMES.append(saved_name)
        try:
            result_tuple = safe_handle_process(str(saved_path))
        except Exception as e:
            tb = traceback.format_exc()
            print("[on_process_uploaded] processing error:", tb)
            return (gr.update(value=f"Exception processing uploaded file: {e}. See server log."),) + tuple([gr.update(visible=False)] * 8) + (None, gr.update(visible=True, choices=TEMPLATE_NAMES))

        # Append update for template_dropdown so new file appears and is selected
        return result_tuple + (gr.update(visible=True, choices=TEMPLATE_NAMES, value=saved_name),)


    process_uploaded_btn.click(
        on_process_uploaded,
        inputs=[file_in],
        outputs=[status_box, snapshot_selector, radar_output, agg_valuations, agg_profitability, agg_leverage,
                sheet_selector, region_check, custom_plot, state, template_dropdown]
    )
    download_processed_btn.click(on_download_processed, inputs=[state], outputs=[download_processed_html])

    # interactions: update charts when sheet/snapshot/region choices change
    def on_sheet_change_custom(sheet_name, selected_regions, st, display_mode):
        try:
            if not st or 'processed' not in st or not sheet_name or sheet_name not in st['processed']:
                return go.Figure()
            return build_custom_series_plot(st['processed'], sheet_name, selected_regions or [], display_mode or 'trend+cycle')
        except Exception:
            print("[on_sheet_change_custom] exception:", traceback.format_exc())
            return go.Figure()


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


    region_check.change(
        lambda sh, regs, st, mode: build_custom_series_plot(st['processed'], sh, regs or [], mode)
            if (st and 'processed' in st and sh in st['processed']) else go.Figure(),
        inputs=[sheet_selector, region_check, state, plot_mode],
        outputs=[custom_plot]
    )

    plot_mode.change(
    lambda sh, regs, st, mode: build_custom_series_plot(st['processed'], sh, regs or [], mode)
        if (st and 'processed' in st and sh in st['processed']) else go.Figure(),
    inputs=[sheet_selector, region_check, state, plot_mode],
    outputs=[custom_plot]
)

    # snapshot selector updates radar + aggregates
    snapshot_selector.change(
        on_snapshot_change,
        inputs=[snapshot_selector, state],
        outputs=[radar_output, agg_valuations, agg_profitability, agg_leverage]
    )


    sheet_selector.change(
        on_sheet_change_custom,
        inputs=[sheet_selector, region_check, state, plot_mode],
        outputs=[custom_plot]
    )

    region_check.change(
        lambda sh, regs, st, mode: build_custom_series_plot(st['processed'], sh, regs or [], mode)
            if (st and 'processed' in st and sh in st['processed']) else go.Figure(),
        inputs=[sheet_selector, region_check, state, plot_mode],
        outputs=[custom_plot]
    )


    plot_mode.change(
        lambda sh, regs, st, mode: build_custom_series_plot(st['processed'], sh, regs or [], mode)
            if (st and 'processed' in st and sh in st['processed']) else go.Figure(),
        inputs=[sheet_selector, region_check, state, plot_mode],
        outputs=[custom_plot]
    )

# Launch
port = 9115
demo.launch(
    debug=True,
    share=False,
    server_port=port,
    server_name='0.0.0.0',
    root_path=f'/studio/workspace/vscode/proxy/{os.getenv("ALTO_STUDIO_USERNAME")}/{port}',
)

