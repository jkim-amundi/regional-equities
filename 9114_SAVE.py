import gradio as gr
import pandas as pd
import lightgbm as lgb
import plotly.graph_objects as go
import os
import re
import pycountry
import shap
import numpy as np
import datetime
import calendar
from functools import partial
import shutil
import base64
from tabulate import tabulate
from pathlib import Path


# -------------------------
# Config / constants
# -------------------------

# Folder where this script lives
APP_DIR = Path(__file__).resolve().parent

# Files live next to the script
DATA_FOLDER = APP_DIR
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

LOGO_PATH = DATA_FOLDER / "Logo_Amundi_investment_solutions_4c.jpg"
MODEL_PATH = DATA_FOLDER / "LTR_EMU_UK_Quarterly_EPU.txt"
DEBUG_PATH = DATA_FOLDER / "debug.txt"


DISPLAY_DECIMALS = 5

region_features = [
    'pe', 'pbv', 'ps', 'pebitda', 'roe', 'roa', 'roc', 'om', 'pm', 'dy', 'em',
    'nde', 'tdte', 'tdta', 'eps12m'
]

global_features = [
    'surprise_US', 'surprise_EURO', 'surprise_UK', 'surprise_JAP',
    'surprise_PAC', 'surprise_GEM', 'surprise_CHINA', 'surprise_AUS',
    'prob_risk_off', 'prob_inflation', 'world_trade',
    'EPU_global', 'EPU_US', 'EPU_Europe', 'EPU_UK', 'EPU_JAP',
    'EPU_Australia', 'EPU_China'
]

feature_cols = [
    'pe', 'pbv', 'ps', 'pebitda', 'roe', 'roa', 'roc', 'om', 'pm', 'dy', 'em',
    'nde', 'tdte', 'tdta', 'eps12m',
    'surprise_US', 'surprise_EURO', 'surprise_UK', 'surprise_JAP',
    'surprise_PAC', 'surprise_GEM', 'surprise_CHINA', 'surprise_AUS',
    'prob_risk_off', 'prob_inflation', 'world_trade',
    'EPU_global', 'EPU_US', 'EPU_Europe', 'EPU_UK', 'EPU_JAP',
    'EPU_Australia', 'EPU_China'
]

with open(DEBUG_PATH, "w") as f:
    f.write("Script started!\n")

feature_fullname = {
    'pe':'Price to Earnings','pbv':'Price to Book','ps':'Price to Sales','pebitda':'Price to EBITDA',
    'roe':'Return on Equity','roa':'Return on Assets','roc':'Return on Capital','om':'Operating Margin',
    'pm':'Profit Margin','dy':'Dividend Yield','em':'Earnings Momentum','nde':'Net Debt to EBITDA',
    'tdte':'Debt to Equity','tdta':'Debt to Assets','eps12m':'12m EPS',
    'surprise_US':'US Economic Surprise','surprise_EURO':'EU Economic Surprise','surprise_JAP':'JP Economic Surprise',
    'surprise_PAC':'PAC Economic Surprise','surprise_GEM':'GEM Economic Surprise', 'surprise_UK' : 'UK Economic Surprise',
    'surprise_CHINA': 'China Economic Surprise', 'surprise_AUS' : 'Australia Economic Surprise',
    'world_trade':'World Trade','prob_inflation':'Prob. Inflation','prob_risk_off':'Prob. Risk-Off','EPU_global' : 'Global Economic Policy Uncertainty','EPU_US': 'US Economic Policy Uncertainty',
    'EPU_Europe':'Europe Economic Policy Uncertainty' , 'EPU_UK' : 'UK Economic Policy Uncertainty',
    'EPU_JAP': 'Japan Economic Policy Uncertainty' , 'EPU_Australia' : 'Australia Economic Policy Uncertainty',
    'EPU_China' : 'China Economic Policy Uncertainty'
}

feature_categories = {
    'pe':'valuation','pbv':'valuation','ps':'valuation','pebitda':'valuation',
    'roe':'profitability','roa':'profitability','roc':'profitability','om':'profitability',
    'pm':'profitability','dy':'profitability','eps12m':'profitability',
    'nde':'leverage','tdte':'leverage','tdta':'leverage',
    'surprise_US':'macro','surprise_EURO':'macro','surprise_UK':'macro','surprise_JAP':'macro',
    'surprise_PAC':'macro','surprise_GEM':'macro','surprise_CHINA':'macro','surprise_AUS':'macro',
    'prob_risk_off':'macro','prob_inflation':'macro','world_trade':'macro',
    'EPU_global':'macro','EPU_US':'macro','EPU_Europe':'macro','EPU_UK':'macro',
    'EPU_JAP':'macro','EPU_Australia':'macro','EPU_China':'macro'
}

category_colors = {'valuation':'#004F9F','profitability':'#39B2B6','leverage':'#E6325E','macro':'#F4A261'}

# -------------------------
# Defaults
# -------------------------
default_data = {
    'region':['US','EMU','UK','Japan','Pacific','GEM'],
    'pe':[0.0535,   0.1032, 0.0773, 0.1275, 0.0640, 0.1343],
    'pbv':[0.0445,  0.0839, 0.0918, 0.1215, 0.0427, 0.1344],
    'ps':[0.0657,   0.0629, 0.1203, 0.1068, 0.0649, 0.1078],
    'pebitda':[0.0683, 0.0681, 0.1069, 0.1264, 0.0614, 0.0908],
    'roe':[0.0011, -0.0010, -0.0002, -0.0033, 0.0035, 0.0083],
    'roa':[-0.0001, -0.0006, -0.0004, -0.0006, 0.0003, 0.0015],
    'roc':[0.0004, -0.0011, -0.0008, -0.0013, 0.0004, 0.0039],
    'om':[0.0031, 0.0010, 0.0031, -0.0011, 0.0090, 0.0028],
    'pm':[0.0033, -0.0036, 0.0011, -0.0030, 0.0006, 0.0017],
    'dy':[-0.0006, -0.0018, -0.0035, -0.0013, -0.0014, -0.0028],
    'em':[0.0007, 0.0013, 0.0052, -0.0020, 0.0124, 0.0053],
    'nde':[0.0261, -0.0233, -0.0879, 0.1383, 0.0778, -0.0631],
    'tdte':[0.0013, 0.0310, 0.0226, 0.0022, 0.0329, -0.0076],
    'tdta':[-0.0072, -0.0029, -0.0108, -0.0051, 0.0194, -0.0041],
    'eps12m':[13.2881, 11.9418, 9.0883, 8.8801, 4.9721, 15.6791]
}
default_df = pd.DataFrame(default_data)

default_globals = {
    'surprise_US':42.3,'surprise_EURO':94.5,'surprise_JAP':62.7,'surprise_PAC':2.9,'surprise_GEM':2.7,
    'surprise_UK':43.9,'surprise_CHINA':-23,'surprise_AUS':24.9,
    'world_trade':0.005,'prob_inflation':0.5,'prob_risk_off':0.5,
    'EPU_global':233.1,'EPU_US':532.4,'EPU_Europe':231.1,'EPU_UK':113.5,'EPU_JAP':134.2,
    'EPU_Australia':47.3,'EPU_China':699.3
}

def _global_value(key):
    return float(default_globals.get(key, 0.0))

def parse_date_from_filename(fname):
    m = re.search(r'(\d{8})(?=\.xlsx?$)', fname, flags=re.IGNORECASE)
    if m:
        s = m.group(1)
        try:
            return datetime.datetime.strptime(s, "%d%m%Y").date()
        except Exception:
            return None
    m = re.search(r'(\d{6})(?=\.xlsx?$)', fname, flags=re.IGNORECASE)
    if m:
        s = m.group(1)
        try:
            return datetime.datetime.strptime(s, "%d%m%y").date()
        except Exception:
            return None
    return None

def is_quarter_end_date(dt):
    if dt is None:
        return False
    return dt.month in (3, 6, 9, 12) and dt.day == calendar.monthrange(dt.year, dt.month)[1]

def next_quarter_label(dt):
    if dt.month == 3:
        return f"Q2 {dt.year}"
    elif dt.month == 6:
        return f"Q3 {dt.year}"
    elif dt.month == 9:
        return f"Q4 {dt.year}"
    elif dt.month == 12:
        return f"Q1 {dt.year + 1}"
    return f"Q? {dt.year}"

def quarter_end_title(dt):
    return f"Predicted Region Ranking for {next_quarter_label(dt)}"

def quarter_end_warning_html(dt=None):
    dt_txt = dt.strftime("%d %b %Y") if dt else "unknown date"
    return (
        "<div style='padding:8px 12px; background:#fff4e5; "
        "border:1px solid #f0ad4e; border-radius:4px; margin:8px 0; font-size:13px'>"
        f"Quarterly predictions are only available at quarter-end "
        f"(31 Mar, 30 Jun, 30 Sep, 31 Dec). Selected file date: <b>{dt_txt}</b>."
        "</div>"
    )

def df_to_compact_html(df, title=None):
    html = df.to_html(index=False, escape=False)
    css = ("<style>"
           ".compact-table{border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px}"
           ".compact-table th,.compact-table td{border:1px solid #999;padding:6px 10px;white-space:nowrap}"
           ".compact-table th{background:#f0f0f0;font-weight:700}"
           "</style>")
    table = html.replace('<table', '<table class="compact-table"')
    title_html = f"<h4 style='margin:0 0 8px 0'>{title}</h4>" if title else ""
    return f"<div style='width:100%;display:flex;justify-content:center;margin:8px 0'><div style='text-align:center'>{css}{title_html}{table}</div></div>"

def build_ranking_html(ranks_df, title="Predicted Region Ranking"):
    return df_to_compact_html(ranks_df, title)

def load_inputs_from_excel(filename):
    folder = DATA_FOLDER
    full_path = os.path.join(folder, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{filename} not found: {filename}")
    sheets = pd.read_excel(full_path, sheet_name=None)
    regions_df = None
    globals_df = None
    for name, df in sheets.items():
        n = name.strip().lower()
        if 'region' in n:
            regions_df = df.copy()
        if 'global' in n:
            globals_df = df.copy()
    if regions_df is None:
        regions_df = list(sheets.values())[0].copy()
    if globals_df is None:
        globals_df = list(sheets.values())[1].copy() if len(sheets) > 1 else pd.DataFrame([{}])

    expected_cols = ['region'] + region_features
    for col in expected_cols:
        if col not in regions_df.columns:
            regions_df[col] = 0.0
    regions_df = regions_df[expected_cols].copy()

    globals_row = globals_df.iloc[0].to_dict() if not globals_df.empty else {}
    for g in global_features:
        globals_row.setdefault(g, default_globals.get(g, 0.0))

    return regions_df, globals_row

def compare_globals(global_features, prev_globals, curr_globals, feature_fullname):
    edits = []
    tol = 10 ** (-DISPLAY_DECIMALS)

    for gf in global_features:
        old_val = prev_globals.get(gf, None)
        new_val = curr_globals.get(gf, None)

        try:
            old_f = float(old_val)
            new_f = float(new_val)

            is_diff = not np.isclose(old_f, new_f, atol=tol)

            old_s = f"{old_f:.{DISPLAY_DECIMALS}f}"
            new_s = f"{new_f:.{DISPLAY_DECIMALS}f}"
        except Exception:
            is_diff = old_val != new_val
            old_s = str(old_val)
            new_s = str(new_val)

        if is_diff:
            label = feature_fullname.get(gf, gf)
            edits.append(f"{gf} ({label}) from {old_s} to {new_s}")

    return edits

def get_templates_and_default(folder=DATA_FOLDER):
    try:
        raw = []
        for root, _, files in os.walk(folder):
            for f in files:
                fl = f.lower()
                if fl.endswith(".xlsx") and (
                    fl.startswith("input_template_data")
                    or fl.startswith("input_data_template")
                ):
                    rel_path = os.path.relpath(os.path.join(root, f), folder)
                    raw.append(rel_path)
    except FileNotFoundError:
        raw = []

    quarter_end_files = []
    for f in raw:
        dt = parse_date_from_filename(os.path.basename(f))
        if dt is not None and is_quarter_end_date(dt):
            quarter_end_files.append((f, dt))

    quarter_end_files = sorted(quarter_end_files, key=lambda x: x[1], reverse=True)

    files_list = [f for f, _ in quarter_end_files]
    default = files_list[0] if files_list else None
    return files_list, default

# Load model
model = lgb.Booster(model_file=str(MODEL_PATH))

def get_features_used_in_splits(model):
    used_features = set()
    model_dict = model.dump_model()
    for tree in model_dict.get('tree_info', []):
        def traverse(node):
            if not node:
                return
            if 'split_feature' in node:
                idx = node['split_feature']
                used_features.add(model_dict['feature_names'][idx])
                traverse(node.get('left_child'))
                traverse(node.get('right_child'))
        traverse(tree.get('tree_structure'))
    return used_features

used_features = get_features_used_in_splits(model)

def create_map(region_ranks_df):
    region_ranks_df = region_ranks_df.copy()
    region_ranks_df['region_norm'] = region_ranks_df['region'].astype(str).str.strip().str.lower()
    region_rank_dict = dict(zip(region_ranks_df['region_norm'], region_ranks_df['predicted_rank']))

    us_countries = ['United States']
    emu_countries = ['Austria','Belgium','Denmark','Finland','France','Germany','Ireland','Italy','Luxembourg','Netherlands','Norway','Poland','Portugal','Spain','Sweden','Switzerland']
    uk_countries = ['United Kingdom']
    japan_countries = ['Japan']
    pacific_countries = ['Australia','Hong Kong','New Zealand','Singapore']
    gem_countries = ['Brazil','Chile','China','Colombia','Czech Republic','Egypt','Greece','Hungary','India','Indonesia','Korea','Kuwait','Malaysia','Mexico','Peru','Philippines','Poland','Qatar','Saudi Arabia','South Africa','Taiwan','Thailand','Turkey','United Arab Emirates']

    country_to_region = {}
    for c in us_countries:
        country_to_region[c] = 'us'
    for c in emu_countries:
        country_to_region[c] = 'emu'
    for c in uk_countries:
        country_to_region[c] = 'uk'
    for c in japan_countries:
        country_to_region[c] = 'japan'
    for c in pacific_countries:
        country_to_region[c] = 'pacific'
    for c in gem_countries:
        if c not in country_to_region:
            country_to_region[c] = 'gem'

    rows = [{'country': country, 'region_norm': rn, 'rank': region_rank_dict.get(rn, None)} for country, rn in country_to_region.items()]
    df = pd.DataFrame(rows)

    def iso3(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except Exception:
            special = {
                'South Korea': 'KOR', 'Korea': 'KOR', 'Taiwan': 'TWN', 'Czech Republic': 'CZE',
                'United Arab Emirates': 'ARE', 'South Africa': 'ZAF', 'Saudi Arabia': 'SAU',
                'Hong Kong': 'HKG', 'United States': 'USA', 'United Kingdom': 'GBR', 'Netherlands': 'NLD'
            }
            return special.get(name, None)

    df['iso_alpha'] = df['country'].apply(iso3)
    df = df[df['iso_alpha'].notnull()].copy()

    blue_iso = df[df['rank'].isin([1, 2])]['iso_alpha'].tolist()
    white_iso = df[df['rank'].isin([3, 4])]['iso_alpha'].tolist()
    red_iso = df[df['rank'].isin([5, 6])]['iso_alpha'].tolist()

    def make_trace(isos, color_hex, name):
        if not isos:
            return None
        sub = df[df['iso_alpha'].isin(isos)]
        return go.Choropleth(
            locations=sub['iso_alpha'], z=[1] * len(sub),
            text=sub['country'] + '<br>Rank: ' + sub['rank'].astype(str),
            colorscale=[[0, color_hex], [1, color_hex]], showscale=False,
            marker_line_color='black', marker_line_width=0.5, name=name, hoverinfo='text'
        )

    fig = go.Figure()
    fig.add_trace(go.Choropleth(locations=[], z=[], colorscale=[[0, 'lightgray'], [1, 'lightgray']], showscale=False, hoverinfo='skip'))
    for tr in [
        make_trace(blue_iso, 'rgb(0,28,75)', 'Rank 1-2 (blue)'),
        make_trace(white_iso, 'rgb(255,255,255)', 'Rank 3-4 (white)'),
        make_trace(red_iso, 'rgb(142,33,56)', 'Rank 5-6 (red)')
    ]:
        if tr is not None:
            fig.add_trace(tr)

    fig.update_layout(
        title_text='Regional Asset Allocation - Quarterly call',
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top',
                    bgcolor='rgba(255,255,255,0.7)', bordercolor='black', borderwidth=1)
    )
    return fig

# SHAP / drivers helper
def compute_shap_and_drivers(model, X, full_df, ranks_df):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        shap_df = pd.DataFrame(shap_vals * 1000, columns=feature_cols, index=full_df['region'].astype(str))

        categories = sorted(set(feature_categories.values()))
        cat_df = pd.DataFrame(index=shap_df.index, columns=categories, data=0.0)
        for cat in categories:
            feats = [f for f, c in feature_categories.items() if c == cat and f in shap_df.columns]
            if feats:
                cat_df[cat] = shap_df[feats].sum(axis=1)
            else:
                cat_df[cat] = 0.0

        regions_in_chart = [r for r in ranks_df['region'].tolist() if r in cat_df.index]
        cat_df = cat_df.reindex(regions_in_chart)

        shap_fig = go.Figure()
        for cat in cat_df.columns:
            shap_fig.add_trace(go.Bar(
                x=cat_df.index.tolist(), y=cat_df[cat].values, name=cat.capitalize(),
                marker_color=category_colors.get(cat),
                hovertemplate='%{x}<br>' + cat.capitalize() + ': %{y:.6f}<extra></extra>'
            ))
        shap_fig.update_layout(
            barmode='relative', title='Pillar Contribution to Ranking',
            xaxis_title='Region', yaxis_title='Sum of SHAP values (impact on prediction)',
            yaxis=dict(ticksuffix='', exponentformat='e'),
            legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
            margin=dict(l=40, r=200, t=60, b=120), height=600,
            plot_bgcolor='white', paper_bgcolor='white'
        )

        shap_df_ordered = shap_df.reindex(regions_in_chart)

        def effect_phrase(val):
            if val > 1e-5:
                return "▲", "drives ranking up", f"contributes positively ({val:.5f})"
            elif val < -1e-5:
                return "▼", "drives ranking down", f"contributes negatively ({val:.5f})"
            else:
                return "◼", "has no net effect", f"neutral ({val:.5f})"

        top_k = 2
        cards = []
        ordered_cats = ['valuation', 'profitability', 'leverage', 'macro']

        for region in shap_df_ordered.index:
            row = shap_df_ordered.loc[region]

            card_html = (
                '<div style="flex: 0 1 18%; min-width:140px; margin:6px; padding:8px; '
                'border:1px solid #e0e0e0; border-radius:6px; background:#ffffff; font-size:13px; line-height:1.15;">'
            )

            card_html += f'<div style="font-weight:700; margin-bottom:8px;">{region} Top features:</div>'

            card_html += '<ul style="margin:6px 0 8px 18px; padding:0;">'
            abs_sorted = row.abs().sort_values(ascending=False)
            top_feats = abs_sorted.index.tolist()[:top_k]
            for f in top_feats:
                signed = row[f]
                emoji, phrase, _ = effect_phrase(signed)
                label = feature_fullname.get(f, f)
                card_html += f'<li style="margin-bottom:6px;">{emoji} {label} {phrase}</li>'
            card_html += '</ul>'

            card_html += '<div style="height:6px;"></div>'
            card_html += '<div style="font-weight:700; margin-bottom:6px;">Top driver by category:</div>'

            for cat in ordered_cats:
                feats_in_cat = [
                    feat for feat, c in feature_categories.items()
                    if c == cat
                    and feat in shap_df_ordered.columns
                    and np.ptp(shap_df_ordered[feat].values) > 1e-4
                ]

                if not feats_in_cat:
                    card_html += f'<div style="margin-top:6px; font-size:13px;">- {cat.capitalize()} overall has no features</div>'
                    continue

                cat_sum = row[feats_in_cat].sum()
                if cat_sum > 0:
                    overall = f"▲ {cat.capitalize()} overall contributes positively."
                elif cat_sum < 0:
                    overall = f"▼ {cat.capitalize()} overall contributes negatively."
                else:
                    overall = f"◼ {cat.capitalize()} overall has no net contribution."

                cat_top_feat = row[feats_in_cat].abs().idxmax()
                cat_top_signed = row[cat_top_feat]
                cat_top_label = feature_fullname.get(cat_top_feat, cat_top_feat)
                if cat_top_signed > 0:
                    top_phrase = "drives ranking up"
                elif cat_top_signed < 0:
                    top_phrase = "drives ranking down"
                else:
                    top_phrase = "has no net effect"

                card_html += f'<div style="margin-top:6px; font-size:13px;">- {overall}</div>'
                card_html += f'<div style="margin-left:12px; margin-top:4px; font-size:13px;">- {cat_top_label} {top_phrase}</div>'

            card_html += '</div>'
            cards.append(card_html)

        card_divs = []
        for card in cards:
            card_div = f'''
            <div style="
                flex: 1 1 20%;
                padding: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background: #fff;
                font-size: 13px;
                line-height: 1.15;
                box-sizing: border-box;
                min-width: 0;
            ">
                {card}
            </div>
            '''
            card_divs.append(card_div)

        drivers_html = f'''
        <div style="
            display: flex;
            flex-wrap: nowrap;
            justify-content: space-between;
            width: 100%;
            box-sizing: border-box;
            gap: 10px;
        ">
            {''.join(card_divs)}
        </div>
        '''

        return shap_fig, drivers_html

    except Exception as e:
        shap_fig = go.Figure()
        shap_fig.add_annotation(text=f"Error computing SHAP: {e}", xref="paper", yref="paper", showarrow=False, x=0.5, y=0.5)
        shap_fig.update_layout(title="SHAP error", plot_bgcolor='white', paper_bgcolor='white')
        drivers_html = f"<div style='color:red;'>Error computing SHAP: {e}</div>"
        return shap_fig, drivers_html

def compute_pillar_pie_chart(model, X, feature_cols):
    shap_values = model.predict(X, pred_contrib=True)
    shap_features = np.abs(shap_values[:, :-1])  # exclude bias column
    df_shap = pd.DataFrame(shap_features, columns=feature_cols)
    pillar_map = {
        'valuation': ['pe','pbv','ps','pebitda'],
        'profitability': ['roe','roa','roc','om','pm','dy','em','eps12m'],
        'macro': [
            'surprise_US','surprise_EURO','surprise_JAP','surprise_PAC','surprise_GEM','surprise_UK','surprise_CHINA','surprise_AUS',
            'world_trade','prob_inflation','prob_risk_off','EPU_global','EPU_US','EPU_Europe','EPU_UK','EPU_JAP','EPU_Australia','EPU_China'
        ],
        'leverage': ['nde','tdte','tdta']
    }
    pillar_importance = {}
    for pillar, feats in pillar_map.items():
        feats_in_data = [f for f in feats if f in df_shap.columns]
        if feats_in_data:
            pillar_importance[pillar] = df_shap[feats_in_data].values.sum()
        else:
            pillar_importance[pillar] = 0.0

    if 'dividend' in pillar_importance and 'profitability' in pillar_importance:
        pillar_importance['profitability'] -= pillar_importance['dividend']

    pie_labels = ['valuation', 'profitability', 'macro', 'leverage']
    pie_values = [pillar_importance.get(p, 0.0) for p in pie_labels]
    fig = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=pie_values,
        marker=dict(colors=[category_colors.get(p, None) for p in pie_labels]),
        hole=0.3
    )])
    fig.update_layout(title="Pillar Importance (SHAP, absolute sum)", margin=dict(l=40, r=40, t=60, b=40))
    return fig

def predict_and_map_with_state(
    display_df,
    surprise_US, surprise_EURO, surprise_UK, surprise_JAP,
    surprise_PAC, surprise_GEM, surprise_CHINA, surprise_AUS,
    prob_risk_off, prob_inflation, world_trade,
    EPU_global, EPU_US, EPU_Europe, EPU_UK, EPU_JAP, EPU_Australia, EPU_China,
    full_df, last_action_state, prev_globals=None, ranking_title=None, selected_file=None):

    if not isinstance(display_df, pd.DataFrame):
        display_df = pd.DataFrame(display_df)

    expected_cols = ['region'] + region_features
    if (
        list(display_df.columns) == list(range(len(display_df.columns)))
        and display_df.shape[1] == len(expected_cols)
    ):
        display_df.columns = expected_cols
    else:
        if 'region' not in display_df.columns:
            display_df = display_df.rename(columns={display_df.columns[0]: 'region'})

    display_df = display_df[['region'] + region_features].copy().reset_index(drop=True)
    prior_df = full_df.copy().reset_index(drop=True)

    selected_dt = parse_date_from_filename(selected_file) if selected_file else None
    if not selected_dt or not is_quarter_end_date(selected_dt):
        warning_html = quarter_end_warning_html(selected_dt)

        blank_map = go.Figure()
        blank_map.add_annotation(
            text="Prediction disabled: quarter-end data required",
            xref="paper", yref="paper", showarrow=False, x=0.5, y=0.5
        )
        blank_map.update_layout(title="Quarterly prediction unavailable")

        blank_shap = go.Figure()
        blank_shap.add_annotation(
            text="Prediction disabled: quarter-end data required",
            xref="paper", yref="paper", showarrow=False, x=0.5, y=0.5
        )
        blank_shap.update_layout(title="Quarterly prediction unavailable")

        blank_pie = go.Figure()
        blank_pie.add_annotation(
            text="Prediction disabled: quarter-end data required",
            xref="paper", yref="paper", showarrow=False, x=0.5, y=0.5
        )
        blank_pie.update_layout(title="Quarterly prediction unavailable")

        return (
            warning_html,
            blank_map,
            full_df,
            display_df,
            blank_shap,
            warning_html,
            warning_html,
            last_action_state,
            prev_globals,
            blank_pie,
            ranking_title
        )

    def norm_region(s):
        return re.sub(r'[^a-z0-9]', '', str(s).strip().lower())

    # Use rounded display values only for manual-change detection and UI cleanliness.
    rounded_prior = prior_df[['region'] + region_features].round(DISPLAY_DECIMALS).reset_index(drop=True)
    rounded_disp = display_df[['region'] + region_features].round(DISPLAY_DECIMALS).reset_index(drop=True)

    prior_map = {norm_region(rounded_prior.loc[i, 'region']): i for i in range(len(rounded_prior))}
    edits = []

    for i in range(len(rounded_disp)):
        disp_region_raw = rounded_disp.loc[i, 'region']
        disp_region_norm = norm_region(disp_region_raw)

        if disp_region_norm in prior_map:
            j = prior_map[disp_region_norm]
        else:
            j = None
            for k, idx in prior_map.items():
                if disp_region_norm in k or k in disp_region_norm:
                    j = idx
                    break
        if j is None:
            j = i if i < len(rounded_prior) else None
        if j is None:
            continue

        for col in region_features:
            try:
                old_val = float(rounded_prior.loc[j, col])
                new_val = float(rounded_disp.loc[i, col])
            except Exception:
                continue

            if not np.isclose(old_val, new_val, atol=1e-6):
                edits.append((rounded_prior.loc[j, 'region'], col, old_val, new_val))
                # Apply the edited value to the exact dataframe used for prediction
                full_df.loc[j, col] = new_val

    curr_globals = {}
    for gf, val in zip(global_features, [
        surprise_US, surprise_EURO, surprise_UK, surprise_JAP,
        surprise_PAC, surprise_GEM, surprise_CHINA, surprise_AUS,
        prob_risk_off, prob_inflation, world_trade,
        EPU_global, EPU_US, EPU_Europe, EPU_UK, EPU_JAP, EPU_Australia, EPU_China
    ]):
        try:
            curr_globals[gf] = float(val)
        except Exception:
            curr_globals[gf] = val

        if prev_globals is None:
            prev_globals = {k: default_globals.get(k, None) for k in global_features}

    global_edits = compare_globals(global_features, prev_globals, curr_globals, feature_fullname)

    def fmt_number(x, key=None):
        try:
            xf = float(x)
            if key in ('prob_inflation', 'prob_risk_off'):
                return f"{xf:.3f}"
            if abs(xf) < 1:
                return f"{xf:.3f}"
            if abs(xf) >= 0.01:
                return f"{xf:.2f}"
        except Exception:
            return str(x)

    edit_parts = []
    if edits:
        for reg, col, old, new in edits:
            label = feature_fullname.get(col, col)
            edit_parts.append(f"{col} ({label}) for {reg} from {_fmt_number(old)} to {_fmt_number(new)}")
    if global_edits:
        edit_parts.extend(global_edits)

    if not edit_parts:
        edit_note_text = ""
    else:
        joined = ", ".join(edit_parts)
        edit_note_text = f"User changed {joined}."
        edit_note_text = (
            f"<div style='padding:8px 12px; background:#fff9cc; "
            f"border:1px solid #ffd54f; border-radius:4px; margin:8px 0; font-size:13px'>"
            f"{edit_note_text}</div>"
        )

    for gf, val in curr_globals.items():
        try:
            full_df[gf] = float(val)
        except Exception:
            full_df[gf] = val

    X = (full_df.reindex(columns=feature_cols).copy().apply(pd.to_numeric, errors='coerce').fillna(0.0))

    pred_log_df = full_df[['region'] + feature_cols].copy()
    print("\n=== Data used for prediction ===")
    print(tabulate(pred_log_df, headers="keys", tablefmt="psql", showindex=False))
    print("=== End prediction data ===\n")

    try:
        pillar_pie_fig = compute_pillar_pie_chart(model, X, feature_cols)
    except Exception as e:
        pillar_pie_fig = go.Figure()
        pillar_pie_fig.add_annotation(text=f"Error: {e}", xref="paper", yref="paper", showarrow=False, x=0.5, y=0.5)
        pillar_pie_fig.update_layout(title="Pillar Pie Chart Error")

    y_pred = model.predict(X)
    full_df['predicted_score'] = y_pred
    full_df['qid'] = 'group1'
    full_df['predicted_rank'] = full_df.groupby('qid')['predicted_score'].rank(ascending=False, method='first').astype(int)

    ranks_df = full_df[['region', 'predicted_rank']].sort_values('predicted_rank').reset_index(drop=True)
    title_to_use = ranking_title if ranking_title else "Predicted Region Ranking"
    ranking_html = build_ranking_html(ranks_df, title=title_to_use)
    map_fig = create_map(ranks_df)

    try:
        shap_fig, drivers_html = compute_shap_and_drivers(model, X, full_df, ranks_df)
    except Exception as e:
        print("DEBUG SHAP error:", e)
        shap_fig = go.Figure()
        drivers_html = "<div style='color:red;'>SHAP error</div>"

    # Keep the table neat while showing the latest exact values rounded for display
    display_out = full_df[['region'] + region_features].round(DISPLAY_DECIMALS)
    last_action_out = 'manual_edit' if edits or global_edits else (last_action_state or 'init')

    return ranking_html, map_fig, full_df, display_out, shap_fig, drivers_html, edit_note_text, last_action_out, curr_globals, pillar_pie_fig, ranking_title

def _logo_html(logo_path=None, max_width_px=300):
    if logo_path and os.path.exists(logo_path):
        try:
            ext = os.path.splitext(logo_path)[1].lower().lstrip('.')
            mime = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'gif': 'image/gif',
                'svg': 'image/svg+xml'
            }.get(ext, 'application/octet-stream')
            with open(logo_path, 'rb') as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            src = f"data:{mime};base64,{b64}"
            return f"<img src='{src}' style='max-width:{max_width_px}px; height:auto; display:block;' draggable='false'/>"
        except Exception:
            return f"<img src='https://via.placeholder.com/{max_width_px}x60?text=Logo' style='max-width:{max_width_px}px; height:auto;' draggable='false'/>"
    else:
        return f"<img src='https://via.placeholder.com/{max_width_px}x60?text=Logo' style='max-width:{max_width_px}px; height:auto;' draggable='false'/>"

# compute initial files and default
files, default_file = get_templates_and_default()
default_file = default_file if default_file else None

def on_file_uploaded(uploaded_file):
    folder = DATA_FOLDER
    empty_note = ""
    title_label = "Predicted Region Ranking (quarter-end only)"

    def _refresh_files(folder=folder):
        return get_templates_and_default(folder)

    if not uploaded_file:
        files, default_file = _refresh_files()
        return (
            gr.update(value=default_df.round(DISPLAY_DECIMALS)),
            default_df.copy(),
            _global_value('surprise_US'), _global_value('surprise_EURO'), _global_value('surprise_UK'),_global_value('surprise_JAP'),
            _global_value('surprise_PAC'), _global_value('surprise_GEM'),
             _global_value('surprise_CHINA'), _global_value('surprise_AUS'),
            _global_value('prob_risk_off'), _global_value('prob_inflation'), _global_value('world_trade'),
            _global_value('EPU_global'), _global_value('EPU_US'), _global_value('EPU_Europe'),
            _global_value('EPU_UK'), _global_value('EPU_JAP'), _global_value('EPU_Australia'), _global_value('EPU_China'),
            empty_note, 'excel_load', title_label, default_globals.copy(),
            "<div style='color:green'>No file uploaded</div>",
            gr.update(choices=files if files else ["No input files found"], value=default_file)
        )

    try:
        incoming_name = None
        if hasattr(uploaded_file, "filename") and uploaded_file.filename:
            incoming_name = uploaded_file.filename
        elif hasattr(uploaded_file, "name") and uploaded_file.name:
            incoming_name = os.path.basename(uploaded_file.name)
        elif isinstance(uploaded_file, dict) and "name" in uploaded_file:
            incoming_name = os.path.basename(uploaded_file["name"])
        else:
            incoming_name = "uploaded.xlsx"

        base, ext = os.path.splitext(incoming_name)
        if not ext:
            ext = ".xlsx"
        dest_name = f"{base}{ext}"
        dest_path = os.path.join(folder, dest_name)

        if hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
            with open(dest_path, "wb") as fout:
                fout.write(data)
        elif isinstance(uploaded_file, dict) and "data" in uploaded_file:
            with open(dest_path, "wb") as fout:
                fout.write(uploaded_file["data"])
        else:
            src_path = getattr(uploaded_file, "name", None)
            if src_path and os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                raise RuntimeError("Unsupported upload object: cannot read bytes")

        saved_name = os.path.basename(dest_path)

        try:
            with open(os.path.join(folder, "upload_debug.txt"), "a") as dbg:
                dbg.write(f"{datetime.datetime.now().isoformat()} saved: {saved_name}\n")
        except Exception:
            pass

    except Exception as e:
        files, default_file = _refresh_files()
        return (
            gr.update(value=default_df.round(DISPLAY_DECIMALS)),
            default_df.copy(),
            _global_value('surprise_US'), _global_value('surprise_EURO'), _global_value('surprise_UK'), _global_value('surprise_JAP'),
            _global_value('surprise_PAC'), _global_value('surprise_GEM'),
             _global_value('surprise_CHINA'), _global_value('surprise_AUS'),
            _global_value('prob_risk_off'),_global_value('prob_inflation'), _global_value('world_trade'), 
            _global_value('EPU_global'), _global_value('EPU_US'), _global_value('EPU_Europe'),
            _global_value('EPU_UK'), _global_value('EPU_JAP'), _global_value('EPU_Australia'), _global_value('EPU_China'),
            f"<div style='color:red'>Error saving upload: {e}</div>", 'init', title_label, default_globals.copy(),
            f"<div style='color:red'>Error saving upload: {e}</div>",
            gr.update(choices=files if files else ["No input files found"], value=default_file)
        )

    try:
        regions_df, globals_row = load_inputs_from_excel(saved_name)
    except Exception as e:
        files, default_file = _refresh_files()
        return (
            gr.update(value=default_df.round(DISPLAY_DECIMALS)),
            default_df.copy(),
            _global_value('surprise_US'), _global_value('surprise_EURO'),  _global_value('surprise_UK'), _global_value('surprise_JAP'),
            _global_value('surprise_PAC'), _global_value('surprise_GEM'),
            _global_value('surprise_CHINA'), _global_value('surprise_AUS'),
             _global_value('prob_risk_off'),_global_value('prob_inflation'),_global_value('world_trade'), 
            _global_value('EPU_global'), _global_value('EPU_US'), _global_value('EPU_Europe'),
            _global_value('EPU_UK'), _global_value('EPU_JAP'), _global_value('EPU_Australia'), _global_value('EPU_China'),
            f"<div style='color:red'>Saved {saved_name} but failed to parse: {e}</div>", 'init', title_label, default_globals.copy(),
            f"<div style='color:red'>Saved {saved_name} but failed to parse: {e}</div>",
            gr.update(choices=files if files else ["No input files found"], value=default_file)
        )

    display = regions_df.round(DISPLAY_DECIMALS)
    full = regions_df.copy()

    dt = parse_date_from_filename(saved_name)
    if dt:
        if is_quarter_end_date(dt):
            title_label = quarter_end_title(dt)
        else:
            title_label = "Predicted Region Ranking (quarter-end only)"

    globals_dict = {
        'surprise_US': globals_row.get('surprise_US', default_globals.get('surprise_US', 0.0)),
        'surprise_EURO': globals_row.get('surprise_EURO', default_globals.get('surprise_EURO', 0.0)),
        'surprise_UK': globals_row.get('surprise_UK', default_globals.get('surprise_UK', 0.0)),
        'surprise_JAP': globals_row.get('surprise_JAP', default_globals.get('surprise_JAP', 0.0)),
        'surprise_PAC': globals_row.get('surprise_PAC', default_globals.get('surprise_PAC', 0.0)),
        'surprise_GEM': globals_row.get('surprise_GEM', default_globals.get('surprise_GEM', 0.0)),
        'surprise_CHINA': globals_row.get('surprise_CHINA', default_globals.get('surprise_CHINA', 0.0)),
        'surprise_AUS': globals_row.get('surprise_AUS', default_globals.get('surprise_AUS', 0.0)),  
        'prob_risk_off': globals_row.get('prob_risk_off', default_globals.get('prob_risk_off', 0.0)),
        'prob_inflation': globals_row.get('prob_inflation', default_globals.get('prob_inflation', 0.0)),
        'world_trade': globals_row.get('world_trade', default_globals.get('world_trade', 0.0)),
        'EPU_global': globals_row.get('EPU_global', default_globals.get('EPU_global', 0.0)),
        'EPU_US': globals_row.get('EPU_US', default_globals.get('EPU_US', 0.0)),
        'EPU_Europe': globals_row.get('EPU_Europe', default_globals.get('EPU_Europe', 0.0)),
        'EPU_UK': globals_row.get('EPU_UK', default_globals.get('EPU_UK', 0.0)),
        'EPU_JAP': globals_row.get('EPU_JAP', default_globals.get('EPU_JAP', 0.0)),
        'EPU_Australia': globals_row.get('EPU_Australia', default_globals.get('EPU_Australia', 0.0)),
        'EPU_China': globals_row.get('EPU_China', default_globals.get('EPU_China', 0.0)),
    }

    files, default_file = _refresh_files()

    if dt and not is_quarter_end_date(dt):
        status_msg = (
            f"<div style='color:#a15c00'>Uploaded and loaded {saved_name}. "
            f"This is not a quarter-end file, so prediction is disabled until a quarter-end file is loaded.</div>"
        )
    else:
        status_msg = f"<div style='color:green'>Uploaded and loaded {saved_name}</div>"

    return (
        gr.update(value=display),
        full,
        float(globals_row.get('surprise_US', default_globals.get('surprise_US', 0.0))),
        float(globals_row.get('surprise_EURO', default_globals.get('surprise_EURO', 0.0))),
        float(globals_row.get('surprise_UK', default_globals.get('surprise_UK', 0.0))),
        float(globals_row.get('surprise_JAP', default_globals.get('surprise_JAP', 0.0))),
        float(globals_row.get('surprise_PAC', default_globals.get('surprise_PAC', 0.0))),
        float(globals_row.get('surprise_GEM', default_globals.get('surprise_GEM', 0.0))),
        float(globals_row.get('surprise_CHINA', default_globals.get('surprise_CHINA', 0.0))),
        float(globals_row.get('surprise_AUS', default_globals.get('surprise_AUS', 0.0))),
        float(globals_row.get('prob_risk_off', default_globals.get('prob_risk_off', 0.0))),
        float(globals_row.get('prob_inflation', default_globals.get('prob_inflation', 0.0))),
        float(globals_row.get('world_trade', default_globals.get('world_trade', 0.0))),
        float(globals_row.get('EPU_global', default_globals.get('EPU_global', 0.0))),
        float(globals_row.get('EPU_US', default_globals.get('EPU_US', 0.0))),
        float(globals_row.get('EPU_Europe', default_globals.get('EPU_Europe', 0.0))),
        float(globals_row.get('EPU_UK', default_globals.get('EPU_UK', 0.0))),
        float(globals_row.get('EPU_JAP', default_globals.get('EPU_JAP', 0.0))),
        float(globals_row.get('EPU_Australia', default_globals.get('EPU_Australia', 0.0))),
        float(globals_row.get('EPU_China', default_globals.get('EPU_China', 0.0))),
        empty_note,
        'excel_load',
        title_label,
        globals_dict,
        load_msg
    )

def on_download_current(selected_file):
    if not selected_file:
        return "<div style='color:red'>No template selected</div>"
    path = os.path.join(DATA_FOLDER, selected_file)
    if not os.path.exists(path):
        return f"<div style='color:red'>File not found: {selected_file}</div>"
    try:
        with open(path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("ascii")
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        data_uri = f"data:{mime};base64,{b64}"
        size = os.path.getsize(path)
        if size < 1024:
            size_label = f"{size} B"
        elif size < 1024**2:
            size_label = f"{size/1024:.1f} KB"
        else:
            size_label = f"{size/1024**2:.1f} MB"
        anchor = f"<div style='padding:8px 0;'><a href='{data_uri}' download='{selected_file}' style='font-weight:700; color:#f58220;'>Click to download {selected_file}</a> — {size_label}</div>"
        return f"<div style='color:green'>{anchor}</div>"
    except Exception as e:
        return f"<div style='color:red'>Unable to prepare download: {e}</div>"

def on_load_excel_clicked(selected_file):
    empty_note = ""
    title_label = "Predicted Region Ranking (quarter-end only)"

    if not selected_file or selected_file == "No input files found" or selected_file is None:
        return gr.update(value=full_df_state.value.round(DISPLAY_DECIMALS)), full_df_state.value, \
            _global_value('surprise_US'), _global_value('surprise_EURO'),  _global_value('surprise_UK'),\
            _global_value('surprise_JAP'), _global_value('surprise_PAC'), _global_value('surprise_GEM'),   \
            _global_value('surprise_CHINA'), _global_value('surprise_AUS'), \
            _global_value('prob_risk_off'),_global_value('prob_inflation'),_global_value('world_trade'),\
            _global_value('EPU_global'), _global_value('EPU_US'), _global_value('EPU_Europe'), \
            _global_value('EPU_UK'), _global_value('EPU_JAP'), _global_value('EPU_Australia'), _global_value('EPU_China'), \
            empty_note, 'excel_load', title_label, prev_globals_state.value, \
            f"<div style='color:red'>No file selected</div>"

    try:
        regions_df, globals_row = load_inputs_from_excel(selected_file)
    except Exception as e:
        print("Load failed:", e)
        return gr.update(value=full_df_state.value.round(DISPLAY_DECIMALS)), full_df_state.value, \
            _global_value('surprise_US'), _global_value('surprise_EURO'), _global_value('surprise_UK'), _global_value('surprise_JAP'), \
            _global_value('surprise_PAC'), _global_value('surprise_GEM'), \
            _global_value('surprise_CHINA'), _global_value('surprise_AUS'), \
            _global_value('prob_risk_off'), _global_value('prob_inflation'), _global_value('world_trade'),\
            _global_value('EPU_global'), _global_value('EPU_US'), _global_value('EPU_Europe'), \
            _global_value('EPU_UK'), _global_value('EPU_JAP'), _global_value('EPU_Australia'), _global_value('EPU_China'), \
            empty_note, 'excel_load', title_label, prev_globals_state.value, \
            f"<div style='color:red'>Error loading {selected_file}: {e}</div>"

    globals_dict = {
        'surprise_US': globals_row.get('surprise_US', default_globals.get('surprise_US', 0.0)),
        'surprise_EURO': globals_row.get('surprise_EURO', default_globals.get('surprise_EURO', 0.0)),
        'surprise_UK': globals_row.get('surprise_UK', default_globals.get('surprise_UK', 0.0)),
        'surprise_JAP': globals_row.get('surprise_JAP', default_globals.get('surprise_JAP', 0.0)),
        'surprise_PAC': globals_row.get('surprise_PAC', default_globals.get('surprise_PAC', 0.0)),
        'surprise_GEM': globals_row.get('surprise_GEM', default_globals.get('surprise_GEM', 0.0)),
        'surprise_CHINA': globals_row.get('surprise_CHINA', default_globals.get('surprise_CHINA', 0.0)),
        'surprise_AUS': globals_row.get('surprise_AUS', default_globals.get('surprise_AUS', 0.0)),
        'prob_risk_off': globals_row.get('prob_risk_off', default_globals.get('prob_risk_off', 0.0)),
        'prob_inflation': globals_row.get('prob_inflation', default_globals.get('prob_inflation', 0.0)),
        'world_trade': globals_row.get('world_trade', default_globals.get('world_trade', 0.0)),
        'EPU_global': globals_row.get('EPU_global', default_globals.get('EPU_global', 0.0)),
        'EPU_US': globals_row.get('EPU_US', default_globals.get('EPU_US', 0.0)),
        'EPU_Europe': globals_row.get('EPU_Europe', default_globals.get('EPU_Europe', 0.0)),
        'EPU_UK': globals_row.get('EPU_UK', default_globals.get('EPU_UK', 0.0)),
        'EPU_JAP': globals_row.get('EPU_JAP', default_globals.get('EPU_JAP', 0.0)),
        'EPU_Australia': globals_row.get('EPU_Australia', default_globals.get('EPU_Australia', 0.0)),
        'EPU_China': globals_row.get('EPU_China', default_globals.get('EPU_China', 0.0)),
    }

    display = regions_df.round(DISPLAY_DECIMALS)
    full = regions_df.copy()

    dt = parse_date_from_filename(selected_file)
    if dt:
        if is_quarter_end_date(dt):
            title_label = quarter_end_title(dt)
        else:
            title_label = "Predicted Region Ranking (quarter-end only)"

    if dt and not is_quarter_end_date(dt):
        load_msg = (
            f"<div style='color:#a15c00'>Loaded {selected_file}. "
            f"This is not a quarter-end file, so prediction is disabled until a quarter-end file is loaded.</div>"
        )
    else:
        load_msg = f"<div style='color:green'>Loaded {selected_file}</div>"

    return (
    gr.update(value=display),
    full,
    float(globals_row.get('surprise_US', default_globals.get('surprise_US', 0.0))),
    float(globals_row.get('surprise_EURO', default_globals.get('surprise_EURO', 0.0))),
    float(globals_row.get('surprise_UK', default_globals.get('surprise_UK', 0.0))),
    float(globals_row.get('surprise_JAP', default_globals.get('surprise_JAP', 0.0))),
    float(globals_row.get('surprise_PAC', default_globals.get('surprise_PAC', 0.0))),
    float(globals_row.get('surprise_GEM', default_globals.get('surprise_GEM', 0.0))),
    float(globals_row.get('surprise_CHINA', default_globals.get('surprise_CHINA', 0.0))),
    float(globals_row.get('surprise_AUS', default_globals.get('surprise_AUS', 0.0))),
    float(globals_row.get('prob_risk_off', default_globals.get('prob_risk_off', 0.0))),
    float(globals_row.get('prob_inflation', default_globals.get('prob_inflation', 0.0))),
    float(globals_row.get('world_trade', default_globals.get('world_trade', 0.0))),
    float(globals_row.get('EPU_global', default_globals.get('EPU_global', 0.0))),
    float(globals_row.get('EPU_US', default_globals.get('EPU_US', 0.0))),
    float(globals_row.get('EPU_Europe', default_globals.get('EPU_Europe', 0.0))),
    float(globals_row.get('EPU_UK', default_globals.get('EPU_UK', 0.0))),
    float(globals_row.get('EPU_JAP', default_globals.get('EPU_JAP', 0.0))),
    float(globals_row.get('EPU_Australia', default_globals.get('EPU_Australia', 0.0))),
    float(globals_row.get('EPU_China', default_globals.get('EPU_China', 0.0))),
    empty_note,
    'excel_load',
    title_label,
    globals_dict,
    load_msg
)

# UI wiring
with gr.Blocks(css="""
/* Title styling */
.title { font-size: 28px; font-weight: 800; margin: 0; line-height: 1.05; text-align:left; }

/* Hide number-input spinners globally (Chrome/Safari/Edge & Firefox) */
input[type="number"], .gr-number input[type="number"] {
  -webkit-appearance: none !important;
  -moz-appearance: textfield !important;
  appearance: none !important;
}
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button,
.gr-number input[type="number"]::-webkit-outer-spin-button,
.gr-number input[type="number"]::-webkit-inner-spin-button {
  -webkit-appearance: none !important;
  display: none !important;
}

/* Layout spacing */
.controls-row { margin-top: 8px; margin-bottom: 12px; }
.controls-row .col-left,
.controls-row .col-mid,
.controls-row .col-right { display:flex; align-items:center; }
.controls-row .col-left { justify-content:flex-start; }
.controls-row .col-mid { flex-direction:column; justify-content:center; align-items:flex-start; gap:8px; padding-left:16px; padding-right:16px; }
.controls-row .col-right { justify-content:center; }

.gr-button[title="Download current template"] { background: #f58220 !important; color: #fff !important; font-weight:700; border:none !important; }
.gr-button[title="Load Template"] { background: #e9e9e9 !important; color:#222 !important; border:none !important; }
""") as demo:

    gr.Markdown("# Cross-Asset Research - Regional Equity Quarterly Allocation Recommendation")

    try:
        initial_full_df
    except NameError:
        initial_full_df = default_df.copy()

    full_df_state = gr.State(initial_full_df.copy())
    display_df = initial_full_df.round(3)

    # Top controls row
    files, default_file = get_templates_and_default()

    with gr.Row(elem_classes="controls-row"):
        with gr.Column(scale=2, elem_classes="col-left"):
            logo_html = _logo_html(LOGO_PATH, max_width_px=280)
            gr.HTML(value=logo_html)

        with gr.Column(scale=4, elem_classes="col-mid"):
            file_selector = gr.Dropdown(
                choices=files if files else ["No input files found"],
                value=default_file,
                label="Choose template"
            )
            load_btn = gr.Button("Load Template", variant="secondary")
            download_btn = gr.Button("Download current template", variant="primary")

        with gr.Column(scale=4, elem_classes="col-right"):
            file_uploader = gr.File(label="Upload Template (drag & drop .xlsx)", file_types=[".xlsx"], interactive=True)

    status_html = gr.HTML(value="")

    predict_btn = gr.Button("Predict Quarterly Rank")

    region_table = gr.Dataframe(
        value=display_df,
        label="Region Features (5 rows)",
        headers=["region"] + region_features,
        datatype=["str"] + ["number"] * len(region_features),
        interactive=True
    )

    edit_note = gr.HTML(value="", label="")

    with gr.Row():
        surprise_US = gr.Number(value=_global_value('surprise_US'), label="surprise_US", precision=DISPLAY_DECIMALS)
        surprise_EURO = gr.Number(value=_global_value('surprise_EURO'), label="surprise_EURO", precision=DISPLAY_DECIMALS)
        surprise_UK = gr.Number(value=_global_value('surprise_UK'), label="surprise_UK", precision=DISPLAY_DECIMALS)
        surprise_JAP = gr.Number(value=_global_value('surprise_JAP'), label="surprise_JAP", precision=DISPLAY_DECIMALS)
        surprise_PAC = gr.Number(value=_global_value('surprise_PAC'), label="surprise_PAC", precision=DISPLAY_DECIMALS)
        surprise_GEM = gr.Number(value=_global_value('surprise_GEM'), label="surprise_GEM", precision=DISPLAY_DECIMALS)
        surprise_CHINA = gr.Number(value=_global_value('surprise_CHINA'), label="surprise_CHINA", precision=DISPLAY_DECIMALS)
        surprise_AUS = gr.Number(value=_global_value('surprise_AUS'), label="surprise_AUS", precision=DISPLAY_DECIMALS)

    with gr.Row():
        EPU_global = gr.Number(value=_global_value('EPU_global'), label="EPU_global", precision=DISPLAY_DECIMALS)
        EPU_US = gr.Number(value=_global_value('EPU_US'), label="EPU_US", precision=DISPLAY_DECIMALS)
        EPU_Europe = gr.Number(value=_global_value('EPU_Europe'), label="EPU_Europe", precision=DISPLAY_DECIMALS)
        EPU_UK = gr.Number(value=_global_value('EPU_UK'), label="EPU_UK", precision=DISPLAY_DECIMALS)
        EPU_JAP = gr.Number(value=_global_value('EPU_JAP'), label="EPU_JAP", precision=DISPLAY_DECIMALS)
        EPU_Australia = gr.Number(value=_global_value('EPU_Australia'), label="EPU_Australia", precision=DISPLAY_DECIMALS)
        EPU_China = gr.Number(value=_global_value('EPU_China'), label="EPU_China", precision=DISPLAY_DECIMALS)

    with gr.Row():
        prob_risk_off = gr.Number(value=_global_value('prob_risk_off'), label="prob_risk_off", precision=DISPLAY_DECIMALS)
        prob_inflation = gr.Number(value=_global_value('prob_inflation'), label="prob_inflation", precision=DISPLAY_DECIMALS)
        world_trade = gr.Number(value=_global_value('world_trade'), label="world_trade", precision=DISPLAY_DECIMALS)

    with gr.Row():
        with gr.Column(scale=1):
            init_title = "Predicted Region Ranking (quarter-end only)"
            if default_file:
                dt0 = parse_date_from_filename(default_file)
                if dt0 and is_quarter_end_date(dt0):
                    init_title = quarter_end_title(dt0)

            output_table = gr.HTML(
                value=build_ranking_html(
                    initial_full_df[['region']].assign(predicted_rank=['-'] * len(initial_full_df)),
                    title=init_title
                ),
                label=init_title
            )
            ranking_title_state = gr.State(value=init_title)
        with gr.Column(scale=3):
            map_output = gr.Plot(label="Regional Asset Allocation")

    metrics_data = {
        "Metric": [
            "Profit margin below trend",
            "Debt-to-assets above trend ",
            "Price-to-EBITDA above trend",
            "Price-to-book values above trend"
        ],
        "Performance / Rank": ["↗", "↗", "↘", "↘"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_html = df_to_compact_html(metrics_df, title="Strongest relationships based on history:<br>their impact  on performance for the following quarter")
    metrics_output = gr.HTML(value=metrics_html, label="Performance Metrics")

    shap_plot = gr.Plot(label="Pillar Contribution to Ranking")
    drivers_html = gr.HTML(label="Drivers by region")
    pillar_pie_plot = gr.Plot(label="Pillar Importance Pie Chart")

    last_action = gr.State(value="init")
    prev_globals_state = gr.State(default_globals.copy())

    file_uploader.upload(
        fn=on_file_uploaded,
        inputs=[file_uploader],
        outputs=[
            region_table, full_df_state,
            surprise_US, surprise_EURO, surprise_UK, surprise_JAP, surprise_PAC, surprise_GEM,
            surprise_CHINA, surprise_AUS,
            prob_risk_off, prob_inflation, world_trade,
            EPU_global, EPU_US, EPU_Europe, EPU_UK, EPU_JAP, EPU_Australia, EPU_China,
            edit_note, last_action, ranking_title_state, prev_globals_state,
            status_html,
            file_selector
        ]
    )

    download_btn.click(fn=on_download_current, inputs=[file_selector], outputs=[status_html])

    load_btn.click(
        fn=on_load_excel_clicked,
        inputs=[file_selector],
        outputs=[
            region_table, full_df_state,
            surprise_US, surprise_EURO, surprise_UK, surprise_JAP, surprise_PAC, surprise_GEM,
            surprise_CHINA, surprise_AUS,
            prob_risk_off,prob_inflation, world_trade,
            EPU_global, EPU_US, EPU_Europe, EPU_UK, EPU_JAP, EPU_Australia, EPU_China,
            edit_note, last_action, ranking_title_state, prev_globals_state,
            status_html
        ]
    )

    def _fmt_number(x):
        try:
            return f"{float(x):.{DISPLAY_DECIMALS}f}"
        except Exception:
            return str(x)

    def _on_global_changed(new_val, prev_globals, last_action_state, key):
        if last_action_state == 'excel_load':
            old = prev_globals.get(key, None) if isinstance(prev_globals, dict) else None
            try:
                old_f = float(old)
                new_f = float(new_val)
                changed = round(old_f, DISPLAY_DECIMALS) != round(new_f, DISPLAY_DECIMALS)
            except Exception:
                changed = old != new_val
            if changed:
                label = feature_fullname.get(key, key)
                note = f"User changed {key} ({label}) from {_fmt_number(old)} to {_fmt_number(new_val)}."
                note_html = (
                    f"<div style='padding:8px 12px; background:#fff9cc; "
                    f"border:1px solid #ffd54f; border-radius:4px; margin:8px 0; font-size:13px'>"
                    f"{note}</div>"
                )
                return note_html, 'manual_edit'
        return "", last_action_state

    def on_region_table_changed(display_df, full_df, last_action_state):
        if last_action_state != 'excel_load':
            return "", last_action_state

        if not isinstance(display_df, pd.DataFrame):
            display_df = pd.DataFrame(display_df)
        expected_cols = ['region'] + region_features
        if (
            list(display_df.columns) == list(range(len(display_df.columns)))
            and display_df.shape[1] == len(expected_cols)
        ):
            display_df.columns = expected_cols
        else:
            if 'region' not in display_df.columns:
                display_df = display_df.rename(columns={display_df.columns[0]: 'region'})
        display_df = display_df[['region'] + region_features].copy().reset_index(drop=True)

        prior_df = full_df.copy().reset_index(drop=True)
        rounded_prior = prior_df[['region'] + region_features].round(3).reset_index(drop=True)
        rounded_disp = display_df[['region'] + region_features].round(3).reset_index(drop=True)

        def norm_region(s):
            return re.sub(r'[^a-z0-9]', '', str(s).strip().lower())

        prior_map = {norm_region(rounded_prior.loc[i, 'region']): i for i in range(len(rounded_prior))}
        edits = []
        for i in range(len(rounded_disp)):
            disp_region_raw = rounded_disp.loc[i, 'region']
            disp_region_norm = norm_region(disp_region_raw)

            if disp_region_norm in prior_map:
                j = prior_map[disp_region_norm]
            else:
                j = None
                for k, idx in prior_map.items():
                    if disp_region_norm in k or k in disp_region_norm:
                        j = idx
                        break
            if j is None:
                j = i if i < len(rounded_prior) else None
            if j is None:
                continue

            for col in region_features:
                try:
                    old_val = float(rounded_prior.loc[j, col])
                    new_val = float(rounded_disp.loc[i, col])
                except Exception:
                    continue

                if not np.isclose(old_val, new_val, atol=1e-6):
                    edits.append((rounded_prior.loc[j, 'region'], col, old_val, new_val))

        if not edits:
            return "", last_action_state

        def _fmt_number(x):
            try:
                return f"{float(x):.{DISPLAY_DECIMALS}f}"
            except Exception:
                return str(x)

        edit_parts = []
        for reg, col, old, new in edits[:8]:
            label = feature_fullname.get(col, col)
            edit_parts.append(f"{col} ({label}) for {reg} from {_fmt_number(old)} to {_fmt_number(new)}")
        if len(edits) > 8:
            edit_parts.append(f"... and {len(edits)-8} more changes")

        joined = ", ".join(edit_parts)
        edit_note_text = f"User changed {joined}."
        edit_note_text = (
            f"<div style='padding:8px 12px; background:#fff9cc; "
            f"border:1px solid #ffd54f; border-radius:4px; margin:8px 0; font-size:13px'>"
            f"{edit_note_text}</div>"
        )
        return edit_note_text, 'manual_edit'

    surprise_US.change(fn=partial(_on_global_changed, key='surprise_US'), inputs=[surprise_US, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    surprise_EURO.change(fn=partial(_on_global_changed, key='surprise_EURO'), inputs=[surprise_EURO, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    surprise_UK.change(fn=partial(_on_global_changed, key='surprise_UK'), inputs=[surprise_UK, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    surprise_JAP.change(fn=partial(_on_global_changed, key='surprise_JAP'), inputs=[surprise_JAP, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    surprise_PAC.change(fn=partial(_on_global_changed, key='surprise_PAC'), inputs=[surprise_PAC, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    surprise_GEM.change(fn=partial(_on_global_changed, key='surprise_GEM'), inputs=[surprise_GEM, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    surprise_CHINA.change(fn=partial(_on_global_changed, key='surprise_CHINA'), inputs=[surprise_CHINA, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    surprise_AUS.change(fn=partial(_on_global_changed, key='surprise_AUS'), inputs=[surprise_AUS, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)

    EPU_global.change(fn=partial(_on_global_changed, key='EPU_global'), inputs=[EPU_global, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    EPU_US.change(fn=partial(_on_global_changed, key='EPU_US'), inputs=[EPU_US, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    EPU_Europe.change(fn=partial(_on_global_changed, key='EPU_Europe'), inputs=[EPU_Europe, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    EPU_UK.change(fn=partial(_on_global_changed, key='EPU_UK'), inputs=[EPU_UK, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    EPU_JAP.change(fn=partial(_on_global_changed, key='EPU_JAP'), inputs=[EPU_JAP, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    EPU_Australia.change(fn=partial(_on_global_changed, key='EPU_Australia'), inputs=[EPU_Australia, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    EPU_China.change(fn=partial(_on_global_changed, key='EPU_China'), inputs=[EPU_China, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)

    prob_risk_off.change(fn=partial(_on_global_changed, key='prob_risk_off'), inputs=[prob_risk_off, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    prob_inflation.change(fn=partial(_on_global_changed, key='prob_inflation'), inputs=[prob_inflation, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    world_trade.change(fn=partial(_on_global_changed, key='world_trade'), inputs=[world_trade, prev_globals_state, last_action], outputs=[edit_note, last_action], queue=False)
    region_table.change(fn=on_region_table_changed, inputs=[region_table, full_df_state, last_action], outputs=[edit_note, last_action], queue=False)

    predict_btn.click(
        fn=predict_and_map_with_state,
        inputs=[
            region_table,
            surprise_US, surprise_EURO, surprise_UK, surprise_JAP,
            surprise_PAC, surprise_GEM, surprise_CHINA, surprise_AUS,
            prob_risk_off, prob_inflation, world_trade,
            EPU_global, EPU_US, EPU_Europe, EPU_UK, EPU_JAP, EPU_Australia, EPU_China,
            full_df_state, last_action, prev_globals_state, ranking_title_state, file_selector
        ],
        outputs=[
            output_table, map_output, full_df_state, region_table,
            shap_plot, drivers_html, edit_note, last_action,
            prev_globals_state, pillar_pie_plot, ranking_title_state
        ]
    )

    port = 9114
    demo.launch(
        debug=True,
        share=False,
        server_port=port,
        server_name='0.0.0.0',
        root_path=f'/studio/workspace/vscode/proxy/{os.getenv("ALTO_STUDIO_USERNAME")}/{port}',
    )
