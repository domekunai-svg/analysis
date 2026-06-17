# -*- coding: utf-8 -*-
"""СоциоГраф — аналитика программы «3Д Коммуникации». Тема светлая/тёмная (переключатель).
Граф D3 встроен (на тёмно-тёплом холсте). Панели — panels.py, грейды — grades.py, тема — theme.py.
Зависимости: streamlit, pandas, numpy, networkx, plotly, python-louvain, scipy, openpyxl"""

import os
import json
import time
import base64
import platform
import traceback
import warnings
import contextlib
from datetime import datetime

import numpy as np
import pandas as pd
import theme


# ───────────────────────── ДИАГНОСТИКА (временный блок) ─────────────────────────
class Debug:
    stages, errors, warns, info = [], [], [], {}

    @classmethod
    def reset(cls):
        cls.stages, cls.errors, cls.warns = [], [], []

    @classmethod
    @contextlib.contextmanager
    def stage(cls, name, fatal=True):
        t0 = time.time()
        try:
            yield
            cls.stages.append((name, "OK", round(time.time() - t0, 2), ""))
        except Exception as e:
            cls.stages.append((name, "ОШИБКА", round(time.time() - t0, 2), f"{type(e).__name__}: {e}"))
            cls.errors.append((name, traceback.format_exc()))
            if fatal:
                raise

    @classmethod
    def note_warning(cls, msg):
        cls.warns.append(str(msg))

    @classmethod
    def report_text(cls):
        L = ["=== ДИАГНОСТИКА СоциоГраф ===", f"время: {datetime.now():%Y-%m-%d %H:%M:%S}", "", "[ОКРУЖЕНИЕ]"]
        L += [f"  {k}: {v}" for k, v in cls.info.items()]
        L += ["", "[ЭТАПЫ]"]
        L += [f"  [{s:7}] {sec:>5}s  {n}" + (f"  — {m}" if m else "") for n, s, sec, m in cls.stages]
        if cls.warns:
            L += ["", "[ПРЕДУПРЕЖДЕНИЯ]"] + ["  " + w for w in cls.warns]
        L += ["", "[ТРЕЙСБЕКИ]" + ("" if cls.errors else " нет")]
        for where, tb in cls.errors:
            L += [f"  --- {where} ---", tb]
        return "\n".join(L)


def _showwarning(message, category, filename, lineno, file=None, line=None):
    Debug.note_warning(f"{category.__name__}: {message} ({os.path.basename(str(filename))}:{lineno})")
warnings.showwarning = _showwarning

_IMPORT_ERR = {}
import streamlit as st
try:
    import networkx as nx
except Exception:
    nx = None; _IMPORT_ERR["networkx"] = traceback.format_exc()
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None; make_subplots = None; _IMPORT_ERR["plotly"] = traceback.format_exc()
try:
    from community import community_louvain
except Exception:
    community_louvain = None; _IMPORT_ERR["python-louvain"] = traceback.format_exc()
try:
    import streamlit.components.v1 as components
except Exception:
    components = None; _IMPORT_ERR["components"] = traceback.format_exc()
try:
    import panels
except Exception:
    panels = None; _IMPORT_ERR["panels"] = traceback.format_exc()
try:
    from grades import grade_dynamics_figure, set_grade_map
except Exception:
    grade_dynamics_figure = set_grade_map = None; _IMPORT_ERR["grades"] = traceback.format_exc()

Debug.info = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
              "networkx": getattr(nx, "__version__", "—"), "streamlit": getattr(st, "__version__", "—"),
              "plotly": "ok" if go else "—", "louvain": "ok" if community_louvain else "—",
              "graph": "встроен в app.py", "panels": "ok" if panels else "—",
              "grades": "ok" if grade_dynamics_figure else "—"}

CORAL = "#e95f3e"; OLIVE = "#6b8e23"; AMBER = "#cf8b22"
st.set_page_config(page_title="3Д Коммуникации", page_icon="🪡", layout="wide", initial_sidebar_state="expanded")


def light(fig, title, height):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=44, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=theme.INK, family="Golos Text, sans-serif"), title=title)
    fig.update_xaxes(gridcolor=theme.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False)
    return fig


# ───────────────────────── ТЕМА / CSS ─────────────────────────
CSS_TMPL = """
<style>
 @import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700&display=swap');
 html{color-scheme:__SCHEME__;}
 html,body,[class*="css"]{font-family:'Golos Text',-apple-system,Segoe UI,sans-serif;}
 .stApp,[data-testid="stAppViewContainer"]{background:__BG__ !important;color:__INK__;}
 [data-testid="stHeader"]{background:__HEADER__ !important;backdrop-filter:blur(8px);border-bottom:1px solid __BORDER__;}
 .block-container{padding:3.4rem 2rem 2rem !important;}
 h1,h2,h3{font-family:'Golos Text',sans-serif !important;color:__INK__ !important;font-weight:600 !important;letter-spacing:-.3px;}
 [data-testid="stMetricValue"]{font-family:'Golos Text',sans-serif;font-size:1.6rem !important;color:#e95f3e !important;}
 [data-testid="stMetricLabel"]{font-size:.72rem !important;color:__MUTED__ !important;text-transform:uppercase;letter-spacing:.8px;}
 [data-testid="stMetric"]{background:__CARD__;backdrop-filter:blur(6px);border:1px solid __BORDER__;border-radius:14px;padding:.7rem 1rem;box-shadow:0 1px 3px rgba(0,0,0,.05);}
 [data-testid="stSidebar"]{background:__SIDEBAR__ !important;border-right:1px solid __BORDER__;}
 [data-testid="stSidebar"] [data-testid="stWidgetLabel"],[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] [data-baseweb="radio"] div{color:__INK__ !important;}
 .kpi-box{background:__CARD__;backdrop-filter:blur(6px);border:1px solid __BORDER__;border-radius:16px;padding:1rem 1.1rem;min-height:96px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
 .kpi-box .kpi-value{font-family:'Golos Text',sans-serif;font-size:1.55rem;font-weight:700;color:#e95f3e;line-height:1.1;}
 .kpi-box .kpi-label{font-size:.68rem;color:__MUTED__;text-transform:uppercase;letter-spacing:.8px;margin-top:4px;}
 .kpi-box .kpi-sub{font-size:.72rem;color:__MUTED__;margin-top:6px;}
 .section-header{font-family:'Golos Text',sans-serif;font-size:.78rem;text-transform:uppercase;letter-spacing:1.6px;color:__MUTED__;border-bottom:1px solid __BORDER__;padding-bottom:6px;margin:1.6rem 0 1rem;font-weight:600;}
 .card{background:__CARD__;backdrop-filter:blur(6px);border:1px solid __BORDER__;border-left:3px solid #e95f3e;padding:.9rem 1.1rem;border-radius:14px;margin:.5rem 0;font-size:.9rem;color:__INK__;line-height:1.6;box-shadow:0 1px 3px rgba(0,0,0,.04);}
 .card strong{color:#e95f3e;}
 .care{border-left-color:#cf8b22;} .care strong{color:#cf8b22;}
 .good{border-left-color:#6b8e23;} .good strong{color:#6b8e23;}
 .muted{color:__MUTED__;font-size:.82rem;}
 .info-box{background:__CARD__;backdrop-filter:blur(6px);border:1px solid __BORDER__;border-radius:14px;padding:.9rem;margin:.6rem 0;font-size:.83rem;color:__MUTED__;line-height:1.6;}
 .hero{font-family:'Golos Text',sans-serif;font-size:1.95rem;font-weight:700;color:__INK__;letter-spacing:-.5px;}
 .stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:none;}
 .stTabs [data-baseweb="tab"]{background:__CARD__;border:1px solid __BORDER__;border-radius:11px;padding:6px 16px !important;color:__MUTED__;font-weight:500;}
 .stTabs [aria-selected="true"]{background:#e95f3e;color:#fff !important;border-color:#e95f3e;}
 .stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none !important;}
 .stButton>button,.stDownloadButton>button{background:__CARD__;color:#e95f3e;border:1px solid __BORDER__;border-radius:10px;}
 .stButton>button:hover,.stDownloadButton>button:hover{border-color:#e95f3e;}
 .sidebar-section{font-size:.7rem;text-transform:uppercase;letter-spacing:1.4px;color:__MUTED__ !important;margin:1rem 0 .3rem;font-weight:600;}
</style>
"""


def inject_theme(dark):
    v = (dict(SCHEME="dark", BG="#262220", CARD="rgba(51,46,41,.72)", BORDER="#3a342f", INK="#f1e8e2",
              MUTED="#a89d95", SIDEBAR="#2a2521", HEADER="rgba(38,34,32,.82)")
         if dark else
         dict(SCHEME="light", BG="#faf9f5", CARD="rgba(255,255,255,.72)", BORDER="#ece0d8", INK="#37312c",
              MUTED="#8c817a", SIDEBAR="#f5efe9", HEADER="rgba(250,249,245,.82)"))
    css = CSS_TMPL
    for k, val in v.items():
        css = css.replace("__" + k + "__", val)
    st.markdown(css, unsafe_allow_html=True)


def localize_widgets():
    if components is None:
        return
    try:
        components.html("<script>setInterval(function(){try{var d=window.parent.document;if(!d)return;"
                        "d.querySelectorAll('li,div,span,button,p').forEach(function(n){if(n.childElementCount===0){"
                        "var t=n.textContent.trim();if(t==='Select all'){n.textContent='Выбрать всё';}"
                        "else if(t==='Choose an option'){n.textContent='Выберите…';}}});}catch(e){}},700);</script>", height=0)
    except Exception:
        pass


def logo_html(base):
    p = os.path.join(base, "logo.png")
    if os.path.exists(p):
        try:
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            return f'<img src="data:image/png;base64,{b64}" style="height:44px;border-radius:10px;vertical-align:middle">'
        except Exception:
            pass
    return ('<svg width="44" height="44" viewBox="0 0 100 100"><rect width="100" height="100" rx="22" fill="#e95f3e"/>'
            '<g fill="none" stroke="#faf2ea" stroke-width="6" stroke-linejoin="round" stroke-linecap="round">'
            '<path d="M50 22 L74 36 L74 64 L50 78 L26 64 L26 36 Z"/><path d="M50 22 L50 50 M50 50 L74 36 M50 50 L26 36"/></g></svg>')


# ───────────────────────── СХЕМА ─────────────────────────
MERITS_PER_MONTH = 10
EMP = {"last":"Фамилия","first":"Имя","mid":"Отчество","id":"Персональный номер",
       "pos":"Должность","company":"Компания","dept":"Отдел","fire":"Дата увольнения"}
TX  = {"date":"Дата","time":"Время","sid":"Номер отправителя","rid":"Номер получателя",
       "value":"Ценность","merits":"Мериты","comment":"Комментарий"}
MM = {1:"Январь",2:"Февраль",3:"Март",4:"Апрель",5:"Май",6:"Июнь",7:"Июль",8:"Август",9:"Сентябрь",10:"Октябрь",11:"Ноябрь",12:"Декабрь"}


# ───────────────────────── ЗАГРУЗКА (новый шаблон HR, 6 листов) ─────────────────────────
def _col(df, *keys):
    for c in df.columns:
        cl = str(c).lower()
        if all(k.lower() in cl for k in keys):
            return c
    return None


@st.cache_data(show_spinner=False)
def _registry(path):
    try:
        r = pd.read_excel(path, sheet_name="3_lineage_registry", header=1)
        cid, cname = _col(r, "canon_id"), _col(r, "каноническая")
        return dict(zip(r[cid].astype(str).str.strip(), r[cname].astype(str).str.strip()))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_employees(path):
    df = pd.read_excel(path, sheet_name="2_employees_clean", header=1)
    g = lambda *k: _col(df, *k)
    o = pd.DataFrame()
    o[EMP["id"]] = df[g("person_id")].astype(str).str.strip()
    for key, src_ in (("last", "фамилия"), ("first", "имя"), ("mid", "отчество")):
        c = g(src_)
        o[EMP[key]] = (df[c].astype(str).str.strip().replace({"nan": None, "None": None}) if c is not None else None)
    gc = g("пол")
    if gc is not None:
        o["Пол"] = df[gc].astype(str).str.strip()
    o[EMP["pos"]] = df[g("должность_норм") or g("должность")].astype(str).str.strip()
    o[EMP["company"]] = df[g("отдел_l1")].astype(str).str.strip()
    o[EMP["dept"]] = df[g("отдел_l2")].astype(str).str.strip()
    fc = g("дата_увольнения")
    o[EMP["fire"]] = pd.to_datetime(df[fc], dayfirst=True, errors="coerce") if fc else pd.NaT
    if g("уровень"):
        o["Уровень"] = df[g("уровень")].astype(str).str.strip()
    if g("трудоустройства"):
        o["Дата_найма"] = pd.to_datetime(df[g("трудоустройства")], dayfirst=True, errors="coerce")
    la = pd.to_datetime(df[g("последней")], dayfirst=True, errors="coerce") if g("последней") else None
    if la is not None:
        o["Посл_активность"] = la
    if g("статус"):
        o["Статус"] = df[g("статус")].astype(str).str.strip()
        if la is not None:
            fired = o["Статус"].eq("уволен") & o[EMP["fire"]].isna()
            o.loc[fired, EMP["fire"]] = la[fired]
    o["full_name"] = (o[EMP["last"]].fillna("") + " " + o[EMP["first"]].fillna("") + " " + o[EMP["mid"]].fillna("")).str.strip()
    o["dept_key"] = o[EMP["company"]].fillna("?") + " / " + o[EMP["dept"]].fillna("?")
    return o


@st.cache_data(show_spinner=False)
def load_transactions(path):
    df = pd.read_excel(path, sheet_name="1_transactions_clean", header=1)
    g = lambda *k: _col(df, *k)
    reg = _registry(path)
    o = pd.DataFrame()
    o["dt"] = pd.to_datetime(df[g("дата")], dayfirst=True, errors="coerce")
    o[TX["date"]] = df[g("дата")]
    if g("время"):
        o[TX["time"]] = df[g("время")]
    o[TX["sid"]] = df[g("sender_id")].astype(str).str.strip()
    o[TX["rid"]] = df[g("receiver_id")].astype(str).str.strip()
    cc = g("value_canonical")
    if cc and reg:
        canon = df[cc].astype(str).str.strip()
        o[TX["value"]] = canon.map(reg).fillna(canon)
    else:
        rv = g("value_raw")
        o[TX["value"]] = df[rv].astype(str).str.strip() if rv else "—"
    o[TX["merits"]] = pd.to_numeric(df[g("merits")], errors="coerce").fillna(0).astype(int)
    o[TX["comment"]] = df[g("comment")] if g("comment") else None
    if g("reaction"):
        o["reaction"] = df[g("reaction")]
    if g("lineage"):
        o["lineage_version"] = df[g("lineage")].astype(str).str.strip()
    o = o[o[TX["sid"]].notna() & o[TX["rid"]].notna() & (o[TX["sid"]] != "nan") & (o[TX["rid"]] != "nan")]
    o["day"] = o["dt"].dt.day; o["year"] = o["dt"].dt.year; o["month"] = o["dt"].dt.month
    o["ym"] = o["dt"].dt.to_period("M").astype(str)
    return o


@st.cache_data(show_spinner=False)
def load_limits(path):
    try:
        r = pd.read_excel(path, sheet_name="4_limit_history", header=1)
        dc, lc = _col(r, "date_from"), _col(r, "лимит")
        r = r.assign(_d=pd.to_datetime(r[dc], dayfirst=True, errors="coerce")).dropna(subset=["_d"]).sort_values("_d")
        breaks, prev = [], None
        for _, row in r.iterrows():
            lim = row[lc]
            if prev is not None and pd.notna(lim) and lim != prev:
                breaks.append((row["_d"], int(lim), int(prev)))
            prev = lim
        return breaks
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_grade_map(path):
    try:
        r = pd.read_excel(path, sheet_name="3_lineage_registry", header=1)
        nc, gc = _col(r, "каноническая"), _col(r, "grade_primary")
        if nc and gc:
            return {str(k).strip(): str(v).strip() for k, v in zip(r[nc], r[gc])
                    if pd.notna(v) and str(v).strip() and str(v).strip().lower() != "nan"}
    except Exception:
        pass
    return {}


@st.cache_data(show_spinner=False)
def resolve_curators(path):
    try:
        emp = pd.read_excel(path, sheet_name="2_employees_clean", header=1)
        org = pd.read_excel(path, sheet_name="5_org_structure", header=1)
    except Exception:
        return dict(cur_to_subs={}, sub_to_curs={}, n_links=0, n_matrix=0, n_resolved=0)

    def nrm(s):
        return " ".join(str(s).strip().lower().replace("ё", "е").split())
    fc, ff, fi, fo = _col(emp, "person_id"), _col(emp, "фамилия"), _col(emp, "имя"), _col(emp, "отчество")
    idx = {}
    for _, r in emp.iterrows():
        idx.setdefault(nrm(r[ff]), []).append((str(r[fc]).strip(),
            nrm(r[fi]) if pd.notna(r[fi]) else "", nrm(r[fo]) if pd.notna(r[fo]) else ""))

    def split_c(raw):
        s = str(raw)
        for sep in (";", ",", "/"):
            s = s.replace(sep, "|")
        s = s.replace(" и ", "|")
        return [p.strip() for p in s.split("|") if p.strip()]

    def match(tok):
        parts = tok.split()
        if len(parts) < 2:
            return []
        fam = nrm(parts[0])
        inits = []
        for w in parts[1:]:
            if "." in w:
                inits += [ch.lower() for ch in w if ch.isalpha()]
            elif w[:1].isalpha():
                inits.append(w[0].lower())
        if not inits:
            return []
        i1 = inits[0]; i2 = inits[1] if len(inits) > 1 else ""
        return [pid for pid, im, ot in idx.get(fam, []) if im[:1] == i1 and (not i2 or not ot or ot[:1] == i2)]

    pc, cc = _col(org, "person_id"), _col(org, "curator_raw")
    cur_to_subs, sub_to_curs, n_links, n_matrix, n_res = {}, {}, 0, 0, 0
    for _, r in org.iterrows():
        raw = r[cc]
        if pd.isna(raw):
            continue
        n_links += 1
        toks = split_c(raw)
        if len(toks) > 1:
            n_matrix += 1
        resolved = []
        for t in toks:
            mm = match(t)
            if len(mm) == 1:
                resolved.append(mm[0])
        if resolved:
            n_res += 1
            sub = str(r[pc]).strip()
            sub_to_curs[sub] = resolved
            for c in resolved:
                cur_to_subs.setdefault(c, set()).add(sub)
    return dict(cur_to_subs={k: list(v) for k, v in cur_to_subs.items()}, sub_to_curs=sub_to_curs,
                n_links=n_links, n_matrix=n_matrix, n_resolved=n_res)


def merge_data(tx, emp):
    m = emp.set_index(EMP["id"]); m = m[~m.index.duplicated(keep="first")]
    cols = ["full_name", EMP["pos"], EMP["company"], EMP["dept"], "dept_key", "Уровень"]
    for pref, idc in (("s", TX["sid"]), ("r", TX["rid"])):
        for c in cols:
            tx[f"{pref}_{c}"] = tx[idc].map(m[c]) if c in m.columns else None
        if EMP["fire"] in m.columns:
            tx[f"{pref}_fire"] = tx[idc].map(m[EMP["fire"]])
    return tx


# ───────────────────────── ФИЛЬТРЫ ─────────────────────────
def sidebar_controls(tx, emp):
    st.sidebar.toggle("🌙 Тёмная тема", key="dark")
    st.sidebar.markdown("## ⚙️ Фильтры")
    st.sidebar.markdown('<div class="sidebar-section">Период</div>', unsafe_allow_html=True)
    dmin = tx["dt"].min(); dmax = tx["dt"].max()
    dmin_d = dmin.date() if pd.notna(dmin) else None
    dmax_d = dmax.date() if pd.notna(dmax) else None
    dr = st.sidebar.date_input("Диапазон дат", (dmin_d, dmax_d), min_value=dmin_d, max_value=dmax_d)
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        d_from, d_to = dr
    else:
        d_from = d_to = (dr if not isinstance(dr, (list, tuple)) else dmin_d)
    years = sorted(tx["year"].dropna().unique().astype(int).tolist())
    sel_years = st.sidebar.multiselect("Год", years, default=[], placeholder="все годы")
    months = sorted(tx["month"].dropna().unique().astype(int).tolist())
    sel_months = st.sidebar.multiselect("Месяц", months, format_func=lambda x: MM.get(x, x), default=[], placeholder="все месяцы")

    st.sidebar.markdown('<div class="sidebar-section">Организация</div>', unsafe_allow_html=True)
    comps = sorted(emp[EMP["company"]].dropna().unique().tolist())
    sel_comps = st.sidebar.multiselect("Компания", comps, default=[], placeholder="все компании")
    dpool = emp[emp[EMP["company"]].isin(sel_comps)] if sel_comps else emp
    depts = sorted(dpool["dept_key"].dropna().unique().tolist())
    sel_depts = st.sidebar.multiselect("Отдел", depts, default=[], placeholder="все отделы")
    epool = dpool[dpool["dept_key"].isin(sel_depts)] if sel_depts else dpool
    emps = sorted(epool["full_name"].dropna().unique().tolist())
    sel_emps = st.sidebar.multiselect("Сотрудники", emps, default=[], placeholder="все сотрудники")
    side = st.sidebar.radio("Сторона признания", ["обе стороны", "только отправитель", "только получатель"], index=0,
                            help="Двусторонний фильтр сохраняет и входящую сторону признания")

    st.sidebar.markdown('<div class="sidebar-section">Ценности</div>', unsafe_allow_html=True)
    vals = sorted(tx[TX["value"]].dropna().unique().tolist())
    sel_vals = st.sidebar.multiselect("Ценности", vals, default=vals)

    st.sidebar.markdown('<div class="sidebar-section">Граф</div>', unsafe_allow_html=True)
    try:
        real_max = max(int(tx.groupby([TX["sid"], TX["rid"]])[TX["merits"]].sum().max()), 10)
    except Exception:
        real_max = 500
    mr = st.sidebar.slider("Сила связи (мериты) — фильтр графа", 1, real_max, (1, real_max), 1,
                           help="Скрывает очень слабые или очень сильные связи, чтобы граф читался. Обычно трогать не нужно.")
    return dict(d_from=d_from, d_to=d_to, years=set(sel_years), months=set(sel_months), values=set(sel_vals),
                companies=set(sel_comps), depts=set(sel_depts), emps=set(sel_emps), side=side, merit_range=mr)


def apply_filters(tx, cfg):
    df = tx
    if cfg["d_from"] is not None and cfg["d_to"] is not None:
        dd = df["dt"].dt.date
        df = df[(dd >= cfg["d_from"]) & (dd <= cfg["d_to"])]
    if cfg["years"]:
        df = df[df["year"].isin(cfg["years"])]
    if cfg["months"]:
        df = df[df["month"].isin(cfg["months"])]
    if cfg["values"]:
        df = df[df[TX["value"]].isin(cfg["values"])]

    def mask(scol, rcol, allowed):
        s_in, r_in = df[scol].isin(allowed), df[rcol].isin(allowed)
        if cfg["side"] == "только отправитель": return s_in
        if cfg["side"] == "только получатель": return r_in
        return s_in | r_in

    if cfg["companies"]:
        df = df[mask("s_" + EMP["company"], "r_" + EMP["company"], cfg["companies"])]
    if cfg["depts"]:
        df = df[mask("s_dept_key", "r_dept_key", cfg["depts"])]
    if cfg["emps"]:
        df = df[mask("s_full_name", "r_full_name", cfg["emps"])]
    return df


# ───────────────────────── KPI ─────────────────────────
def compute_funnel(emp, fd):
    active = emp[emp[EMP["fire"]].isna()]
    n_active = len(active); active_ids = set(active[EMP["id"]])
    fd = fd[fd[TX["sid"]].isin(active_ids) & fd[TX["rid"]].isin(active_ids)]
    senders = fd.groupby(TX["sid"]).size()
    n_senders = int((senders >= 1).sum()); n_regular = int((senders > 1).sum())
    receivers = set(fd[TX["rid"]].unique())
    involved = set(fd[TX["sid"]].unique()) | receivers
    n_involved = len(involved); total = int(fd[TX["merits"]].sum())
    cshare = round(fd[TX["comment"]].notna().mean() * 100, 1) if (TX["comment"] in fd.columns and len(fd)) else 0
    return dict(n_involved=n_involved, n_senders=n_senders, n_regular=n_regular, n_receivers=len(receivers),
                involve_share=round(n_involved / n_active * 100, 1) if n_active else 0,
                send_share=round(n_senders / n_involved * 100, 1) if n_involved else 0,
                comment_share=cshare,
                avg_recv=round(total / len(receivers), 1) if receivers else 0)


def render_funnel(f):
    st.markdown('<div class="section-header">Вовлечённость в программу признания</div>', unsafe_allow_html=True)
    cards = [
        ("Участвуют в признании", f"{f['n_involved']:,}", f"{f['involve_share']}% активных · отдали или получили благодарность"),
        ("Отправляют благодарности", f"{f['n_senders']:,}", f"{f['send_share']}% участников проявляют активную позицию"),
        ("Регулярно участвуют", f"{f['n_regular']:,}", "отправили больше одной благодарности"),
        ("Получают признание", f"{f['n_receivers']:,}", "хотя бы одна благодарность за период"),
        ("Глубина — с комментарием", f"{f['comment_share']}%", "доля благодарностей с текстом — насколько признание развёрнуто"),
        ("Среднее признание на человека", f"{f['avg_recv']}", "мериты на получателя — см. распределение"),
    ]
    for c, (lab, val, sub) in zip(st.columns(6), cards):
        c.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div>'
                   f'<div class="kpi-label">{lab}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


# ───────────────────────── ПУЛЬС ─────────────────────────
def _mark_limits(fig, limits):
    for d, new, old in (limits or []):
        try:
            fig.add_vline(x=d, line_dash="dash", line_color="#a89d95", line_width=1.5,
                          annotation_text=f"лимит {old}→{new}", annotation_position="top",
                          annotation_font_size=10, annotation_font_color="#8c817a")
        except Exception:
            pass


def render_temporal(fd, emp, limits=None):
    st.markdown('<div class="section-header">Пульс программы</div>', unsafe_allow_html=True)
    if go is None:
        st.warning("plotly недоступен (см. диагностику)."); return
    fdt = fd.dropna(subset=["dt"])
    if len(fdt) == 0:
        st.info("Нет данных с датами."); return

    if panels is not None:
        dfig = panels.daily_pulse_fig(fd, emp)
        if dfig is not None:
            _mark_limits(dfig, limits)
            st.plotly_chart(dfig, use_container_width=True)

    monthly = fdt.groupby("ym").agg(Актов=(TX["merits"], "size")).reset_index()
    if len(monthly) >= 2:
        monthly["d"] = pd.PeriodIndex(monthly["ym"], freq="M").to_timestamp()
        last_p = fdt["dt"].max().to_period("M")
        in_last = fdt["dt"].dt.to_period("M") == last_p
        trimmed = (fdt.loc[in_last, "dt"].dt.day.max() if in_last.any() else 31) < 25
        plot = monthly.iloc[:-1] if trimmed and len(monthly) > 2 else monthly
        fig = go.Figure(go.Scatter(x=plot["d"], y=plot["Актов"], mode="lines+markers", line=dict(color=CORAL, width=2.5)))
        light(fig, "Помесячный объём — число актов" + (" (последний неполный месяц скрыт)" if trimmed else ""), 260)
        _mark_limits(fig, limits)
        st.plotly_chart(fig, use_container_width=True)
    if limits:
        d, new, old = limits[-1]
        st.markdown(f'<div class="info-box">Вертикальная линия — смена лимита голосов на один акт '
                    f'<strong>{old}→{new}</strong> ({d:%d.%m.%Y}). Объём показан в актах (от лимита не зависит); '
                    f'сумма меритов после смены растёт частично из-за лимита, а не активности.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, vertical_alignment="center")
    short = {1:"Янв",2:"Фев",3:"Мар",4:"Апр",5:"Май",6:"Июн",7:"Июл",8:"Авг",9:"Сен",10:"Окт",11:"Ноя",12:"Дек"}
    with c1:
        _yr = fdt["dt"].dt.year.rename("y"); _mo = fdt["dt"].dt.month.rename("mo")
        by_ym = fdt.groupby([_yr, _mo]).size().rename("acts").reset_index()
        typical = by_ym.groupby("mo")["acts"].mean()
        yrs = by_ym["y"].nunique()
        figs = go.Figure(go.Bar(x=[short[m] for m in typical.index], y=typical.round(0).values, marker_color=AMBER))
        light(figs, f"Типичный месяц — среднее число актов (по {yrs} годам)", 260)
        st.plotly_chart(figs, use_container_width=True)
    with c2:
        late = (fd["day"] >= 25).mean(); expected = (31 - 25 + 1) / 30.4
        ratio = late / expected if expected else 0
        klass, txt = ("good", "распределено ровно") if ratio < 1.3 else ("care", "признание стягивается к концу месяца")
        st.markdown(f'<div class="card {klass}"><strong>Ритм внутри месяца.</strong><br>'
                    f'{late*100:.0f}% актов приходится на дни 25–31 (ожидаемо ~{expected*100:.0f}%). Это {txt}.<br>'
                    f'<span class="muted">Сильная концентрация в конце месяца — возможный признак «дозакрытия лимита» '
                    f'по привычке. Стоит сопоставить с первыми месяцами программы.</span></div>', unsafe_allow_html=True)


# ───────────────────────── ЦЕННОСТИ ─────────────────────────
def render_values(fd, emp):
    st.markdown('<div class="section-header">Карта ценностей</div>', unsafe_allow_html=True)
    vc = fd.groupby(TX["value"]).agg(Меритов=(TX["merits"], "sum")).reset_index()
    if len(vc) == 0:
        st.info("Нет ценностей в выборке."); return
    vc["Доля, %"] = (vc["Меритов"] / vc["Меритов"].sum() * 100).round(1)
    vc = vc.sort_values("Доля, %", ascending=False).reset_index(drop=True)
    K = len(vc); top3 = vc.head(3)["Доля, %"].sum()
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        if go is not None:
            fig = go.Figure(go.Bar(x=vc["Доля, %"], y=vc[TX["value"]], orientation="h", marker_color=CORAL))
            light(fig, "Доля меритов по ценностям", 380)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(vc, use_container_width=True, hide_index=True)
    with c2:
        st.markdown(f'<div class="card"><strong>Признание держится на {K} ценностях;</strong> '
                    f'на три ведущие приходится {top3:.0f}% всех меритов '
                    f'({", ".join(vc.head(3)[TX["value"]])}).<br>'
                    f'<span class="muted">Чем выше доля немногих ценностей, тем у́же язык признания в компании.</span></div>',
                    unsafe_allow_html=True)
        if panels is not None:
            panels.render_value_people(fd, emp)
    if len(fd) < 200:
        st.caption("Выборка невелика — доли по отдельным ценностям могут заметно колебаться.")


def render_grade_dynamics(tx, emp):
    st.markdown('<div class="section-header">Темпоральная динамика градовой структуры</div>', unsafe_allow_html=True)
    if go is None or grade_dynamics_figure is None:
        st.info("Модуль грейдов недоступен (см. диагностику)."); return
    st.markdown('<div class="info-box"><strong>Грады</strong> — типы ценностей по Болтански–Тевено: за какой принцип '
                'благодарят. Надёжность и функция (индустриальный), забота и поддержка (патриархальный), результат '
                '(рыночный), общее дело (гражданский), идея и вдохновение, репутация и имя, проектная гибкость. '
                'Графики показывают, как менялась доля этих типов по годам.<br>'
                '<span class="muted">Сопоставление конкретных ценностей с типами — предварительное; читать как '
                'гипотезу о языке признания, не как точную классификацию.</span></div>', unsafe_allow_html=True)
    comps = sorted(emp[EMP["company"]].dropna().unique().tolist())
    if not comps:
        return
    default = "Тепловое Оборудование" if "Тепловое Оборудование" in comps else comps[0]
    unit = st.selectbox("Подразделение", comps, index=comps.index(default))
    fig = grade_dynamics_figure(go, make_subplots, tx, EMP["company"], unit)
    st.plotly_chart(fig, use_container_width=True)


def render_value_evolution(fd):
    st.markdown('<div class="section-header">Эволюция ценностей</div>', unsafe_allow_html=True)
    if go is None:
        return
    fdt = fd.dropna(subset=["dt"])
    if len(fdt) < 2:
        st.info("Недостаточно данных."); return
    vy = fdt.groupby([fdt["dt"].dt.year.rename("y"), TX["value"]]).size().reset_index(name="n")
    tab = vy.pivot(index="y", columns=TX["value"], values="n").fillna(0)
    share = tab.div(tab.sum(axis=1), axis=0) * 100
    years = sorted(share.index)
    if len(years) < 2:
        st.info("Нужно ≥2 года для динамики."); return
    topv = share.loc[years[-1]].sort_values(ascending=False).head(7).index.tolist()
    WARM = ["#e95f3e", "#5e7d16", "#c9871f", "#c0492f", "#8a9a3f", "#b5743a", "#6b8e23"]
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        fig = go.Figure()
        for i, v in enumerate(topv):
            fig.add_trace(go.Scatter(x=[str(y) for y in years], y=[round(share.loc[y, v], 1) for y in years],
                                     name=v[:24], mode="lines+markers", line=dict(color=WARM[i % len(WARM)], width=2)))
        light(fig, "Доля ведущих ценностей по годам, %", 340)
        fig.update_layout(legend=dict(orientation="h", y=-0.28, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        delta = (share.loc[years[-1]] - share.loc[years[0]]).sort_values()
        st.markdown(f'<div class="card"><strong>Что поднялось и просело</strong> (с {years[0]} к {years[-1]}):<br>'
                    f'↑ <strong>{delta.index[-1]}</strong> (+{delta.iloc[-1]:.0f} пп) · '
                    f'↓ <strong>{delta.index[0]}</strong> ({delta.iloc[0]:.0f} пп).<br>'
                    f'<span class="muted">Формулировки ценностей менялись тремя волнами: 25 исходных названий сведены '
                    f'к 17 устойчивым. Ниже — доли этих волн по годам.</span></div>', unsafe_allow_html=True)
        if "lineage_version" in fdt.columns:
            gen = fdt.groupby([fdt["dt"].dt.year.rename("y"), "lineage_version"]).size().reset_index(name="n")
            gt = gen.pivot(index="y", columns="lineage_version", values="n").fillna(0)
            gs = gt.div(gt.sum(axis=1), axis=0) * 100
            figg = go.Figure()
            for i, g in enumerate([c for c in ["gen1", "gen2", "gen3"] if c in gs.columns]):
                figg.add_trace(go.Bar(x=[str(y) for y in gs.index], y=gs[g].round(0).values, name=g,
                                      marker_color=["#cf8b22", "#6b8e23", "#e95f3e"][i % 3]))
            light(figg, "Доли волн формулировок по годам, %", 220)
            figg.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(figg, use_container_width=True)


def render_tenure(fd, emp):
    st.markdown('<div class="section-header">Стаж и онбординг</div>', unsafe_allow_html=True)
    if "Дата_найма" not in emp.columns:
        st.info("Нет даты найма в данных."); return
    e = emp[emp[EMP["fire"]].isna()].copy().dropna(subset=["Дата_найма"])
    if len(e) == 0:
        st.info("Нет дат найма у активных."); return
    ref = fd["dt"].max() if ("dt" in fd.columns and fd["dt"].notna().any()) else pd.Timestamp.now()
    e["ten"] = (ref - e["Дата_найма"]).dt.days / 365.25
    labels = ["<6 мес", "6–12 мес", "1–2 года", "2–3 года", "3–5 лет", "5+ лет"]
    e["coh"] = pd.cut(e["ten"], [-1, 0.5, 1, 2, 3, 5, 100], labels=labels)
    sent = fd.groupby(TX["sid"]).agg(n=("dt", "size"), mo=("ym", "nunique"))
    sent["rate"] = sent["n"] / sent["mo"].clip(lower=1)
    e = e.set_index(EMP["id"]); e["rate"] = sent["rate"].reindex(e.index).fillna(0)
    n_active = max(int(emp[EMP["fire"]].isna().sum()), 1)
    a, b, cc = st.columns(3)
    a.metric("Медианный стаж", f"{e['ten'].median():.1f} лет")
    b.metric("Новичков (<6 мес)", f"{int((e['coh'] == '<6 мес').sum())}")
    cc.metric("Известен стаж", f"{len(e)/n_active*100:.0f}% активных")
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        if go is not None:
            cr = e.groupby("coh", observed=False)["rate"].mean().reindex(labels)
            fig = go.Figure(go.Bar(x=labels, y=cr.round(2).fillna(0).values, marker_color=CORAL))
            light(fig, "Средняя активность (актов/мес на человека) по стажу", 300)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        new_rate = e[e["coh"] == "<6 мес"]["rate"].mean()
        rest_rate = e[e["coh"] != "<6 мес"]["rate"].mean()
        n_new = int((e["coh"] == "<6 мес").sum())
        if n_new >= 3 and rest_rate:
            ratio = new_rate / rest_rate if rest_rate else 0
            if ratio > 1.3:
                msg, kl = f"Новички активнее остальных в ~{ratio:.1f}× — на старте часто благодарят чаще, это нормальное включение в практику.", "care"
            elif ratio < 0.7:
                msg, kl = "Новички заметно менее активны — стоит посмотреть на онбординг в программу.", "care"
            else:
                msg, kl = "Активность новичков близка к остальным — ровное включение.", "good"
            st.markdown(f'<div class="card {kl}"><strong>Онбординг.</strong> {n_new} новичков (&lt;6 мес): '
                        f'{new_rate:.1f} актов/мес против {rest_rate:.1f} у остальных. {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card"><strong>Онбординг.</strong> Новичков (&lt;6 мес) сейчас мало ({n_new}) — '
                        f'для устойчивого вывода нужна большая когорта.</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Дата найма известна не у всех — см. покрытие выше. '
                    'Полноценные ALP-фазы жизненного цикла участия — следующий шаг на этих данных.</div>', unsafe_allow_html=True)


# ───────────────────────── СИГНАЛЫ ВНИМАНИЯ (помесячно) ─────────────────────────
def compute_leave_monthly(tx_all, emp):
    active_ids = set(emp[emp[EMP["fire"]].isna()][EMP["id"]])
    a = tx_all.dropna(subset=["dt"])
    a = a[a[TX["sid"]].isin(active_ids)]
    if len(a) == 0:
        return dict(months=[], counts=[], current_month=None, rows=[], detail={})
    per = a.groupby([a[TX["sid"]], a["dt"].dt.to_period("M")]).size().unstack(fill_value=0)
    months = sorted(per.columns)
    if months:
        last = months[-1]
        last_days = a[a["dt"].dt.to_period("M") == last]["dt"].dt.day.max()
        if last_days < 25 and len(months) > 1:
            months = months[:-1]
            per = per[months]
    counts, sig_month, sig_detail, month_signals = [], {}, {}, {}
    for i, mo in enumerate(months):
        if i < 3:
            counts.append(0); continue
        base = per[months[i-3:i]].mean(axis=1)
        cur = per[mo]
        flag = (base >= 2) & (cur < 0.5 * base)
        counts.append(int(flag.sum()))
        month_signals[str(mo)] = list(per.index[flag])
        for pid in per.index[flag]:
            sig_month[pid] = str(mo)
            sig_detail[pid] = (int(cur[pid]), round(float(base[pid]), 1))
    cur_mo = str(months[-1]) if months else None
    rows = [pid for pid, mo in sig_month.items() if mo == cur_mo]
    return dict(months=[str(m) for m in months], counts=counts, current_month=cur_mo,
                rows=rows, detail=sig_detail, month_signals=month_signals)


def validate_leave_on_fired(tx_all, emp):
    fired = emp[emp[EMP["fire"]].notna()]; silent, total = 0, 0
    for _, row in fired.iterrows():
        pid, fdate = row[EMP["id"]], row[EMP["fire"]]
        sub = tx_all[(tx_all[TX["sid"]] == pid) & (tx_all["dt"] < fdate)]
        if len(sub) < 10:
            continue
        total += 1
        gap = (fdate - sub["dt"].max()).days
        gaps = sub.sort_values("dt")["dt"].diff().dt.days.dropna()
        med = gaps.median() if len(gaps) else 30
        if gap > max(med * 2.5, 45):
            silent += 1
    return silent, total


def render_leave(tx_all, emp, closed_view):
    st.markdown('<div class="section-header">Ранние сигналы внимания</div>', unsafe_allow_html=True)
    lv = compute_leave_monthly(tx_all, emp)
    sil, tot = validate_leave_on_fired(tx_all, emp)
    if go is not None and lv["months"]:
        mm = emp.set_index(EMP["id"]); mm = mm[~mm.index.duplicated(keep="first")]
        ms = lv.get("month_signals", {})
        names = []
        for mo in lv["months"]:
            pids = ms.get(mo, [])
            nm = [str(mm["full_name"].get(p, p)) for p in pids[:14]]
            extra = len(pids) - len(nm)
            names.append(("<br>".join(nm) + (f"<br>…ещё {extra}" if extra > 0 else "")) if nm else "—")
        fig = go.Figure(go.Bar(x=lv["months"], y=lv["counts"], customdata=names, marker_color=AMBER,
            hovertemplate="<b>%{x}</b><br>Сигналов: %{y}<br><br><b>Кто:</b><br>%{customdata}<extra></extra>"))
        light(fig, "Сколько участников в каждом месяце снизили активность вдвое+ относительно своих 3 предыдущих месяцев", 240)
        st.plotly_chart(fig, use_container_width=True, key=f"leave_chart_{closed_view}")
    n_now = len(lv["rows"])
    st.markdown(
        f'<div class="card care"><strong>Сигнал внимания, не прогноз ухода.</strong><br>'
        f'В {lv["current_month"]}: <strong>{n_now}</strong> участников заметно снизили активность относительно '
        f'своих же предыдущих месяцев (индикатор помесячный, а не «в целом за период»).<br>'
        f'<span class="muted">Возможные причины: сезонный спад, смена роли или проекта, уход близкого коллеги, '
        f'перегрузка — и лишь в числе прочего эмоциональная усталость. Рекомендация: при желании мягко '
        f'поинтересоваться состоянием. Не используется для кадровых решений.</span></div>', unsafe_allow_html=True)
    if tot:
        st.markdown(f'<div class="card"><strong>Историческая основа.</strong> Из {tot} ранее уволившихся '
                    f'<strong>{sil} ({sil/tot*100:.0f}%)</strong> заметно замолкали перед уходом — затухание участия '
                    f'стоит замечать вовремя.</div>', unsafe_allow_html=True)
    if closed_view and lv["rows"]:
        m = emp.set_index(EMP["id"]); m = m[~m.index.duplicated(keep="first")]
        det = lv.get("detail", {})
        rows = [{"ФИО": m["full_name"].get(p, p), "Отдел": m[EMP["dept"]].get(p, ""),
                 "Актов в этом месяце": det.get(p, (0, 0))[0], "Обычно за месяц": det.get(p, (0, 0))[1]} for p in lv["rows"]]
        with st.expander(f"Кто подаёт сигнал в {lv['current_month']} ({len(rows)})"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Только для HR/доверенного руководителя. Рамка заботы: поддержка, не санкции.")


# ───────────────────────── ГРАФ (расчёт) ─────────────────────────
def build_graph(fd, emp, merit_range):
    if nx is None:
        return None, None
    m = emp.set_index(EMP["id"]); m = m[~m.index.duplicated(keep="first")]
    agg = fd.groupby([TX["sid"], TX["rid"]]).agg(total=(TX["merits"], "sum"), n=(TX["merits"], "size")).reset_index()
    lo, hi = merit_range
    agg = agg[(agg["total"] >= lo) & (agg["total"] <= hi) & (agg[TX["sid"]] != agg[TX["rid"]])]

    def attrs(eid):
        if eid in m.index:
            r = m.loc[eid]
            nm = f"{r.get(EMP['last'],'')} {str(r.get(EMP['first'],''))[:1]}.".strip()
            return dict(label=nm, dept=str(r.get(EMP["dept"], "")), company=str(r.get(EMP["company"], "")),
                        position=str(r.get(EMP["pos"], "")), role=str(r.get("Уровень", "")))
        return dict(label=eid, dept="", company="", position="", role="")

    G = nx.DiGraph()
    for _, row in agg.iterrows():
        s, rr = row[TX["sid"]], row[TX["rid"]]
        if s not in G: G.add_node(s, **attrs(s))
        if rr not in G: G.add_node(rr, **attrs(rr))
        G.add_edge(s, rr, weight=float(row["total"]), msgs=int(row["n"]))
    for u, v in G.edges():
        G[u][v]["mutual"] = G.has_edge(v, u)
    return G, None


def graph_metrics(G):
    if G is None or G.number_of_nodes() == 0:
        return {}
    mt = {"in_strength": dict(G.in_degree(weight="weight")), "out_strength": dict(G.out_degree(weight="weight"))}
    try: mt["pagerank"] = nx.pagerank(G, weight="weight", max_iter=200)
    except Exception: mt["pagerank"] = {n: 1.0/G.number_of_nodes() for n in G.nodes()}; Debug.note_warning("pagerank fallback")
    UG = G.to_undirected(); n = G.number_of_nodes()
    try: mt["betweenness"] = nx.betweenness_centrality(UG, k=min(200, n), normalized=True, seed=42)
    except Exception: mt["betweenness"] = {x: 0.0 for x in G.nodes()}; Debug.note_warning("betweenness fallback")
    try: mt["is_cut"] = {x: int(x in set(nx.articulation_points(UG))) for x in G.nodes()}
    except Exception: mt["is_cut"] = {x: 0 for x in G.nodes()}
    if community_louvain is not None:
        try:
            part = community_louvain.best_partition(UG, weight="weight", random_state=42)
            mt["communities"] = part; mt["modularity"] = community_louvain.modularity(part, UG, weight="weight")
        except Exception:
            mt["communities"] = {x: 0 for x in G.nodes()}; mt["modularity"] = 0.0; Debug.note_warning("louvain fallback")
    else:
        mt["communities"] = {x: 0 for x in G.nodes()}; mt["modularity"] = 0.0
    try: mt["reciprocity"] = nx.reciprocity(G) if G.number_of_edges() else 0.0
    except Exception: mt["reciprocity"] = 0.0
    return mt


def render_vertical(fd, emp):
    st.markdown('<div class="section-header">Вертикаль признания · руководители и специалисты</div>', unsafe_allow_html=True)
    if "s_Уровень" not in fd.columns or "Уровень" not in emp.columns:
        st.info("Нет данных об уровне (рук/спец)."); return
    sub = fd.dropna(subset=["s_Уровень", "r_Уровень"])
    sub = sub[sub["s_Уровень"].isin(["рук", "спец"]) & sub["r_Уровень"].isin(["рук", "спец"])]
    if len(sub) == 0:
        st.info("Нет вертикальных данных в выборке."); return
    fl = sub.groupby(["s_Уровень", "r_Уровень"]).size()
    cell = lambda a, b: int(fl.get((a, b), 0))
    spsp, up, down, rr = cell("спец", "спец"), cell("спец", "рук"), cell("рук", "спец"), cell("рук", "рук")
    tot = spsp + up + down + rr
    pct = lambda x: (x / tot * 100) if tot else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Горизонталь спец↔спец", f"{pct(spsp):.0f}%", help=f"специалисты благодарят специалистов · {spsp:,} актов")
    c2.metric("Вверх: спец→рук", f"{pct(up):.0f}%", help=f"специалисты благодарят руководителей · {up:,} актов")
    c3.metric("Вниз: рук→спец", f"{pct(down):.0f}%", help=f"руководители благодарят специалистов · {down:,} актов")
    c4.metric("Среди руковод. рук↔рук", f"{pct(rr):.0f}%", help=f"руководители благодарят руководителей · {rr:,} актов")

    active = emp[emp[EMP["fire"]].isna()]
    nruk = max(int((active["Уровень"] == "рук").sum()), 1)
    nspec = max(int((active["Уровень"] == "спец").sum()), 1)
    recv_ruk, recv_spec = up + rr, spsp + down
    sent_ruk, sent_spec = down + rr, spsp + up

    g1, g2 = st.columns([3, 2], vertical_alignment="center")
    with g1:
        if go is not None:
            fig = go.Figure(go.Heatmap(z=[[rr, down], [up, spsp]],
                x=["получатель: рук", "получатель: спец"], y=["отправитель: рук", "отправитель: спец"],
                text=[[rr, down], [up, spsp]], texttemplate="%{text}",
                colorscale=[[0, "#fbeee6"], [1, "#e95f3e"]], showscale=False))
            light(fig, "Кто кого признаёт (число актов)", 280)
            st.plotly_chart(fig, use_container_width=True)
            figb = go.Figure()
            figb.add_trace(go.Bar(name="Руководители", x=["получает на 1 чел", "отдаёт на 1 чел"],
                                  y=[round(recv_ruk / nruk, 1), round(sent_ruk / nruk, 1)], marker_color="#e95f3e"))
            figb.add_trace(go.Bar(name="Специалисты", x=["получает на 1 чел", "отдаёт на 1 чел"],
                                  y=[round(recv_spec / nspec, 1), round(sent_spec / nspec, 1)], marker_color="#6b8e23"))
            light(figb, "Признание на одного человека (делим на число людей уровня — чтобы сравнивать честно)", 260)
            figb.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(figb, use_container_width=True)
    with g2:
        vert = up + down
        lead = "горизонтально, внутри своего уровня" if (spsp + rr) > vert else "по вертикали, между уровнями"
        st.markdown(f'<div class="card"><strong>Геометрия признания.</strong> Признание идёт преимущественно {lead}: '
                    f'горизонталь {pct(spsp) + pct(rr):.0f}%, вертикаль {pct(vert):.0f}% '
                    f'(вверх {pct(up):.0f}% · вниз {pct(down):.0f}%).<br>'
                    f'<span class="muted">Перекос вверх — внимание снизу к руководителям; перекос вниз — руководители '
                    f'замечают подчинённых. Ни один полюс не «лучше» — это разные стили вертикальной культуры.</span></div>',
                    unsafe_allow_html=True)
        active_ruk = set(active[active["Уровень"] == "рук"][EMP["id"]])
        got_from_spec = set(sub[sub["s_Уровень"] == "спец"][TX["rid"]]) & active_ruk
        no_up = len(active_ruk - got_from_spec)
        if active_ruk:
            st.markdown(f'<div class="card care"><strong>Руководители без признания снизу.</strong> '
                        f'{no_up} из {len(active_ruk)} руководителей за период не получили ни одной благодарности '
                        f'от специалистов.<br><span class="muted">Сигнал внимания, не оценка: возможны разные причины '
                        f'(роль вне прямого контакта, фрейм отдела, специфика функции). Читать через парную процедуру.</span></div>',
                        unsafe_allow_html=True)


def render_curator(fd, emp, cur):
    st.markdown('<div class="section-header">Кураторская вертикаль · руководитель и подчинённые</div>', unsafe_allow_html=True)
    cts = cur.get("cur_to_subs", {}) if cur else {}
    if not cts:
        st.info("Кураторские связи не разрешены (нет листа оргструктуры)."); return
    active = set(emp[emp[EMP["fire"]].isna()][EMP["id"]])
    out_t = fd.groupby(TX["sid"])[TX["rid"]].apply(set)
    in_s = fd.groupby(TX["rid"])[TX["sid"]].apply(set)
    mgrs = [(c, set(s) & active) for c, s in cts.items() if c in active]
    mgrs = [(c, s) for c, s in mgrs if s]
    if not mgrs:
        st.info("Нет руководителей с активными подчинёнными в выборке."); return
    dri_list, up_list, mc_list, no_recog, no_up = [], [], [], 0, 0
    for c, subs in mgrs:
        sent, recv = out_t.get(c, set()), in_s.get(c, set())
        dri = len(subs & sent) / len(subs)
        up = len(subs & recv) / len(subs)
        dri_list.append(dri); up_list.append(up)
        if dri == 0: no_recog += 1
        if up == 0: no_up += 1
        group = subs | {c}
        gi = fd[fd[TX["sid"]].isin(group) & fd[TX["rid"]].isin(group)]
        if len(gi):
            mc_list.append(float(((gi[TX["sid"]] == c) | (gi[TX["rid"]] == c)).mean()))
    n_m = len(mgrs)
    a, b, c4, d = st.columns(4)
    a.metric("Руководителей (привязано)", f"{n_m}")
    b.metric("Признают подчинённых", f"{np.mean(dri_list)*100:.0f}%", help="средняя доля подчинённых, которых руководитель поблагодарил хотя бы раз")
    c4.metric("Признаны подчинёнными", f"{np.mean(up_list)*100:.0f}%", help="средняя доля подчинённых, признавших своего руководителя")
    d.metric("Руковод.-центричность", f"{np.mean(mc_list)*100:.0f}%" if mc_list else "—", help="доля внутригрупповых актов с участием руководителя")

    mm = emp.set_index(EMP["id"]); mm = mm[~mm.index.duplicated(keep="first")]
    keys = ["0% (никого)", "<50%", "50–99%", "100% (всех)"]
    bnames = {k: [] for k in keys}
    for (cid, _subs), v in zip(mgrs, dri_list):
        k = "0% (никого)" if v == 0 else "<50%" if v < 0.5 else "50–99%" if v < 1 else "100% (всех)"
        bnames[k].append(str(mm["full_name"].get(cid, cid)))
    bcust = []
    for k in keys:
        nm = bnames[k][:16]; extra = len(bnames[k]) - len(nm)
        bcust.append(("<br>".join(nm) + (f"<br>…ещё {extra}" if extra > 0 else "")) if nm else "—")
    g1, g2 = st.columns([3, 2], vertical_alignment="center")
    with g1:
        if go is not None:
            fig = go.Figure(go.Bar(x=keys, y=[len(bnames[k]) for k in keys], customdata=bcust, marker_color=CORAL,
                hovertemplate="<b>%{x}</b><br>Руководителей: %{y}<br><br><b>Кто:</b><br>%{customdata}<extra></extra>"))
            light(fig, "Сколько руководителей какой доле подчинённых дали признание", 300)
            st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown(f'<div class="card care"><strong>Руководители без признания подчинённых.</strong> '
                    f'{no_recog} из {n_m} руководителей за период не поблагодарили ни одного своего подчинённого.<br>'
                    f'<span class="muted">Сигнал внимания, не оценка: возможны причины (стиль управления, '
                    f'дистанционная роль, признание вне платформы). Читать через парную процедуру.</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="card"><strong>Руководители без признания снизу.</strong> '
                    f'{no_up} из {n_m} не получили ни одной благодарности от своих подчинённых за период.</div>',
                    unsafe_allow_html=True)
    st.caption(f"Кураторы привязаны автоматически: {cur.get('n_resolved',0)} из {cur.get('n_links',0)} связей "
               f"(матричных, с двойным куратором — {cur.get('n_matrix',0)}). Непривязанные — вне расчёта; "
               f"метрики читать с поправкой на покрытие.")


def render_network_health(G, mt):
    st.markdown('<div class="section-header">Сеть признания</div>', unsafe_allow_html=True)
    if G is None or G.number_of_nodes() == 0:
        st.info("Граф пуст."); return
    n, e = G.number_of_nodes(), G.number_of_edges()
    if n < 15:
        st.markdown(f'<div class="card care"><strong>Выборка мала ({n} человек в сети).</strong> '
                    f'Структурные метрики (взаимность, сообщества, незаменимые связки) на такой выборке '
                    f'недостоверны и не рассчитываются.<br><span class="muted">Снимите фильтры или возьмите более '
                    f'крупное подразделение/период, чтобы увидеть структуру сети.</span></div>', unsafe_allow_html=True)
        return
    vals = sorted(mt.get("pagerank", {}).values(), reverse=True)
    top10 = sum(vals[:max(1, n // 10)]) / (sum(vals) or 1)
    rec = mt.get("reciprocity", 0); n_comm = len(set(mt.get("communities", {}).values()))
    n_cut = int(sum(mt.get("is_cut", {}).values()))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Сотрудников в сети", f"{n:,}"); c2.metric("Связей признания", f"{e:,}")
    c3.metric("Взаимность", f"{rec:.2f}", help="Доля связей, где признание идёт в обе стороны")
    c4.metric("Сообществ", f"{n_comm}", help="Группы плотного взаимного признания")
    dist = "распределённое" if top10 < 0.35 else ("умеренно сосредоточенное" if top10 < 0.55 else "сильно сосредоточенное")
    st.markdown(f'<div class="card"><strong>Распределённость признания.</strong> На верхние 10% узлов приходится '
                f'{top10*100:.0f}% признания — сеть {dist}.<br><span class="muted">Распределённое — ресурс устойчивости. '
                f'Сосредоточенное часто отражает естественные центры; тревожно лишь при падении их активности.</span></div>',
                unsafe_allow_html=True)
    if n_cut:
        st.markdown(f'<div class="card care"><strong>Незаменимые связки.</strong> {n_cut} узлов соединяют части сети, '
                    f'которые иначе распались бы.<br><span class="muted">Структурная характеристика позиции, не оценка '
                    f'человека. Полезно, чтобы межотделовые связи держались не на одном человеке.</span></div>', unsafe_allow_html=True)
    if n < 30:
        st.caption("Сеть небольшая (меньше 30 человек) — деление на сообщества и «незаменимые связки» приблизительны.")


def render_analyst(G, mt):
    st.markdown('<div class="section-header">Аналитический режим (числа)</div>', unsafe_allow_html=True)
    if G is None or G.number_of_nodes() == 0:
        st.info("Граф пуст."); return
    rows = [{"ФИО": G.nodes[x].get("label", ""), "Компания": G.nodes[x].get("company", ""),
             "Отдел": G.nodes[x].get("dept", ""), "PageRank": round(mt["pagerank"].get(x, 0), 5),
             "Betweenness": round(mt["betweenness"].get(x, 0), 5), "Cut-vertex": mt.get("is_cut", {}).get(x, 0),
             "Сообщество": mt["communities"].get(x, 0), "Входящих": round(mt["in_strength"].get(x, 0), 1),
             "Исходящих": round(mt["out_strength"].get(x, 0), 1)} for x in G.nodes()]
    df = pd.DataFrame(rows).sort_values("PageRank", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True, height=420)
    st.download_button("📥 Метрики (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "metrics.csv", "text/csv")


# ───────────────────────── ГРАФ (D3, встроен) ─────────────────────────
PALETTE = ["#e95f3e","#5e7d16","#c9871f","#c0492f","#8a9a3f","#b5743a","#6b8e23","#d98a5a",
           "#9aa83f","#a85d3a","#e0a23e","#90ae3c","#cf8b22","#8a6d3b","#d2693e","#7d8b2f"]


def social_graph_html(G, mt):
    comm = mt.get("communities", {}); pr = mt.get("pagerank", {})
    nodes = [dict(id=str(n), label=G.nodes[n].get("label", str(n)), dept=G.nodes[n].get("dept", ""),
                  company=G.nodes[n].get("company", ""), position=G.nodes[n].get("position", ""), role=G.nodes[n].get("role", ""),
                  community=comm.get(n, 0), pagerank=pr.get(n, 0),
                  ins=mt.get("in_strength", {}).get(n, 0), outs=mt.get("out_strength", {}).get(n, 0))
             for n in G.nodes()]
    edges = [dict(source=str(u), target=str(v), weight=d.get("weight", 1), mutual=bool(d.get("mutual", False)))
             for u, v, d in G.edges(data=True)]
    n_comm = max(1, len(set(comm.values())))
    colors = (PALETTE * (n_comm // len(PALETTE) + 1))[:max(n_comm, 1)]
    return (_SOCIAL.replace("__NODES__", json.dumps(nodes)).replace("__LINKS__", json.dumps(edges))
            .replace("__COLORS__", json.dumps(colors)))


def hierarchy_html(G_people, mt_people):
    pr = mt_people.get("pagerank", {})
    people = [dict(id=str(n), label=G_people.nodes[n].get("label", str(n)),
                   dept=G_people.nodes[n].get("dept", ""), company=G_people.nodes[n].get("company", ""),
                   deptkey=(str(G_people.nodes[n].get("company", "")) + " / " + str(G_people.nodes[n].get("dept", ""))),
                   position=G_people.nodes[n].get("position", ""), role=G_people.nodes[n].get("role", ""), pagerank=pr.get(n, 0),
                   ins=mt_people.get("in_strength", {}).get(n, 0), outs=mt_people.get("out_strength", {}).get(n, 0))
              for n in G_people.nodes()]
    pe = [dict(source=str(u), target=str(v), weight=d.get("weight", 1)) for u, v, d in G_people.edges(data=True)]
    return _HIER.replace("__PEOPLE__", json.dumps(people)).replace("__PEDGES__", json.dumps(pe))


def org_graph_html(emp, curators):
    cts = curators.get("cur_to_subs", {}) if curators else {}
    stc = curators.get("sub_to_curs", {}) if curators else {}
    if not cts:
        return None
    m = emp.set_index(EMP["id"]); m = m[~m.index.duplicated(keep="first")]
    people = set(cts.keys()) | set(stc.keys())
    nsub = {c: len(v) for c, v in cts.items()}
    nodes = []
    for pid in people:
        if pid in m.index:
            r = m.loc[pid]
            label = f"{r.get(EMP['last'],'')} {str(r.get(EMP['first'],''))[:1]}.".strip()
            nodes.append(dict(id=pid, label=label, role=str(r.get("Уровень", "")), dept=str(r.get(EMP["dept"], "")),
                              company=str(r.get(EMP["company"], "")), position=str(r.get(EMP["pos"], "")), nsub=nsub.get(pid, 0)))
        else:
            nodes.append(dict(id=pid, label=pid, role="", dept="", company="", position="", nsub=nsub.get(pid, 0)))
    edges = [dict(source=sub, target=cur) for sub, curs in stc.items() for cur in curs if sub in people and cur in people]
    return _ORG.replace("__ONODES__", json.dumps(nodes)).replace("__OEDGES__", json.dumps(edges))


_SOCIAL = "<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n<style> body{margin:0;background:#262220;font-family:'Golos Text',-apple-system,sans-serif;overflow:hidden} #viz{width:100%;height:100vh} .controls{position:absolute;top:12px;right:12px;z-index:1000;display:flex;gap:6px} .btn{background:#332e29;color:#e9d9c8;border:1px solid #3a342f;padding:6px 14px;border-radius:9px;cursor:pointer;font-size:12px;font-weight:600;font-family:'Golos Text',sans-serif} .btn:hover{background:#3a342f} .btn.on{background:#90ae3c;color:#262220;border-color:#90ae3c} #bc{position:absolute;top:14px;left:14px;color:#f3724f;font-family:'Golos Text',sans-serif;font-size:13px;font-weight:600} #tip{position:absolute;background:#2a2521;border:1px solid #3a342f;color:#f1e8e2;padding:10px 14px;border-radius:10px;font-size:12px;pointer-events:none;opacity:0;transition:opacity .2s;max-width:240px;line-height:1.6}</style></head><body>\n<div class=\"controls\">\n <button class=\"btn\" onclick=\"rz()\">↺ Сброс</button>\n <button class=\"btn\" onclick=\"tl()\">Метки</button>\n <button class=\"btn\" id=\"cb\" onclick=\"tc()\">Роль</button>\n <button class=\"btn\" id=\"mb\" onclick=\"tm()\">Только взаимные</button>\n <button class=\"btn\" onclick=\"tp()\">Физика</button>\n</div>\n<div id=\"tip\"></div><svg id=\"viz\"></svg>\n<script>\nfunction CW(){const v=document.getElementById(\"viz\");return (v&&v.clientWidth)||innerWidth;}\nfunction CH(){const v=document.getElementById(\"viz\");return (v&&v.clientHeight)||innerHeight;}\nconst nodes=__NODES__,allLinks=__LINKS__,colors=__COLORS__;\nlet links=allLinks.slice(), mutualOnly=false, labelsOn=true;\nlet selected=new Set(),colorMode=\"comm\";\nconst byId={}; nodes.forEach(n=>byId[n.id]=n);\nconst adj={}; nodes.forEach(n=>adj[n.id]=new Set());\nallLinks.forEach(l=>{adj[l.source].add(l.target);adj[l.target].add(l.source);});\nconst prv=nodes.map(n=>n.pagerank),mn=Math.min(...prv),mx=Math.max(...prv),rg=(mx-mn)||1;\nconst R=d=>3+27*Math.pow((d.pagerank-mn)/rg,0.7);\nconst RMIN=3,RMAX=30,NB=5;\nfunction rb(id){const n=byId[id];if(!n)return 0;return Math.max(0,Math.min(NB-1,Math.floor((R(n)-RMIN)/(RMAX-RMIN)*NB)));}\nfunction eid(x){return (x&&x.id!==undefined)?x.id:x;}\nfunction ekey(l){const s=eid(l.source),t=eid(l.target);return s<t?s+\"|\"+t:t+\"|\"+s;}\nconst svg=d3.select(\"#viz\").attr(\"width\",\"100%\").attr(\"height\",\"100%\");\nconst defs=svg.append(\"defs\");\nfunction mk(prefix,color){for(let b=0;b<NB;b++){const sz=7+b*3;\n  defs.append(\"marker\").attr(\"id\",prefix+b).attr(\"viewBox\",\"0 -5 10 10\").attr(\"refX\",9).attr(\"refY\",0)\n   .attr(\"markerUnits\",\"userSpaceOnUse\").attr(\"markerWidth\",sz).attr(\"markerHeight\",sz).attr(\"orient\",\"auto\")\n   .append(\"path\").attr(\"d\",\"M0,-4L8,0L0,4\").attr(\"fill\",color);}}\nmk(\"arr\",\"#5a524b\"); mk(\"arrm\",\"#90ae3c\");\nconst g=svg.append(\"g\");\nconst zoom=d3.zoom().scaleExtent([.05,12]).on(\"zoom\",e=>g.attr(\"transform\",e.transform));\nsvg.call(zoom);\nsvg.on(\"click\",()=>{selected.clear();refresh();});\nlet link=g.append(\"g\"), node, labels;\nfunction drawLinks(){\n link.selectAll(\"line\").remove();\n window._le=link.selectAll(\"line\").data(links).join(\"line\")\n   .attr(\"stroke\",d=>d.mutual?\"#90ae3c\":\"#3a342f\").attr(\"stroke-opacity\",.55)\n   .attr(\"stroke-width\",d=>Math.sqrt(d.weight)*.6+.4)\n   .attr(\"marker-end\",d=>\"url(#\"+(d.mutual?\"arrm\":\"arr\")+rb(eid(d.target))+\")\");\n}\ndrawLinks();\nfunction nodeFill(d){return colorMode===\"role\"?(d.role===\"рук\"?\"#e95f3e\":\"#6b8e23\"):colors[d.community%colors.length];}\nnode=g.append(\"g\").selectAll(\"circle\").data(nodes).join(\"circle\")\n .attr(\"r\",R).attr(\"fill\",nodeFill).attr(\"stroke\",\"#262220\").attr(\"stroke-width\",1.5).attr(\"cursor\",\"pointer\")\n .on(\"click\",(e,d)=>{e.stopPropagation(); if(e.ctrlKey||e.metaKey){selected.has(d.id)?selected.delete(d.id):selected.add(d.id);} else {selected=new Set([d.id]);} refresh();})\n .on(\"mouseover\",(e,d)=>{const t=document.getElementById(\"tip\");\n   t.innerHTML=`<strong>${d.label}</strong><br>${d.position}<br><span style=\"color:#f3724f\">${d.company}</span> / ${d.dept}<hr style=\"border-color:#3a342f;margin:6px 0\">Входящих: ${d.ins.toFixed(0)} · Исходящих: ${d.outs.toFixed(0)}<br><em style=\"color:#a89d95\">клик — связи · Ctrl+клик — путь между узлами</em>`;\n   t.style.opacity=1;t.style.left=(e.pageX+12)+\"px\";t.style.top=(e.pageY-10)+\"px\";})\n .on(\"mouseout\",()=>document.getElementById(\"tip\").style.opacity=0)\n .call(d3.drag().on(\"start\",(e,d)=>{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;})\n   .on(\"drag\",(e,d)=>{d.fx=e.x;d.fy=e.y;}).on(\"end\",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));\nlabels=g.append(\"g\").selectAll(\"text\").data(nodes).join(\"text\").attr(\"fill\",\"#a89d95\").attr(\"font-size\",\"10px\").attr(\"text-anchor\",\"middle\").attr(\"dy\",d=>-R(d)-3).attr(\"pointer-events\",\"none\")\n .text(d=>d.label.length>18?d.label.slice(0,18)+\"…\":d.label);\nfunction bfs(a,b){const prev={},q=[a],seen=new Set([a]);\n while(q.length){const cur=q.shift();if(cur===b){const p=[b];let c=b;while(c!==a){c=prev[c];p.unshift(c);}return p;}\n   adj[cur].forEach(nx=>{if(!seen.has(nx)){seen.add(nx);prev[nx]=cur;q.push(nx);}});}\n return null;}\nlet pathNodes=new Set(),pathEdges=new Set();\nfunction recomputePaths(){pathNodes=new Set();pathEdges=new Set();const sel=[...selected];\n for(let i=0;i<sel.length;i++)for(let j=i+1;j<sel.length;j++){const p=bfs(sel[i],sel[j]);if(!p)continue;\n   p.forEach(x=>pathNodes.add(x));for(let k=0;k+1<p.length;k++){const a=p[k],b=p[k+1];pathEdges.add(a<b?a+\"|\"+b:b+\"|\"+a);}}}\nfunction markSel(){node.attr(\"stroke\",d=>selected.has(d.id)?\"#faf2ea\":\"#262220\").attr(\"stroke-width\",d=>selected.has(d.id)?2.5:1.5);}\nfunction refresh(){\n if(selected.size===0){node.attr(\"opacity\",1);markSel();labels.style(\"opacity\",labelsOn?1:0);window._le.attr(\"stroke-opacity\",.55).attr(\"stroke\",d=>d.mutual?\"#90ae3c\":\"#3a342f\");return;}\n if(selected.size===1){const sid=[...selected][0],near=adj[sid];\n   node.attr(\"opacity\",d=>(d.id===sid||near.has(d.id))?1:.12);\n   labels.style(\"opacity\",d=>(labelsOn&&(d.id===sid||near.has(d.id)))?1:0);\n   window._le.attr(\"stroke-opacity\",l=>(eid(l.source)===sid||eid(l.target)===sid)?.95:.05)\n            .attr(\"stroke\",l=>(eid(l.source)===sid||eid(l.target)===sid)?\"#f3724f\":(l.mutual?\"#90ae3c\":\"#3a342f\"));\n   markSel();return;}\n recomputePaths();\n node.attr(\"opacity\",d=>pathNodes.has(d.id)?1:.1);\n labels.style(\"opacity\",d=>(labelsOn&&pathNodes.has(d.id))?1:0);\n window._le.attr(\"stroke-opacity\",l=>pathEdges.has(ekey(l))?.95:.04).attr(\"stroke\",l=>pathEdges.has(ekey(l))?\"#f3724f\":(l.mutual?\"#90ae3c\":\"#3a342f\"));\n markSel();}\nconst sim=d3.forceSimulation(nodes)\n .force(\"link\",d3.forceLink(links).id(d=>d.id).distance(70))\n .force(\"charge\",d3.forceManyBody().strength(-200))\n .force(\"center\",d3.forceCenter(CW()/2,CH()/2))\n .force(\"collision\",d3.forceCollide().radius(d=>R(d)+3))\n .on(\"tick\",()=>{window._le.each(function(d){const s=d.source,t=d.target,dx=t.x-s.x,dy=t.y-s.y,dist=Math.hypot(dx,dy)||1,r=R(t),ux=dx/dist,uy=dy/dist;\n     d3.select(this).attr(\"x1\",s.x).attr(\"y1\",s.y).attr(\"x2\",t.x-ux*r).attr(\"y2\",t.y-uy*r);});\n   node.attr(\"cx\",d=>d.x).attr(\"cy\",d=>d.y);labels.attr(\"x\",d=>d.x).attr(\"y\",d=>d.y);});\nfunction rz(){svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity);}\nfunction tl(){labelsOn=!labelsOn;refresh();}\nfunction tm(){mutualOnly=!mutualOnly;document.getElementById(\"mb\").classList.toggle(\"on\",mutualOnly);\n  links=mutualOnly?allLinks.filter(l=>l.mutual):allLinks.slice();\n  sim.force(\"link\",d3.forceLink(links).id(d=>d.id).distance(70));drawLinks();refresh();sim.alpha(.3).restart();}\nlet po=true;function tp(){po=!po;po?sim.alpha(.3).restart():sim.stop();}\n\ntry{new ResizeObserver(()=>{if(CW()>1&&CH()>1){sim.force(\"center\",d3.forceCenter(CW()/2,CH()/2)).alpha(.25).restart();}}).observe(document.getElementById(\"viz\"));}catch(e){}\n\nfunction tc(){colorMode=colorMode===\"comm\"?\"role\":\"comm\";var b=document.getElementById(\"cb\");if(b)b.classList.toggle(\"on\",colorMode===\"role\");node.attr(\"fill\",nodeFill);}\n\n</script></body></html>"
_HIER = "<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n<style> body{margin:0;background:#262220;font-family:'Golos Text',-apple-system,sans-serif;overflow:hidden} #viz{width:100%;height:100vh} .controls{position:absolute;top:12px;right:12px;z-index:1000;display:flex;gap:6px} .btn{background:#332e29;color:#e9d9c8;border:1px solid #3a342f;padding:6px 14px;border-radius:9px;cursor:pointer;font-size:12px;font-weight:600;font-family:'Golos Text',sans-serif} .btn:hover{background:#3a342f} .btn.on{background:#90ae3c;color:#262220;border-color:#90ae3c} #bc{position:absolute;top:14px;left:14px;color:#f3724f;font-family:'Golos Text',sans-serif;font-size:13px;font-weight:600} #tip{position:absolute;background:#2a2521;border:1px solid #3a342f;color:#f1e8e2;padding:10px 14px;border-radius:10px;font-size:12px;pointer-events:none;opacity:0;transition:opacity .2s;max-width:240px;line-height:1.6}</style></head><body>\n<div id=\"bc\">Уровень: Компании</div>\n<div class=\"controls\">\n <button class=\"btn\" onclick=\"back()\">← Назад</button>\n <button class=\"btn\" onclick=\"home()\">↺ Компании</button>\n <button class=\"btn\" onclick=\"allDepts()\">Все отделы ГК</button>\n <button class=\"btn\" onclick=\"rz()\">⊕ Сброс</button>\n <button class=\"btn\" onclick=\"tp()\">Физика</button>\n</div>\n<div id=\"tip\"></div><svg id=\"viz\"></svg>\n<script>\nfunction CW(){const v=document.getElementById(\"viz\");return (v&&v.clientWidth)||innerWidth;}\nfunction CH(){const v=document.getElementById(\"viz\");return (v&&v.clientHeight)||innerHeight;}\nconst PEOPLE=__PEOPLE__,PE=__PEDGES__;\nconst byId={}; PEOPLE.forEach(p=>byId[p.id]=p);\nconst peSet=new Set(); PE.forEach(l=>peSet.add(l.source+\">\"+l.target));\nlet nodes=[],links=[],level=\"companies\",sim,navStack=[{t:\"companies\"}],selected=new Set(),adj={},curMax=1,curById={};\nconst svg=d3.select(\"#viz\").attr(\"width\",\"100%\").attr(\"height\",\"100%\");\nconst defs=svg.append(\"defs\"),NB=5;\nfunction mk(prefix,color){for(let b=0;b<NB;b++){const sz=7+b*3;\n  defs.append(\"marker\").attr(\"id\",prefix+b).attr(\"viewBox\",\"0 -5 10 10\").attr(\"refX\",9).attr(\"refY\",0)\n   .attr(\"markerUnits\",\"userSpaceOnUse\").attr(\"markerWidth\",sz).attr(\"markerHeight\",sz).attr(\"orient\",\"auto\")\n   .append(\"path\").attr(\"d\",\"M0,-4L8,0L0,4\").attr(\"fill\",color);}}\nmk(\"arr\",\"#5a524b\"); mk(\"arrm\",\"#90ae3c\");\nconst g=svg.append(\"g\");\nconst zoom=d3.zoom().scaleExtent([.05,12]).on(\"zoom\",e=>g.attr(\"transform\",e.transform));svg.call(zoom);\nsvg.on(\"click\",()=>{if(level===\"people\"){selected.clear();hl();}});\nlet le,ne,la;\nfunction Rof(d){return d.person?6:(8+22*Math.sqrt(d.size/curMax));}\nfunction rb(d){return Math.max(0,Math.min(NB-1,Math.floor((Rof(d)-6)/(30-6)*NB)));}\nfunction eid(x){return (x&&x.id!==undefined)?x.id:x;}\nfunction aggregate(keyFn,labelFn,extra){\n const groups={}; PEOPLE.forEach(p=>{const k=keyFn(p);(groups[k]=groups[k]||[]).push(p);});\n const nd=Object.entries(groups).map(([k,mem])=>Object.assign({id:k,label:labelFn(k),size:mem.length,members:new Set(mem.map(m=>m.id))},extra?extra(k):{}));\n const ew={}; PE.forEach(l=>{const a=byId[l.source],b=byId[l.target];if(!a||!b)return;const ka=keyFn(a),kb=keyFn(b);if(ka===kb)return;const key=ka+\"||\"+kb;ew[key]=(ew[key]||0)+l.weight;});\n const ed=Object.entries(ew).map(([k,w])=>{const p=k.split(\"||\");return{source:p[0],target:p[1],weight:w,mutual:!!ew[p[1]+\"||\"+p[0]]};});\n return {nodes:nd,links:ed};\n}\nfunction makeCompanies(){const a=aggregate(p=>p.company||\"—\",k=>k);return{nodes:a.nodes,links:a.links,level:\"companies\",crumb:\"Компании\"};}\nfunction makeAllDepts(){const a=aggregate(p=>p.deptkey||\"—\",k=>k.split(\" / \").pop());return{nodes:a.nodes,links:a.links,level:\"alldepts\",crumb:\"Все отделы ГК\"};}\nfunction makeDepts(company){\n const sub=PEOPLE.filter(p=>(p.company||\"—\")===company);const ids=new Set(sub.map(p=>p.id));\n const groups={};sub.forEach(p=>{(groups[p.deptkey]=groups[p.deptkey]||[]).push(p);});\n const nd=Object.entries(groups).map(([k,mem])=>({id:k,label:k.split(\" / \").pop(),size:mem.length,members:new Set(mem.map(m=>m.id))}));\n const ew={};PE.forEach(l=>{if(!ids.has(l.source)||!ids.has(l.target))return;const a=byId[l.source],b=byId[l.target];if(a.deptkey===b.deptkey)return;const key=a.deptkey+\"||\"+b.deptkey;ew[key]=(ew[key]||0)+l.weight;});\n const ed=Object.entries(ew).map(([k,w])=>{const p=k.split(\"||\");return{source:p[0],target:p[1],weight:w,mutual:!!ew[p[1]+\"||\"+p[0]]};});\n return{nodes:nd,links:ed,level:\"depts\",crumb:company+\" → отделы\"};\n}\nfunction makePeople(deptkey){\n const sub=PEOPLE.filter(p=>p.deptkey===deptkey);const ids=new Set(sub.map(p=>p.id));\n const nd=sub.map(p=>({id:p.id,label:p.label,size:1,person:true,position:p.position,company:p.company,dept:p.dept,ins:p.ins,outs:p.outs,role:p.role}));\n const ed=PE.filter(l=>ids.has(l.source)&&ids.has(l.target)).map(l=>({source:l.source,target:l.target,weight:l.weight,mutual:peSet.has(l.target+\">\"+l.source)}));\n return{nodes:nd,links:ed,level:\"people\",crumb:deptkey};\n}\nfunction build(desc){\n if(desc.t===\"companies\")return makeCompanies();\n if(desc.t===\"alldepts\")return makeAllDepts();\n if(desc.t===\"depts\")return makeDepts(desc.company);\n return makePeople(desc.deptkey);\n}\nfunction go(desc,push){\n if(push)navStack.push(desc);\n const r=build(desc);nodes=r.nodes;links=r.links;level=r.level;selected.clear();\n curMax=Math.max(1,...nodes.map(n=>n.size));\n adj={};curById={};nodes.forEach(n=>{adj[n.id]=new Set();curById[n.id]=n;});links.forEach(l=>{adj[l.source].add(l.target);adj[l.target].add(l.source);});\n document.getElementById(\"bc\").textContent=\"Уровень: \"+navStack.map(crumbOf).join(\"  ›  \");\n sim&&sim.stop();init();recenter();\n}\nfunction recenter(){svg.call(zoom.transform,d3.zoomIdentity);setTimeout(()=>{if(sim){sim.force(\"center\",d3.forceCenter(CW()/2,CH()/2)).alpha(.5).restart();}svg.call(zoom.transform,d3.zoomIdentity);},90);}\nfunction crumbOf(d){return d.t===\"companies\"?\"Компании\":d.t===\"alldepts\"?\"Все отделы\":d.t===\"depts\"?d.company:d.deptkey.split(\" / \").pop();}\nfunction back(){if(navStack.length>1){navStack.pop();go(navStack[navStack.length-1],false);}}\nfunction home(){navStack=[{t:\"companies\"}];go(navStack[0],false);}\nfunction allDepts(){navStack=[{t:\"companies\"},{t:\"alldepts\"}];go(navStack[1],false);}\nfunction hl(){\n if(level!==\"people\"||selected.size===0){ne.attr(\"opacity\",1);window._lh&&window._lh.attr(\"stroke-opacity\",.6);return;}\n const sid=[...selected][0],near=adj[sid];\n ne.attr(\"opacity\",d=>(d.id===sid||near.has(d.id))?1:.12);\n window._lh.attr(\"stroke-opacity\",l=>(eid(l.source)===sid||eid(l.target)===sid)?.95:.05).attr(\"stroke\",l=>(eid(l.source)===sid||eid(l.target)===sid)?\"#f3724f\":(l.mutual?\"#90ae3c\":\"#3a342f\"));\n}\nfunction init(){\n g.selectAll(\"*\").remove();\n le=g.append(\"g\"); window._lh=le.selectAll(\"line\").data(links).join(\"line\").attr(\"stroke\",d=>d.mutual?\"#90ae3c\":\"#3a342f\").attr(\"stroke-opacity\",.6).attr(\"stroke-width\",d=>Math.sqrt(d.weight)*.4+.6).attr(\"marker-end\",d=>\"url(#\"+(d.mutual?\"arrm\":\"arr\")+rb(curById[eid(d.target)]||{size:1,person:false})+\")\");\n ne=g.append(\"g\").selectAll(\"circle\").data(nodes).join(\"circle\")\n  .attr(\"r\",Rof).attr(\"fill\",d=>d.person?(d.role===\"рук\"?\"#e95f3e\":\"#6b8e23\"):(level===\"companies\"?\"#e95f3e\":\"#cf8b22\")).attr(\"stroke\",\"#262220\").attr(\"stroke-width\",2).attr(\"cursor\",\"pointer\")\n  .on(\"click\",(e,d)=>{e.stopPropagation();\n     if(level===\"companies\")go({t:\"depts\",company:d.id},true);\n     else if(level===\"depts\"||level===\"alldepts\")go({t:\"people\",deptkey:d.id},true);\n     else {selected=new Set([d.id]);hl();}})\n  .on(\"mouseover\",(e,d)=>{const t=document.getElementById(\"tip\");\n    t.innerHTML=d.person?`<strong>${d.label}</strong><br>${d.position}<br><span style=\"color:#f3724f\">${d.company}</span> / ${d.dept}<br>Входящих: ${d.ins.toFixed(0)} · Исходящих: ${d.outs.toFixed(0)}`:`<strong>${d.label}</strong><br>Участников: ${d.size}<br><em style=\"color:#a89d95\">клик — раскрыть</em>`;\n    t.style.opacity=1;t.style.left=(e.pageX+12)+\"px\";t.style.top=(e.pageY-10)+\"px\";})\n  .on(\"mouseout\",()=>document.getElementById(\"tip\").style.opacity=0)\n  .call(d3.drag().on(\"start\",(e,d)=>{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;})\n   .on(\"drag\",(e,d)=>{d.fx=e.x;d.fy=e.y;}).on(\"end\",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));\n la=g.append(\"g\").selectAll(\"text\").data(nodes).join(\"text\").attr(\"fill\",\"#a89d95\").attr(\"font-size\",\"10px\").attr(\"text-anchor\",\"middle\").attr(\"dy\",d=>-Rof(d)-3).attr(\"pointer-events\",\"none\").text(d=>d.label&&d.label.length>22?d.label.slice(0,22)+\"…\":d.label);\n sim=d3.forceSimulation(nodes).force(\"link\",d3.forceLink(links).id(d=>d.id).distance(level===\"people\"?70:150))\n  .force(\"charge\",d3.forceManyBody().strength(level===\"people\"?-160:-360)).force(\"center\",d3.forceCenter(CW()/2,CH()/2))\n  .force(\"collision\",d3.forceCollide().radius(d=>Rof(d)+6))\n  .on(\"tick\",()=>{window._lh.each(function(d){const s=d.source,t=d.target,dx=t.x-s.x,dy=t.y-s.y,dist=Math.hypot(dx,dy)||1,r=Rof(t),ux=dx/dist,uy=dy/dist;\n      d3.select(this).attr(\"x1\",s.x).attr(\"y1\",s.y).attr(\"x2\",t.x-ux*r).attr(\"y2\",t.y-uy*r);});\n    ne.attr(\"cx\",d=>d.x).attr(\"cy\",d=>d.y);la.attr(\"x\",d=>d.x).attr(\"y\",d=>d.y);});\n}\nfunction rz(){svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity);}\nlet po=true;function tp(){po=!po;po?sim.alpha(.3).restart():sim.stop();}\ngo(navStack[0],false);\n\ntry{new ResizeObserver(()=>{if(CW()>1&&CH()>1){if(sim){sim.force(\"center\",d3.forceCenter(CW()/2,CH()/2)).alpha(.35).restart();}svg.call(zoom.transform,d3.zoomIdentity);}}).observe(document.getElementById(\"viz\"));}catch(e){}\n\n</script></body></html>"
_ORG = "<!DOCTYPE html><html><head><meta charset=\"utf-8\">\n<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n<style> body{margin:0;background:#262220;font-family:'Golos Text',-apple-system,sans-serif;overflow:hidden} #viz{width:100%;height:100vh} .controls{position:absolute;top:12px;right:12px;z-index:1000;display:flex;gap:6px} .btn{background:#332e29;color:#e9d9c8;border:1px solid #3a342f;padding:6px 14px;border-radius:9px;cursor:pointer;font-size:12px;font-weight:600;font-family:'Golos Text',sans-serif} .btn:hover{background:#3a342f} .btn.on{background:#90ae3c;color:#262220;border-color:#90ae3c} #bc{position:absolute;top:14px;left:14px;color:#f3724f;font-family:'Golos Text',sans-serif;font-size:13px;font-weight:600} #tip{position:absolute;background:#2a2521;border:1px solid #3a342f;color:#f1e8e2;padding:10px 14px;border-radius:10px;font-size:12px;pointer-events:none;opacity:0;transition:opacity .2s;max-width:240px;line-height:1.6}</style></head><body>\n<div class=\"controls\">\n <button class=\"btn\" onclick=\"rz()\">↺ Сброс</button>\n <button class=\"btn\" onclick=\"tl()\">Метки</button>\n <button class=\"btn\" onclick=\"tp()\">Физика</button>\n</div>\n<div id=\"tip\"></div><svg id=\"viz\"></svg>\n<script>\nfunction CW(){const v=document.getElementById(\"viz\");return (v&&v.clientWidth)||innerWidth;}\nfunction CH(){const v=document.getElementById(\"viz\");return (v&&v.clientHeight)||innerHeight;}\nconst nodes=__ONODES__, links=__OEDGES__;\nconst byId={}; nodes.forEach(n=>byId[n.id]=n);\nconst adj={}; nodes.forEach(n=>adj[n.id]=new Set());\nlinks.forEach(l=>{adj[l.source].add(l.target);adj[l.target].add(l.source);});\nconst mx=Math.max(1,...nodes.map(n=>n.nsub||0));\nconst R=d=>6+18*Math.sqrt((d.nsub||0)/mx);\nfunction eid(x){return (x&&x.id!==undefined)?x.id:x;}\nfunction rcol(d){return d.role===\"рук\"?\"#e95f3e\":(d.role===\"спец\"?\"#6b8e23\":\"#a89d95\");}\nconst svg=d3.select(\"#viz\").attr(\"width\",\"100%\").attr(\"height\",\"100%\");\nconst defs=svg.append(\"defs\");\ndefs.append(\"marker\").attr(\"id\",\"oarr\").attr(\"viewBox\",\"0 -5 10 10\").attr(\"refX\",9).attr(\"refY\",0).attr(\"markerUnits\",\"userSpaceOnUse\").attr(\"markerWidth\",9).attr(\"markerHeight\",9).attr(\"orient\",\"auto\").append(\"path\").attr(\"d\",\"M0,-4L8,0L0,4\").attr(\"fill\",\"#5a524b\");\nconst g=svg.append(\"g\");\nconst zoom=d3.zoom().scaleExtent([.04,12]).on(\"zoom\",e=>g.attr(\"transform\",e.transform));svg.call(zoom);\nlet selected=null,labelsOn=false;\nsvg.on(\"click\",()=>{selected=null;refresh();});\nconst link=g.append(\"g\").selectAll(\"line\").data(links).join(\"line\").attr(\"stroke\",\"#3a342f\").attr(\"stroke-opacity\",.5).attr(\"stroke-width\",1).attr(\"marker-end\",\"url(#oarr)\");\nconst node=g.append(\"g\").selectAll(\"circle\").data(nodes).join(\"circle\").attr(\"r\",R).attr(\"fill\",rcol).attr(\"stroke\",\"#262220\").attr(\"stroke-width\",1.5).attr(\"cursor\",\"pointer\")\n  .on(\"click\",(e,d)=>{e.stopPropagation();selected=(selected===d.id?null:d.id);refresh();})\n  .on(\"mouseover\",(e,d)=>{const t=document.getElementById(\"tip\");\n    t.innerHTML=`<strong>${d.label}</strong><br>${d.position}<br><span style=\"color:#f3724f\">${d.company}</span> / ${d.dept}<br>${d.role===\"рук\"?(\"Подчинённых: \"+d.nsub):\"специалист\"}`;\n    t.style.opacity=1;t.style.left=(e.pageX+12)+\"px\";t.style.top=(e.pageY-10)+\"px\";})\n  .on(\"mouseout\",()=>document.getElementById(\"tip\").style.opacity=0)\n  .call(d3.drag().on(\"start\",(e,d)=>{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}).on(\"drag\",(e,d)=>{d.fx=e.x;d.fy=e.y;}).on(\"end\",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));\nconst labels=g.append(\"g\").selectAll(\"text\").data(nodes).join(\"text\").attr(\"fill\",\"#a89d95\").attr(\"font-size\",\"9px\").attr(\"text-anchor\",\"middle\").attr(\"dy\",d=>-R(d)-3).attr(\"pointer-events\",\"none\").text(d=>d.label);\nlabels.style(\"opacity\",0);\nfunction refresh(){\n  if(!selected){node.attr(\"opacity\",1);labels.style(\"opacity\",labelsOn?1:0);link.attr(\"stroke-opacity\",.5).attr(\"stroke\",\"#3a342f\");return;}\n  const near=adj[selected];\n  node.attr(\"opacity\",d=>(d.id===selected||near.has(d.id))?1:.12);\n  labels.style(\"opacity\",d=>(d.id===selected||near.has(d.id))?1:0);\n  link.attr(\"stroke-opacity\",l=>(eid(l.source)===selected||eid(l.target)===selected)?.95:.05).attr(\"stroke\",l=>(eid(l.source)===selected||eid(l.target)===selected)?\"#f3724f\":\"#3a342f\");\n}\nconst sim=d3.forceSimulation(nodes).force(\"link\",d3.forceLink(links).id(d=>d.id).distance(48)).force(\"charge\",d3.forceManyBody().strength(-150)).force(\"center\",d3.forceCenter(CW()/2,CH()/2)).force(\"collision\",d3.forceCollide().radius(d=>R(d)+3))\n  .on(\"tick\",()=>{link.each(function(d){const s=d.source,t=d.target,dx=t.x-s.x,dy=t.y-s.y,dist=Math.hypot(dx,dy)||1,r=R(t);d3.select(this).attr(\"x1\",s.x).attr(\"y1\",s.y).attr(\"x2\",t.x-dx/dist*r).attr(\"y2\",t.y-dy/dist*r);});node.attr(\"cx\",d=>d.x).attr(\"cy\",d=>d.y);labels.attr(\"x\",d=>d.x).attr(\"y\",d=>d.y);});\nfunction rz(){svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity);}\nfunction tl(){labelsOn=!labelsOn;refresh();}\nlet po=true;function tp(){po=!po;po?sim.alpha(.3).restart():sim.stop();}\ntry{new ResizeObserver(()=>{if(CW()>1&&CH()>1){sim.force(\"center\",d3.forceCenter(CW()/2,CH()/2)).alpha(.25).restart();}}).observe(document.getElementById(\"viz\"));}catch(e){}\n\n</script></body></html>"


# ───────────────────────── ДИАГНОСТИКА UI ─────────────────────────
def render_diagnostics():
    st.markdown("---")
    has_err = len(Debug.errors) > 0
    title = "🔧 Диагностика — ЕСТЬ ОШИБКИ (раскройте и скопируйте)" if has_err else "🔧 Диагностика (временный блок)"
    with st.expander(title, expanded=has_err):
        st.caption("Временный блок для отладки при загрузке на GitHub / Streamlit Cloud. Скопируйте текст ниже.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Окружение**"); st.json(Debug.info)
        with c2:
            st.markdown("**Этапы**")
            if Debug.stages:
                st.dataframe(pd.DataFrame(Debug.stages, columns=["этап", "статус", "сек", "сообщение"]),
                             use_container_width=True, hide_index=True)
        if Debug.warns:
            st.markdown("**Предупреждения**"); st.code("\n".join(Debug.warns[:50]), language="text")
        if has_err:
            st.markdown("**Трейсбеки**")
            for where, tb in Debug.errors:
                st.code(f"# {where}\n{tb}", language="python")
        st.markdown("**Полный отчёт для копирования**")
        st.text_area("report", Debug.report_text(), height=240, label_visibility="collapsed")


# ───────────────────────── MAIN ─────────────────────────
def main():
    Debug.reset()
    for k, v in _IMPORT_ERR.items():
        Debug.errors.append((f"import {k}", v))

    dark = bool(st.session_state.get("dark", False))
    theme.set_dark(dark)
    inject_theme(dark)
    localize_widgets()

    base = os.path.dirname(os.path.abspath(__file__))
    st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;padding:.3rem 0 .2rem">
        {logo_html(base)}
        <div><span style="font-size:1.1rem;font-weight:700;color:#e95f3e;">3Д Коммуникации</span>
        <span style="color:#a89d95;font-size:.85rem;margin-left:8px;">Социальные технологии для бизнеса</span></div>
        </div>
        <div class="hero">Живая ткань признания</div>
        <div class="muted" style="margin-bottom:.4rem">ГК Термекс · программа «3Д Коммуникации»</div>""",
        unsafe_allow_html=True)

    data_path = os.path.join(base, "data_clean.xlsx")
    miss = [] if os.path.exists(data_path) else ["data_clean.xlsx"]
    if miss:
        st.error(f"❌ Не найдены файлы: {', '.join(miss)}")
        Debug.errors.append(("файлы", f"отсутствуют: {miss}")); render_diagnostics(); return

    try:
        with st.spinner("Загрузка данных…"):
            with Debug.stage("load_employees"): emp = load_employees(data_path)
            with Debug.stage("load_transactions"): tx_raw = load_transactions(data_path)
            with Debug.stage("load_limits"): LIMITS = load_limits(data_path)
            with Debug.stage("merge_data"): tx = merge_data(tx_raw.copy(), emp)
            if set_grade_map: set_grade_map(load_grade_map(data_path))
            with Debug.stage("resolve_curators"): CURATORS = resolve_curators(data_path)
            Debug.info["employees"] = len(emp); Debug.info["transactions"] = len(tx)
            try: Debug.info["период"] = f"{tx['dt'].min():%Y-%m} … {tx['dt'].max():%Y-%m}"
            except Exception: pass

        cfg = sidebar_controls(tx, emp)
        with Debug.stage("apply_filters"):
            fd = apply_filters(tx, cfg)
        if len(fd) == 0:
            st.warning("⚠️ Нет данных под выбранные фильтры"); render_diagnostics(); return

        n_comp = fd["s_" + EMP["company"]].nunique(); n_dep = fd["s_dept_key"].nunique()
        n_ppl = pd.Index(fd[TX["sid"]]).append(pd.Index(fd[TX["rid"]])).nunique()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Транзакций", f"{len(fd):,}")
        c2.metric("Меритов", f"{int(fd[TX['merits']].sum()):,}",
                  help="суммарные голоса; потолок на один акт менялся 6→10 (01.09.2025) — сравнивать периоды осторожно")
        c3.metric("Компаний", f"{n_comp}"); c4.metric("Отделов", f"{n_dep}"); c5.metric("Сотрудников", f"{n_ppl:,}")

        # Условное предупреждение об охвате — всплывает только при низком покрытии (универсально для разных компаний)
        if TX["comment"] in fd.columns and len(fd):
            cov = fd[TX["comment"]].notna().mean()
            if cov < 0.30:
                st.caption(f"⚠️ Комментарии заполнены лишь в {cov*100:.0f}% актов — выводы по тексту и «глубине» "
                           f"ограничены: это голос пишущих, не всего коллектива.")

        with Debug.stage("build_graph"):
            G, _ = build_graph(fd, emp, cfg["merit_range"])
        with Debug.stage("graph_metrics"):
            mt = graph_metrics(G)

        t_org, t_net, t_pass, t_an = st.tabs(["Организация", "Сеть признания", "Меритпаспорт", "Аналитик"])

        with t_org:
            with Debug.stage("render_funnel", fatal=False): render_funnel(compute_funnel(emp, fd))
            with Debug.stage("render_temporal", fatal=False): render_temporal(fd, emp, LIMITS)
            with Debug.stage("render_values", fatal=False): render_values(fd, emp)
            with Debug.stage("render_value_evolution", fatal=False): render_value_evolution(fd)
            with Debug.stage("render_grade_dynamics", fatal=False): render_grade_dynamics(tx, emp)
            with Debug.stage("render_tenure", fatal=False): render_tenure(fd, emp)
            st.markdown('<div class="section-header">Рейтинг охвата</div>', unsafe_allow_html=True)
            if panels is not None:
                with Debug.stage("render_rating", fatal=False): panels.render_rating(fd, emp)
            with Debug.stage("render_leave", fatal=False): render_leave(tx, emp, closed_view=True)

        with t_net:
            st.markdown('<div class="section-header">Визуализация сети</div>', unsafe_allow_html=True)
            if components is not None and G is not None and G.number_of_nodes() > 0:
                st.markdown('<div class="info-box">Стрелка показывает направление благодарности (толще к более '
                            'признаваемым узлам). <span style="color:#6b8e23">Оливковые связи — взаимные</span>. '
                            'Клик по узлу подсвечивает его связи; <strong>Ctrl+клик</strong> по нескольким узлам — '
                            'подсветка пути между ними. «Только взаимные» оставляет двусторонние.<br>'
                            'Граф крупный — удобнее начать с фильтра по компании или отделу слева.</div>', unsafe_allow_html=True)
                gt = st.tabs(["Социальный граф", "Компании → отделы → люди", "Оргструктура (рук/спец, матрица)"])
                with gt[0]:
                    with Debug.stage("social_graph", fatal=False):
                        components.html(social_graph_html(G, mt), height=1040, scrolling=False)
                with gt[1]:
                    with Debug.stage("hierarchy", fatal=False):
                        components.html(hierarchy_html(G, mt), height=1040, scrolling=False)
                with gt[2]:
                    with Debug.stage("org_graph", fatal=False):
                        oh = org_graph_html(emp, CURATORS)
                        if oh:
                            st.markdown('<div class="info-box">Формальная структура подчинения: '
                                        '<span style="color:#e95f3e">руководители</span> · '
                                        '<span style="color:#6b8e23">специалисты</span>. Стрелка — «подчиняется»; '
                                        'двойные рёбра — матричное (двойное) подчинение; размер узла — число подчинённых. '
                                        'Клик по узлу — подсветить связи. Это формальная иерархия, отдельно от сети признания.</div>',
                                        unsafe_allow_html=True)
                            components.html(oh, height=1040, scrolling=False)
                        else:
                            st.info("Кураторские связи не разрешены (нет листа оргструктуры).")
            elif components is None:
                st.warning("streamlit.components недоступен (см. диагностику).")
            with Debug.stage("render_network_health", fatal=False): render_network_health(G, mt)
            with Debug.stage("render_vertical", fatal=False): render_vertical(fd, emp)
            with Debug.stage("render_curator", fatal=False): render_curator(fd, emp, CURATORS)

        with t_pass:
            st.markdown('<div class="section-header">Меритпаспорт сотрудника</div>', unsafe_allow_html=True)
            if panels is not None:
                with Debug.stage("meritpassport", fatal=False): panels.render_meritpassport(fd, tx, emp)

        with t_an:
            with Debug.stage("render_analyst", fatal=False): render_analyst(G, mt)

    except Exception:
        st.error("Произошла ошибка — подробности в блоке диагностики ниже.")
        if not Debug.errors or Debug.errors[-1][0] != "main":
            Debug.errors.append(("main", traceback.format_exc()))

    render_diagnostics()


if __name__ == "__main__":
    main()
