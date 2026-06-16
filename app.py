# -*- coding: utf-8 -*-
"""СоциоГраф — аналитика программы «3Д Коммуникации». Графика D3 в graph_viz.py,
панели в panels.py, грейды в grades.py.
Зависимости: streamlit, pandas, numpy, networkx, plotly, python-louvain, scipy, openpyxl"""

import os
import time
import platform
import traceback
import warnings
import contextlib
from datetime import datetime

import numpy as np
import pandas as pd


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
    from graph_viz import social_graph_html, hierarchy_html
except Exception:
    social_graph_html = hierarchy_html = None; _IMPORT_ERR["graph_viz"] = traceback.format_exc()
try:
    import panels
except Exception:
    panels = None; _IMPORT_ERR["panels"] = traceback.format_exc()
try:
    from grades import grade_dynamics_figure
except Exception:
    grade_dynamics_figure = None; _IMPORT_ERR["grades"] = traceback.format_exc()

Debug.info = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
              "networkx": getattr(nx, "__version__", "—"), "streamlit": getattr(st, "__version__", "—"),
              "plotly": "ok" if go else "—", "louvain": "ok" if community_louvain else "—",
              "graph_viz": "ok" if social_graph_html else "—", "panels": "ok" if panels else "—",
              "grades": "ok" if grade_dynamics_figure else "—"}


# ───────────────────────── СТИЛЬ ─────────────────────────
st.set_page_config(page_title="3Д Коммуникации", page_icon="🕸️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
 @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
 html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
 .main{background:#0d1117;} .block-container{padding:1.5rem 2rem;}
 h1,h2,h3{font-family:'IBM Plex Mono',monospace !important;color:#58a6ff !important;font-weight:600 !important;letter-spacing:-0.5px;}
 [data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace;font-size:1.7rem !important;color:#58a6ff !important;}
 [data-testid="stMetricLabel"]{font-size:0.72rem !important;color:#8b949e !important;text-transform:uppercase;letter-spacing:1px;}
 .kpi-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem 1.1rem;min-height:96px;}
 .kpi-box .kpi-value{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;color:#58a6ff;line-height:1.1;}
 .kpi-box .kpi-label{font-size:0.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
 .kpi-box .kpi-sub{font-size:0.72rem;color:#6e7681;margin-top:6px;}
 .section-header{font-family:'IBM Plex Mono',monospace;font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;color:#8b949e;border-bottom:1px solid #30363d;padding-bottom:6px;margin:1.5rem 0 1rem;}
 .card{background:#161b22;border:1px solid #30363d;border-left:3px solid #58a6ff;padding:0.9rem 1.1rem;border-radius:6px;margin:0.5rem 0;font-size:0.9rem;color:#c9d1d9;line-height:1.6;}
 .card strong{color:#58a6ff;} .care{border-left-color:#d29922;} .care strong{color:#d29922;}
 .good{border-left-color:#3fb950;} .good strong{color:#3fb950;} .muted{color:#8b949e;font-size:0.82rem;}
 .info-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:0.9rem;margin:0.6rem 0;font-size:0.83rem;color:#8b949e;line-height:1.6;}
 .stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#8b949e;}
 .stTabs [aria-selected="true"]{color:#58a6ff !important;}
 div[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
 .sidebar-section{font-size:0.7rem;text-transform:uppercase;letter-spacing:1.5px;color:#8b949e;margin:1rem 0 0.3rem;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────── СХЕМА ─────────────────────────
MERITS_PER_MONTH = 10
EMP = {"last":"Фамилия","first":"Имя","mid":"Отчество","id":"Персональный номер",
       "pos":"Должность","company":"Компания","dept":"Отдел","fire":"Дата увольнения"}
TX  = {"date":"Дата","time":"Время","sid":"Номер отправителя","rid":"Номер получателя",
       "value":"Ценность","merits":"Мериты","comment":"Комментарий"}


# ───────────────────────── ЗАГРУЗКА ─────────────────────────
@st.cache_data(show_spinner=False)
def load_employees(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": None, "None": None})
    if EMP["fire"] in df.columns:
        df[EMP["fire"]] = pd.to_datetime(df[EMP["fire"]], dayfirst=True, errors="coerce")
    df[EMP["id"]] = pd.to_numeric(df[EMP["id"]], errors="coerce").fillna(-1).astype(int).astype(str).str.strip()
    df["full_name"] = (df[EMP["last"]].fillna("") + " " + df[EMP["first"]].fillna("") + " " +
                       df[EMP["mid"]].fillna("")).str.strip()
    df["dept_key"] = df[EMP["company"]].fillna("?") + " / " + df[EMP["dept"]].fillna("?")
    return df


@st.cache_data(show_spinner=False)
def load_transactions(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    if TX["date"] in df.columns:
        df["dt"] = pd.to_datetime(df[TX["date"]], dayfirst=True, errors="coerce")
    elif TX["time"] in df.columns:
        df["dt"] = pd.to_datetime(df[TX["time"]], errors="coerce")
    else:
        df["dt"] = pd.NaT
    df[TX["merits"]] = pd.to_numeric(df[TX["merits"]], errors="coerce").fillna(0).astype(int)
    df[TX["sid"]] = pd.to_numeric(df[TX["sid"]], errors="coerce").fillna(-1).astype(int).astype(str).str.strip()
    df[TX["rid"]] = pd.to_numeric(df[TX["rid"]], errors="coerce").fillna(-1).astype(int).astype(str).str.strip()
    n0 = len(df)
    df = df[(df[TX["sid"]] != "-1") & (df[TX["rid"]] != "-1")]
    if n0 - len(df):
        Debug.note_warning(f"отброшено {n0-len(df)} актов с нераспознанным id")
    if TX["value"] in df.columns:
        df[TX["value"]] = df[TX["value"]].astype(str).str.strip()
    df["day"] = df["dt"].dt.day
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    df["ym"] = df["dt"].dt.to_period("M").astype(str)
    return df


def merge_data(tx, emp):
    m = emp.set_index(EMP["id"]); m = m[~m.index.duplicated(keep="first")]
    cols = ["full_name", EMP["pos"], EMP["company"], EMP["dept"], "dept_key"]
    for pref, idc in (("s", TX["sid"]), ("r", TX["rid"])):
        for c in cols:
            tx[f"{pref}_{c}"] = tx[idc].map(m[c]) if c in m.columns else None
        if EMP["fire"] in m.columns:
            tx[f"{pref}_fire"] = tx[idc].map(m[EMP["fire"]])
    return tx


# ───────────────────────── ФИЛЬТРЫ ─────────────────────────
def sidebar_controls(tx, emp):
    st.sidebar.markdown("## ⚙️ Фильтры")
    st.sidebar.markdown('<div class="sidebar-section">Период</div>', unsafe_allow_html=True)
    years = sorted(tx["year"].dropna().unique().astype(int).tolist())
    sel_years = st.sidebar.multiselect("Год", years, default=years)
    mm = {1:"Янв",2:"Фев",3:"Мар",4:"Апр",5:"Май",6:"Июн",7:"Июл",8:"Авг",9:"Сен",10:"Окт",11:"Ноя",12:"Дек"}
    months = sorted(tx["month"].dropna().unique().astype(int).tolist())
    sel_months = st.sidebar.multiselect("Месяц", months, format_func=lambda x: mm.get(x, x), default=months)
    st.sidebar.markdown('<div class="sidebar-section">Ценности</div>', unsafe_allow_html=True)
    vals = sorted(tx[TX["value"]].dropna().unique().tolist())
    sel_vals = st.sidebar.multiselect("Ценности", vals, default=vals)
    st.sidebar.markdown('<div class="sidebar-section">Организация</div>', unsafe_allow_html=True)
    comps = sorted(emp[EMP["company"]].dropna().unique().tolist())
    sel_comps = st.sidebar.multiselect("Компания", comps, default=comps)
    depts = sorted(emp["dept_key"].dropna().unique().tolist())
    sel_depts = st.sidebar.multiselect("Отдел (компания / отдел)", depts, default=[])
    emps = sorted(emp["full_name"].dropna().unique().tolist())
    sel_emps = st.sidebar.multiselect("Сотрудники", emps, default=[])
    st.sidebar.markdown('<div class="sidebar-section">Сторона признания</div>', unsafe_allow_html=True)
    side = st.sidebar.radio("Учитывать акты, где выбранные лица —",
                            ["обе стороны", "только отправитель", "только получатель"], index=0)
    st.sidebar.markdown('<div class="sidebar-section">Граф</div>', unsafe_allow_html=True)
    group = st.sidebar.radio("Группировка социального графа", ["По компаниям", "По отделам"], index=0)
    try:
        real_max = max(int(tx.groupby([TX["sid"], TX["rid"]])[TX["merits"]].sum().max()), 10)
    except Exception:
        real_max = 500
    mr = st.sidebar.slider("Диапазон меритов на связь", 1, real_max, (1, real_max), 1)
    return dict(years=set(sel_years), months=set(sel_months), values=set(sel_vals),
                companies=set(sel_comps), depts=set(sel_depts), emps=set(sel_emps),
                side=side, group=group, merit_range=mr)


def apply_filters(tx, cfg):
    df = tx
    if cfg["years"]:  df = df[df["year"].isin(cfg["years"])]
    if cfg["months"]: df = df[df["month"].isin(cfg["months"])]
    if cfg["values"]: df = df[df[TX["value"]].isin(cfg["values"])]

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
    months = fd[["year", "month"]].drop_duplicates().shape[0] or 1
    emitted = n_involved * MERITS_PER_MONTH * months
    return dict(n_involved=n_involved, n_senders=n_senders, n_regular=n_regular, n_receivers=len(receivers),
                involve_share=round(n_involved / n_active * 100, 1) if n_active else 0,
                send_share=round(n_senders / n_involved * 100, 1) if n_involved else 0,
                budget_use=round(total / emitted * 100, 1) if emitted else 0,
                avg_recv=round(total / len(receivers), 1) if receivers else 0)


def render_funnel(f):
    st.markdown('<div class="section-header">Вовлечённость в программу признания</div>', unsafe_allow_html=True)
    cards = [
        ("Участвуют в признании", f"{f['n_involved']:,}", f"{f['involve_share']}% активных · отдали или получили благодарность"),
        ("Отправляют благодарности", f"{f['n_senders']:,}", f"{f['send_share']}% участников проявляют активную позицию"),
        ("Регулярно участвуют", f"{f['n_regular']:,}", "отправили больше одной благодарности"),
        ("Получают признание", f"{f['n_receivers']:,}", "хотя бы одна благодарность за период"),
        ("Используется бюджет признания", f"{f['budget_use']}%", "доля переданных меритов от доступных за период"),
        ("Среднее признание на человека", f"{f['avg_recv']}", "мериты на получателя — см. распределение"),
    ]
    for c, (lab, val, sub) in zip(st.columns(6), cards):
        c.markdown(f'<div class="kpi-box"><div class="kpi-value">{val}</div>'
                   f'<div class="kpi-label">{lab}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


# ───────────────────────── ПУЛЬС ─────────────────────────
def render_temporal(fd, emp):
    st.markdown('<div class="section-header">Пульс программы во времени</div>', unsafe_allow_html=True)
    if go is None:
        st.warning("plotly недоступен (см. диагностику)."); return
    fdt = fd.dropna(subset=["dt"])
    if len(fdt) == 0:
        st.info("Нет данных с датами."); return

    if panels is not None:
        dfig = panels.daily_pulse_fig(fd, emp)
        if dfig is not None:
            st.plotly_chart(dfig, use_container_width=True)

    monthly = fdt.groupby("ym").agg(Актов=(TX["merits"], "size")).reset_index()
    if len(monthly) >= 2:
        last_p = fdt["dt"].max().to_period("M")
        in_last = fdt["dt"].dt.to_period("M") == last_p
        trimmed = (fdt.loc[in_last, "dt"].dt.day.max() if in_last.any() else 31) < 25
        plot = monthly.iloc[:-1] if trimmed and len(monthly) > 2 else monthly
        fig = go.Figure(go.Scatter(x=plot["ym"], y=plot["Актов"], mode="lines+markers", line=dict(color="#58a6ff", width=2)))
        fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=36, b=10),
                          paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                          title="Помесячный объём" + (" (последний неполный месяц скрыт)" if trimmed else ""))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    names = {1:"Янв",2:"Фев",3:"Мар",4:"Апр",5:"Май",6:"Июн",7:"Июл",8:"Авг",9:"Сен",10:"Окт",11:"Ноя",12:"Дек"}
    with c1:
        _yr = fdt["dt"].dt.year.rename("y"); _mo = fdt["dt"].dt.month.rename("mo")
        by_ym = fdt.groupby([_yr, _mo]).size().rename("acts").reset_index()
        typical = by_ym.groupby("mo")["acts"].mean()
        yrs = by_ym["y"].nunique()
        figs = go.Figure(go.Bar(x=[names[m] for m in typical.index], y=typical.round(0).values, marker_color="#79c0ff"))
        figs.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=40, b=10),
                           paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                           title=f"Типичный месяц — среднее число актов (по {yrs} годам)")
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
    c1, c2 = st.columns([3, 2])
    with c1:
        if go is not None:
            fig = go.Figure(go.Bar(x=vc["Доля, %"], y=vc[TX["value"]], orientation="h", marker_color="#3fb950"))
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
                              paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                              title="Доля меритов по ценностям", yaxis=dict(autorange="reversed"))
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


def render_grade_dynamics(tx, emp):
    st.markdown('<div class="section-header">Темпоральная динамика градовой структуры</div>', unsafe_allow_html=True)
    if go is None or grade_dynamics_figure is None:
        st.info("Модуль грейдов недоступен (см. диагностику)."); return
    comps = sorted(emp[EMP["company"]].dropna().unique().tolist())
    if not comps:
        return
    default = "Тепловое Оборудование" if "Тепловое Оборудование" in comps else comps[0]
    unit = st.selectbox("Подразделение", comps, index=comps.index(default))
    fig = grade_dynamics_figure(go, make_subplots, tx, EMP["company"], unit)
    st.plotly_chart(fig, use_container_width=True)


# ───────────────────────── LEAVE ─────────────────────────
def compute_leave(tx_all, emp):
    active_ids = set(emp[emp[EMP["fire"]].isna()][EMP["id"]])
    maxd = tx_all["dt"].max()
    if pd.isna(maxd):
        return dict(count=0, checked=0, rows=[])
    w90 = tx_all[tx_all["dt"] > maxd - pd.Timedelta(days=90)]
    wp = tx_all[(tx_all["dt"] <= maxd - pd.Timedelta(days=90)) & (tx_all["dt"] > maxd - pd.Timedelta(days=270))]
    s90, sp = w90.groupby(TX["sid"]).size(), wp.groupby(TX["sid"]).size()
    sent_total = tx_all.groupby(TX["sid"]).size()
    pool = list(active_ids & set(sent_total[sent_total >= 10].index))
    rows = [p for p in pool if (sp.get(p, 0) / 180.0) > 0 and (s90.get(p, 0) / 90.0) < 0.5 * (sp.get(p, 0) / 180.0)]
    return dict(count=len(rows), checked=len(pool), rows=rows)


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
    st.markdown('<div class="section-header">Ранние сигналы затухания (care)</div>', unsafe_allow_html=True)
    lv = compute_leave(tx_all, emp); sil, tot = validate_leave_on_fired(tx_all, emp)
    st.markdown(
        f'<div class="card care"><strong>Сигнал внимания, не прогноз ухода.</strong><br>'
        f'У <strong>{lv["count"]}</strong> из {lv["checked"]} регулярных участников активность за последний квартал '
        f'снизилась более чем вдвое относительно их собственной обычной активности.<br>'
        f'<span class="muted">Возможные причины: сезонный спад, смена роли или проекта, уход близкого коллеги, '
        f'перегрузка — и лишь в числе прочего эмоциональная усталость. Рекомендация: при желании мягко '
        f'поинтересоваться состоянием. Не используется для кадровых решений.</span></div>', unsafe_allow_html=True)
    if tot:
        st.markdown(f'<div class="card"><strong>Обоснование на истории компании.</strong> Из {tot} ранее уволившихся '
                    f'(с заметной историей участия) <strong>{sil} ({sil/tot*100:.0f}%)</strong> заметно замолкали '
                    f'в последнем квартале перед уходом.</div>', unsafe_allow_html=True)
    if closed_view and lv["rows"]:
        m = emp.set_index(EMP["id"]); m = m[~m.index.duplicated(keep="first")]
        names = [{"ФИО": m["full_name"].get(p, p), "Компания": m[EMP["company"]].get(p, ""),
                  "Отдел": m[EMP["dept"]].get(p, "")} for p in lv["rows"]]
        with st.expander(f"Список для бережного внимания ({len(names)})"):
            st.dataframe(pd.DataFrame(names), use_container_width=True, hide_index=True)
            st.caption("Только для HR/доверенного руководителя. Care-рамка: поддержка, не санкции.")


# ───────────────────────── ГРАФ ─────────────────────────
def build_graph(fd, emp, group_by, merit_range):
    if nx is None:
        return None, None
    m = emp.set_index(EMP["id"]); m = m[~m.index.duplicated(keep="first")]
    gcol = EMP["company"] if group_by == "По компаниям" else "dept_key"
    agg = fd.groupby([TX["sid"], TX["rid"]]).agg(total=(TX["merits"], "sum"), n=(TX["merits"], "size")).reset_index()
    lo, hi = merit_range
    agg = agg[(agg["total"] >= lo) & (agg["total"] <= hi) & (agg[TX["sid"]] != agg[TX["rid"]])]

    def attrs(eid):
        if eid in m.index:
            r = m.loc[eid]
            nm = f"{r.get(EMP['last'],'')} {str(r.get(EMP['first'],''))[:1]}.".strip()
            return dict(label=nm, dept=str(r.get(EMP["dept"], "")), company=str(r.get(EMP["company"], "")),
                        position=str(r.get(EMP["pos"], "")), group=str(r.get(gcol, "")))
        return dict(label=eid, dept="", company="", position="", group="")

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


def render_network_health(G, mt):
    st.markdown('<div class="section-header">Здоровье сети признания</div>', unsafe_allow_html=True)
    if G is None or G.number_of_nodes() == 0:
        st.info("Граф пуст."); return
    n, e = G.number_of_nodes(), G.number_of_edges()
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


def render_distributions(G, mt):
    st.markdown('<div class="section-header">Структурные позиции (распределения)</div>', unsafe_allow_html=True)
    if G is None or G.number_of_nodes() == 0:
        return
    pr = pd.Series(mt.get("pagerank", {})); bc = pd.Series(mt.get("betweenness", {}))
    div = {}
    for node in G.nodes():
        neigh = set(G.successors(node)) | set(G.predecessors(node))
        div[node] = len({G.nodes[x].get("dept", "") for x in neigh}) / len(neigh) if neigh else 0
    div = pd.Series(div)
    wide = int((pr >= pr.quantile(0.9)).sum()) if len(pr) else 0
    brok = int((bc >= bc.quantile(0.9)).sum()) if len(bc) else 0
    dvs = int((div >= div.quantile(0.9)).sum()) if len(div) else 0
    st.markdown(f'<div class="card"><strong>Широкий охват признания.</strong> ~{wide} сотрудников получают признание '
                f'из многих частей сети.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><strong>Посреднические позиции.</strong> ~{brok} сотрудников связывают разные '
                f'подразделения. Ресурс интеграции; риск перегрузки.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card good"><strong>Широкая сеть контактов.</strong> ~{dvs} сотрудников признают коллег '
                f'из многих отделов.</div>', unsafe_allow_html=True)


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

    st.markdown("""<div style="padding:1rem 0 .3rem">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:1.7rem;font-weight:600;color:#58a6ff;">🕸️ 3Д Коммуникации</span>
        <span style="color:#8b949e;font-size:.85rem;margin-left:12px;">Социальные технологии для бизнеса · ГК Термекс</span>
        </div>""", unsafe_allow_html=True)

    base = os.path.dirname(os.path.abspath(__file__))
    emp_path, tx_path = os.path.join(base, "employees.xlsx"), os.path.join(base, "dataset.xlsx")
    miss = [f for f, p in [("employees.xlsx", emp_path), ("dataset.xlsx", tx_path)] if not os.path.exists(p)]
    if miss:
        st.error(f"❌ Не найдены файлы: {', '.join(miss)}")
        Debug.errors.append(("файлы", f"отсутствуют: {miss}")); render_diagnostics(); return

    try:
        with st.spinner("Загрузка данных…"):
            with Debug.stage("load_employees"): emp = load_employees(emp_path)
            with Debug.stage("load_transactions"): tx_raw = load_transactions(tx_path)
            with Debug.stage("merge_data"): tx = merge_data(tx_raw.copy(), emp)
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
        c1.metric("Транзакций", f"{len(fd):,}"); c2.metric("Меритов", f"{int(fd[TX['merits']].sum()):,}")
        c3.metric("Компаний", f"{n_comp}"); c4.metric("Отделов", f"{n_dep}"); c5.metric("Сотрудников", f"{n_ppl:,}")

        with Debug.stage("build_graph"):
            G, _ = build_graph(fd, emp, cfg["group"], cfg["merit_range"])
        with Debug.stage("graph_metrics"):
            mt = graph_metrics(G)

        t_org, t_net, t_pass, t_an = st.tabs(["🏢 Организация", "🌐 Сеть и подразделения", "🪪 Меритпаспорт", "📊 Аналитик"])

        with t_org:
            with Debug.stage("render_funnel", fatal=False): render_funnel(compute_funnel(emp, fd))
            with Debug.stage("render_temporal", fatal=False): render_temporal(fd, emp)
            with Debug.stage("render_values", fatal=False): render_values(fd, emp)
            with Debug.stage("render_grade_dynamics", fatal=False): render_grade_dynamics(tx, emp)
            st.markdown('<div class="section-header">Рейтинг охвата</div>', unsafe_allow_html=True)
            if panels is not None:
                with Debug.stage("render_rating", fatal=False): panels.render_rating(fd, emp)
            with Debug.stage("render_leave", fatal=False): render_leave(tx, emp, closed_view=False)

        with t_net:
            with Debug.stage("render_network_health", fatal=False): render_network_health(G, mt)
            st.markdown('<div class="section-header">Визуализация сети</div>', unsafe_allow_html=True)
            if components is not None and social_graph_html is not None and G is not None and G.number_of_nodes() > 0:
                st.markdown('<div class="info-box">Стрелка показывает направление благодарности. '
                            '<span style="color:#3fb950">Зелёные связи — взаимные</span>. Клик по узлу подсвечивает '
                            'его связи. Кнопка «Только взаимные» оставляет двусторонние.</div>', unsafe_allow_html=True)
                gt = st.tabs(["🌀 Социальный граф", "🏢 Компании → отделы → люди"])
                with gt[0]:
                    with Debug.stage("social_graph", fatal=False):
                        components.html(social_graph_html(G, mt), height=720, scrolling=False)
                with gt[1]:
                    with Debug.stage("hierarchy", fatal=False):
                        components.html(hierarchy_html(G, mt, None), height=720, scrolling=False)
            elif social_graph_html is None:
                st.warning("graph_viz.py не загрузился (см. диагностику).")
            with Debug.stage("render_distributions", fatal=False): render_distributions(G, mt)

        with t_pass:
            st.markdown('<div class="section-header">Меритпаспорт сотрудника</div>', unsafe_allow_html=True)
            if panels is not None:
                with Debug.stage("meritpassport", fatal=False): panels.render_meritpassport(fd, tx, emp)
            with Debug.stage("leave_closed", fatal=False): render_leave(tx, emp, closed_view=True)

        with t_an:
            with Debug.stage("render_analyst", fatal=False): render_analyst(G, mt)

    except Exception:
        st.error("Произошла ошибка — подробности в блоке диагностики ниже.")
        if not Debug.errors or Debug.errors[-1][0] != "main":
            Debug.errors.append(("main", traceback.format_exc()))

    render_diagnostics()


if __name__ == "__main__":
    main()
