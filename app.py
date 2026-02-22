# -*- coding: utf-8 -*-
"""
СоциоГраф 6.0
Два источника данных: employees.xlsx + dataset.xlsx
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from community import community_louvain
import streamlit.components.v1 as components
from scipy.spatial.distance import pdist, squareform

st.set_page_config(
    page_title="СоциоГраф",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background: #0d1117; }
    .block-container { padding: 1.5rem 2rem; }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #58a6ff !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }

    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem !important;
        color: #58a6ff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        color: #c9d1d9;
    }
    .metric-card strong { color: #58a6ff; }

    .kpi-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .kpi-box {
        flex: 1;
        min-width: 160px;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        position: relative;
    }
    .kpi-box .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #58a6ff;
        line-height: 1.1;
    }
    .kpi-box .kpi-label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .kpi-box .kpi-info {
        position: absolute;
        top: 10px;
        right: 12px;
        color: #8b949e;
        font-size: 0.75rem;
        cursor: help;
        border: 1px solid #30363d;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-style: italic;
    }
    .kpi-box .kpi-info:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        right: 0;
        top: 22px;
        background: #1f2937;
        border: 1px solid #30363d;
        color: #c9d1d9;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        width: 260px;
        z-index: 999;
        line-height: 1.5;
        white-space: normal;
        font-style: normal;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin: 1.5rem 0 1rem;
    }

    .stButton > button {
        background: #161b22;
        color: #58a6ff;
        border: 1px solid #58a6ff;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        padding: 0.4rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #58a6ff;
        color: #0d1117;
    }

    .stDataFrame { border: 1px solid #30363d; border-radius: 6px; }
    .stDataFrame th { background: #161b22 !important; color: #8b949e !important; }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }

    .sidebar-section {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8b949e;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    .info-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        color: #8b949e;
        line-height: 1.6;
    }

    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

# ========================= КОНСТАНТЫ =========================

TOTAL_COMPANY_EMPLOYEES = 1145  # Для показателя лояльности
MERITS_PER_MONTH = 10           # Голосов в месяц на участника

EMP_COLS = {
    "last_name": "Фамилия",
    "first_name": "Имя",
    "middle_name": "Отчество",
    "gender": "Пол",
    "emp_id": "Персональный номер",
    "position": "Должность",
    "company": "Компания",
    "dept": "Отдел",
    "fire_date": "Дата увольнения",
}

TX_COLS = {
    "time": "Время",
    "sender_id": "Номер отправителя",
    "receiver_id": "Номер получателя",
    "value": "Ценность",
    "merits": "Мериты",
    "comment": "Комментарий",
}


# ========================= ЗАГРУЗКА ДАННЫХ =========================

@st.cache_data(show_spinner=False)
def load_employees(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # Убираем пробелы во всех строковых колонках (решает 'БЕЛАРУСЬ АКВАТЕРМЕКС ')
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", None)

    # Дата увольнения — может быть строкой "01.03.2024" или датой
    if EMP_COLS["fire_date"] in df.columns:
        df[EMP_COLS["fire_date"]] = pd.to_datetime(
            df[EMP_COLS["fire_date"]], dayfirst=True, errors="coerce"
        )

    df[EMP_COLS["emp_id"]] = df[EMP_COLS["emp_id"]].astype(str).str.strip()

    df["full_name"] = (
        df[EMP_COLS["last_name"]].fillna("") + " " +
        df[EMP_COLS["first_name"]].fillna("") + " " +
        df[EMP_COLS["middle_name"]].fillna("")
    ).str.strip()

    return df

@st.cache_data(show_spinner=False)
def load_transactions(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # Парсим дату — в файле есть отдельная колонка "Дата"
    if "Дата" in df.columns:
        df["dt"] = pd.to_datetime(df["Дата"], dayfirst=True, errors="coerce")
    elif TX_COLS["time"] in df.columns:
        df["dt"] = pd.to_datetime(df[TX_COLS["time"]], errors="coerce")
    else:
        df["dt"] = pd.NaT

    df[TX_COLS["merits"]] = pd.to_numeric(df[TX_COLS["merits"]], errors="coerce").fillna(0).astype(int)
    df[TX_COLS["sender_id"]] = df[TX_COLS["sender_id"]].astype(str).str.strip()
    df[TX_COLS["receiver_id"]] = df[TX_COLS["receiver_id"]].astype(str).str.strip()

    # Ценности — убираем лишние пробелы и приводим к нижнему регистру
    if TX_COLS["value"] in df.columns:
        df[TX_COLS["value"]] = df[TX_COLS["value"]].astype(str).str.strip()

    return df

def merge_data(tx_df, emp_df):
    """Объединяем транзакции с данными сотрудников"""
    emp_map = emp_df.set_index(EMP_COLS["emp_id"])

    lookup_cols = ["full_name", EMP_COLS["position"], EMP_COLS["company"], EMP_COLS["dept"]]

    def enrich(df, id_col, prefix):
        for col in lookup_cols:
            if col in emp_map.columns:
                df[f"{prefix}_{col}"] = df[id_col].map(emp_map[col])
            else:
                df[f"{prefix}_{col}"] = None
        return df

    tx_df = enrich(tx_df, TX_COLS["sender_id"], "sender")
    tx_df = enrich(tx_df, TX_COLS["receiver_id"], "receiver")

    # Дата увольнения
    if EMP_COLS["fire_date"] in emp_map.columns:
        tx_df["sender_fire"]   = tx_df[TX_COLS["sender_id"]].map(emp_map[EMP_COLS["fire_date"]])
        tx_df["receiver_fire"] = tx_df[TX_COLS["receiver_id"]].map(emp_map[EMP_COLS["fire_date"]])

    # Год и месяц берём из колонки dt (создана в load_transactions)
    tx_df["year"]  = tx_df["dt"].dt.year
    tx_df["month"] = tx_df["dt"].dt.month

    return tx_df


# ========================= ФИЛЬТРАЦИЯ =========================

def sidebar_controls(tx_df, emp_df):
    st.sidebar.markdown("## ⚙️ Фильтры")

    # --- ГОД ---
    st.sidebar.markdown('<div class="sidebar-section">Период</div>', unsafe_allow_html=True)
    years = sorted(tx_df["year"].dropna().unique().astype(int).tolist())
    selected_years = st.sidebar.multiselect("Год", options=years, default=years)

    months_map = {1:"Январь",2:"Февраль",3:"Март",4:"Апрель",5:"Май",6:"Июнь",
                  7:"Июль",8:"Август",9:"Сентябрь",10:"Октябрь",11:"Ноябрь",12:"Декабрь"}
    months_present = sorted(tx_df["month"].dropna().unique().astype(int).tolist())
    selected_months = st.sidebar.multiselect(
        "Месяц",
        options=months_present,
        format_func=lambda x: months_map.get(x, str(x)),
        default=months_present
    )

    # --- ЦЕННОСТИ ---
    st.sidebar.markdown('<div class="sidebar-section">Ценности</div>', unsafe_allow_html=True)
    values_list = sorted(tx_df[TX_COLS["value"]].dropna().unique().tolist())
    selected_values = st.sidebar.multiselect("Ценности", options=values_list, default=values_list)

    # --- КОМПАНИЯ ---
    st.sidebar.markdown('<div class="sidebar-section">Организация</div>', unsafe_allow_html=True)
    companies = sorted(emp_df[EMP_COLS["company"]].dropna().unique().tolist())
    selected_companies = st.sidebar.multiselect("Компания", options=companies, default=companies)

    depts_all = sorted(emp_df[EMP_COLS["dept"]].dropna().unique().tolist())
    selected_depts = st.sidebar.multiselect("Отдел", options=depts_all, default=depts_all)

    emps_all = sorted(emp_df["full_name"].dropna().unique().tolist())
    selected_emps = st.sidebar.multiselect("Сотрудники", options=emps_all, default=[])

    # --- ГРАФ ---
    st.sidebar.markdown('<div class="sidebar-section">Граф</div>', unsafe_allow_html=True)
    graph_group = st.sidebar.radio(
        "Группировка графа",
        options=["По компаниям", "По отделам"],
        index=1
    )

    # Считаем реальный максимум суммарных меритов между парами
    try:
        pair_merits = (tx_df.groupby([TX_COLS["sender_id"], TX_COLS["receiver_id"]])[TX_COLS["merits"]]
                       .sum())
        real_max = max(int(pair_merits.max()), 10)
    except:
        real_max = 500

    merit_range = st.sidebar.slider(
        "Диапазон меритов на связь",
        min_value=1, max_value=real_max,
        value=(1, real_max), step=1,
        help=f"Суммарные мериты между парой сотрудников. Макс. в данных: {real_max}"
    )

    return {
        "years": set(selected_years),
        "months": set(selected_months),
        "values": set(selected_values),
        "companies": set(selected_companies),
        "depts": set(selected_depts),
        "emps": set(selected_emps),
        "graph_group": graph_group,
        "merit_range": merit_range,
    }


def apply_filters(tx_df, emp_df, cfg):
    df = tx_df.copy()

    if cfg["years"]:
        df = df[df["year"].isin(cfg["years"])]
    if cfg["months"]:
        df = df[df["month"].isin(cfg["months"])]
    if cfg["values"]:
        df = df[df[TX_COLS["value"]].isin(cfg["values"])]

    # Фильтр по компании (через отправителя)
    if cfg["companies"]:
        sender_company_col = f"sender_{EMP_COLS['company']}"
        if sender_company_col in df.columns:
            df = df[df[sender_company_col].isin(cfg["companies"])]

    if cfg["depts"]:
        sender_dept_col = f"sender_{EMP_COLS['dept']}"
        if sender_dept_col in df.columns:
            df = df[df[sender_dept_col].isin(cfg["depts"])]

    if cfg["emps"]:
        sender_name_col = "sender_full_name"
        if sender_name_col in df.columns:
            df = df[df[sender_name_col].isin(cfg["emps"])]

    return df


# ========================= МЕТРИКИ =========================

def compute_kpis(tx_df, emp_df, filtered_df):
    """Вычисляем 6 KPI (без учёта уволенных)"""
    # Активные сотрудники (без даты увольнения)
    active_emp = emp_df[emp_df[EMP_COLS["fire_date"]].isna()]
    n_active = len(active_emp)
    active_ids = set(active_emp[EMP_COLS["emp_id"]].astype(str))

    # Участники программы = активные сотрудники, встречающиеся в транзакциях
    all_tx_ids = set(tx_df[TX_COLS["sender_id"]].astype(str)) | set(tx_df[TX_COLS["receiver_id"]].astype(str))
    program_ids = active_ids & all_tx_ids
    n_program = len(program_ids)

    # Из отфильтрованных транзакций (активные)
    fd = filtered_df[
        filtered_df[TX_COLS["sender_id"]].isin(active_ids) &
        filtered_df[TX_COLS["receiver_id"]].isin(active_ids)
    ]

    sender_counts = fd.groupby(TX_COLS["sender_id"]).size()
    receivers_set = set(fd[TX_COLS["receiver_id"]].unique())

    n_senders = (sender_counts >= 1).sum()
    n_senders_gt1 = (sender_counts > 1).sum()
    n_receivers = len(receivers_set)
    total_merits = fd[TX_COLS["merits"]].sum()

    involved = set(fd[TX_COLS["sender_id"]].unique()) | receivers_set
    n_involved = len(involved)

    # 1. Доля вовлечённости
    kpi1 = round(n_involved / n_program * 100, 1) if n_program > 0 else 0

    # 2. Показатель лояльности
    kpi2 = round(n_senders_gt1 / TOTAL_COMPANY_EMPLOYEES * 100, 1)

    # 3. ER (вовлечённость в программу)
    kpi3 = round(n_senders / n_program * 100, 1) if n_program > 0 else 0

    # 4. Голосов использовано — считаем кол-во месяцев * участников * 10
    if len(fd) > 0 and "year" in fd.columns and "month" in fd.columns:
        period_months = fd[["year","month"]].drop_duplicates().shape[0]
    else:
        period_months = 1
    emitted = n_program * MERITS_PER_MONTH * period_months
    kpi4 = round(total_merits / emitted * 100, 1) if emitted > 0 else 0

    # 5. Степень полезности
    kpi5 = round(total_merits / n_receivers, 1) if n_receivers > 0 else 0

    # 6. Степень активности
    kpi6 = round(total_merits / n_senders, 1) if n_senders > 0 else 0

    return {
        "kpi1": kpi1, "kpi2": kpi2, "kpi3": kpi3,
        "kpi4": kpi4, "kpi5": kpi5, "kpi6": kpi6,
        "n_involved": n_involved, "n_program": n_program,
        "n_senders": n_senders, "n_senders_gt1": n_senders_gt1,
        "n_receivers": n_receivers, "total_merits": total_merits,
    }


def render_kpis(kpis):
    st.markdown('<div class="section-header">Ключевые метрики программы</div>', unsafe_allow_html=True)

    tooltips = [
        ("Доля вовлечённости", f"{kpis['kpi1']}%",
         "Показывает насколько полно сотрудники используют возможности для позитивной оценки коллег. Отношение вовлечённых к участникам программы."),
        ("Показатель лояльности", f"{kpis['kpi2']}%",
         f"Процент отправителей (>1 благодарности) от всех {TOTAL_COMPANY_EMPLOYEES} сотрудников компании. Отражает вовлечение в культуру благодарения."),
        ("ER программы", f"{kpis['kpi3']}%",
         "Отражает активную позицию участников программы. Отношение отправителей хотя бы одной благодарности к числу участников 3Д."),
        ("Голосов использовано", f"{kpis['kpi4']}%",
         "Насколько полно используются возможности для оценки коллег. Отношение использованных голосов к общему числу эмитированных за период."),
        ("Степень полезности", f"{kpis['kpi5']}",
         "Среднее количество голосов, полученное сотрудником. Отношение суммы баллов к числу получивших хотя бы 1 благодарность."),
        ("Степень активности", f"{kpis['kpi6']}",
         "Среднее количество голосов, переданных сотрудником в качестве благодарности. Отношение суммы баллов к числу отправивших хотя бы 1 благодарность."),
    ]

    cols = st.columns(6)
    for i, (label, value, tip) in enumerate(tooltips):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-box">
                <span class="kpi-info" title="{tip}">i</span>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ========================= ТОПЫ =========================

def render_tops(filtered_df, emp_df):
    emp_map = emp_df.set_index(EMP_COLS["emp_id"])

    def get_info(emp_id):
        if emp_id in emp_map.index:
            r = emp_map.loc[emp_id]
            name = f"{r.get(EMP_COLS['last_name'],'')} {r.get(EMP_COLS['first_name'],'')} {r.get(EMP_COLS['middle_name'],'')}".strip()
            return {
                "ФИО": name,
                "Должность": r.get(EMP_COLS["position"], ""),
                "Отдел": r.get(EMP_COLS["dept"], ""),
                "Компания": r.get(EMP_COLS["company"], ""),
            }
        return {"ФИО": emp_id, "Должность": "", "Отдел": "", "Компания": ""}

    st.markdown('<div class="section-header">Топ сотрудников</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Полезность (получено меритов)",
        "🚀 Активность (отправлено меритов)",
        "👥 Охват — получили от стольких",
        "📣 Охват — поблагодарили стольких"
    ])

    N = 20

    # --- Полезность ---
    with tab1:
        recv = (filtered_df.groupby(TX_COLS["receiver_id"])[TX_COLS["merits"]]
                .sum().reset_index().sort_values(TX_COLS["merits"], ascending=False).head(N))
        rows = []
        for rank, (_, row) in enumerate(recv.iterrows(), 1):
            info = get_info(row[TX_COLS["receiver_id"]])
            rows.append({"#": rank, **info, "Мериты": int(row[TX_COLS["merits"]])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Активность ---
    with tab2:
        sent = (filtered_df.groupby(TX_COLS["sender_id"])[TX_COLS["merits"]]
                .sum().reset_index().sort_values(TX_COLS["merits"], ascending=False).head(N))
        rows = []
        for rank, (_, row) in enumerate(sent.iterrows(), 1):
            info = get_info(row[TX_COLS["sender_id"]])
            rows.append({"#": rank, **info, "Мериты": int(row[TX_COLS["merits"]])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Охват: получили от стольких ---
    with tab3:
        reach_recv = (filtered_df.groupby(TX_COLS["receiver_id"])[TX_COLS["sender_id"]]
                      .nunique().reset_index()
                      .rename(columns={TX_COLS["sender_id"]: "Уникальных отправителей"})
                      .sort_values("Уникальных отправителей", ascending=False).head(N))
        rows = []
        for rank, (_, row) in enumerate(reach_recv.iterrows(), 1):
            info = get_info(row[TX_COLS["receiver_id"]])
            rows.append({"#": rank, **info, "Уник. отправителей": int(row["Уникальных отправителей"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Охват: поблагодарили стольких ---
    with tab4:
        reach_sent = (filtered_df.groupby(TX_COLS["sender_id"])[TX_COLS["receiver_id"]]
                      .nunique().reset_index()
                      .rename(columns={TX_COLS["receiver_id"]: "Уникальных получателей"})
                      .sort_values("Уникальных получателей", ascending=False).head(N))
        rows = []
        for rank, (_, row) in enumerate(reach_sent.iterrows(), 1):
            info = get_info(row[TX_COLS["sender_id"]])
            rows.append({"#": rank, **info, "Уник. получателей": int(row["Уникальных получателей"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ========================= ПОСТРОЕНИЕ ГРАФОВ =========================

def build_graph(tx_df, emp_df, group_by, merit_range):
    """group_by: 'company' или 'dept'"""
    emp_map = emp_df.set_index(EMP_COLS["emp_id"])

    group_col = EMP_COLS["company"] if group_by == "По компаниям" else EMP_COLS["dept"]

    # Агрегация по парам sender-receiver
    agg = (tx_df.groupby([TX_COLS["sender_id"], TX_COLS["receiver_id"]])
           .agg(total_merits=(TX_COLS["merits"], "sum"), n_tx=(TX_COLS["merits"], "count"))
           .reset_index())

    min_m, max_m = merit_range
    agg = agg[(agg["total_merits"] >= min_m) & (agg["total_merits"] <= max_m)]

    # Граф людей
    G_people = nx.DiGraph()
    for _, row in agg.iterrows():
        sid, rid = row[TX_COLS["sender_id"]], row[TX_COLS["receiver_id"]]
        if sid == rid:
            continue

        def node_attrs(eid):
            if eid in emp_map.index:
                r = emp_map.loc[eid]
                name = f"{r.get(EMP_COLS['last_name'],'')} {r.get(EMP_COLS['first_name'],'')[:1]}.".strip()
                return {
                    "label": name,
                    "dept": str(r.get(EMP_COLS["dept"], "")),
                    "company": str(r.get(EMP_COLS["company"], "")),
                    "position": str(r.get(EMP_COLS["position"], "")),
                    "group": str(r.get(group_col, "")),
                }
            return {"label": eid, "dept": "", "company": "", "position": "", "group": ""}

        if sid not in G_people:
            G_people.add_node(sid, **node_attrs(sid))
        if rid not in G_people:
            G_people.add_node(rid, **node_attrs(rid))

        w = float(row["total_merits"])
        G_people.add_edge(sid, rid, weight=w, length=1.0/max(w,0.01), msgs=int(row["n_tx"]))

    # Граф групп
    group_agg = {}
    for u, v, data in G_people.edges(data=True):
        gu = G_people.nodes[u].get("group", "")
        gv = G_people.nodes[v].get("group", "")
        key = (gu, gv)
        if key not in group_agg:
            group_agg[key] = {"weight": 0, "people": 0}
        group_agg[key]["weight"] += data["weight"]
        group_agg[key]["people"] += 1

    group_members = {}
    for node in G_people.nodes():
        g = G_people.nodes[node].get("group", "")
        if g not in group_members:
            group_members[g] = []
        group_members[g].append(node)

    G_groups = nx.DiGraph()
    for g, members in group_members.items():
        G_groups.add_node(g, label=g, type="group", size=len(members), members=members)

    for (gu, gv), data in group_agg.items():
        if gu != gv:
            G_groups.add_edge(gu, gv, weight=data["weight"], people=data["people"])

    return G_people, G_groups, group_members


# ========================= МЕТРИКИ ГРАФА =========================

def calculate_graph_metrics(G):
    if G.number_of_nodes() == 0:
        return {}
    metrics = {}
    metrics["in_strength"] = dict(G.in_degree(weight="weight"))
    metrics["out_strength"] = dict(G.out_degree(weight="weight"))
    try:
        metrics["pagerank"] = nx.pagerank(G, weight="weight", max_iter=100)
    except:
        metrics["pagerank"] = {n: 1.0/G.number_of_nodes() for n in G.nodes()}
    UG = G.to_undirected()
    try:
        metrics["betweenness"] = nx.betweenness_centrality(UG, weight="length", normalized=True)
    except:
        metrics["betweenness"] = {n: 0.0 for n in G.nodes()}
    try:
        metrics["closeness"] = nx.closeness_centrality(UG, distance="length")
    except:
        metrics["closeness"] = {n: 0.0 for n in G.nodes()}
    try:
        metrics["clustering"] = nx.clustering(UG, weight="weight")
    except:
        metrics["clustering"] = {n: 0.0 for n in G.nodes()}
    try:
        metrics["eigenvector"] = nx.eigenvector_centrality(UG, weight="weight", max_iter=200)
    except:
        metrics["eigenvector"] = {n: 0.0 for n in G.nodes()}
    try:
        metrics["constraint"] = nx.constraint(UG, weight="weight")
    except:
        metrics["constraint"] = {n: 0.0 for n in G.nodes()}
    try:
        metrics["core_number"] = nx.core_number(UG)
    except:
        metrics["core_number"] = {n: 0 for n in G.nodes()}
    try:
        bridges = list(nx.bridges(UG))
        bridge_nodes = set()
        for a, b in bridges:
            bridge_nodes.add(a); bridge_nodes.add(b)
        metrics["is_bridge"] = {n: 1 if n in bridge_nodes else 0 for n in G.nodes()}
    except:
        metrics["is_bridge"] = {n: 0 for n in G.nodes()}
    try:
        metrics["load"] = nx.load_centrality(UG, weight="length")
    except:
        metrics["load"] = {n: 0.0 for n in G.nodes()}
    dept_diversity = {}
    for node in G.nodes():
        neighbors = set(G.neighbors(node)) | set(G.predecessors(node))
        if len(neighbors) == 0:
            dept_diversity[node] = 0.0
        else:
            depts = set(G.nodes[n].get("dept","") for n in neighbors)
            dept_diversity[node] = len(depts) / len(neighbors)
    metrics["dept_diversity"] = dept_diversity
    try:
        part = community_louvain.best_partition(UG, weight="weight")
        mod = community_louvain.modularity(part, UG, weight="weight")
        metrics["communities"] = part
        metrics["modularity"] = mod
    except:
        metrics["communities"] = {n: 0 for n in G.nodes()}
        metrics["modularity"] = 0.0
    metrics["reciprocity"] = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0
    return metrics


# ========================= СОЦИАЛЬНЫЙ ГРАФ (Force-Directed) =========================

def create_social_graph_viz(G, metrics):
    nodes_data = []
    for node in G.nodes():
        nd = G.nodes[node]
        comm = metrics.get("communities", {}).get(node, 0)
        nodes_data.append({
            "id": str(node),
            "label": nd.get("label", str(node)),
            "dept": nd.get("dept", ""),
            "company": nd.get("company", ""),
            "position": nd.get("position", ""),
            "group": nd.get("group", ""),
            "community": comm,
            "pagerank": metrics.get("pagerank", {}).get(node, 0),
            "in_strength": metrics.get("in_strength", {}).get(node, 0),
            "out_strength": metrics.get("out_strength", {}).get(node, 0),
        })

    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "source": str(u), "target": str(v),
            "weight": data.get("weight", 1),
        })

    n_comm = max(1, len(set(metrics.get("communities", {}).values())))
    colors = ["#58a6ff","#f78166","#3fb950","#d2a8ff","#ffa657",
              "#79c0ff","#ff7b72","#56d364","#bc8cff","#e3b341",
              "#39d353","#ff9b8c","#a5d6ff","#cfba7c","#7ee787"]

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin:0; background:#0d1117; font-family:'IBM Plex Sans',sans-serif; overflow:hidden; }}
        #viz {{ width:100%; height:100vh; }}
        .controls {{ position:absolute; top:12px; right:12px; z-index:1000; display:flex; gap:6px; }}
        .btn {{ background:#161b22; color:#58a6ff; border:1px solid #30363d; padding:6px 14px;
                border-radius:5px; cursor:pointer; font-size:12px; font-family:'IBM Plex Mono',monospace; }}
        .btn:hover {{ background:#21262d; }}
        .label {{ fill:#8b949e; font-size:10px; pointer-events:none; text-anchor:middle; }}
        #tooltip {{ position:absolute; background:#161b22; border:1px solid #30363d;
                    color:#c9d1d9; padding:10px 14px; border-radius:6px; font-size:12px;
                    pointer-events:none; opacity:0; transition:opacity 0.2s; max-width:220px; line-height:1.6; }}
    </style></head><body>
    <div class="controls">
        <button class="btn" onclick="resetZoom()">↺ Сброс</button>
        <button class="btn" onclick="toggleLabels()">Метки</button>
        <button class="btn" onclick="togglePhysics()">Физика</button>
    </div>
    <div id="tooltip"></div>
    <svg id="viz"></svg>
    <script>
    const W = window.innerWidth, H = window.innerHeight;
    const nodes = {json.dumps(nodes_data)};
    const links = {json.dumps(edges_data)};
    const colors = {json.dumps(colors[:n_comm])};

    const svg = d3.select("#viz").attr("width",W).attr("height",H);
    const g = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.05,12]).on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);

    const linkEls = g.append("g").selectAll("line").data(links).join("line")
        .attr("stroke","#30363d").attr("stroke-opacity",0.6)
        .attr("stroke-width", d => Math.sqrt(d.weight)*0.6 + 0.3);

    const nodeEls = g.append("g").selectAll("circle").data(nodes).join("circle")
        .attr("r", d => 4 + Math.sqrt(d.pagerank * 800))
        .attr("fill", d => colors[d.community % colors.length])
        .attr("stroke","#0d1117").attr("stroke-width",1.5)
        .attr("cursor","pointer")
        .on("mouseover", (event, d) => {{
            const tip = document.getElementById("tooltip");
            tip.innerHTML = `<strong>${{d.label}}</strong><br>
                ${{d.position}}<br>
                <span style="color:#58a6ff">${{d.company}}</span> / ${{d.dept}}<br>
                <hr style="border-color:#30363d;margin:6px 0">
                PageRank: ${{d.pagerank.toFixed(4)}}<br>
                Входящих: ${{d.in_strength.toFixed(0)}} · Исходящих: ${{d.out_strength.toFixed(0)}}`;
            tip.style.opacity = 1;
            tip.style.left = (event.pageX+12)+"px";
            tip.style.top = (event.pageY-10)+"px";
        }})
        .on("mouseout", () => {{ document.getElementById("tooltip").style.opacity = 0; }})
        .call(d3.drag()
            .on("start",(e,d)=>{{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x;d.fy=d.y; }})
            .on("drag",(e,d)=>{{ d.fx=e.x;d.fy=e.y; }})
            .on("end",(e,d)=>{{ if(!e.active) sim.alphaTarget(0); d.fx=null;d.fy=null; }}));

    const labelEls = g.append("g").selectAll("text").data(nodes).join("text")
        .attr("class","label").attr("dy",-8)
        .text(d => d.label.length>18 ? d.label.slice(0,18)+"…" : d.label);

    const sim = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d=>d.id).distance(70))
        .force("charge", d3.forceManyBody().strength(-180))
        .force("center", d3.forceCenter(W/2, H/2))
        .force("collision", d3.forceCollide().radius(14))
        .on("tick", () => {{
            linkEls.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
                   .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
            nodeEls.attr("cx",d=>d.x).attr("cy",d=>d.y);
            labelEls.attr("x",d=>d.x).attr("y",d=>d.y);
        }});

    function resetZoom(){{ svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity); }}
    let labelsOn=true;
    function toggleLabels(){{ labelsOn=!labelsOn; labelEls.style("opacity", labelsOn?1:0); }}
    let physOn=true;
    function togglePhysics(){{ physOn=!physOn; physOn ? sim.alpha(0.3).restart() : sim.stop(); }}
    </script></body></html>"""
    return html


# ========================= ФУНКЦИОНАЛЬНАЯ СЕТЬ (Иерархическая) =========================

def create_functional_network_viz(G_groups, G_people, group_members, metrics_groups, metrics_people):
    group_nodes = []
    for node in G_groups.nodes():
        nd = G_groups.nodes[node]
        group_nodes.append({
            "id": f"g_{node}", "original_id": node,
            "label": nd.get("label", str(node)),
            "type": "group", "size": nd.get("size", 1),
            "members": nd.get("members", []),
            "in_strength": metrics_groups.get("in_strength",{}).get(node,0),
            "out_strength": metrics_groups.get("out_strength",{}).get(node,0),
        })

    group_edges = []
    for u, v, data in G_groups.edges(data=True):
        group_edges.append({
            "source": f"g_{u}", "target": f"g_{v}",
            "weight": data.get("weight",1), "people": data.get("people",0),
        })

    people_nodes = []
    for node in G_people.nodes():
        nd = G_people.nodes[node]
        people_nodes.append({
            "id": f"p_{node}", "original_id": node,
            "label": nd.get("label", str(node)),
            "dept": nd.get("dept",""), "company": nd.get("company",""),
            "position": nd.get("position",""), "group": nd.get("group",""),
            "type": "person",
            "in_strength": metrics_people.get("in_strength",{}).get(node,0),
            "out_strength": metrics_people.get("out_strength",{}).get(node,0),
            "pagerank": metrics_people.get("pagerank",{}).get(node,0),
        })

    people_edges = []
    for u, v, data in G_people.edges(data=True):
        people_edges.append({
            "source": f"p_{u}", "target": f"p_{v}",
            "weight": data.get("weight",1),
        })

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin:0; background:#0d1117; font-family:'IBM Plex Sans',sans-serif; overflow:hidden; }}
        #viz {{ width:100%; height:100vh; }}
        .controls {{ position:absolute; top:12px; right:12px; display:flex; gap:6px; z-index:1000; }}
        .btn {{ background:#161b22; color:#58a6ff; border:1px solid #30363d; padding:6px 14px;
                border-radius:5px; cursor:pointer; font-size:12px; font-family:'IBM Plex Mono',monospace; }}
        .btn:hover {{ background:#21262d; }}
        #breadcrumb {{ position:absolute; top:14px; left:14px; color:#58a6ff;
                        font-family:'IBM Plex Mono',monospace; font-size:13px; }}
        #tooltip {{ position:absolute; background:#161b22; border:1px solid #30363d;
                    color:#c9d1d9; padding:10px 14px; border-radius:6px; font-size:12px;
                    pointer-events:none; opacity:0; transition:opacity 0.2s; max-width:230px; line-height:1.6; }}
    </style></head><body>
    <div id="breadcrumb">Уровень: Группы</div>
    <div class="controls">
        <button class="btn" onclick="goHome()">↺ Домой</button>
        <button class="btn" onclick="resetZoom()">⊕ Сброс</button>
        <button class="btn" onclick="toggleLabels()">Метки</button>
        <button class="btn" onclick="togglePhysics()">Физика</button>
    </div>
    <div id="tooltip"></div>
    <svg id="viz"></svg>
    <script>
    const W = window.innerWidth, H = window.innerHeight;
    const groupNodesData = {json.dumps(group_nodes)};
    const groupEdgesData = {json.dumps(group_edges)};
    const peopleNodesData = {json.dumps(people_nodes)};
    const peopleEdgesData = {json.dumps(people_edges)};

    let nodes = [...groupNodesData];
    let links = [...groupEdgesData];
    let level = "groups";
    let sim;

    const svg = d3.select("#viz").attr("width",W).attr("height",H);
    const g = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.05,12]).on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);

    let linkEls, nodeEls, labelEls;

    function initSim() {{
        g.selectAll("*").remove();

        linkEls = g.append("g").selectAll("line").data(links).join("line")
            .attr("stroke","#30363d").attr("stroke-opacity",0.7)
            .attr("stroke-width", d => Math.sqrt(d.weight)*0.5 + 0.5);

        nodeEls = g.append("g").selectAll("circle").data(nodes).join("circle")
            .attr("r", d => d.type==="group" ? Math.sqrt(d.size)*4+10 : 6)
            .attr("fill", d => d.type==="group" ? "#58a6ff" : "#3fb950")
            .attr("stroke","#0d1117").attr("stroke-width",2)
            .attr("cursor","pointer")
            .on("click", (e,d) => {{ if(level==="groups" && d.type==="group") expandGroup(d); }})
            .on("dblclick", (e,d) => {{ if(level==="people") goHome(); }})
            .on("mouseover", (event, d) => {{
                const tip = document.getElementById("tooltip");
                if(d.type==="group") {{
                    tip.innerHTML = `<strong>${{d.label}}</strong><br>
                        Участников: ${{d.size}}<br>
                        Входящих меритов: ${{d.in_strength.toFixed(0)}}<br>
                        Исходящих меритов: ${{d.out_strength.toFixed(0)}}<br>
                        <em style="color:#8b949e">Клик — раскрыть</em>`;
                }} else {{
                    tip.innerHTML = `<strong>${{d.label}}</strong><br>
                        ${{d.position}}<br>
                        <span style="color:#58a6ff">${{d.company}}</span> / ${{d.dept}}<br>
                        <hr style="border-color:#30363d;margin:6px 0">
                        Входящих: ${{d.in_strength.toFixed(0)}} · Исходящих: ${{d.out_strength.toFixed(0)}}<br>
                        <em style="color:#8b949e">Двойной клик — назад</em>`;
                }}
                tip.style.opacity=1;
                tip.style.left=(event.pageX+12)+"px";
                tip.style.top=(event.pageY-10)+"px";
            }})
            .on("mouseout", () => {{ document.getElementById("tooltip").style.opacity=0; }})
            .call(d3.drag()
                .on("start",(e,d)=>{{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x;d.fy=d.y; }})
                .on("drag",(e,d)=>{{ d.fx=e.x;d.fy=e.y; }})
                .on("end",(e,d)=>{{ if(!e.active) sim.alphaTarget(0); d.fx=null;d.fy=null; }}));

        labelEls = g.append("g").selectAll("text").data(nodes).join("text")
            .attr("fill","#8b949e").attr("font-size","10px")
            .attr("text-anchor","middle").attr("dy",-10).attr("pointer-events","none")
            .text(d => d.label && d.label.length>20 ? d.label.slice(0,20)+"…" : d.label);

        if(sim) sim.stop();
        sim = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d=>d.id).distance(level==="groups"?160:80))
            .force("charge", d3.forceManyBody().strength(-280))
            .force("center", d3.forceCenter(W/2, H/2))
            .force("collision", d3.forceCollide().radius(d => d.type==="group" ? Math.sqrt(d.size)*4+15 : 12))
            .on("tick", () => {{
                linkEls.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
                       .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
                nodeEls.attr("cx",d=>d.x).attr("cy",d=>d.y);
                labelEls.attr("x",d=>d.x).attr("y",d=>d.y);
            }});
    }}

    function expandGroup(gNode) {{
        level = "people";
        const members = gNode.members || [];
        nodes = peopleNodesData.filter(n => members.includes(n.original_id));
        const mids = new Set(nodes.map(n=>n.id));
        links = peopleEdgesData.filter(l => mids.has(l.source) && mids.has(l.target));
        document.getElementById("breadcrumb").textContent = `Уровень: ${{gNode.label}} (двойной клик — назад)`;
        sim.stop(); initSim();
    }}

    function goHome() {{
        level = "groups";
        nodes = [...groupNodesData];
        links = [...groupEdgesData];
        document.getElementById("breadcrumb").textContent = "Уровень: Группы";
        sim.stop(); initSim();
    }}

    function resetZoom(){{ svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity); }}
    let labelsOn=true;
    function toggleLabels(){{ labelsOn=!labelsOn; labelEls.style("opacity", labelsOn?1:0); }}
    let physOn=true;
    function togglePhysics(){{ physOn=!physOn; physOn ? sim.alpha(0.3).restart() : sim.stop(); }}

    initSim();
    </script></body></html>"""
    return html


# ========================= ПРОДВИНУТАЯ СТАТИСТИКА =========================

def render_advanced_stats(G_people, metrics):
    st.markdown('<div class="section-header">Продвинутая социальная статистика</div>', unsafe_allow_html=True)

    nodes_metrics = []
    for node in G_people.nodes():
        nd = G_people.nodes[node]
        nodes_metrics.append({
            "id": node,
            "ФИО": nd.get("label",""),
            "Компания": nd.get("company",""),
            "Отдел": nd.get("dept",""),
            "PageRank": round(metrics["pagerank"].get(node,0),5),
            "Betweenness": round(metrics["betweenness"].get(node,0),5),
            "Closeness": round(metrics["closeness"].get(node,0),4),
            "Clustering": round(metrics["clustering"].get(node,0),4),
            "Eigenvector": round(metrics["eigenvector"].get(node,0),4),
            "Constraint": round(metrics["constraint"].get(node,0),4),
            "K-core": metrics["core_number"].get(node,0),
            "Bridge": metrics["is_bridge"].get(node,0),
            "Load": round(metrics["load"].get(node,0),5),
            "DeptDiv": round(metrics["dept_diversity"].get(node,0),3),
            "In": round(metrics["in_strength"].get(node,0),1),
            "Out": round(metrics["out_strength"].get(node,0),1),
        })
    df_m = pd.DataFrame(nodes_metrics)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""<div class="metric-card">
        <strong>PageRank</strong> — влиятельность в сети<br>
        <strong>Betweenness</strong> — посредничество между группами<br>
        <strong>Closeness</strong> — близость к центру сети<br>
        <strong>Clustering</strong> — плотность связей вокруг узла<br>
        <strong>Eigenvector</strong> — связан с влиятельными людьми<br>
        <strong>Constraint</strong> — ограниченность (↓ = больше структурных дыр)<br>
        <strong>K-core</strong> — принадлежность к ядру активности<br>
        <strong>Bridge</strong> — мост между сообществами<br>
        <strong>DeptDiv</strong> — разнообразие отделов в связях
        </div>""", unsafe_allow_html=True)

    with col2:
        avg_cl = df_m["Clustering"].mean()
        avg_cn = df_m["Constraint"].mean()
        n_br = int(df_m["Bridge"].sum())
        max_core = int(df_m["K-core"].max())
        mod = metrics.get("modularity",0)
        rec = metrics.get("reciprocity",0)

        st.markdown(f"""<div class="metric-card">
        <strong>Средняя кластеризация:</strong> {avg_cl:.3f}
        {"&nbsp;✅ высокая" if avg_cl > 0.3 else "&nbsp;⚠️ низкая"}<br>
        <strong>Средний Constraint:</strong> {avg_cn:.3f}
        {"&nbsp;✅ много структурных дыр" if avg_cn < 0.5 else "&nbsp;⚠️ высокий"}<br>
        <strong>Мостов в сети:</strong> {n_br}<br>
        <strong>Макс. K-core:</strong> {max_core}<br>
        <strong>Модулярность:</strong> {mod:.3f}<br>
        <strong>Взаимность:</strong> {rec:.3f}
        </div>""", unsafe_allow_html=True)

    st.markdown("#### 🏆 Топ по метрикам")
    tabs = st.tabs(["Opinion Leaders", "Brokers", "Gatekeepers", "Influencers", "Diverse Networks", "Core Members"])
    with tabs[0]:
        st.dataframe(df_m.nlargest(15,"PageRank")[["ФИО","Компания","Отдел","PageRank","In","Out"]], use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(df_m.nlargest(15,"Betweenness")[["ФИО","Компания","Отдел","Betweenness","Bridge"]], use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(df_m.nsmallest(15,"Constraint")[["ФИО","Компания","Отдел","Constraint","Betweenness"]], use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(df_m.nlargest(15,"Eigenvector")[["ФИО","Компания","Отдел","Eigenvector","PageRank"]], use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(df_m.nlargest(15,"DeptDiv")[["ФИО","Компания","Отдел","DeptDiv","Out"]], use_container_width=True, hide_index=True)
    with tabs[5]:
        st.dataframe(df_m.nlargest(15,"K-core")[["ФИО","Компания","Отдел","K-core","PageRank"]], use_container_width=True, hide_index=True)

    st.markdown("#### 📋 Полная таблица метрик")
    st.dataframe(df_m.sort_values("PageRank", ascending=False), use_container_width=True, hide_index=True, height=400)

    # Экспорт
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = df_m.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Скачать метрики (CSV)", csv, "metrics.csv", "text/csv")
    with col2:
        graph_data = {
            "nodes": [{"id":str(n),"label":G_people.nodes[n].get("label",""),
                       "dept":G_people.nodes[n].get("dept",""),
                       "company":G_people.nodes[n].get("company",""),
                       "pagerank":float(metrics["pagerank"].get(n,0)),
                       "community":int(metrics["communities"].get(n,0))} for n in G_people.nodes()],
            "edges": [{"source":str(u),"target":str(v),"weight":float(d.get("weight",1))}
                      for u,v,d in G_people.edges(data=True)],
            "stats": {"modularity": float(metrics.get("modularity",0)),
                      "reciprocity": float(metrics.get("reciprocity",0))}
        }
        st.download_button("📥 Скачать граф (JSON)",
                           json.dumps(graph_data, indent=2, ensure_ascii=False),
                           "graph.json", "application/json")


# ========================= MAIN =========================

def main():
    st.markdown("""
    <div style="padding:1.5rem 0 0.5rem">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:600;color:#58a6ff;">
            🕸️ СоциоГраф 6.0
        </span>
        <span style="color:#8b949e;font-size:0.85rem;margin-left:12px;">
            Аналитика программы 3Д Коммуникации · ГК Термекс
        </span>
    </div>
    """, unsafe_allow_html=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    emp_path = os.path.join(base_dir, "employees.xlsx")
    tx_path  = os.path.join(base_dir, "dataset.xlsx")

    missing = []
    if not os.path.exists(emp_path): missing.append("employees.xlsx")
    if not os.path.exists(tx_path):  missing.append("dataset.xlsx")
    if missing:
        st.error(f"❌ Файлы не найдены: {', '.join(missing)}\nПоложите их рядом с app.py")
        st.stop()

    with st.spinner("Загрузка данных..."):
        emp_df = load_employees(emp_path)
        tx_raw = load_transactions(tx_path)
        tx_df  = merge_data(tx_raw, emp_df)

    # --- Фильтры ---
    cfg = sidebar_controls(tx_df, emp_df)
    filtered = apply_filters(tx_df, emp_df, cfg)

    if len(filtered) == 0:
        st.warning("⚠️ Нет данных для выбранных фильтров")
        st.stop()

    # --- Шапка: 5 метрик ---
    n_companies = filtered[f"sender_{EMP_COLS['company']}"].nunique()
    n_depts     = filtered[f"sender_{EMP_COLS['dept']}"].nunique()
    n_emps      = pd.Index(filtered[TX_COLS["sender_id"]]).append(pd.Index(filtered[TX_COLS["receiver_id"]])).nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Транзакций",  f"{len(filtered):,}")
    c2.metric("Меритов",     f"{filtered[TX_COLS['merits']].sum():,}")
    c3.metric("Компаний",    f"{n_companies}")
    c4.metric("Отделов",     f"{n_depts}")
    c5.metric("Сотрудников", f"{n_emps:,}")

    # --- Строим граф ---
    with st.spinner("Строим граф..."):
        G_people, G_groups, group_members = build_graph(
            filtered, emp_df, cfg["graph_group"], cfg["merit_range"]
        )
        if G_people.number_of_nodes() == 0:
            st.warning("⚠️ Граф пуст после применения фильтров")
            st.stop()
        metrics_people = calculate_graph_metrics(G_people)
        metrics_groups = calculate_graph_metrics(G_groups)

    st.markdown(f"""<div class="metric-card" style="margin:0.5rem 0">
    <strong>Граф:</strong> {G_groups.number_of_nodes()} групп · {G_people.number_of_nodes()} сотрудников · {G_people.number_of_edges()} связей &nbsp;|&nbsp;
    <strong>Модулярность:</strong> {metrics_people.get('modularity',0):.3f} &nbsp;|&nbsp;
    <strong>Взаимность:</strong> {metrics_people.get('reciprocity',0):.3f}
    </div>""", unsafe_allow_html=True)

    # --- Визуализации ---
    st.markdown('<div class="section-header">Визуализация сети</div>', unsafe_allow_html=True)

    tab_social, tab_func = st.tabs(["🌀 Социальный граф", "🌐 Функциональная сеть"])

    with tab_social:
        st.markdown("""<div class="info-box">
        <strong>Социальный граф</strong> — все сотрудники одновременно.<br>
        Цвет узла = сообщество (алгоритм Louvain) · Размер = PageRank · Толщина связи = мериты<br>
        Наведите на узел для подробностей · Перетаскивайте · Скролл — зум
        </div>""", unsafe_allow_html=True)
        components.html(create_social_graph_viz(G_people, metrics_people), height=720, scrolling=False)

    with tab_func:
        st.markdown("""<div class="info-box">
        <strong>Функциональная сеть</strong> — иерархия: группы → люди.<br>
        <strong>Клик на группу</strong> — раскрыть сотрудников · <strong>Двойной клик</strong> — вернуться назад<br>
        Размер группы = число участников · Наведите на узел для подробностей
        </div>""", unsafe_allow_html=True)
        components.html(
            create_functional_network_viz(G_groups, G_people, group_members, metrics_groups, metrics_people),
            height=720, scrolling=False
        )

    # --- KPI ---
    kpis = compute_kpis(tx_raw, emp_df, filtered)
    render_kpis(kpis)

    # --- Топы ---
    render_tops(filtered, emp_df)

    # --- Продвинутая статистика ---
    render_advanced_stats(G_people, metrics_people)


if __name__ == "__main__":
    main()
