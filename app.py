# -*- coding: utf-8 -*-
"""
🕸️ СоциоГраф 8.0
==========================================================
V8: Расширенный датасет
- ФИО из 3 колонок (Фамилия, Имя, Отчество)
- Уникальный ID = Компания_Номер
- Отдел №1 + Отдел/подразделение №2
- Номер компании → фильтр + атрибут узла
- Обратная совместимость: если колонки старого формата — работает как раньше

Запуск: streamlit run streamlit_app_v8.py
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

st.set_page_config(
    page_title="СоциоГраф",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #00d4ff; }
    .stButton button {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.5rem 2rem; font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .info-box {
        background: rgba(123, 44, 191, 0.1);
        border: 2px solid #7b2cbf;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .role-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ========================= МАППИНГ КОЛОНОК =========================
# Новый расширенный формат
COLS_NEW = {
    "date":           "Дата",
    "time":           "Время",
    "s_last":         "Фамилия Отправителя",
    "s_first":        "Имя Отправителя",
    "s_middle":       "Отчество Отправителя",
    "s_num":          "Порядковый номер отправителя",
    "s_company_num":  "Номер компании отправителя",
    "s_role":         "Должность Отправителя",
    "s_dept1":        "Отдел/подразделение №1 Отправителя",
    "s_dept2":        "Отдел/подразделение №2 Отправителя",
    "r_last":         "Фамилия Получателя",
    "r_first":        "Имя Получателя",
    "r_middle":       "Отчество Получателя",
    "r_num":          "Порядковый номер получателя",
    "r_company_num":  "Номер компании получателя",
    "r_role":         "Должность Получателя",
    "r_dept1":        "Отдел/подразделение №1 Получателя",
    "r_dept2":        "Отдел/подразделение №2 Получателя",
    "value":          "Ценность",
    "merits":         "Мериты",
    "comment":        "Комментарий",
}

# Унифицированные имена внутри приложения (после load_df)
# ВАЖНО: dept1 в датасете = Компания, dept2 = Отдел внутри компании
C = {
    "date": "_date", "time": "_time", "dt": "dt",
    "s_fio": "_s_fio", "s_id": "_s_id", "s_company_num": "_s_company_num",
    "s_role": "_s_role", "s_company": "_s_company", "s_dept": "_s_dept",
    "r_fio": "_r_fio", "r_id": "_r_id", "r_company_num": "_r_company_num",
    "r_role": "_r_role", "r_company": "_r_company", "r_dept": "_r_dept",
    "value": "_value", "merits": "_merits", "comment": "_comment",
}

HR_NAMES = {
    "influence_index": "Индекс влиятельности",
    "gf": "Коэфф. признания", "vu": "Коэфф. использования голосов",
    "si": "Коэфф. устойчивости", "cii": "Коэфф. интеграции",
    "ci": "Коэфф. концентрации источников", "sar": "Коэфф. соц. активности",
    "dept_div": "Коэфф. кросс-функциональности", "idd": "Коэфф. кросс-функц. доверия",
    "evr_recv": "Равномерность получения", "evr_send": "Равномерность отправки",
    "betweenness_norm": "Индекс посредничества", "closeness": "Индекс доступности",
    "clustering": "Плотность окружения", "k_core": "Глубина интеграции",
}

SOCIAL_ROLES = {
    "leader_integrator":      {"name": "Лидер-интегратор",       "color": "#FFD700", "icon": "👑"},
    "internal_leader":        {"name": "Внутренний лидер",       "color": "#FF8C00", "icon": "🏆"},
    "connector":              {"name": "Связующее звено",        "color": "#00CED1", "icon": "🔗"},
    "strategic_broker":       {"name": "Стратегический посредник","color": "#9370DB", "icon": "🌉"},
    "network_builder":        {"name": "Строитель связей",       "color": "#32CD32", "icon": "🏗️"},
    "quiet_engine":           {"name": "Тихий двигатель",        "color": "#87CEEB", "icon": "⚙️"},
    "unrecognized_ambassador":{"name": "Посол без ответа",       "color": "#DDA0DD", "icon": "📡"},
    "inner_focus":            {"name": "Внутренний фокус",       "color": "#A9A9A9", "icon": "🔒"},
    "quiet_presence":         {"name": "Тихое участие",          "color": "#696969", "icon": "🌫️"},
}

# Справочник: номер компании → название
COMPANY_MAP = {
    "1":  "Корпорация Термекс",
    "2":  "Торговый дом ТЕРМЕКС",
    "3":  "Тепловое Оборудование",
    "4":  "Термекс ГазПро",
    "5":  "Термекс Энерджи",
    "6":  'ТОО "Термекс Сары-Арка"',
    "7":  "БЕЛАРУСЬ АКВАТЕРМЕКС",
    "8":  "Thermex.ge",
    "9":  "Thermex MLD",
    "10": "Термекс-Сервис",
    "11": "Центр Сервисных Компетенций",
    "12": "Heateq Technology",
}


# ========================= ЗАГРУЗКА ДАННЫХ =========================

def _safe_str(val):
    """Безопасное преобразование в строку, NaN → пустая строка"""
    if pd.isna(val):
        return ""
    return str(val).strip()


@st.cache_data(show_spinner=False)
def load_df(path_or_file):
    df = pd.read_excel(path_or_file, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # --- Определяем формат: новый или старый ---
    has_new_format = COLS_NEW["s_last"] in df.columns

    if has_new_format:
        # === НОВЫЙ ФОРМАТ ===
        # ФИО: собираем из 3 колонок → "Фамилия И.О."
        df[C["s_fio"]] = df.apply(
            lambda r: _build_fio(r.get(COLS_NEW["s_last"]),
                                 r.get(COLS_NEW["s_first"]),
                                 r.get(COLS_NEW["s_middle"])), axis=1)
        df[C["r_fio"]] = df.apply(
            lambda r: _build_fio(r.get(COLS_NEW["r_last"]),
                                 r.get(COLS_NEW["r_first"]),
                                 r.get(COLS_NEW["r_middle"])), axis=1)

        # Уникальный ID: компания_номер
        df[C["s_id"]] = df.apply(
            lambda r: _build_uid(r.get(COLS_NEW["s_company_num"]),
                                 r.get(COLS_NEW["s_num"])), axis=1)
        df[C["r_id"]] = df.apply(
            lambda r: _build_uid(r.get(COLS_NEW["r_company_num"]),
                                 r.get(COLS_NEW["r_num"])), axis=1)

        # Номер компании (сохраняем отдельно)
        df[C["s_company_num"]] = df[COLS_NEW["s_company_num"]].apply(_safe_str)
        df[C["r_company_num"]] = df[COLS_NEW["r_company_num"]].apply(_safe_str)

        # Должность
        df[C["s_role"]] = df[COLS_NEW["s_role"]].apply(_safe_str) if COLS_NEW["s_role"] in df.columns else ""
        df[C["r_role"]] = df[COLS_NEW["r_role"]].apply(_safe_str) if COLS_NEW["r_role"] in df.columns else ""

        # Компания = Отдел/подразделение №1 из датасета
        df[C["s_company"]] = df[COLS_NEW["s_dept1"]].apply(_safe_str) if COLS_NEW["s_dept1"] in df.columns else ""
        df[C["r_company"]] = df[COLS_NEW["r_dept1"]].apply(_safe_str) if COLS_NEW["r_dept1"] in df.columns else ""

        # Отдел = Отдел/подразделение №2 из датасета
        df[C["s_dept"]] = df[COLS_NEW["s_dept2"]].apply(_safe_str) if COLS_NEW["s_dept2"] in df.columns else ""
        df[C["r_dept"]] = df[COLS_NEW["r_dept2"]].apply(_safe_str) if COLS_NEW["r_dept2"] in df.columns else ""

        # Ценность, Мериты, Комментарий
        df[C["value"]] = df[COLS_NEW["value"]].apply(_safe_str) if COLS_NEW["value"] in df.columns else ""
        if COLS_NEW["merits"] in df.columns:
            df[C["merits"]] = pd.to_numeric(df[COLS_NEW["merits"]], errors="coerce").fillna(0).astype(int)
        else:
            df[C["merits"]] = 1
        df[C["comment"]] = df[COLS_NEW["comment"]].apply(_safe_str) if COLS_NEW["comment"] in df.columns else ""

        # Дата / Время
        date_col = COLS_NEW["date"]
        time_col = COLS_NEW["time"]

    else:
        # === СТАРЫЙ ФОРМАТ (обратная совместимость) ===
        OLD = {
            "date": "Дата", "time": "Вермя",
            "sender": "ФИО Отправителя", "sender_id": "№ Отправителя",
            "sender_role": "Должномть Отправителя", "sender_dept": "Отдел Отправителя",
            "receiver": "ФИО Получателя", "receiver_id": "№ Получателя",
            "receiver_role": "Должномть Получателя", "receiver_dept": "Отдел Получателя",
            "value": "Ценность", "merits": "Мериты (сила)", "comment": "Комментарий",
        }
        # Нечувствительный маппинг
        col_lower = {c.lower(): c for c in df.columns}
        def _get(key):
            target = OLD.get(key, "")
            return col_lower.get(target.lower(), None)

        s_col = _get("sender")
        df[C["s_fio"]] = df[s_col].apply(_safe_str) if s_col else ""
        df[C["r_fio"]] = df[_get("receiver")].apply(_safe_str) if _get("receiver") else ""
        df[C["s_id"]] = df[_get("sender_id")].apply(_safe_str) if _get("sender_id") else ""
        df[C["r_id"]] = df[_get("receiver_id")].apply(_safe_str) if _get("receiver_id") else ""
        df[C["s_company_num"]] = ""
        df[C["r_company_num"]] = ""
        df[C["s_role"]] = df[_get("sender_role")].apply(_safe_str) if _get("sender_role") else ""
        df[C["r_role"]] = df[_get("receiver_role")].apply(_safe_str) if _get("receiver_role") else ""
        df[C["s_company"]] = df[_get("sender_dept")].apply(_safe_str) if _get("sender_dept") else ""
        df[C["s_dept"]] = ""
        df[C["r_company"]] = df[_get("receiver_dept")].apply(_safe_str) if _get("receiver_dept") else ""
        df[C["r_dept"]] = ""
        df[C["value"]] = df[_get("value")].apply(_safe_str) if _get("value") else ""
        m_col = _get("merits")
        if m_col:
            df[C["merits"]] = pd.to_numeric(df[m_col], errors="coerce").fillna(0).astype(int)
        else:
            df[C["merits"]] = 1
        df[C["comment"]] = df[_get("comment")].apply(_safe_str) if _get("comment") else ""
        date_col = _get("date")
        time_col = _get("time")

    # --- Парсинг даты ---
    def parse_dt(row):
        d = row.get(date_col, None) if date_col else None
        t = row.get(time_col, "00:00:00") if time_col else "00:00:00"
        if pd.isna(d):
            return pd.NaT
        ds = str(d)
        ts = str(t) if not pd.isna(t) else "00:00:00"
        return pd.to_datetime(f"{ds} {ts}", dayfirst=True, errors="coerce")

    df["dt"] = df.apply(parse_dt, axis=1)

    return df


def _build_fio(last, first, middle):
    """Фамилия И.О. — компактный формат для отображения"""
    last = _safe_str(last)
    first = _safe_str(first)
    middle = _safe_str(middle)
    if not last:
        return first or "—"
    parts = [last]
    if first:
        parts.append(first[0] + ".")
    if middle:
        parts.append(middle[0] + ".")
    return " ".join(parts)


def _build_uid(company_num, person_num):
    """Уникальный ID: компания_номер"""
    cn = _safe_str(company_num)
    pn = _safe_str(person_num)
    if cn and pn:
        return f"{cn}_{pn}"
    return pn or cn or "unknown"


# ========================= ПОСТРОЕНИЕ ГРАФОВ =========================

def build_hierarchical_graph(df: pd.DataFrame, merit_range: tuple = (1, 50),
                             dept_level: str = "dept"):
    """
    dept_level: "company" — группировка узлов по компаниям
                "dept"    — группировка узлов по отделам внутри компаний
    """
    df = df[df[C["s_id"]] != df[C["r_id"]]].copy()

    # Определяем какое поле использовать для группировки в граф отделов
    if dept_level == "company":
        df["_s_grp"] = df[C["s_company"]]
        df["_r_grp"] = df[C["r_company"]]
    else:
        df["_s_grp"] = df[C["s_dept"]]
        df["_r_grp"] = df[C["r_dept"]]

    # Агрегация на уровне пар сотрудников
    person_agg = (
        df.groupby([C["s_id"], C["r_id"]], dropna=False)
        .agg(
            total_merits=(C["merits"], "sum"),
            n_msgs=("dt", "count"),
            s_fio=(C["s_fio"], "first"),
            r_fio=(C["r_fio"], "first"),
            s_grp=("_s_grp", "first"),
            r_grp=("_r_grp", "first"),
            s_company=(C["s_company"], "first"),
            r_company=(C["r_company"], "first"),
        )
        .reset_index()
    )

    min_m, max_m = merit_range
    person_agg = person_agg[
        (person_agg["total_merits"] >= min_m) & (person_agg["total_merits"] <= max_m)
    ].copy()

    G_people = nx.DiGraph()
    for _, row in person_agg.iterrows():
        sid, rid = row[C["s_id"]], row[C["r_id"]]
        sname, rname = row["s_fio"], row["r_fio"]
        sdept, rdept = row["s_grp"], row["r_grp"]
        scomp, rcomp = row["s_company"], row["r_company"]

        if sid not in G_people:
            G_people.add_node(sid, label=sname, dept=sdept, company=scomp, type="person")
        if rid not in G_people:
            G_people.add_node(rid, label=rname, dept=rdept, company=rcomp, type="person")

        w = float(row["total_merits"])
        G_people.add_edge(sid, rid, weight=w, length=1.0 / max(w, 0.01), msgs=int(row["n_msgs"]))

    # Граф отделов/компаний
    dept_agg = (
        person_agg.groupby(["s_grp", "r_grp"])
        .agg(total_merits=("total_merits", "sum"), n_people=("total_merits", "count"))
        .reset_index()
    )
    G_depts = nx.DiGraph()
    dept_members = {}
    for node in G_people.nodes():
        dept = G_people.nodes[node].get("dept", "")
        dept_members.setdefault(dept, []).append(node)
    for dept, members in dept_members.items():
        G_depts.add_node(dept, label=dept, type="dept", size=len(members), members=members)
    for _, row in dept_agg.iterrows():
        sd, rd = row["s_grp"], row["r_grp"]
        if sd != rd:
            G_depts.add_edge(sd, rd, weight=float(row["total_merits"]), people=int(row["n_people"]))

    return G_people, G_depts, dept_members


# ========================= ГРАФОВЫЕ МЕТРИКИ =========================

def calculate_graph_metrics(G: nx.DiGraph):
    if G.number_of_nodes() == 0:
        return {}
    metrics = {}
    metrics['in_strength'] = dict(G.in_degree(weight="weight"))
    metrics['out_strength'] = dict(G.out_degree(weight="weight"))
    try: metrics['pagerank'] = nx.pagerank(G, weight="weight", max_iter=100)
    except: metrics['pagerank'] = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
    UG = G.to_undirected()
    try: metrics['betweenness'] = nx.betweenness_centrality(UG, weight='length', normalized=True)
    except: metrics['betweenness'] = {n: 0.0 for n in G.nodes()}
    try: metrics['closeness'] = nx.closeness_centrality(UG, distance='length')
    except: metrics['closeness'] = {n: 0.0 for n in G.nodes()}
    try: metrics['clustering'] = nx.clustering(UG, weight='weight')
    except: metrics['clustering'] = {n: 0.0 for n in G.nodes()}
    try: metrics['constraint'] = nx.constraint(UG, weight='weight')
    except: metrics['constraint'] = {n: 0.0 for n in G.nodes()}
    try: metrics['core_number'] = nx.core_number(UG)
    except: metrics['core_number'] = {n: 0 for n in G.nodes()}
    try:
        bridges = list(nx.bridges(UG))
        bn = set()
        for u, v in bridges: bn.add(u); bn.add(v)
        metrics['is_bridge'] = {n: 1 if n in bn else 0 for n in G.nodes()}
    except: metrics['is_bridge'] = {n: 0 for n in G.nodes()}
    # DeptDiv
    dd = {}
    for node in G.nodes():
        nbs = set(G.neighbors(node)) | set(G.predecessors(node))
        if not nbs: dd[node] = 0.0
        else:
            depts = set(G.nodes[n].get('dept', '') for n in nbs if G.nodes[n].get('dept', ''))
            dd[node] = len(depts) / max(len(nbs), 1)
    metrics['dept_diversity'] = dd
    try:
        part = community_louvain.best_partition(UG, weight="weight")
        metrics['communities'] = part
        metrics['modularity'] = community_louvain.modularity(part, UG, weight="weight")
    except:
        metrics['communities'] = {n: 0 for n in G.nodes()}
        metrics['modularity'] = 0.0
    metrics['reciprocity'] = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0
    return metrics


# ========================= HR-МЕТРИКИ =========================

def calculate_hr_metrics(G, df, graph_metrics, merits_per_month=10, total_employees=0):
    nodes = list(G.nodes())
    if not nodes: return {}
    hr = {}
    ins = graph_metrics.get('in_strength', {})
    outs = graph_metrics.get('out_strength', {})

    # GF
    all_r = [ins.get(n, 0) for n in nodes]
    avg_r = np.mean(all_r) if np.mean(all_r) > 0 else 1.0
    hr['gf'] = {n: ins.get(n, 0) / avg_r for n in nodes}

    # SI
    dfc = df.copy()
    dfc['_month'] = dfc['dt'].dt.to_period('M')
    tm = max(dfc['_month'].nunique(), 1)
    sm = dfc.groupby(C["s_id"])['_month'].nunique().to_dict()
    hr['si'] = {n: sm.get(n, 0) / tm for n in nodes}

    # CII
    cii = {}
    for n in nodes:
        nd = G.nodes[n].get('dept', '')
        succ = list(G.neighbors(n))
        cii[n] = sum(1 for s in succ if G.nodes[s].get('dept', '') != nd) / max(len(succ), 1) if succ else 0.0
    hr['cii'] = cii

    # CI
    ci = {}
    for n in nodes:
        preds = list(G.predecessors(n))
        if not preds: ci[n] = 0.0
        else:
            iw = sorted([(G[p][n].get('weight', 0)) for p in preds], reverse=True)
            ci[n] = sum(iw[:3]) / ins.get(n, 1) if ins.get(n, 0) > 0 else 0.0
    hr['ci'] = ci

    # SAR
    hr['sar'] = {n: (ins.get(n, 0) + outs.get(n, 0)) / 10.0 for n in nodes}

    # VU
    if merits_per_month > 0 and tm > 0:
        avail = merits_per_month * tm
        hr['vu'] = {n: min(outs.get(n, 0) / avail, 1.0) for n in nodes}
    else:
        hr['vu'] = {n: 0.0 for n in nodes}

    # IDD
    total_depts = max(len(set(G.nodes[n].get('dept', '') for n in nodes)), 1)
    idd = {}
    for n in nodes:
        preds = list(G.predecessors(n))
        idd[n] = len(set(G.nodes[p].get('dept', '') for p in preds)) / total_depts if preds else 0.0
    hr['idd'] = idd

    # Influence Index
    def norm(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx - mn > 0 else 1.0
        return {k: (v - mn) / rng for k, v in d.items()}

    pr_n = norm(graph_metrics.get('pagerank', {n: 0 for n in nodes}))
    idd_n = norm(hr['idd']); cii_n = norm(hr['cii'])
    si_n = norm(hr['si']); gf_n = norm(hr['gf'])

    fa = dfc.groupby(C["s_id"])['_month'].min().to_dict()
    lm = dfc['_month'].max()
    tenure = {}
    for n in nodes:
        fm = fa.get(n, lm)
        tenure[n] = max((lm - fm).n, 0) if not pd.isna(fm) and hasattr(lm - fm, 'n') else 0
    mt = max(tenure.values()) if tenure and max(tenure.values()) > 0 else 1.0
    t_n = {k: v / mt for k, v in tenure.items()}

    hr['influence_index'] = {
        n: pr_n.get(n, 0) * 0.25 + idd_n.get(n, 0) * 0.20 + cii_n.get(n, 0) * 0.15 +
           si_n.get(n, 0) * 0.15 + t_n.get(n, 0) * 0.10 + gf_n.get(n, 0) * 0.15
        for n in nodes
    }
    return hr


# ========================= СОЦИАЛЬНЫЕ РОЛИ =========================

def assign_social_roles(G, gm, hr):
    roles = {}
    nodes = list(G.nodes())
    if not nodes: return roles
    gf = hr.get('gf', {}); vu = hr.get('vu', {}); cii = hr.get('cii', {})
    betw = gm.get('betweenness', {}); dd = gm.get('dept_diversity', {})
    bridge = gm.get('is_bridge', {}); con = gm.get('constraint', {})
    gf_m = np.median([gf.get(n, 0) for n in nodes]) or 0.5
    vu_m = np.median([vu.get(n, 0) for n in nodes]) or 0.1
    cii_m = np.median([cii.get(n, 0) for n in nodes]) or 0.2
    b75 = np.percentile([betw.get(n, 0) for n in nodes], 75) or 0.05
    for n in nodes:
        g, v, c = gf.get(n, 0), vu.get(n, 0), cii.get(n, 0)
        b, d, br, cn = betw.get(n, 0), dd.get(n, 0), bridge.get(n, 0), con.get(n, 1.0)
        gh, gl = g > gf_m * 1.3, g < gf_m * 0.5
        vh, vl = v > vu_m * 1.3 or v > 0.15, v < vu_m * 0.5 or v < 0.05
        ch, cz = c > cii_m * 1.3 or c > 0.3, c < 0.02
        bh = b > b75
        if bh and d > 0.3 and (br == 1 or cn < 0.4): roles[n] = "strategic_broker"
        elif gh and vh and ch: roles[n] = "leader_integrator"
        elif gh and vh and not ch: roles[n] = "internal_leader"
        elif not gh and not gl and ch: roles[n] = "connector"
        elif cn < 0.35 and d > 0.4: roles[n] = "network_builder"
        elif gl and vh and ch: roles[n] = "unrecognized_ambassador"
        elif gl and vh and not ch: roles[n] = "quiet_engine"
        elif cz and not gl: roles[n] = "inner_focus"
        elif gl and vl: roles[n] = "quiet_presence"
        else: roles[n] = "connector"
    return roles


# ========================= EvR =========================

def calculate_evenness(vals):
    arr = np.array(sorted(vals))
    n = len(arr)
    if n == 0 or arr.sum() == 0: return 0.0
    idx = np.arange(1, n + 1)
    gini = (2 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1) / n
    return max(0.0, min(1.0, 1.0 - gini))


# ========================= D3 ВИЗУАЛИЗАЦИИ =========================

def create_hierarchical_d3_viz(G_depts, G_people, dept_members, metrics_depts, metrics_people):
    dept_nodes = [{"id": f"dept_{n}", "original_id": n, "label": G_depts.nodes[n].get("label", str(n)),
                   "type": "dept", "size": G_depts.nodes[n].get("size", 1),
                   "members": G_depts.nodes[n].get("members", []),
                   "in_strength": metrics_depts.get("in_strength", {}).get(n, 0),
                   "out_strength": metrics_depts.get("out_strength", {}).get(n, 0)} for n in G_depts.nodes()]
    dept_edges = [{"source": f"dept_{u}", "target": f"dept_{v}", "weight": d.get("weight", 1),
                   "people": d.get("people", 0)} for u, v, d in G_depts.edges(data=True)]
    ppl_nodes = [{"id": f"person_{n}", "original_id": n, "label": G_people.nodes[n].get("label", str(n)),
                  "dept": G_people.nodes[n].get("dept", ""), "type": "person",
                  "in_strength": metrics_people.get("in_strength", {}).get(n, 0),
                  "out_strength": metrics_people.get("out_strength", {}).get(n, 0),
                  "pagerank": metrics_people.get("pagerank", {}).get(n, 0)} for n in G_people.nodes()]
    ppl_edges = [{"source": f"person_{u}", "target": f"person_{v}", "weight": d.get("weight", 1),
                  "msgs": d.get("msgs", 0)} for u, v, d in G_people.edges(data=True)]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>body{{margin:0;background:#0a0e27;font-family:'Segoe UI',sans-serif;overflow:hidden;}}#viz{{width:100%;height:100vh;}}.controls{{position:absolute;top:10px;right:10px;z-index:1000;}}.btn{{background:linear-gradient(90deg,#00d4ff,#7b2cbf);color:white;border:none;padding:8px 15px;margin:2px;border-radius:5px;cursor:pointer;font-weight:600;font-size:12px;}}.btn:hover{{opacity:0.8;}}.node{{cursor:pointer;stroke:#fff;stroke-width:2px;}}.node.dept{{fill:#7b2cbf;}}.node.person{{fill:#00d4ff;}}.link{{stroke:#999;stroke-opacity:0.4;}}.label{{fill:white;font-size:11px;pointer-events:none;text-anchor:middle;text-shadow:0 0 3px #000;}}#breadcrumb{{position:absolute;top:10px;left:10px;color:#00d4ff;font-size:16px;font-weight:bold;text-shadow:0 0 10px rgba(0,212,255,0.8);}}#info{{position:absolute;bottom:10px;left:10px;color:white;font-size:12px;background:rgba(0,0,0,0.7);padding:10px;border-radius:5px;max-width:300px;}}</style></head><body>
    <div id="breadcrumb">Уровень: Отделы</div><div id="info">Загрузка...</div>
    <div class="controls"><button class="btn" onclick="resetView()">🏠 Домой</button><button class="btn" onclick="resetZoom()">🔍 Зум</button><button class="btn" onclick="toggleLabels()">🏷️ Метки</button><button class="btn" onclick="togglePhysics()">⚡ Физика</button></div><svg id="viz"></svg>
    <script>const W=window.innerWidth,H=window.innerHeight;const DN={json.dumps(dept_nodes)};const DL={json.dumps(dept_edges)};const PN={json.dumps(ppl_nodes)};const PL={json.dumps(ppl_edges)};let nodes=[...DN],links=[...DL],lvl="depts";const svg=d3.select("#viz").attr("width",W).attr("height",H);const g=svg.append("g");const zm=d3.zoom().scaleExtent([0.1,10]).on("zoom",e=>g.attr("transform",e.transform));svg.call(zm);let le,ne,lb,sim;
    function init(){{g.selectAll("*").remove();le=g.append("g").selectAll("line").data(links).join("line").attr("class","link").attr("stroke-width",d=>Math.sqrt(d.weight)/2);ne=g.append("g").selectAll("circle").data(nodes).join("circle").attr("class",d=>`node ${{d.type}}`).attr("r",d=>d.type==="dept"?Math.sqrt(d.size)*5+10:6).on("click",(e,d)=>{{e.stopPropagation();if(lvl==="depts"&&d.type==="dept")expand(d);}}).on("dblclick",(e,d)=>{{e.stopPropagation();if(lvl==="people")collapse();}}).on("mouseover",(e,d)=>{{let i=`<strong>${{d.label}}</strong><br>`;i+=d.type==="dept"?`Сотрудников: ${{d.size}}<br>Вх: ${{d.in_strength.toFixed(1)}} Исх: ${{d.out_strength.toFixed(1)}}`:`Отдел: ${{d.dept}}<br>Вх: ${{d.in_strength.toFixed(1)}} Исх: ${{d.out_strength.toFixed(1)}}`;document.getElementById("info").innerHTML=i;}}).call(d3.drag().on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}}).on("drag",(e,d)=>{{d.fx=e.x;d.fy=e.y;}}).on("end",(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));lb=g.append("g").selectAll("text").data(nodes).join("text").attr("class","label").attr("dy",-10).text(d=>d.label.length>20?d.label.slice(0,20)+"...":d.label);sim=d3.forceSimulation(nodes).force("link",d3.forceLink(links).id(d=>d.id).distance(lvl==="depts"?150:80)).force("charge",d3.forceManyBody().strength(-300)).force("center",d3.forceCenter(W/2,H/2)).force("collision",d3.forceCollide().radius(d=>d.type==="dept"?Math.sqrt(d.size)*5+15:10)).on("tick",()=>{{le.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);ne.attr("cx",d=>d.x).attr("cy",d=>d.y);lb.attr("x",d=>d.x).attr("y",d=>d.y);}});}}
    function expand(dn){{lvl="people";const m=dn.members||[];nodes=PN.filter(n=>m.includes(n.original_id));const ids=new Set(nodes.map(n=>n.id));links=PL.filter(l=>ids.has(l.source)&&ids.has(l.target));document.getElementById("breadcrumb").textContent=`${{dn.label}} (dbl-click назад)`;sim.stop();init();}}
    function collapse(){{lvl="depts";nodes=[...DN];links=[...DL];document.getElementById("breadcrumb").textContent="Уровень: Отделы";sim.stop();init();}}
    function resetView(){{collapse();}}function resetZoom(){{svg.transition().duration(750).call(zm.transform,d3.zoomIdentity);}}let lv=true;function toggleLabels(){{lv=!lv;lb.style("opacity",lv?1:0);}}let pv=true;function togglePhysics(){{pv=!pv;if(pv)sim.alpha(.3).restart();else sim.stop();}}init();</script></body></html>"""
    return html


def create_force_d3_viz(G, metrics):
    nd = [{"id":str(n),"label":G.nodes[n].get("label",""),"dept":G.nodes[n].get("dept",""),
           "community":int(metrics.get("communities",{}).get(n,0)),
           "pagerank":float(metrics.get("pagerank",{}).get(n,0)),
           "in_strength":float(metrics.get("in_strength",{}).get(n,0)),
           "out_strength":float(metrics.get("out_strength",{}).get(n,0))} for n in G.nodes()]
    ed = [{"source":str(u),"target":str(v),"weight":float(d.get("weight",1))} for u,v,d in G.edges(data=True)]
    nc = max(len(set(metrics.get("communities",{}).values())),1)
    cl = ["#00d4ff","#7b2cbf","#ff006e","#ffbe0b","#8ac926","#3a86ff","#fb5607","#06ffa5","#8338ec","#e9c46a"]
    nodes_json = json.dumps(nd, ensure_ascii=False)
    edges_json = json.dumps(ed, ensure_ascii=False)
    colors_json = json.dumps(cl[:max(nc,1)])

    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin:0; padding:0; background:#0a0e27; overflow:hidden; }
        svg { width:100%; height:100%; display:block; }
        .node { cursor:pointer; stroke:#fff; stroke-width:1.5px; }
        .link { stroke:#999; stroke-opacity:.3; }
        .label { fill:white; font-size:10px; pointer-events:none; text-anchor:middle;
                 text-shadow:0 0 3px #000; }
        .controls { position:absolute; top:10px; right:10px; z-index:1000; }
        .btn { background:linear-gradient(90deg,#00d4ff,#7b2cbf); color:white;
               border:none; padding:8px 15px; margin:2px; border-radius:5px;
               cursor:pointer; font-size:12px; }
        #info { position:absolute; bottom:10px; left:10px; color:white;
                background:rgba(0,0,0,.7); padding:10px; border-radius:5px;
                font-size:12px; max-width:300px; }
    </style></head><body>
    <div class="controls">
        <button class="btn" onclick="resetZoom()">🔍 Зум</button>
        <button class="btn" onclick="toggleLabels()">🏷️ Метки</button>
        <button class="btn" onclick="togglePhysics()">⚡ Физика</button>
    </div>
    <div id="info">Наведите на узел для информации</div>
    <svg id="viz"></svg>
    <script>
    (function() {
        var W = Math.max(document.documentElement.clientWidth || 800, 800);
        var H = Math.max(document.documentElement.clientHeight || 600, 600);
        var nodes = """ + nodes_json + """;
        var links = """ + edges_json + """;
        var colors = """ + colors_json + """;

        var svg = d3.select("#viz").attr("width", W).attr("height", H)
            .attr("viewBox", "0 0 " + W + " " + H);
        var g = svg.append("g");
        var zoom = d3.zoom().scaleExtent([0.1, 10])
            .on("zoom", function(event) { g.attr("transform", event.transform); });
        svg.call(zoom);

        var linkEl = g.append("g").selectAll("line").data(links).join("line")
            .attr("class", "link")
            .attr("stroke-width", function(d) { return Math.max(Math.sqrt(d.weight) / 2, 0.5); });

        var nodeEl = g.append("g").selectAll("circle").data(nodes).join("circle")
            .attr("class", "node")
            .attr("r", function(d) { return Math.max(3 + Math.sqrt(d.pagerank * 1000), 4); })
            .attr("fill", function(d) { return colors[d.community % colors.length]; })
            .on("mouseover", function(event, d) {
                document.getElementById("info").innerHTML =
                    "<strong>" + d.label + "</strong><br>" +
                    "Отдел: " + d.dept + "<br>" +
                    "Сообщество: " + d.community + "<br>" +
                    "Вх: " + d.in_strength.toFixed(1) + " Исх: " + d.out_strength.toFixed(1);
            })
            .call(d3.drag()
                .on("start", function(event, d) {
                    if (!event.active) sim.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on("drag", function(event, d) { d.fx = event.x; d.fy = event.y; })
                .on("end", function(event, d) {
                    if (!event.active) sim.alphaTarget(0);
                    d.fx = null; d.fy = null;
                })
            );

        var labelEl = g.append("g").selectAll("text").data(nodes).join("text")
            .attr("class", "label").attr("dy", -8)
            .text(function(d) { return d.label.length > 15 ? d.label.slice(0,15) + "..." : d.label; });

        var sim = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(function(d) { return d.id; }).distance(70))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(W / 2, H / 2))
            .force("collision", d3.forceCollide().radius(15))
            .on("tick", function() {
                linkEl.attr("x1", function(d){return d.source.x;}).attr("y1", function(d){return d.source.y;})
                      .attr("x2", function(d){return d.target.x;}).attr("y2", function(d){return d.target.y;});
                nodeEl.attr("cx", function(d){return d.x;}).attr("cy", function(d){return d.y;});
                labelEl.attr("x", function(d){return d.x;}).attr("y", function(d){return d.y;});
            });

        window.resetZoom = function() {
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        };
        var labelsOn = true;
        window.toggleLabels = function() {
            labelsOn = !labelsOn;
            labelEl.style("opacity", labelsOn ? 1 : 0);
        };
        var physicsOn = true;
        window.togglePhysics = function() {
            physicsOn = !physicsOn;
            if (physicsOn) sim.alpha(0.3).restart(); else sim.stop();
        };
    })();
    </script></body></html>"""
    return html


# ========================= SIDEBAR & FILTERING =========================

def sidebar_controls(df: pd.DataFrame):
    st.sidebar.header("⚙️ Настройки")

    # --- Компания (= Отдел/подразделение №1 в датасете) ---
    all_companies = sorted(set(
        df[C["s_company"]].dropna().unique().tolist() + df[C["r_company"]].dropna().unique().tolist()
    ))
    all_companies = [c for c in all_companies if c]
    selected_companies = st.sidebar.multiselect("🏭 Компания", options=all_companies, default=all_companies)

    # --- Отдел (= Отдел/подразделение №2 в датасете, зависит от выбранных компаний) ---
    df_comp = df[df[C["s_company"]].isin(selected_companies) | df[C["r_company"]].isin(selected_companies)]
    all_depts = sorted(set(
        df_comp[C["s_dept"]].dropna().unique().tolist() + df_comp[C["r_dept"]].dropna().unique().tolist()
    ))
    all_depts = [d for d in all_depts if d]
    if all_depts:
        selected_depts = st.sidebar.multiselect("🏢 Отдел", options=all_depts, default=all_depts)
    else:
        selected_depts = []

    # --- Группировка графа ---
    dept_level = "dept"
    if all_depts:
        dept_level = st.sidebar.radio(
            "Группировка графа", options=["company", "dept"],
            format_func=lambda x: "По компаниям" if x == "company" else "По отделам",
            index=1,  # по умолчанию: по отделам
            horizontal=True
        )
    else:
        dept_level = "company"

    # --- Сотрудники (зависят от компаний + отделов) ---
    df_dept_filt = df_comp
    if selected_depts:
        df_dept_filt = df_comp[
            df_comp[C["s_dept"]].isin(selected_depts) | df_comp[C["r_dept"]].isin(selected_depts) |
            (df_comp[C["s_dept"]] == "") | (df_comp[C["r_dept"]] == "")
        ]

    # --- Период: год / месяц ---
    st.sidebar.markdown("### 📅 Период")
    df_dates = df["dt"].dropna()
    available_years = sorted(df_dates.dt.year.unique().tolist())
    selected_years = st.sidebar.multiselect("Год", options=available_years, default=available_years)

    month_names = {1:"Январь",2:"Февраль",3:"Март",4:"Апрель",5:"Май",6:"Июнь",
                   7:"Июль",8:"Август",9:"Сентябрь",10:"Октябрь",11:"Ноябрь",12:"Декабрь"}
    df_in_y = df_dates[df_dates.dt.year.isin(selected_years)] if selected_years else df_dates
    avail_m = sorted(df_in_y.dt.month.unique().tolist())
    m_opts = [month_names.get(m, str(m)) for m in avail_m]
    sel_m_names = st.sidebar.multiselect("Месяц", options=m_opts, default=m_opts)
    n2m = {v: k for k, v in month_names.items()}
    selected_months = [n2m.get(mn, 0) for mn in sel_m_names]

    st.sidebar.markdown("---")
    all_ppl = sorted(set(
        df_dept_filt[C["s_fio"]].dropna().unique().tolist() +
        df_dept_filt[C["r_fio"]].dropna().unique().tolist()
    ))
    all_ppl = [p for p in all_ppl if p and p != "—"]
    selected_people = st.sidebar.multiselect(
        "👤 Сотрудники", options=all_ppl, default=[],
        help="Пусто = все сотрудники"
    )

    st.sidebar.markdown("---")

    # --- Ценности ---
    values_list = sorted(df[C["value"]].dropna().unique().tolist())
    values_list = [v for v in values_list if v]
    selected_values = st.sidebar.multiselect("⭐ Ценности", options=values_list, default=values_list)

    # --- Мериты ---
    st.sidebar.markdown("### 💎 Мериты на связь")
    merit_range = st.sidebar.slider("Диапазон", min_value=1, max_value=50, value=(1, 50), step=1)

    # --- Параметры ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Параметры программы")
    mpm = st.sidebar.number_input("Меритов в месяц", min_value=1, max_value=100, value=10)
    te = st.sidebar.number_input("Всего сотрудников", min_value=0, max_value=10000, value=0,
                                 help="Для LR. 0 = не считать")
    st.sidebar.markdown("---")
    show_stats = st.sidebar.checkbox("📊 Продвинутая статистика", value=True)

    return {
        "companies": set(selected_companies),
        "depts": set(selected_depts),
        "dept_level": dept_level,
        "years": selected_years, "months": selected_months,
        "people": selected_people,
        "values": set(selected_values),
        "merit_range": merit_range,
        "merits_per_month": mpm, "total_employees": te,
        "show_social_stats": show_stats,
    }


def filter_df(df, cfg):
    m = pd.Series(True, index=df.index)
    if cfg["companies"]:
        m &= (df[C["s_company"]].isin(cfg["companies"]) | df[C["r_company"]].isin(cfg["companies"]))
    if cfg["years"]:
        m &= df["dt"].dt.year.isin(cfg["years"])
    if cfg["months"]:
        m &= df["dt"].dt.month.isin(cfg["months"])
    m &= df[C["value"]].isin(cfg["values"])
    if cfg["depts"]:
        m &= (df[C["s_dept"]].isin(cfg["depts"]) | df[C["r_dept"]].isin(cfg["depts"]) |
               (df[C["s_dept"]] == "") | (df[C["r_dept"]] == ""))
    if cfg["people"]:
        m &= (df[C["s_fio"]].isin(cfg["people"]) | df[C["r_fio"]].isin(cfg["people"]))
    return df.loc[m].copy()


# ========================= MAIN =========================

def main():
    st.markdown("""<div style='text-align:center;padding:2rem 0;'>
        <h1 style='font-size:3rem;'>🕸️ СоциоГраф 8.0</h1>
        <p style='font-size:1.2rem;color:#00d4ff;'>Иерархическая визуализация + HR-аналитика + Социальные роли</p>
    </div>""", unsafe_allow_html=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "dataset.xlsx")
    if os.path.exists(local_path):
        df = load_df(local_path)
    else:
        st.error("❌ Файл dataset.xlsx не найден рядом со скриптом.")
        st.stop()

    cfg = sidebar_controls(df)
    df_f = filter_df(df, cfg)
    if len(df_f) == 0:
        st.warning("⚠️ Нет данных для выбранных фильтров")
        st.stop()

    # Верхние метрики
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("📊 Транзакций", f"{len(df_f):,}")
    with c2:
        uniq = pd.Index(df_f[C["s_id"]]).append(pd.Index(df_f[C["r_id"]])).nunique()
        st.metric("👥 Сотрудников", f"{uniq:,}")
    with c3: st.metric("⭐ Меритов", f"{df_f[C['merits']].sum():,}")
    with c4:
        n_depts = len(set(df_f[C['s_dept']].unique().tolist() + df_f[C['r_dept']].unique().tolist()) - {""})
        st.metric("🏢 Отделов", f"{n_depts:,}")
    with c5:
        n_comp = len(set(df_f[C["s_company"]].unique().tolist() + df_f[C["r_company"]].unique().tolist()) - {""})
        if n_comp > 0:
            st.metric("🏭 Компаний", f"{n_comp}")

    # Графы
    with st.spinner("🔄 Строим структуру..."):
        G_people, G_depts, dept_members = build_hierarchical_graph(df_f, cfg["merit_range"], cfg["dept_level"])
        if G_depts.number_of_nodes() == 0 or G_people.number_of_nodes() == 0:
            st.warning("⚠️ Граф пуст"); st.stop()
        m_d = calculate_graph_metrics(G_depts)
        m_p = calculate_graph_metrics(G_people)

    with st.spinner("📐 HR-метрики..."):
        hr = calculate_hr_metrics(G_people, df_f, m_p, cfg["merits_per_month"], cfg["total_employees"])
        roles = assign_social_roles(G_people, m_p, hr)

    # Сводка
    ns = df_f[C["s_id"]].nunique()
    lr_t = ""
    if cfg["total_employees"] > 0:
        lr_t = f" | <strong>LR:</strong> {ns / cfg['total_employees']:.2f}"
    st.markdown(f"""<div class='metric-card'>
        <strong>Граф:</strong> {G_depts.number_of_nodes()} отделов, {G_people.number_of_nodes()} сотрудников, {G_people.number_of_edges()} связей |
        <strong>Модулярность:</strong> {m_p.get('modularity',0):.3f} | <strong>Взаимность:</strong> {m_p.get('reciprocity',0):.3f}{lr_t}
    </div>""", unsafe_allow_html=True)

    # ===== ВИЗУАЛИЗАЦИИ =====
    st.markdown("---"); st.header("🗺️ Визуализации")
    tv1, tv2 = st.tabs(["🌐 Иерархическая сеть", "🌀 Force-Directed"])
    with tv1:
        st.markdown("""<div class='info-box'><strong>🌐 Иерархическая сеть</strong><br>
        🖱️ Клик на отдел → люди &nbsp; 🖱️ Double-click → назад &nbsp; 🔍 Scroll → зум</div>""", unsafe_allow_html=True)
        components.html(create_hierarchical_d3_viz(G_depts, G_people, dept_members, m_d, m_p), height=800, scrolling=False)
    with tv2:
        st.markdown("<div class='info-box'><strong>🌀 Force-Directed</strong> — Цвет=сообщество • Размер=влиятельность</div>", unsafe_allow_html=True)
        components.html(create_force_d3_viz(G_people, m_p), height=700, scrolling=False)

    # ===== РОЛИ (кликабельные) =====
    if cfg["show_social_stats"]:
        st.markdown("---"); st.header("🎭 Социальные роли")
        st.markdown("*Нажмите на роль для топа сотрудников*")
        rc = {}
        for r in roles.values(): rc[r] = rc.get(r, 0) + 1
        if "selected_role" not in st.session_state: st.session_state.selected_role = None
        sr = sorted(rc.items(), key=lambda x: -x[1])
        nc = min(len(sr), 5)
        cols = st.columns(nc)
        for i, (rk, cnt) in enumerate(sr):
            ri = SOCIAL_ROLES.get(rk, {"name": rk, "icon": "❓", "color": "#888"})
            with cols[i % nc]:
                if st.button(f"{ri['icon']} {ri['name']} ({cnt})", key=f"rb_{rk}", use_container_width=True):
                    st.session_state.selected_role = None if st.session_state.selected_role == rk else rk
                    st.rerun()

        if st.session_state.selected_role:
            sel = st.session_state.selected_role
            ri = SOCIAL_ROLES.get(sel, {"name": sel, "icon": "❓"})
            st.markdown(f"### {ri['icon']} Топ: {ri['name']}")
            rd = []
            for n, rk in roles.items():
                if rk == sel:
                    nd = G_people.nodes[n]
                    rd.append({"ФИО": nd.get("label",""), "Отдел": nd.get("dept",""),
                               "Компания": nd.get("company",""),
                               "Влиятельность": round(hr['influence_index'].get(n,0), 3),
                               "GF": round(hr['gf'].get(n,0), 2), "VU": round(hr['vu'].get(n,0), 2),
                               "SI": round(hr['si'].get(n,0), 2), "CII": round(hr['cii'].get(n,0), 2),
                               "Кросс-ф.": round(m_p['dept_diversity'].get(n,0), 3),
                               "Посредн.": round(m_p['betweenness'].get(n,0), 4),
                               "Получено": round(m_p['in_strength'].get(n,0), 1),
                               "Отправлено": round(m_p['out_strength'].get(n,0), 1)})
            if rd: st.dataframe(pd.DataFrame(rd).sort_values("Влиятельность", ascending=False), use_container_width=True, hide_index=True)
            else: st.info("Нет сотрудников")
        else:
            with st.expander("📋 Все роли", expanded=False):
                ad = []
                for n, rk in roles.items():
                    ri = SOCIAL_ROLES.get(rk, {"name": rk, "icon": "❓"})
                    nd = G_people.nodes[n]
                    ad.append({"ФИО": nd.get("label",""), "Отдел": nd.get("dept",""),
                               "Компания": nd.get("company",""),
                               "Роль": f"{ri['icon']} {ri['name']}",
                               "GF": round(hr['gf'].get(n,0), 2), "VU": round(hr['vu'].get(n,0), 2),
                               "CII": round(hr['cii'].get(n,0), 2)})
                st.dataframe(pd.DataFrame(ad).sort_values("GF", ascending=False), use_container_width=True, hide_index=True, height=400)

        # ===== ТОПЫ =====
        st.markdown("---"); st.header("🏆 Топы значимости")
        t1, t2, t3, t4 = st.tabs(["👑 Влиятельность", "🌉 Посредничество", "🔗 Кросс-функц.", "🤝 Поддержка"])

        with t1:
            st.markdown("**Влиятельность** — признание + широта + устойчивость + стаж")
            td = [{"ФИО": G_people.nodes[n].get("label",""), "Отдел": G_people.nodes[n].get("dept",""),
                   "Компания": G_people.nodes[n].get("company",""),
                   "Индекс": round(hr['influence_index'].get(n,0), 3),
                   "GF": round(hr['gf'].get(n,0), 2), "SI": round(hr['si'].get(n,0), 2),
                   "CII": round(hr['cii'].get(n,0), 2)} for n in G_people.nodes()]
            st.dataframe(pd.DataFrame(td).sort_values("Индекс", ascending=False).head(20), use_container_width=True, hide_index=True)

        with t2:
            st.markdown("**Посредничество** — мосты и брокеры")
            td = []
            for n in G_people.nodes():
                bw = m_p['betweenness'].get(n,0); br = m_p['is_bridge'].get(n,0)
                cn = m_p['constraint'].get(n,1); dd = m_p['dept_diversity'].get(n,0)
                st_type = "🌉 Мост" if br==1 else ("🔀 Брокер" if cn<0.4 and dd>0.3 else ("↔️ Посредник" if bw>0.01 else "—"))
                td.append({"ФИО": G_people.nodes[n].get("label",""), "Отдел": G_people.nodes[n].get("dept",""),
                           "Компания": G_people.nodes[n].get("company",""),
                           "Посредн.": round(bw,4), "Тип": st_type, "Кросс-ф.": round(dd,2),
                           "Мост": "да" if br==1 else "—"})
            st.dataframe(pd.DataFrame(td).sort_values("Посредн.", ascending=False).head(20), use_container_width=True, hide_index=True)

        with t3:
            st.markdown("**Кросс-функциональность** — разнообразие связей")
            td = [{"ФИО": G_people.nodes[n].get("label",""), "Отдел": G_people.nodes[n].get("dept",""),
                   "Компания": G_people.nodes[n].get("company",""),
                   "Кросс-ф.": round(m_p['dept_diversity'].get(n,0), 3),
                   "IDD": round(hr['idd'].get(n,0), 3), "CII": round(hr['cii'].get(n,0), 2),
                   "Отправлено": round(m_p['out_strength'].get(n,0), 1)} for n in G_people.nodes()]
            st.dataframe(pd.DataFrame(td).sort_values("Кросс-ф.", ascending=False).head(20), use_container_width=True, hide_index=True)

        with t4:
            st.markdown("**Поддержка** — наставничество и надёжное плечо")
            sv = {"Наставничество","Надёжное плечо","наставничество","надёжное плечо","надежное плечо","Надежное плечо"}
            ds = df_f[df_f[C["value"]].isin(sv)]
            if len(ds) > 0:
                sr2 = ds.groupby(C["r_id"])[C["merits"]].sum()
                tr2 = df_f.groupby(C["r_id"])[C["merits"]].sum()
                td = []
                for n in G_people.nodes():
                    s = sr2.get(n,0); t = tr2.get(n,0)
                    if s > 0:
                        td.append({"ФИО": G_people.nodes[n].get("label",""), "Отдел": G_people.nodes[n].get("dept",""),
                                   "Компания": G_people.nodes[n].get("company",""),
                                   "Поддержка": int(s), "MSI": round(s/t if t>0 else 0, 2), "Всего": int(t)})
                if td: st.dataframe(pd.DataFrame(td).sort_values("Поддержка", ascending=False).head(20), use_container_width=True, hide_index=True)
                else: st.info("Нет данных")
            else: st.info("Нет благодарностей за поддержку")

        # ===== СТАТИСТИКА =====
        st.markdown("---"); st.header("📊 Продвинутая статистика")
        ca, cb = st.columns(2)
        with ca:
            st.subheader("🎯 Метрики")
            st.markdown("""<div class='metric-card'>
            <strong>Влиятельность</strong> — признание+широта+устойчивость+стаж<br>
            <strong>Посредничество</strong> — соединение групп<br>
            <strong>Доступность</strong> — скорость информации<br>
            <strong>Плотность окружения</strong> — связанность соседей<br>
            <strong>GF</strong> — мериты/среднее<br>
            <strong>SI</strong> — доля активных месяцев<br>
            <strong>CII</strong> — внешние связи<br>
            <strong>CI</strong> — концентрация источников<br>
            <strong>Кросс-функц.</strong> — разнообразие отделов
            </div>""", unsafe_allow_html=True)
        with cb:
            st.subheader("📈 Сеть")
            nl = list(G_people.nodes())
            nn = G_people.number_of_nodes(); ne = G_people.number_of_edges()
            dens = ne / (nn * (nn-1)) if nn > 1 else 0
            iv = [m_p["in_strength"].get(n,0) for n in nl]; ov = [m_p["out_strength"].get(n,0) for n in nl]
            er, es = calculate_evenness(iv), calculate_evenness(ov)
            st.markdown(f"""<div class='metric-card'>
            <strong>Плотность:</strong> {dens:.4f}<br>
            <strong>Равномерн. получения:</strong> {er:.3f} {'✅' if er>=.6 else '⚠️' if er>=.4 else '🔴'}<br>
            <strong>Равномерн. отправки:</strong> {es:.3f} {'✅' if es>=.6 else '⚠️' if es>=.4 else '🔴'}<br>
            <strong>Плотн. окруж. (ср.):</strong> {np.mean([m_p["clustering"].get(n,0) for n in nl]):.3f}<br>
            <strong>Мостов:</strong> {sum(1 for n in nl if m_p["is_bridge"].get(n,0)==1)}<br>
            <strong>K-core (макс.):</strong> {max([m_p["core_number"].get(n,0) for n in nl]) if nl else 0}
            </div>""", unsafe_allow_html=True)

        # Отделы
        st.markdown("### 🏢 Отделы")
        dst = []
        for dept, members in dept_members.items():
            if not members: continue
            snd = set(m for m in members if m_p['out_strength'].get(m,0) > 0)
            er2 = len(snd)/len(members) if members else 0
            ci2 = np.mean([hr['cii'].get(m,0) for m in members])
            di = [m_p['in_strength'].get(m,0) for m in members]
            evr2 = calculate_evenness(di)
            te2 = 0; ep = {}
            for m in members:
                ec = sum(1 for nb in G_people.neighbors(m) if G_people.nodes[nb].get('dept','')!=dept)
                ep[m] = ec; te2 += ec
            bdi = sum(sorted(ep.values(), reverse=True)[:2])/te2 if te2>0 else 0
            dst.append({"Отдел": dept, "Чел.": len(members), "ER": round(er2,2), "CII": round(ci2,2),
                        "EvR": round(evr2,2), "BDI": round(bdi,2),
                        "Вх.": round(m_d['in_strength'].get(dept,0),1),
                        "Исх.": round(m_d['out_strength'].get(dept,0),1)})
        st.dataframe(pd.DataFrame(dst).sort_values("Чел.", ascending=False), use_container_width=True, hide_index=True)

        # Полная таблица
        st.markdown("### 📋 Полная таблица")
        fm = []
        for n in G_people.nodes():
            nd = G_people.nodes[n]; rk = roles.get(n,"")
            ri = SOCIAL_ROLES.get(rk, {"name":"—","icon":""})
            fm.append({"ФИО": nd.get("label",""), "Отдел": nd.get("dept",""), "Компания": nd.get("company",""),
                "Роль": f"{ri['icon']} {ri['name']}",
                "Влият.": round(hr['influence_index'].get(n,0), 3),
                "GF": round(hr['gf'].get(n,0), 2), "VU": round(hr['vu'].get(n,0), 2),
                "SI": round(hr['si'].get(n,0), 2), "CII": round(hr['cii'].get(n,0), 2),
                "CI": round(hr['ci'].get(n,0), 2),
                "Кросс-ф.": round(m_p['dept_diversity'].get(n,0), 3),
                "Посредн.": round(m_p['betweenness'].get(n,0), 4),
                "Доступн.": round(m_p['closeness'].get(n,0), 3),
                "Плотн.": round(m_p['clustering'].get(n,0), 3),
                "K-core": m_p['core_number'].get(n,0),
                "Мост": "да" if m_p['is_bridge'].get(n,0)==1 else "",
                "Получ.": round(m_p['in_strength'].get(n,0), 1),
                "Отправл.": round(m_p['out_strength'].get(n,0), 1)})
        df_full = pd.DataFrame(fm).sort_values("Влият.", ascending=False)
        st.dataframe(df_full, use_container_width=True, hide_index=True, height=400)

    # ===== ЭКСПОРТ =====
    st.markdown("---"); st.subheader("💾 Экспорт")
    e1, e2 = st.columns(2)
    with e1:
        if cfg["show_social_stats"]:
            st.download_button("📥 Метрики (CSV)", df_full.to_csv(index=False).encode('utf-8-sig'),
                               "sociograph_v8.csv", "text/csv")
    with e2:
        gd = {"nodes": [{"id":str(n),"label":G_people.nodes[n].get("label",""),
                          "dept":G_people.nodes[n].get("dept",""), "company":G_people.nodes[n].get("company",""),
                          "influence":float(hr.get('influence_index',{}).get(n,0)),
                          "role":SOCIAL_ROLES.get(roles.get(n,""),{}).get("name",""),
                          "community":int(m_p.get("communities",{}).get(n,0))} for n in G_people.nodes()],
              "edges": [{"source":str(u),"target":str(v),"weight":float(d.get("weight",1))} for u,v,d in G_people.edges(data=True)],
              "stats": {"modularity":float(m_p.get("modularity",0)),"reciprocity":float(m_p.get("reciprocity",0)),
                        "n_communities":len(set(m_p.get("communities",{}).values()))}}
        st.download_button("📥 Граф (JSON)", json.dumps(gd, indent=2, ensure_ascii=False), "graph_v8.json", "application/json")


if __name__ == "__main__":
    main()