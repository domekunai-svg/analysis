# -*- coding: utf-8 -*-
"""
🕸️ СоциоГраф 6.0
==========================================================
Обновления V6:
- HR-метрики: GF, VU, SI, CII, CI, EvR, SAR, EI
- Составной Индекс влиятельности (вместо чистого PageRank)
- 4 HR-топа: Влиятельность, Посредничество, Кросс-функциональность, Поддержка
- 9 социальных ролей с HR-названиями
- Блок алертов: отрицательная динамика, тени, хрупкий мост
- Убраны Load, Triadic, Eigenvector (дублирование)
- HR-понятные названия метрик

Запуск: streamlit run streamlit_app_v6.py
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

# Стили
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
    .alert-red {
        background: rgba(231, 76, 60, 0.15);
        border-left: 4px solid #e74c3c;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        color: #ff6b6b;
    }
    .alert-yellow {
        background: rgba(243, 156, 18, 0.15);
        border-left: 4px solid #f39c12;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        color: #feca57;
    }
    .alert-green {
        background: rgba(39, 174, 96, 0.15);
        border-left: 4px solid #27ae60;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        color: #6bff9e;
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

COLS = {
    "date": "Дата", "time": "Вермя",
    "sender": "ФИО Отправителя", "sender_id": "№ Отправителя",
    "sender_role": "Должномть Отправителя", "sender_dept": "Отдел Отправителя",
    "receiver": "ФИО Получателя", "receiver_id": "№ Получателя",
    "receiver_role": "Должномть Получателя", "receiver_dept": "Отдел Получателя",
    "value": "Ценность", "merits": "Мериты (сила)", "comment": "Комментарий",
}

# HR-названия метрик (внутреннее имя -> отображаемое)
HR_NAMES = {
    "influence_index": "Индекс влиятельности",
    "gf": "Коэфф. признания",
    "vu": "Коэфф. использования голосов",
    "si": "Коэфф. устойчивости",
    "cii": "Коэфф. интеграции",
    "ci": "Коэфф. концентрации источников",
    "sar": "Коэфф. соц. активности",
    "dept_div": "Коэфф. кросс-функциональности",
    "idd": "Коэфф. кросс-функц. доверия",
    "evr_recv": "Равномерность получения",
    "evr_send": "Равномерность отправки",
    "betweenness_norm": "Индекс посредничества",
    "closeness": "Индекс доступности",
    "clustering": "Плотность окружения",
    "k_core": "Глубина интеграции",
    "pagerank": "PageRank (скрытый)",
    "in_strength": "Получено меритов",
    "out_strength": "Отправлено меритов",
}

# Социальные роли
SOCIAL_ROLES = {
    "leader_integrator": {"name": "Лидер-интегратор", "color": "#FFD700", "icon": "👑"},
    "internal_leader": {"name": "Внутренний лидер", "color": "#FF8C00", "icon": "🏆"},
    "connector": {"name": "Связующее звено", "color": "#00CED1", "icon": "🔗"},
    "strategic_broker": {"name": "Стратегический посредник", "color": "#9370DB", "icon": "🌉"},
    "network_builder": {"name": "Строитель связей", "color": "#32CD32", "icon": "🏗️"},
    "quiet_engine": {"name": "Тихий двигатель", "color": "#87CEEB", "icon": "⚙️"},
    "unrecognized_ambassador": {"name": "Посол без ответа", "color": "#DDA0DD", "icon": "📡"},
    "inner_focus": {"name": "Внутренний фокус", "color": "#A9A9A9", "icon": "🔒"},
    "quiet_presence": {"name": "Тихое участие", "color": "#696969", "icon": "🌫️"},
}


@st.cache_data(show_spinner=False)
def load_df(path_or_file):
    df = pd.read_excel(path_or_file, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for key, name in COLS.items():
        for c in df.columns:
            if c.lower() == name.lower():
                rename_map[c] = name
    if rename_map:
        df = df.rename(columns=rename_map)

    def parse_dt(row):
        d = row.get(COLS["date"], None)
        t = row.get(COLS["time"], "00:00:00")
        if pd.isna(d):
            return pd.NaT
        ds = str(d)
        ts = str(t) if not pd.isna(t) else "00:00:00"
        return pd.to_datetime(f"{ds} {ts}", dayfirst=True, errors="coerce")

    df["dt"] = df.apply(parse_dt, axis=1)

    if COLS["merits"] in df.columns:
        df[COLS["merits"]] = pd.to_numeric(df[COLS["merits"]], errors="coerce").fillna(0).astype(int)
    else:
        df[COLS["merits"]] = 1

    for c in [COLS["sender"], COLS["receiver"], COLS["sender_dept"],
              COLS["receiver_dept"], COLS["value"], COLS["comment"]]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

# ========================= ПОСТРОЕНИЕ ГРАФОВ =========================

def build_hierarchical_graph(df: pd.DataFrame, merit_range: tuple = (1, 100), allow_self: bool = False):
    """Построение иерархического графа (отделы + люди) с фильтром по диапазону меритов"""
    if not allow_self:
        df = df[df[COLS["sender_id"]] != df[COLS["receiver_id"]]].copy()

    person_agg = (
        df.groupby([
            COLS["sender_id"], COLS["receiver_id"],
            COLS["sender"], COLS["receiver"],
            COLS["sender_dept"], COLS["receiver_dept"]
        ], dropna=False)
        .agg(total_merits=(COLS["merits"], "sum"), n_msgs=("dt", "count"))
        .reset_index()
    )

    min_merit, max_merit = merit_range
    person_agg = person_agg[
        (person_agg["total_merits"] >= min_merit) &
        (person_agg["total_merits"] <= max_merit)
    ].copy()

    G_people = nx.DiGraph()
    for _, row in person_agg.iterrows():
        sid, rid = row[COLS["sender_id"]], row[COLS["receiver_id"]]
        sname, rname = row[COLS["sender"]], row[COLS["receiver"]]
        sdept, rdept = row[COLS["sender_dept"]], row[COLS["receiver_dept"]]

        if sid not in G_people:
            G_people.add_node(sid, label=sname, dept=sdept, type="person")
        if rid not in G_people:
            G_people.add_node(rid, label=rname, dept=rdept, type="person")

        w = float(row["total_merits"])
        length = 1.0 / max(w, 0.01)
        G_people.add_edge(sid, rid, weight=w, length=length, msgs=int(row["n_msgs"]))

    dept_agg = (
        person_agg.groupby([COLS["sender_dept"], COLS["receiver_dept"]])
        .agg(total_merits=("total_merits", "sum"), n_people=("total_merits", "count"))
        .reset_index()
    )

    G_depts = nx.DiGraph()
    dept_members = {}
    for node in G_people.nodes():
        dept = G_people.nodes[node].get("dept", "")
        if dept not in dept_members:
            dept_members[dept] = []
        dept_members[dept].append(node)

    for dept, members in dept_members.items():
        G_depts.add_node(dept, label=dept, type="dept", size=len(members), members=members)

    for _, row in dept_agg.iterrows():
        s_dept = row[COLS["sender_dept"]]
        r_dept = row[COLS["receiver_dept"]]
        if s_dept != r_dept:
            w = float(row["total_merits"])
            G_depts.add_edge(s_dept, r_dept, weight=w, people=int(row["n_people"]))

    return G_people, G_depts, dept_members


# ========================= ГРАФОВЫЕ МЕТРИКИ =========================

def calculate_graph_metrics(G: nx.DiGraph):
    """Расчет графовых метрик (оставляем прагматичные, убираем дубли)"""
    if G.number_of_nodes() == 0:
        return {}

    metrics = {}

    # Базовые
    metrics['in_strength'] = dict(G.in_degree(weight="weight"))
    metrics['out_strength'] = dict(G.out_degree(weight="weight"))

    try:
        metrics['pagerank'] = nx.pagerank(G, weight="weight", max_iter=100)
    except Exception:
        metrics['pagerank'] = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

    UG = G.to_undirected()

    # Betweenness — посредничество
    try:
        metrics['betweenness'] = nx.betweenness_centrality(UG, weight='length', normalized=True)
    except Exception:
        metrics['betweenness'] = {n: 0.0 for n in G.nodes()}

    # Closeness — доступность
    try:
        metrics['closeness'] = nx.closeness_centrality(UG, distance='length')
    except Exception:
        metrics['closeness'] = {n: 0.0 for n in G.nodes()}

    # Clustering — плотность окружения
    try:
        metrics['clustering'] = nx.clustering(UG, weight='weight')
    except Exception:
        metrics['clustering'] = {n: 0.0 for n in G.nodes()}

    # Constraint (Burt's structural holes)
    try:
        metrics['constraint'] = nx.constraint(UG, weight='weight')
    except Exception:
        metrics['constraint'] = {n: 0.0 for n in G.nodes()}

    # K-core — глубина интеграции
    try:
        metrics['core_number'] = nx.core_number(UG)
    except Exception:
        metrics['core_number'] = {n: 0 for n in G.nodes()}

    # Bridge — мост (бинарный)
    try:
        bridges = list(nx.bridges(UG))
        bridge_nodes = set()
        for u, v in bridges:
            bridge_nodes.add(u)
            bridge_nodes.add(v)
        metrics['is_bridge'] = {n: 1 if n in bridge_nodes else 0 for n in G.nodes()}
    except Exception:
        metrics['is_bridge'] = {n: 0 for n in G.nodes()}

    # DeptDiv — кросс-функциональность (общая)
    dept_diversity = {}
    for node in G.nodes():
        neighbors = set(G.neighbors(node)) | set(G.predecessors(node))
        if len(neighbors) == 0:
            dept_diversity[node] = 0.0
        else:
            depts = set()
            for n in neighbors:
                dept = G.nodes[n].get('dept', '')
                if dept:
                    depts.add(dept)
            dept_diversity[node] = len(depts) / max(len(neighbors), 1)
    metrics['dept_diversity'] = dept_diversity

    # Communities (Louvain)
    try:
        part = community_louvain.best_partition(UG, weight="weight")
        mod = community_louvain.modularity(part, UG, weight="weight")
        metrics['communities'] = part
        metrics['modularity'] = mod
    except Exception:
        metrics['communities'] = {n: 0 for n in G.nodes()}
        metrics['modularity'] = 0.0

    metrics['reciprocity'] = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0

    return metrics


# ========================= HR-МЕТРИКИ =========================

def calculate_hr_metrics(G: nx.DiGraph, df: pd.DataFrame, graph_metrics: dict,
                         merits_per_month: int = 10, total_employees: int = 0):
    """Расчёт HR-метрик поверх графовых"""

    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}

    hr = {}
    in_str = graph_metrics.get('in_strength', {})
    out_str = graph_metrics.get('out_strength', {})

    # --- GF (Gratitude Factor) ---
    all_received = [in_str.get(n, 0) for n in nodes]
    avg_received = np.mean(all_received) if len(all_received) > 0 and np.mean(all_received) > 0 else 1.0
    hr['gf'] = {n: in_str.get(n, 0) / avg_received for n in nodes}

    # --- SI (Stability Index) ---
    # Доля активных месяцев (отправка)
    df_copy = df.copy()
    df_copy['_month'] = df_copy['dt'].dt.to_period('M')
    total_months = df_copy['_month'].nunique()
    if total_months == 0:
        total_months = 1

    sender_months = df_copy.groupby(COLS["sender_id"])['_month'].nunique().to_dict()
    hr['si'] = {n: sender_months.get(n, 0) / total_months for n in nodes}

    # --- CII (Corporate Integration Index) ---
    cii = {}
    for node in nodes:
        node_dept = G.nodes[node].get('dept', '')
        successors = list(G.neighbors(node))
        if len(successors) == 0:
            cii[node] = 0.0
        else:
            external = sum(1 for s in successors if G.nodes[s].get('dept', '') != node_dept)
            cii[node] = external / len(successors)
    hr['cii'] = cii

    # --- CI (Concentration Index) ---
    ci = {}
    for node in nodes:
        predecessors = list(G.predecessors(node))
        if len(predecessors) == 0:
            ci[node] = 0.0
        else:
            incoming_weights = [(p, G[p][node].get('weight', 0)) for p in predecessors]
            incoming_weights.sort(key=lambda x: x[1], reverse=True)
            top3_sum = sum(w for _, w in incoming_weights[:3])
            total_in = in_str.get(node, 0)
            ci[node] = top3_sum / total_in if total_in > 0 else 0.0
    hr['ci'] = ci

    # --- SAR (Social Activity Rate) ---
    hr['sar'] = {n: (in_str.get(n, 0) + out_str.get(n, 0)) / 10.0 for n in nodes}

    # --- VU (Votes Used) ---
    if merits_per_month > 0 and total_months > 0:
        available = merits_per_month * total_months
        hr['vu'] = {n: min(out_str.get(n, 0) / available, 1.0) if available > 0 else 0.0 for n in nodes}
    else:
        hr['vu'] = {n: 0.0 for n in nodes}

    # --- IDD (In-Degree Diversity) - от скольких отделов получает ---
    total_depts = len(set(G.nodes[n].get('dept', '') for n in nodes))
    if total_depts == 0:
        total_depts = 1
    idd = {}
    for node in nodes:
        predecessors = list(G.predecessors(node))
        if len(predecessors) == 0:
            idd[node] = 0.0
        else:
            depts_in = set(G.nodes[p].get('dept', '') for p in predecessors)
            idd[node] = len(depts_in) / total_depts
    hr['idd'] = idd

    # --- Influence Index (составной) ---
    # Нормализуем компоненты к [0, 1]
    def normalize_dict(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx - mn > 0 else 1.0
        return {k: (v - mn) / rng for k, v in d.items()}

    pr_norm = normalize_dict(graph_metrics.get('pagerank', {n: 0 for n in nodes}))
    idd_norm = normalize_dict(hr['idd'])
    cii_norm = normalize_dict(hr['cii'])
    si_norm = normalize_dict(hr['si'])
    gf_norm = normalize_dict(hr['gf'])

    # Tenure — месяцы с первой благодарности
    first_activity = df_copy.groupby(COLS["sender_id"])['_month'].min().to_dict()
    last_month = df_copy['_month'].max()
    tenure = {}
    for n in nodes:
        fm = first_activity.get(n, last_month)
        if pd.isna(fm):
            tenure[n] = 0.0
        else:
            months_active = (last_month - fm).n if hasattr(last_month - fm, 'n') else 0
            tenure[n] = max(months_active, 0)
    max_tenure = max(tenure.values()) if tenure and max(tenure.values()) > 0 else 1.0
    tenure_norm = {k: v / max_tenure for k, v in tenure.items()}

    hr['influence_index'] = {}
    for n in nodes:
        hr['influence_index'][n] = (
            pr_norm.get(n, 0) * 0.25 +
            idd_norm.get(n, 0) * 0.20 +
            cii_norm.get(n, 0) * 0.15 +
            si_norm.get(n, 0) * 0.15 +
            tenure_norm.get(n, 0) * 0.10 +
            gf_norm.get(n, 0) * 0.15
        )

    return hr


# ========================= СОЦИАЛЬНЫЕ РОЛИ =========================

def assign_social_roles(G, graph_metrics, hr_metrics):
    """Определение социальных ролей по комбинации показателей"""
    roles = {}
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return roles

    gf = hr_metrics.get('gf', {})
    vu = hr_metrics.get('vu', {})
    cii = hr_metrics.get('cii', {})
    betw = graph_metrics.get('betweenness', {})
    dept_div = graph_metrics.get('dept_diversity', {})
    bridge = graph_metrics.get('is_bridge', {})
    constraint = graph_metrics.get('constraint', {})

    # Пороги (медианы)
    gf_vals = [gf.get(n, 0) for n in nodes]
    vu_vals = [vu.get(n, 0) for n in nodes]
    cii_vals = [cii.get(n, 0) for n in nodes]
    betw_vals = [betw.get(n, 0) for n in nodes]

    gf_med = np.median(gf_vals) if gf_vals else 0.5
    vu_med = np.median(vu_vals) if vu_vals else 0.1
    cii_med = np.median(cii_vals) if cii_vals else 0.2
    betw_p75 = np.percentile(betw_vals, 75) if betw_vals else 0.05

    for n in nodes:
        g = gf.get(n, 0)
        v = vu.get(n, 0)
        c = cii.get(n, 0)
        b = betw.get(n, 0)
        dd = dept_div.get(n, 0)
        br = bridge.get(n, 0)
        con = constraint.get(n, 1.0)

        g_high = g > gf_med * 1.3
        g_low = g < gf_med * 0.5
        v_high = v > vu_med * 1.3 or v > 0.15
        v_low = v < vu_med * 0.5 or v < 0.05
        c_high = c > cii_med * 1.3 or c > 0.3
        c_zero = c < 0.02
        b_high = b > betw_p75

        # Порядок проверки: от специфичных к общим
        if b_high and dd > 0.3 and (br == 1 or con < 0.4):
            roles[n] = "strategic_broker"
        elif g_high and v_high and c_high:
            roles[n] = "leader_integrator"
        elif g_high and v_high and not c_high:
            roles[n] = "internal_leader"
        elif not g_high and not g_low and c_high:
            roles[n] = "connector"
        elif con < 0.35 and dd > 0.4:
            roles[n] = "network_builder"
        elif g_low and v_high and c_high:
            roles[n] = "unrecognized_ambassador"
        elif g_low and v_high and not c_high:
            roles[n] = "quiet_engine"
        elif c_zero and not g_low:
            roles[n] = "inner_focus"
        elif g_low and v_low:
            roles[n] = "quiet_presence"
        else:
            roles[n] = "connector"  # default

    return roles


# ========================= АЛЕРТЫ =========================

def calculate_alerts(G, df, graph_metrics, hr_metrics, dept_members):
    """Расчёт HR-алертов"""
    alerts = {"critical": [], "warning": [], "positive": []}

    gf = hr_metrics.get('gf', {})
    vu = hr_metrics.get('vu', {})
    si = hr_metrics.get('si', {})
    cii = hr_metrics.get('cii', {})

    # --- Алерт: Тени (>60 дней без получения) ---
    df_copy = df.copy()
    last_date = df_copy['dt'].max()
    last_received = df_copy.groupby(COLS["receiver_id"])['dt'].max().to_dict()

    for node in G.nodes():
        lr = last_received.get(node, pd.NaT)
        label = G.nodes[node].get('label', str(node))
        if pd.isna(lr):
            alerts["critical"].append(
                f"🔴 «Тень»: {label} — ни одной благодарности за весь период"
            )
        elif (last_date - lr).days > 60:
            days = (last_date - lr).days
            alerts["warning"].append(
                f"⚠️ «Тень»: {label} — {days} дней без благодарности"
            )

    # --- Алерт: Неучастие (VU < 0.10 И SI < 0.15) ---
    for node in G.nodes():
        label = G.nodes[node].get('label', str(node))
        if vu.get(node, 0) < 0.10 and si.get(node, 0) < 0.15:
            alerts["warning"].append(
                f"⚠️ «Неучастие»: {label} — практически не отправляет благодарности"
            )

    # --- Алерт: Хрупкий мост (BDI > 0.70 по отделу) ---
    for dept, members in dept_members.items():
        if len(members) < 3:
            continue
        external_links = {}
        total_external = 0
        for m in members:
            ext_count = 0
            for neighbor in G.neighbors(m):
                if G.nodes[neighbor].get('dept', '') != dept:
                    ext_count += 1
            external_links[m] = ext_count
            total_external += ext_count

        if total_external > 0:
            top2 = sorted(external_links.values(), reverse=True)[:2]
            bdi = sum(top2) / total_external
            if bdi > 0.70:
                alerts["warning"].append(
                    f"⚠️ «Хрупкий мост»: отдел «{dept}» — {bdi:.0%} внешних связей через 1-2 человек"
                )

    # --- Позитивные ---
    top_gf = sorted(gf.items(), key=lambda x: x[1], reverse=True)[:3]
    for node, val in top_gf:
        label = G.nodes[node].get('label', str(node))
        if val > 1.5:
            alerts["positive"].append(
                f"✅ {label} — коэфф. признания {val:.2f} (значительно выше среднего)"
            )

    return alerts


# ========================= EvR (РАВНОМЕРНОСТЬ) =========================

def calculate_evenness(values_list):
    """Коэффициент Джини -> EvR = 1 - Gini"""
    arr = np.array(sorted(values_list))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n
    return max(0.0, min(1.0, 1.0 - gini))

# ========================= ИЕРАРХИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ (D3) =========================

def create_hierarchical_d3_viz(G_depts, G_people, dept_members, metrics_depts, metrics_people):
    """D3.js иерархическая визуализация — без изменений каркаса"""
    dept_nodes = []
    for node in G_depts.nodes():
        nd = G_depts.nodes[node]
        dept_nodes.append({
            "id": f"dept_{node}", "original_id": node,
            "label": nd.get("label", str(node)), "type": "dept",
            "size": nd.get("size", 1), "members": nd.get("members", []),
            "in_strength": metrics_depts.get("in_strength", {}).get(node, 0),
            "out_strength": metrics_depts.get("out_strength", {}).get(node, 0),
        })
    dept_edges = []
    for u, v, data in G_depts.edges(data=True):
        dept_edges.append({"source": f"dept_{u}", "target": f"dept_{v}",
                           "weight": data.get("weight", 1), "people": data.get("people", 0)})

    people_nodes = []
    for node in G_people.nodes():
        nd = G_people.nodes[node]
        people_nodes.append({
            "id": f"person_{node}", "original_id": node,
            "label": nd.get("label", str(node)), "dept": nd.get("dept", ""), "type": "person",
            "in_strength": metrics_people.get("in_strength", {}).get(node, 0),
            "out_strength": metrics_people.get("out_strength", {}).get(node, 0),
            "pagerank": metrics_people.get("pagerank", {}).get(node, 0),
        })
    people_edges = []
    for u, v, data in G_people.edges(data=True):
        people_edges.append({"source": f"person_{u}", "target": f"person_{v}",
                             "weight": data.get("weight", 1), "msgs": data.get("msgs", 0)})

    html = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin:0; padding:0; background:#0a0e27; font-family:'Segoe UI',Tahoma,sans-serif; overflow:hidden; }}
        #viz {{ width:100%; height:100vh; }}
        .controls {{ position:absolute; top:10px; right:10px; z-index:1000; }}
        .btn {{ background:linear-gradient(90deg,#00d4ff,#7b2cbf); color:white; border:none;
                padding:8px 15px; margin:2px; border-radius:5px; cursor:pointer; font-weight:600; font-size:12px; }}
        .btn:hover {{ opacity:0.8; }}
        .node {{ cursor:pointer; stroke:#fff; stroke-width:2px; }}
        .node.dept {{ fill:#7b2cbf; }}
        .node.person {{ fill:#00d4ff; }}
        .link {{ stroke:#999; stroke-opacity:0.4; }}
        .label {{ fill:white; font-size:11px; pointer-events:none; text-anchor:middle; text-shadow:0 0 3px #000; }}
        #breadcrumb {{ position:absolute; top:10px; left:10px; color:#00d4ff; font-size:16px;
                       font-weight:bold; text-shadow:0 0 10px rgba(0,212,255,0.8); }}
        #info {{ position:absolute; bottom:10px; left:10px; color:white; font-size:12px;
                 background:rgba(0,0,0,0.7); padding:10px; border-radius:5px; max-width:300px; }}
    </style></head><body>
    <div id="breadcrumb">Уровень: Отделы</div>
    <div id="info">Загрузка...</div>
    <div class="controls">
        <button class="btn" onclick="resetView()">🏠 Домой</button>
        <button class="btn" onclick="resetZoom()">🔍 Сбросить зум</button>
        <button class="btn" onclick="toggleLabels()">🏷️ Метки</button>
        <button class="btn" onclick="togglePhysics()">⚡ Физика</button>
    </div>
    <svg id="viz"></svg>
    <script>
        const width=window.innerWidth, height=window.innerHeight;
        const deptNodesData={json.dumps(dept_nodes)};
        const deptLinksData={json.dumps(dept_edges)};
        const peopleNodesData={json.dumps(people_nodes)};
        const peopleLinksData={json.dumps(people_edges)};
        let nodes=[...deptNodesData], links=[...deptLinksData];
        let currentLevel="depts", expandedDept=null;
        const svg=d3.select("#viz").attr("width",width).attr("height",height);
        const g=svg.append("g");
        const zoom=d3.zoom().scaleExtent([0.1,10]).on("zoom",(event)=>{{g.attr("transform",event.transform);}});
        svg.call(zoom);
        let linkElements, nodeElements, labels, simulation;
        function initSimulation(){{
            g.selectAll("*").remove();
            linkElements=g.append("g").selectAll("line").data(links).join("line")
                .attr("class","link").attr("stroke-width",d=>Math.sqrt(d.weight)/2);
            nodeElements=g.append("g").selectAll("circle").data(nodes).join("circle")
                .attr("class",d=>`node ${{d.type}}`)
                .attr("r",d=>{{ if(d.type==="dept") return Math.sqrt(d.size)*5+10; return 6; }})
                .on("click",handleNodeClick).on("dblclick",handleNodeDoubleClick)
                .on("mouseover",showNodeInfo)
                .call(d3.drag().on("start",dragstarted).on("drag",dragged).on("end",dragended));
            labels=g.append("g").selectAll("text").data(nodes).join("text")
                .attr("class","label").attr("dy",-10)
                .text(d=>d.label.length>20?d.label.slice(0,20)+"...":d.label);
            simulation=d3.forceSimulation(nodes)
                .force("link",d3.forceLink(links).id(d=>d.id).distance(d=>currentLevel==="depts"?150:80))
                .force("charge",d3.forceManyBody().strength(-300))
                .force("center",d3.forceCenter(width/2,height/2))
                .force("collision",d3.forceCollide().radius(d=>{{
                    if(d.type==="dept") return Math.sqrt(d.size)*5+15; return 10;}}))
                .on("tick",ticked);
        }}
        function ticked(){{
            linkElements.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
                .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
            nodeElements.attr("cx",d=>d.x).attr("cy",d=>d.y);
            labels.attr("x",d=>d.x).attr("y",d=>d.y);
        }}
        function dragstarted(e,d){{ if(!e.active)simulation.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y; }}
        function dragged(e,d){{ d.fx=e.x;d.fy=e.y; }}
        function dragended(e,d){{ if(!e.active)simulation.alphaTarget(0);d.fx=null;d.fy=null; }}
        function handleNodeClick(e,d){{ e.stopPropagation(); if(currentLevel==="depts"&&d.type==="dept") expandDept(d); }}
        function handleNodeDoubleClick(e,d){{ e.stopPropagation(); if(currentLevel==="people") collapseToDepts(); }}
        function expandDept(dn){{
            currentLevel="people"; expandedDept=dn.original_id;
            const members=dn.members||[];
            nodes=peopleNodesData.filter(n=>members.includes(n.original_id));
            const mids=new Set(nodes.map(n=>n.id));
            links=peopleLinksData.filter(l=>mids.has(l.source)&&mids.has(l.target));
            document.getElementById("breadcrumb").textContent=`Уровень: ${{dn.label}} (double-click для возврата)`;
            simulation.stop(); initSimulation();
        }}
        function collapseToDepts(){{
            currentLevel="depts"; expandedDept=null;
            nodes=[...deptNodesData]; links=[...deptLinksData];
            document.getElementById("breadcrumb").textContent="Уровень: Отделы";
            simulation.stop(); initSimulation();
        }}
        function resetView(){{ collapseToDepts(); }}
        function resetZoom(){{ svg.transition().duration(750).call(zoom.transform,d3.zoomIdentity); }}
        let labelsVisible=true;
        function toggleLabels(){{ labelsVisible=!labelsVisible; labels.style("opacity",labelsVisible?1:0); }}
        let physicsEnabled=true;
        function togglePhysics(){{ physicsEnabled=!physicsEnabled;
            if(physicsEnabled){{ simulation.alpha(0.3).restart(); }}else{{ simulation.stop(); }} }}
        function showNodeInfo(e,d){{
            let info=`<strong>${{d.label}}</strong><br>`;
            if(d.type==="dept"){{
                info+=`Тип: Отдел<br>Сотрудников: ${{d.size}}<br>Входящие: ${{d.in_strength.toFixed(1)}}<br>Исходящие: ${{d.out_strength.toFixed(1)}}`;
            }}else{{
                info+=`Отдел: ${{d.dept}}<br>PageRank: ${{d.pagerank.toFixed(4)}}<br>Входящие: ${{d.in_strength.toFixed(1)}}<br>Исходящие: ${{d.out_strength.toFixed(1)}}`;
            }}
            document.getElementById("info").innerHTML=info;
        }}
        initSimulation();
    </script></body></html>"""
    return html


def create_force_d3_viz(G, metrics):
    """Force-directed визуализация — без изменений каркаса"""
    nodes_data = []
    for node in G.nodes():
        nd = G.nodes[node]
        comm = metrics.get("communities", {}).get(node, 0)
        nodes_data.append({
            "id": str(node), "label": nd.get("label", str(node)),
            "dept": nd.get("dept", ""), "community": comm,
            "pagerank": metrics.get("pagerank", {}).get(node, 0),
            "in_strength": metrics.get("in_strength", {}).get(node, 0),
            "out_strength": metrics.get("out_strength", {}).get(node, 0),
        })
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({"source": str(u), "target": str(v), "weight": data.get("weight", 1)})
    n_communities = len(set(metrics.get("communities", {}).values()))
    colors = ["#00d4ff","#7b2cbf","#ff006e","#ffbe0b","#8ac926",
              "#ff006e","#3a86ff","#fb5607","#06ffa5","#8338ec"]

    html = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin:0; padding:0; background:#0a0e27; font-family:sans-serif; overflow:hidden; }}
        #viz {{ width:100%; height:100vh; }}
        .node {{ cursor:pointer; stroke:#fff; stroke-width:1.5px; }}
        .link {{ stroke:#999; stroke-opacity:0.3; }}
        .label {{ fill:white; font-size:10px; pointer-events:none; text-anchor:middle; text-shadow:0 0 3px #000; }}
        .controls {{ position:absolute; top:10px; right:10px; z-index:1000; }}
        .btn {{ background:linear-gradient(90deg,#00d4ff,#7b2cbf); color:white; border:none;
                padding:8px 15px; margin:2px; border-radius:5px; cursor:pointer; font-size:12px; }}
        #info {{ position:absolute; bottom:10px; left:10px; color:white; background:rgba(0,0,0,0.7);
                 padding:10px; border-radius:5px; font-size:12px; max-width:300px; }}
    </style></head><body>
    <div class="controls">
        <button class="btn" onclick="resetZoom()">🔍 Сбросить зум</button>
        <button class="btn" onclick="toggleLabels()">🏷️ Метки</button>
        <button class="btn" onclick="togglePhysics()">⚡ Физика</button>
    </div>
    <div id="info">Наведите на узел для информации</div>
    <svg id="viz"></svg>
    <script>
        const width=window.innerWidth, height=window.innerHeight;
        const nodes={json.dumps(nodes_data)};
        const links={json.dumps(edges_data)};
        const colors={json.dumps(colors[:max(n_communities,1)])};
        const svg=d3.select("#viz").attr("width",width).attr("height",height);
        const g=svg.append("g");
        const zoom=d3.zoom().scaleExtent([0.1,10]).on("zoom",(event)=>g.attr("transform",event.transform));
        svg.call(zoom);
        const linkElements=g.append("g").selectAll("line").data(links).join("line")
            .attr("class","link").attr("stroke-width",d=>Math.sqrt(d.weight)/2);
        const nodeElements=g.append("g").selectAll("circle").data(nodes).join("circle")
            .attr("class","node").attr("r",d=>3+Math.sqrt(d.pagerank*1000))
            .attr("fill",d=>colors[d.community%colors.length])
            .on("mouseover",showInfo)
            .call(d3.drag().on("start",dragstarted).on("drag",dragged).on("end",dragended));
        const labels=g.append("g").selectAll("text").data(nodes).join("text")
            .attr("class","label").attr("dy",-8)
            .text(d=>d.label.length>15?d.label.slice(0,15)+"...":d.label);
        const simulation=d3.forceSimulation(nodes)
            .force("link",d3.forceLink(links).id(d=>d.id).distance(70))
            .force("charge",d3.forceManyBody().strength(-200))
            .force("center",d3.forceCenter(width/2,height/2))
            .force("collision",d3.forceCollide().radius(15))
            .on("tick",ticked);
        function ticked(){{
            linkElements.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
                .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
            nodeElements.attr("cx",d=>d.x).attr("cy",d=>d.y);
            labels.attr("x",d=>d.x).attr("y",d=>d.y);
        }}
        function dragstarted(e,d){{ if(!e.active)simulation.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y; }}
        function dragged(e,d){{ d.fx=e.x;d.fy=e.y; }}
        function dragended(e,d){{ if(!e.active)simulation.alphaTarget(0);d.fx=null;d.fy=null; }}
        function showInfo(e,d){{
            document.getElementById("info").innerHTML=
                `<strong>${{d.label}}</strong><br>Отдел: ${{d.dept}}<br>PageRank: ${{d.pagerank.toFixed(4)}}<br>Сообщество: ${{d.community}}<br>Входящие: ${{d.in_strength.toFixed(1)}}<br>Исходящие: ${{d.out_strength.toFixed(1)}}`;
        }}
        function resetZoom(){{ svg.transition().duration(750).call(zoom.transform,d3.zoomIdentity); }}
        let labelsVisible=true;
        function toggleLabels(){{ labelsVisible=!labelsVisible; labels.style("opacity",labelsVisible?1:0); }}
        let physicsEnabled=true;
        function togglePhysics(){{ physicsEnabled=!physicsEnabled;
            if(physicsEnabled){{ simulation.alpha(0.3).restart(); }}else{{ simulation.stop(); }} }}
    </script></body></html>"""
    return html

# ========================= SIDEBAR & FILTERING =========================

def sidebar_controls(df: pd.DataFrame):
    st.sidebar.header("⚙️ Настройки")

    min_dt = pd.to_datetime(df["dt"]).min()
    max_dt = pd.to_datetime(df["dt"]).max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        min_dt = pd.to_datetime("2000-01-01")
        max_dt = pd.to_datetime("2100-01-01")

    period = st.sidebar.date_input(
        "📅 Период",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(), max_value=max_dt.date(),
    )
    if isinstance(period, tuple):
        start_date, end_date = period
    else:
        start_date, end_date = period, period

    st.sidebar.markdown("---")

    values_list = sorted(df[COLS["value"]].dropna().unique().tolist())
    selected_values = st.sidebar.multiselect("Ценности", options=values_list, default=values_list)

    st.sidebar.markdown("### 💎 Мериты на связь")
    max_merits_possible = 1000
    merit_range = st.sidebar.slider(
        "Диапазон меритов", min_value=1, max_value=max_merits_possible,
        value=(1, max_merits_possible), step=1,
        help="Выберите минимальное и максимальное количество меритов для фильтрации связей"
    )
    st.sidebar.markdown(f"*Связи с {merit_range[0]} по {merit_range[1]} меритов*")

    allow_self = st.sidebar.checkbox("Самонаграждения", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Параметры программы")
    merits_per_month = st.sidebar.number_input(
        "Меритов в месяц на сотрудника", min_value=1, max_value=100, value=10,
        help="Лимит меритов, которые сотрудник может отправить в месяц"
    )
    total_employees = st.sidebar.number_input(
        "Всего сотрудников в компании", min_value=0, max_value=10000, value=0,
        help="Общее число сотрудников (для расчёта LR). 0 = не рассчитывать"
    )

    st.sidebar.markdown("---")
    show_social_stats = st.sidebar.checkbox(
        "📊 Показать продвинутую статистику", value=True,
        help="Включает расчет и отображение глубоких социологических метрик"
    )

    return {
        "start": pd.to_datetime(start_date),
        "end": pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        "values": set(selected_values),
        "merit_range": merit_range,
        "allow_self": allow_self,
        "merits_per_month": merits_per_month,
        "total_employees": total_employees,
        "show_social_stats": show_social_stats,
    }


def filter_df(df: pd.DataFrame, cfg):
    m = (df["dt"] >= cfg["start"]) & (df["dt"] <= cfg["end"])
    m &= df[COLS["value"]].isin(cfg["values"])
    return df.loc[m].copy()

# ========================= MAIN =========================

def main():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3rem;'>🕸️ СоциоГраф 6.0</h1>
            <p style='font-size: 1.2rem; color: #00d4ff;'>
                Иерархическая визуализация + HR-аналитика + Социальные роли
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Загрузка данных
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "dataset.xlsx")
    if os.path.exists(local_path):
        df = load_df(local_path)
    else:
        st.error("❌ Встроенный файл dataset.xlsx не найден. Положите dataset.xlsx рядом со скриптом.")
        st.stop()

    cfg = sidebar_controls(df)
    df_filtered = filter_df(df, cfg)

    if len(df_filtered) == 0:
        st.warning("⚠️ Нет данных для выбранных фильтров")
        st.stop()

    # === Верхние метрики ===
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Транзакций", f"{len(df_filtered):,}")
    with col2:
        uniq = pd.Index(df_filtered[COLS["sender_id"]]).append(
            pd.Index(df_filtered[COLS["receiver_id"]])).nunique()
        st.metric("👥 Сотрудников", f"{uniq:,}")
    with col3:
        st.metric("⭐ Меритов", f"{df_filtered[COLS['merits']].sum():,}")
    with col4:
        st.metric("🏢 Отделов", f"{df_filtered[COLS['sender_dept']].nunique():,}")

    # === Построение графов ===
    with st.spinner("🔄 Строим иерархическую структуру..."):
        G_people, G_depts, dept_members = build_hierarchical_graph(
            df_filtered, cfg["merit_range"], cfg["allow_self"]
        )
        if G_depts.number_of_nodes() == 0 or G_people.number_of_nodes() == 0:
            st.warning("⚠️ Граф пуст после применения фильтров")
            st.stop()
        metrics_depts = calculate_graph_metrics(G_depts)
        metrics_people = calculate_graph_metrics(G_people)

    # === HR-метрики ===
    with st.spinner("📐 Рассчитываем HR-метрики..."):
        hr_metrics = calculate_hr_metrics(
            G_people, df_filtered, metrics_people,
            merits_per_month=cfg["merits_per_month"],
            total_employees=cfg["total_employees"]
        )
        social_roles = assign_social_roles(G_people, metrics_people, hr_metrics)
        alerts = calculate_alerts(G_people, df_filtered, metrics_people, hr_metrics, dept_members)

    # === Сетевая сводка ===
    n_senders = df_filtered[COLS["sender_id"]].nunique()
    lr_text = ""
    if cfg["total_employees"] > 0:
        lr = n_senders / cfg["total_employees"]
        lr_text = f" | <strong>LR (лояльность):</strong> {lr:.2f}"

    st.markdown(f"""
        <div class='metric-card'>
            <strong>Граф:</strong> {G_depts.number_of_nodes()} отделов,
            {G_people.number_of_nodes()} сотрудников,
            {G_people.number_of_edges()} связей |
            <strong>Модулярность:</strong> {metrics_people.get('modularity', 0):.3f} |
            <strong>Взаимность:</strong> {metrics_people.get('reciprocity', 0):.3f}{lr_text}
        </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # ВИЗУАЛИЗАЦИИ (сохранены из V5)
    # ============================================================
    st.markdown("---")
    st.header("🗺️ Визуализации")

    tab_viz1, tab_viz2 = st.tabs(["🌐 Иерархическая сеть", "🌀 Force-Directed"])

    with tab_viz1:
        st.markdown("""
        <div class='info-box'>
            <strong>🌐 Иерархическая интерактивная сеть</strong><br><br>
            🖱️ <strong>Клик на отдел</strong> — раскрывает людей отдела<br>
            🖱️ <strong>Double-click</strong> — возврат к отделам<br>
            🔍 <strong>Scroll</strong> — зум &nbsp; ✋ <strong>Drag</strong> — перетащить
        </div>""", unsafe_allow_html=True)
        html_hier = create_hierarchical_d3_viz(G_depts, G_people, dept_members, metrics_depts, metrics_people)
        components.html(html_hier, height=800, scrolling=False)

    with tab_viz2:
        st.markdown("""
        <div class='info-box'>
            <strong>🌀 Force-Directed Layout</strong><br>
            Цвет = сообщество (Louvain) • Размер = PageRank
        </div>""", unsafe_allow_html=True)
        html_force = create_force_d3_viz(G_people, metrics_people)
        components.html(html_force, height=700, scrolling=False)

    # ============================================================
    # АЛЕРТЫ
    # ============================================================
    if alerts["critical"] or alerts["warning"]:
        st.markdown("---")
        st.header("🚨 Алерты")

        if alerts["critical"]:
            for a in alerts["critical"][:10]:
                st.markdown(f"<div class='alert-red'>{a}</div>", unsafe_allow_html=True)
        if alerts["warning"]:
            for a in alerts["warning"][:15]:
                st.markdown(f"<div class='alert-yellow'>{a}</div>", unsafe_allow_html=True)
        if alerts["positive"]:
            for a in alerts["positive"][:5]:
                st.markdown(f"<div class='alert-green'>{a}</div>", unsafe_allow_html=True)

    # ============================================================
    # СОЦИАЛЬНЫЕ РОЛИ
    # ============================================================
    if cfg["show_social_stats"]:
        st.markdown("---")
        st.header("🎭 Социальные роли")

        # Подсчёт ролей
        role_counts = {}
        for r in social_roles.values():
            role_counts[r] = role_counts.get(r, 0) + 1

        role_cols = st.columns(min(len(role_counts), 5))
        for i, (role_key, count) in enumerate(sorted(role_counts.items(), key=lambda x: -x[1])):
            role_info = SOCIAL_ROLES.get(role_key, {"name": role_key, "icon": "❓", "color": "#888"})
            with role_cols[i % len(role_cols)]:
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.05); border-left:4px solid {role_info["color"]};
                     padding:8px; border-radius:5px; margin:3px 0;'>
                    <span style='font-size:1.3rem;'>{role_info["icon"]}</span>
                    <strong style='color:{role_info["color"]};'>{role_info["name"]}</strong><br>
                    <span style='color:#aaa; font-size:0.9rem;'>{count} чел.</span>
                </div>""", unsafe_allow_html=True)

        # Таблица ролей
        with st.expander("📋 Роли по сотрудникам", expanded=False):
            roles_data = []
            for node, role_key in social_roles.items():
                ri = SOCIAL_ROLES.get(role_key, {"name": role_key, "icon": "❓"})
                roles_data.append({
                    "ФИО": G_people.nodes[node].get("label", ""),
                    "Отдел": G_people.nodes[node].get("dept", ""),
                    "Роль": f"{ri['icon']} {ri['name']}",
                    "Признание (GF)": round(hr_metrics['gf'].get(node, 0), 2),
                    "Голоса (VU)": round(hr_metrics['vu'].get(node, 0), 2),
                    "Интеграция (CII)": round(hr_metrics['cii'].get(node, 0), 2),
                })
            st.dataframe(pd.DataFrame(roles_data).sort_values("Признание (GF)", ascending=False),
                         use_container_width=True, hide_index=True, height=400)

        # ============================================================
        # ТОПЫ (HR)
        # ============================================================
        st.markdown("---")
        st.header("🏆 Топы значимости")

        tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs([
            "👑 Влиятельность", "🌉 Посредничество",
            "🔗 Кросс-функциональность", "🤝 Поддержка и наставничество"
        ])

        # --- Топ 1: Влиятельность ---
        with tab_t1:
            st.markdown("**Топ влиятельности** — составной индекс: признание + широта связей + устойчивость + стаж в программе")
            top_data = []
            for node in G_people.nodes():
                top_data.append({
                    "ФИО": G_people.nodes[node].get("label", ""),
                    "Отдел": G_people.nodes[node].get("dept", ""),
                    "Индекс влиятельности": round(hr_metrics['influence_index'].get(node, 0), 3),
                    "Признание (GF)": round(hr_metrics['gf'].get(node, 0), 2),
                    "Устойчивость (SI)": round(hr_metrics['si'].get(node, 0), 2),
                    "Интеграция (CII)": round(hr_metrics['cii'].get(node, 0), 2),
                })
            df_top = pd.DataFrame(top_data).sort_values("Индекс влиятельности", ascending=False)
            st.dataframe(df_top.head(20), use_container_width=True, hide_index=True)

        # --- Топ 2: Посредничество ---
        with tab_t2:
            st.markdown("**Топ посредничества** — кто соединяет группы и отделы (мосты и брокеры информации)")
            top_data = []
            for node in G_people.nodes():
                bw = metrics_people['betweenness'].get(node, 0)
                br = metrics_people['is_bridge'].get(node, 0)
                con = metrics_people['constraint'].get(node, 1)
                dd = metrics_people['dept_diversity'].get(node, 0)
                subtype = "—"
                if br == 1:
                    subtype = "🌉 Мост"
                elif con < 0.4 and dd > 0.3:
                    subtype = "🔀 Брокер информации"
                elif bw > 0.01:
                    subtype = "↔️ Посредник"
                top_data.append({
                    "ФИО": G_people.nodes[node].get("label", ""),
                    "Отдел": G_people.nodes[node].get("dept", ""),
                    "Индекс посредничества": round(bw, 4),
                    "Тип": subtype,
                    "Кросс-функц.": round(dd, 2),
                    "Мост": "да" if br == 1 else "—",
                })
            df_top = pd.DataFrame(top_data).sort_values("Индекс посредничества", ascending=False)
            st.dataframe(df_top.head(20), use_container_width=True, hide_index=True)

        # --- Топ 3: Кросс-функциональность ---
        with tab_t3:
            st.markdown("**Топ кросс-функциональности** — кто взаимодействует с наибольшим числом разных отделов")
            top_data = []
            for node in G_people.nodes():
                top_data.append({
                    "ФИО": G_people.nodes[node].get("label", ""),
                    "Отдел": G_people.nodes[node].get("dept", ""),
                    "Кросс-функц. (общая)": round(metrics_people['dept_diversity'].get(node, 0), 3),
                    "Кросс-функц. доверие (IDD)": round(hr_metrics['idd'].get(node, 0), 3),
                    "Интеграция (CII)": round(hr_metrics['cii'].get(node, 0), 2),
                    "Отправлено": round(metrics_people['out_strength'].get(node, 0), 1),
                })
            df_top = pd.DataFrame(top_data).sort_values("Кросс-функц. (общая)", ascending=False)
            st.dataframe(df_top.head(20), use_container_width=True, hide_index=True)

        # --- Топ 4: Поддержка и наставничество ---
        with tab_t4:
            st.markdown("**Топ поддержки** — кого ценят за наставничество, надёжное плечо и поддержку")
            support_values = {"Наставничество", "Надёжное плечо", "наставничество", "надёжное плечо",
                              "надежное плечо", "Надежное плечо"}
            df_support = df_filtered[df_filtered[COLS["value"]].isin(support_values)]
            if len(df_support) > 0:
                support_recv = df_support.groupby(COLS["receiver_id"])[COLS["merits"]].sum()
                total_recv = df_filtered.groupby(COLS["receiver_id"])[COLS["merits"]].sum()
                top_data = []
                for node in G_people.nodes():
                    s_recv = support_recv.get(node, 0)
                    t_recv = total_recv.get(node, 0)
                    msi = s_recv / t_recv if t_recv > 0 else 0.0
                    if s_recv > 0:
                        top_data.append({
                            "ФИО": G_people.nodes[node].get("label", ""),
                            "Отдел": G_people.nodes[node].get("dept", ""),
                            "Мериты за поддержку": int(s_recv),
                            "Доля поддержки (MSI)": round(msi, 2),
                            "Всего получено": int(t_recv),
                        })
                df_top = pd.DataFrame(top_data).sort_values("Мериты за поддержку", ascending=False)
                st.dataframe(df_top.head(20), use_container_width=True, hide_index=True)
            else:
                st.info("Нет благодарностей за ценности поддержки в выбранном периоде")

        # ============================================================
        # ПРОДВИНУТАЯ СТАТИСТИКА (расширенная из V5)
        # ============================================================
        st.markdown("---")
        st.header("📊 Продвинутая социальная статистика")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🎯 Описание метрик")
            st.markdown("""
            <div class='metric-card'>
            <strong>Индекс влиятельности</strong> — составной: признание + широта + устойчивость + стаж<br>
            <strong>Индекс посредничества</strong> — кто соединяет разрозненные группы<br>
            <strong>Индекс доступности</strong> — насколько быстро информация достигает сотрудника<br>
            <strong>Плотность окружения</strong> — насколько тесно связаны коллеги друг с другом<br>
            <strong>Глубина интеграции</strong> — к какому ядру сети принадлежит (K-core)<br>
            <strong>Коэфф. признания (GF)</strong> — полученные мериты / среднее<br>
            <strong>Коэфф. устойчивости (SI)</strong> — доля активных месяцев<br>
            <strong>Коэфф. интеграции (CII)</strong> — доля внешних благодарностей<br>
            <strong>Коэфф. концентрации (CI)</strong> — зависимость от узкого круга<br>
            <strong>Кросс-функциональность</strong> — разнообразие отделов в связях
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.subheader("📈 Сетевые показатели")
            nodes_list = list(G_people.nodes())
            avg_clustering = np.mean([metrics_people["clustering"].get(n, 0) for n in nodes_list])
            avg_constraint = np.mean([metrics_people["constraint"].get(n, 0) for n in nodes_list])
            n_bridges = sum(1 for n in nodes_list if metrics_people["is_bridge"].get(n, 0) == 1)
            max_core = max([metrics_people["core_number"].get(n, 0) for n in nodes_list]) if nodes_list else 0

            # EvR на уровне сети
            in_vals = [metrics_people["in_strength"].get(n, 0) for n in nodes_list]
            out_vals = [metrics_people["out_strength"].get(n, 0) for n in nodes_list]
            evr_recv = calculate_evenness(in_vals)
            evr_send = calculate_evenness(out_vals)

            # Density
            n_nodes = G_people.number_of_nodes()
            n_edges = G_people.number_of_edges()
            density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

            st.markdown(f"""
            <div class='metric-card'>
            <strong>Плотность сети:</strong> {density:.4f}<br><br>
            <strong>Равномерность получения (EvR):</strong> {evr_recv:.3f}
            {'  ✅' if evr_recv >= 0.6 else '  ⚠️' if evr_recv >= 0.4 else '  🔴'}<br>
            <strong>Равномерность отправки (EvR):</strong> {evr_send:.3f}
            {'  ✅' if evr_send >= 0.6 else '  ⚠️' if evr_send >= 0.4 else '  🔴'}<br><br>
            <strong>Средняя плотность окружения:</strong> {avg_clustering:.3f}<br>
            <strong>Средний Constraint:</strong> {avg_constraint:.3f}<br>
            <strong>Мостов в сети:</strong> {n_bridges}<br>
            <strong>Максимальный K-core:</strong> {max_core}
            </div>
            """, unsafe_allow_html=True)

        # === Статистика по отделам (расширенная) ===
        st.markdown("### 🏢 Статистика по отделам")
        dept_stats = []
        for dept, members in dept_members.items():
            if len(members) == 0:
                continue
            # ER
            senders_in_dept = set()
            for m in members:
                if metrics_people['out_strength'].get(m, 0) > 0:
                    senders_in_dept.add(m)
            er = len(senders_in_dept) / len(members) if len(members) > 0 else 0

            # CII средний
            cii_avg = np.mean([hr_metrics['cii'].get(m, 0) for m in members])

            # EvR внутри отдела
            dept_in = [metrics_people['in_strength'].get(m, 0) for m in members]
            dept_evr = calculate_evenness(dept_in)

            # BDI
            total_ext = 0
            ext_per_member = {}
            for m in members:
                ext_c = sum(1 for nb in G_people.neighbors(m)
                            if G_people.nodes[nb].get('dept', '') != dept)
                ext_per_member[m] = ext_c
                total_ext += ext_c
            if total_ext > 0:
                top2 = sorted(ext_per_member.values(), reverse=True)[:2]
                bdi = sum(top2) / total_ext
            else:
                bdi = 0.0

            dept_stats.append({
                "Отдел": dept,
                "Сотрудников": len(members),
                "Вовлечённость (ER)": round(er, 2),
                "Интеграция (CII)": round(cii_avg, 2),
                "Равномерность (EvR)": round(dept_evr, 2),
                "Хрупкость (BDI)": round(bdi, 2),
                "Входящие": round(sum(metrics_depts['in_strength'].get(dept, 0) for _ in [1]), 1),
                "Исходящие": round(sum(metrics_depts['out_strength'].get(dept, 0) for _ in [1]), 1),
            })
        df_dept_stats = pd.DataFrame(dept_stats).sort_values("Сотрудников", ascending=False)
        st.dataframe(df_dept_stats, use_container_width=True, hide_index=True)

        # === Полная таблица метрик ===
        st.markdown("### 📋 Полная таблица метрик")
        full_metrics = []
        for node in G_people.nodes():
            nd = G_people.nodes[node]
            role_key = social_roles.get(node, "")
            ri = SOCIAL_ROLES.get(role_key, {"name": "—", "icon": ""})
            full_metrics.append({
                "ФИО": nd.get("label", ""),
                "Отдел": nd.get("dept", ""),
                "Роль": f"{ri['icon']} {ri['name']}",
                "Влиятельность": round(hr_metrics['influence_index'].get(node, 0), 3),
                "Признание (GF)": round(hr_metrics['gf'].get(node, 0), 2),
                "Голоса (VU)": round(hr_metrics['vu'].get(node, 0), 2),
                "Устойчивость (SI)": round(hr_metrics['si'].get(node, 0), 2),
                "Интеграция (CII)": round(hr_metrics['cii'].get(node, 0), 2),
                "Концентрация (CI)": round(hr_metrics['ci'].get(node, 0), 2),
                "Кросс-функц.": round(metrics_people['dept_diversity'].get(node, 0), 3),
                "Посредничество": round(metrics_people['betweenness'].get(node, 0), 4),
                "Доступность": round(metrics_people['closeness'].get(node, 0), 3),
                "Плотн. окруж.": round(metrics_people['clustering'].get(node, 0), 3),
                "K-core": metrics_people['core_number'].get(node, 0),
                "Мост": "да" if metrics_people['is_bridge'].get(node, 0) == 1 else "",
                "Получено": round(metrics_people['in_strength'].get(node, 0), 1),
                "Отправлено": round(metrics_people['out_strength'].get(node, 0), 1),
            })
        df_full = pd.DataFrame(full_metrics).sort_values("Влиятельность", ascending=False)
        st.dataframe(df_full, use_container_width=True, hide_index=True, height=400)

    # ============================================================
    # ЭКСПОРТ (сохранён из V5, расширен)
    # ============================================================
    st.markdown("---")
    st.subheader("💾 Экспорт данных")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if cfg["show_social_stats"]:
            csv = df_full.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Скачать все метрики (CSV)", csv,
                               "sociograph_metrics_v6.csv", "text/csv")

    with col_e2:
        graph_data = {
            "nodes": [
                {"id": str(n), "label": G_people.nodes[n].get("label", ""),
                 "dept": G_people.nodes[n].get("dept", ""),
                 "influence": float(hr_metrics.get('influence_index', {}).get(n, 0)),
                 "role": SOCIAL_ROLES.get(social_roles.get(n, ""), {}).get("name", ""),
                 "community": int(metrics_people.get("communities", {}).get(n, 0))}
                for n in G_people.nodes()
            ],
            "edges": [
                {"source": str(u), "target": str(v), "weight": float(data.get("weight", 1))}
                for u, v, data in G_people.edges(data=True)
            ],
            "stats": {
                "modularity": float(metrics_people.get("modularity", 0)),
                "reciprocity": float(metrics_people.get("reciprocity", 0)),
                "n_communities": len(set(metrics_people.get("communities", {}).values()))
            }
        }
        json_str = json.dumps(graph_data, indent=2, ensure_ascii=False)
        st.download_button("📥 Скачать граф (JSON)", json_str,
                           "network_graph_v6.json", "application/json")


if __name__ == "__main__":
    main()