# -*- coding: utf-8 -*-
"""
🕸️ СоциоГраф 7.0
==========================================================
V7 изменения:
- Убраны алерты
- Убраны самонаграждения (всегда фильтруются)
- Социальные роли кликабельны → топ по каждой роли
- Диапазон меритов: 1–50
- Новые фильтры: по периодам (год/месяц), по отделам, по сотрудникам

Запуск: streamlit run streamlit_app_v7.py
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

COLS = {
    "date": "Дата", "time": "Вермя",
    "sender": "ФИО Отправителя", "sender_id": "№ Отправителя",
    "sender_role": "Должномть Отправителя", "sender_dept": "Отдел Отправителя",
    "receiver": "ФИО Получателя", "receiver_id": "№ Получателя",
    "receiver_role": "Должномть Получателя", "receiver_dept": "Отдел Получателя",
    "value": "Ценность", "merits": "Мериты (сила)", "comment": "Комментарий",
}

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

def build_hierarchical_graph(df: pd.DataFrame, merit_range: tuple = (1, 50)):
    """Построение графа. Самонаграждения всегда отфильтрованы."""
    # Всегда убираем самонаграждения
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
        s_dept, r_dept = row[COLS["sender_dept"]], row[COLS["receiver_dept"]]
        if s_dept != r_dept:
            w = float(row["total_merits"])
            G_depts.add_edge(s_dept, r_dept, weight=w, people=int(row["n_people"]))
    return G_people, G_depts, dept_members


# ========================= ГРАФОВЫЕ МЕТРИКИ =========================

def calculate_graph_metrics(G: nx.DiGraph):
    if G.number_of_nodes() == 0:
        return {}
    metrics = {}
    metrics['in_strength'] = dict(G.in_degree(weight="weight"))
    metrics['out_strength'] = dict(G.out_degree(weight="weight"))
    try:
        metrics['pagerank'] = nx.pagerank(G, weight="weight", max_iter=100)
    except Exception:
        metrics['pagerank'] = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
    UG = G.to_undirected()
    try:
        metrics['betweenness'] = nx.betweenness_centrality(UG, weight='length', normalized=True)
    except Exception:
        metrics['betweenness'] = {n: 0.0 for n in G.nodes()}
    try:
        metrics['closeness'] = nx.closeness_centrality(UG, distance='length')
    except Exception:
        metrics['closeness'] = {n: 0.0 for n in G.nodes()}
    try:
        metrics['clustering'] = nx.clustering(UG, weight='weight')
    except Exception:
        metrics['clustering'] = {n: 0.0 for n in G.nodes()}
    try:
        metrics['constraint'] = nx.constraint(UG, weight='weight')
    except Exception:
        metrics['constraint'] = {n: 0.0 for n in G.nodes()}
    try:
        metrics['core_number'] = nx.core_number(UG)
    except Exception:
        metrics['core_number'] = {n: 0 for n in G.nodes()}
    try:
        bridges = list(nx.bridges(UG))
        bridge_nodes = set()
        for u, v in bridges:
            bridge_nodes.add(u)
            bridge_nodes.add(v)
        metrics['is_bridge'] = {n: 1 if n in bridge_nodes else 0 for n in G.nodes()}
    except Exception:
        metrics['is_bridge'] = {n: 0 for n in G.nodes()}
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

def calculate_hr_metrics(G, df, graph_metrics, merits_per_month=10, total_employees=0):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}
    hr = {}
    in_str = graph_metrics.get('in_strength', {})
    out_str = graph_metrics.get('out_strength', {})

    # GF
    all_received = [in_str.get(n, 0) for n in nodes]
    avg_received = np.mean(all_received) if np.mean(all_received) > 0 else 1.0
    hr['gf'] = {n: in_str.get(n, 0) / avg_received for n in nodes}

    # SI
    df_copy = df.copy()
    df_copy['_month'] = df_copy['dt'].dt.to_period('M')
    total_months = df_copy['_month'].nunique()
    if total_months == 0:
        total_months = 1
    sender_months = df_copy.groupby(COLS["sender_id"])['_month'].nunique().to_dict()
    hr['si'] = {n: sender_months.get(n, 0) / total_months for n in nodes}

    # CII
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

    # CI
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

    # SAR
    hr['sar'] = {n: (in_str.get(n, 0) + out_str.get(n, 0)) / 10.0 for n in nodes}

    # VU
    if merits_per_month > 0 and total_months > 0:
        available = merits_per_month * total_months
        hr['vu'] = {n: min(out_str.get(n, 0) / available, 1.0) if available > 0 else 0.0 for n in nodes}
    else:
        hr['vu'] = {n: 0.0 for n in nodes}

    # IDD
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

    # Influence Index
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
            roles[n] = "connector"
    return roles


# ========================= EvR =========================

def calculate_evenness(values_list):
    arr = np.array(sorted(values_list))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n
    return max(0.0, min(1.0, 1.0 - gini))


# ========================= D3 ВИЗУАЛИЗАЦИИ =========================

def create_hierarchical_d3_viz(G_depts, G_people, dept_members, metrics_depts, metrics_people):
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
        body {{ margin:0; background:#0a0e27; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
        #viz {{ width:100%; height:100vh; }}
        .controls {{ position:absolute; top:10px; right:10px; z-index:1000; }}
        .btn {{ background:linear-gradient(90deg,#00d4ff,#7b2cbf); color:white; border:none;
                padding:8px 15px; margin:2px; border-radius:5px; cursor:pointer; font-weight:600; font-size:12px; }}
        .btn:hover {{ opacity:0.8; }}
        .node {{ cursor:pointer; stroke:#fff; stroke-width:2px; }}
        .node.dept {{ fill:#7b2cbf; }} .node.person {{ fill:#00d4ff; }}
        .link {{ stroke:#999; stroke-opacity:0.4; }}
        .label {{ fill:white; font-size:11px; pointer-events:none; text-anchor:middle; text-shadow:0 0 3px #000; }}
        #breadcrumb {{ position:absolute; top:10px; left:10px; color:#00d4ff; font-size:16px; font-weight:bold; text-shadow:0 0 10px rgba(0,212,255,0.8); }}
        #info {{ position:absolute; bottom:10px; left:10px; color:white; font-size:12px; background:rgba(0,0,0,0.7); padding:10px; border-radius:5px; max-width:300px; }}
    </style></head><body>
    <div id="breadcrumb">Уровень: Отделы</div><div id="info">Загрузка...</div>
    <div class="controls">
        <button class="btn" onclick="resetView()">🏠 Домой</button>
        <button class="btn" onclick="resetZoom()">🔍 Сбросить зум</button>
        <button class="btn" onclick="toggleLabels()">🏷️ Метки</button>
        <button class="btn" onclick="togglePhysics()">⚡ Физика</button>
    </div><svg id="viz"></svg>
    <script>
        const width=window.innerWidth,height=window.innerHeight;
        const deptNodesData={json.dumps(dept_nodes)};const deptLinksData={json.dumps(dept_edges)};
        const peopleNodesData={json.dumps(people_nodes)};const peopleLinksData={json.dumps(people_edges)};
        let nodes=[...deptNodesData],links=[...deptLinksData],currentLevel="depts",expandedDept=null;
        const svg=d3.select("#viz").attr("width",width).attr("height",height);const g=svg.append("g");
        const zoom=d3.zoom().scaleExtent([0.1,10]).on("zoom",e=>g.attr("transform",e.transform));svg.call(zoom);
        let linkEl,nodeEl,labels,sim;
        function initSim(){{g.selectAll("*").remove();
            linkEl=g.append("g").selectAll("line").data(links).join("line").attr("class","link").attr("stroke-width",d=>Math.sqrt(d.weight)/2);
            nodeEl=g.append("g").selectAll("circle").data(nodes).join("circle").attr("class",d=>`node ${{d.type}}`).attr("r",d=>d.type==="dept"?Math.sqrt(d.size)*5+10:6).on("click",onClick).on("dblclick",onDbl).on("mouseover",showInfo).call(d3.drag().on("start",ds).on("drag",dr).on("end",de));
            labels=g.append("g").selectAll("text").data(nodes).join("text").attr("class","label").attr("dy",-10).text(d=>d.label.length>20?d.label.slice(0,20)+"...":d.label);
            sim=d3.forceSimulation(nodes).force("link",d3.forceLink(links).id(d=>d.id).distance(currentLevel==="depts"?150:80)).force("charge",d3.forceManyBody().strength(-300)).force("center",d3.forceCenter(width/2,height/2)).force("collision",d3.forceCollide().radius(d=>d.type==="dept"?Math.sqrt(d.size)*5+15:10)).on("tick",tick);}}
        function tick(){{linkEl.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);nodeEl.attr("cx",d=>d.x).attr("cy",d=>d.y);labels.attr("x",d=>d.x).attr("y",d=>d.y);}}
        function ds(e,d){{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}}
        function dr(e,d){{d.fx=e.x;d.fy=e.y;}}function de(e,d){{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}
        function onClick(e,d){{e.stopPropagation();if(currentLevel==="depts"&&d.type==="dept")expand(d);}}
        function onDbl(e,d){{e.stopPropagation();if(currentLevel==="people")collapse();}}
        function expand(dn){{currentLevel="people";expandedDept=dn.original_id;const m=dn.members||[];nodes=peopleNodesData.filter(n=>m.includes(n.original_id));const ids=new Set(nodes.map(n=>n.id));links=peopleLinksData.filter(l=>ids.has(l.source)&&ids.has(l.target));document.getElementById("breadcrumb").textContent=`Уровень: ${{dn.label}} (double-click для возврата)`;sim.stop();initSim();}}
        function collapse(){{currentLevel="depts";expandedDept=null;nodes=[...deptNodesData];links=[...deptLinksData];document.getElementById("breadcrumb").textContent="Уровень: Отделы";sim.stop();initSim();}}
        function resetView(){{collapse();}}function resetZoom(){{svg.transition().duration(750).call(zoom.transform,d3.zoomIdentity);}}
        let lv=true;function toggleLabels(){{lv=!lv;labels.style("opacity",lv?1:0);}}
        let pv=true;function togglePhysics(){{pv=!pv;if(pv)sim.alpha(0.3).restart();else sim.stop();}}
        function showInfo(e,d){{let i=`<strong>${{d.label}}</strong><br>`;if(d.type==="dept")i+=`Сотрудников: ${{d.size}}<br>Входящие: ${{d.in_strength.toFixed(1)}}<br>Исходящие: ${{d.out_strength.toFixed(1)}}`;else i+=`Отдел: ${{d.dept}}<br>Входящие: ${{d.in_strength.toFixed(1)}}<br>Исходящие: ${{d.out_strength.toFixed(1)}}`;document.getElementById("info").innerHTML=i;}}
        initSim();
    </script></body></html>"""
    return html


def create_force_d3_viz(G, metrics):
    nodes_data = []
    for node in G.nodes():
        nd = G.nodes[node]; comm = metrics.get("communities", {}).get(node, 0)
        nodes_data.append({"id": str(node), "label": nd.get("label", str(node)), "dept": nd.get("dept", ""), "community": comm,
            "pagerank": metrics.get("pagerank", {}).get(node, 0), "in_strength": metrics.get("in_strength", {}).get(node, 0), "out_strength": metrics.get("out_strength", {}).get(node, 0)})
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({"source": str(u), "target": str(v), "weight": data.get("weight", 1)})
    n_comm = max(len(set(metrics.get("communities", {}).values())), 1)
    colors = ["#00d4ff","#7b2cbf","#ff006e","#ffbe0b","#8ac926","#ff006e","#3a86ff","#fb5607","#06ffa5","#8338ec"]
    html = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><script src="https://d3js.org/d3.v7.min.js"></script>
    <style>body{{margin:0;background:#0a0e27;font-family:sans-serif;overflow:hidden;}}#viz{{width:100%;height:100vh;}}.node{{cursor:pointer;stroke:#fff;stroke-width:1.5px;}}.link{{stroke:#999;stroke-opacity:0.3;}}.label{{fill:white;font-size:10px;pointer-events:none;text-anchor:middle;text-shadow:0 0 3px #000;}}.controls{{position:absolute;top:10px;right:10px;z-index:1000;}}.btn{{background:linear-gradient(90deg,#00d4ff,#7b2cbf);color:white;border:none;padding:8px 15px;margin:2px;border-radius:5px;cursor:pointer;font-size:12px;}}#info{{position:absolute;bottom:10px;left:10px;color:white;background:rgba(0,0,0,0.7);padding:10px;border-radius:5px;font-size:12px;max-width:300px;}}</style></head><body>
    <div class="controls"><button class="btn" onclick="resetZoom()">🔍 Сбросить зум</button><button class="btn" onclick="toggleLabels()">🏷️ Метки</button><button class="btn" onclick="togglePhysics()">⚡ Физика</button></div>
    <div id="info">Наведите на узел</div><svg id="viz"></svg>
    <script>
        const w=window.innerWidth,h=window.innerHeight;const nodes={json.dumps(nodes_data)};const links={json.dumps(edges_data)};const colors={json.dumps(colors[:n_comm])};
        const svg=d3.select("#viz").attr("width",w).attr("height",h);const g=svg.append("g");const zoom=d3.zoom().scaleExtent([0.1,10]).on("zoom",e=>g.attr("transform",e.transform));svg.call(zoom);
        const le=g.append("g").selectAll("line").data(links).join("line").attr("class","link").attr("stroke-width",d=>Math.sqrt(d.weight)/2);
        const ne=g.append("g").selectAll("circle").data(nodes).join("circle").attr("class","node").attr("r",d=>3+Math.sqrt(d.pagerank*1000)).attr("fill",d=>colors[d.community%colors.length]).on("mouseover",si).call(d3.drag().on("start",ds).on("drag",dr).on("end",de));
        const lb=g.append("g").selectAll("text").data(nodes).join("text").attr("class","label").attr("dy",-8).text(d=>d.label.length>15?d.label.slice(0,15)+"...":d.label);
        const sim=d3.forceSimulation(nodes).force("link",d3.forceLink(links).id(d=>d.id).distance(70)).force("charge",d3.forceManyBody().strength(-200)).force("center",d3.forceCenter(w/2,h/2)).force("collision",d3.forceCollide().radius(15)).on("tick",()=>{{le.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);ne.attr("cx",d=>d.x).attr("cy",d=>d.y);lb.attr("x",d=>d.x).attr("y",d=>d.y);}});
        function ds(e,d){{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}}
        function dr(e,d){{d.fx=e.x;d.fy=e.y;}}function de(e,d){{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}
        function si(e,d){{document.getElementById("info").innerHTML=`<strong>${{d.label}}</strong><br>Отдел: ${{d.dept}}<br>Сообщество: ${{d.community}}<br>Входящие: ${{d.in_strength.toFixed(1)}}<br>Исходящие: ${{d.out_strength.toFixed(1)}}`;}}
        function resetZoom(){{svg.transition().duration(750).call(zoom.transform,d3.zoomIdentity);}}
        let lv=true;function toggleLabels(){{lv=!lv;lb.style("opacity",lv?1:0);}}
        let pv=true;function togglePhysics(){{pv=!pv;if(pv)sim.alpha(0.3).restart();else sim.stop();}}
    </script></body></html>"""
    return html


# ========================= SIDEBAR & FILTERING =========================

def sidebar_controls(df: pd.DataFrame):
    st.sidebar.header("⚙️ Настройки")

    # --- Период: год / месяц ---
    st.sidebar.markdown("### 📅 Период")
    df_dates = df["dt"].dropna()
    available_years = sorted(df_dates.dt.year.unique().tolist())
    selected_years = st.sidebar.multiselect(
        "Год", options=available_years, default=available_years
    )

    # Месяцы (зависят от выбранных годов)
    month_names = {1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
                   5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
                   9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"}
    df_in_years = df_dates[df_dates.dt.year.isin(selected_years)] if selected_years else df_dates
    available_months = sorted(df_in_years.dt.month.unique().tolist())
    month_options = [f"{month_names.get(m, m)}" for m in available_months]
    selected_month_names = st.sidebar.multiselect(
        "Месяц", options=month_options, default=month_options
    )
    # Обратная карта: название → номер
    name_to_num = {v: k for k, v in month_names.items()}
    selected_months = [name_to_num.get(mn, 0) for mn in selected_month_names]

    st.sidebar.markdown("---")

    # --- Отделы ---
    all_depts = sorted(set(
        df[COLS["sender_dept"]].dropna().unique().tolist() +
        df[COLS["receiver_dept"]].dropna().unique().tolist()
    ))
    selected_depts = st.sidebar.multiselect(
        "🏢 Отделы", options=all_depts, default=all_depts
    )

    # --- Сотрудники ---
    # Фильтруем список сотрудников по выбранным отделам
    df_dept_filtered = df[
        (df[COLS["sender_dept"]].isin(selected_depts)) |
        (df[COLS["receiver_dept"]].isin(selected_depts))
    ]
    all_people = sorted(set(
        df_dept_filtered[COLS["sender"]].dropna().unique().tolist() +
        df_dept_filtered[COLS["receiver"]].dropna().unique().tolist()
    ))
    selected_people = st.sidebar.multiselect(
        "👤 Сотрудники", options=all_people, default=[],
        help="Оставьте пустым для выбора всех сотрудников выбранных отделов"
    )

    st.sidebar.markdown("---")

    # --- Ценности ---
    values_list = sorted(df[COLS["value"]].dropna().unique().tolist())
    selected_values = st.sidebar.multiselect("⭐ Ценности", options=values_list, default=values_list)

    # --- Мериты: 1–50 ---
    st.sidebar.markdown("### 💎 Мериты на связь")
    merit_range = st.sidebar.slider(
        "Диапазон меритов", min_value=1, max_value=50,
        value=(1, 50), step=1,
        help="Фильтрация связей по суммарному количеству меритов"
    )
    st.sidebar.markdown(f"*Связи с {merit_range[0]} по {merit_range[1]} меритов*")

    # --- Параметры программы ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Параметры программы")
    merits_per_month = st.sidebar.number_input(
        "Меритов в месяц на сотрудника", min_value=1, max_value=100, value=10,
        help="Лимит меритов в месяц"
    )
    total_employees = st.sidebar.number_input(
        "Всего сотрудников в компании", min_value=0, max_value=10000, value=0,
        help="Для расчёта LR. 0 = не рассчитывать"
    )

    st.sidebar.markdown("---")
    show_social_stats = st.sidebar.checkbox(
        "📊 Показать продвинутую статистику", value=True
    )

    return {
        "years": selected_years,
        "months": selected_months,
        "depts": set(selected_depts),
        "people": selected_people,  # пустой список = все
        "values": set(selected_values),
        "merit_range": merit_range,
        "merits_per_month": merits_per_month,
        "total_employees": total_employees,
        "show_social_stats": show_social_stats,
    }


def filter_df(df: pd.DataFrame, cfg):
    m = pd.Series(True, index=df.index)

    # Год + Месяц
    if cfg["years"]:
        m &= df["dt"].dt.year.isin(cfg["years"])
    if cfg["months"]:
        m &= df["dt"].dt.month.isin(cfg["months"])

    # Ценности
    m &= df[COLS["value"]].isin(cfg["values"])

    # Отделы
    m &= (
        df[COLS["sender_dept"]].isin(cfg["depts"]) |
        df[COLS["receiver_dept"]].isin(cfg["depts"])
    )

    # Сотрудники (если выбраны конкретные)
    if cfg["people"]:
        m &= (
            df[COLS["sender"]].isin(cfg["people"]) |
            df[COLS["receiver"]].isin(cfg["people"])
        )

    return df.loc[m].copy()


# ========================= MAIN =========================

def main():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3rem;'>🕸️ СоциоГраф 7.0</h1>
            <p style='font-size: 1.2rem; color: #00d4ff;'>
                Иерархическая визуализация + HR-аналитика + Социальные роли
            </p>
        </div>
    """, unsafe_allow_html=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "dataset.xlsx")
    if os.path.exists(local_path):
        df = load_df(local_path)
    else:
        st.error("❌ Файл dataset.xlsx не найден рядом со скриптом.")
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

    # === Графы ===
    with st.spinner("🔄 Строим структуру..."):
        G_people, G_depts, dept_members = build_hierarchical_graph(
            df_filtered, cfg["merit_range"]
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

    # === Сводка ===
    n_senders = df_filtered[COLS["sender_id"]].nunique()
    lr_text = ""
    if cfg["total_employees"] > 0:
        lr = n_senders / cfg["total_employees"]
        lr_text = f" | <strong>LR:</strong> {lr:.2f}"

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
    # ВИЗУАЛИЗАЦИИ
    # ============================================================
    st.markdown("---")
    st.header("🗺️ Визуализации")
    tab_viz1, tab_viz2 = st.tabs(["🌐 Иерархическая сеть", "🌀 Force-Directed"])

    with tab_viz1:
        st.markdown("""<div class='info-box'>
            <strong>🌐 Иерархическая сеть</strong><br>
            🖱️ Клик на отдел → люди &nbsp; 🖱️ Double-click → назад &nbsp;
            🔍 Scroll → зум &nbsp; ✋ Drag → перетащить
        </div>""", unsafe_allow_html=True)
        html_hier = create_hierarchical_d3_viz(G_depts, G_people, dept_members, metrics_depts, metrics_people)
        components.html(html_hier, height=800, scrolling=False)

    with tab_viz2:
        st.markdown("""<div class='info-box'>
            <strong>🌀 Force-Directed</strong> — Цвет = сообщество • Размер = влиятельность
        </div>""", unsafe_allow_html=True)
        html_force = create_force_d3_viz(G_people, metrics_people)
        components.html(html_force, height=700, scrolling=False)

    # ============================================================
    # СОЦИАЛЬНЫЕ РОЛИ — кликабельные
    # ============================================================
    if cfg["show_social_stats"]:
        st.markdown("---")
        st.header("🎭 Социальные роли")
        st.markdown("*Нажмите на роль, чтобы увидеть топ сотрудников с этой ролью*")

        # Подсчёт ролей
        role_counts = {}
        for r in social_roles.values():
            role_counts[r] = role_counts.get(r, 0) + 1

        # Инициализация session_state
        if "selected_role" not in st.session_state:
            st.session_state.selected_role = None

        # Кнопки ролей
        sorted_roles = sorted(role_counts.items(), key=lambda x: -x[1])
        n_cols = min(len(sorted_roles), 5)
        role_cols = st.columns(n_cols)

        for i, (role_key, count) in enumerate(sorted_roles):
            role_info = SOCIAL_ROLES.get(role_key, {"name": role_key, "icon": "❓", "color": "#888"})
            with role_cols[i % n_cols]:
                is_selected = st.session_state.selected_role == role_key
                border_style = f"border: 2px solid {role_info['color']};" if is_selected else ""
                if st.button(
                    f"{role_info['icon']} {role_info['name']} ({count})",
                    key=f"role_btn_{role_key}",
                    use_container_width=True
                ):
                    if st.session_state.selected_role == role_key:
                        st.session_state.selected_role = None  # toggle off
                    else:
                        st.session_state.selected_role = role_key
                    st.rerun()

        # Таблица по выбранной роли
        if st.session_state.selected_role:
            sel_role = st.session_state.selected_role
            ri = SOCIAL_ROLES.get(sel_role, {"name": sel_role, "icon": "❓", "color": "#888"})
            st.markdown(f"### {ri['icon']} Топ: {ri['name']}")

            role_data = []
            for node, rk in social_roles.items():
                if rk == sel_role:
                    role_data.append({
                        "ФИО": G_people.nodes[node].get("label", ""),
                        "Отдел": G_people.nodes[node].get("dept", ""),
                        "Влиятельность": round(hr_metrics['influence_index'].get(node, 0), 3),
                        "Признание (GF)": round(hr_metrics['gf'].get(node, 0), 2),
                        "Голоса (VU)": round(hr_metrics['vu'].get(node, 0), 2),
                        "Устойчивость (SI)": round(hr_metrics['si'].get(node, 0), 2),
                        "Интеграция (CII)": round(hr_metrics['cii'].get(node, 0), 2),
                        "Кросс-функц.": round(metrics_people['dept_diversity'].get(node, 0), 3),
                        "Посредничество": round(metrics_people['betweenness'].get(node, 0), 4),
                        "Получено": round(metrics_people['in_strength'].get(node, 0), 1),
                        "Отправлено": round(metrics_people['out_strength'].get(node, 0), 1),
                    })
            if role_data:
                df_role = pd.DataFrame(role_data).sort_values("Влиятельность", ascending=False)
                st.dataframe(df_role, use_container_width=True, hide_index=True)
            else:
                st.info("Нет сотрудников с данной ролью")
        else:
            # Общая таблица ролей
            with st.expander("📋 Все роли по сотрудникам", expanded=False):
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

        with tab_t1:
            st.markdown("**Топ влиятельности** — признание + широта связей + устойчивость + стаж")
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
            st.dataframe(pd.DataFrame(top_data).sort_values("Индекс влиятельности", ascending=False).head(20),
                         use_container_width=True, hide_index=True)

        with tab_t2:
            st.markdown("**Топ посредничества** — мосты и брокеры информации")
            top_data = []
            for node in G_people.nodes():
                bw = metrics_people['betweenness'].get(node, 0)
                br = metrics_people['is_bridge'].get(node, 0)
                con = metrics_people['constraint'].get(node, 1)
                dd = metrics_people['dept_diversity'].get(node, 0)
                subtype = "—"
                if br == 1: subtype = "🌉 Мост"
                elif con < 0.4 and dd > 0.3: subtype = "🔀 Брокер"
                elif bw > 0.01: subtype = "↔️ Посредник"
                top_data.append({
                    "ФИО": G_people.nodes[node].get("label", ""), "Отдел": G_people.nodes[node].get("dept", ""),
                    "Посредничество": round(bw, 4), "Тип": subtype, "Кросс-функц.": round(dd, 2),
                    "Мост": "да" if br == 1 else "—",
                })
            st.dataframe(pd.DataFrame(top_data).sort_values("Посредничество", ascending=False).head(20),
                         use_container_width=True, hide_index=True)

        with tab_t3:
            st.markdown("**Топ кросс-функциональности** — разнообразие связей между отделами")
            top_data = []
            for node in G_people.nodes():
                top_data.append({
                    "ФИО": G_people.nodes[node].get("label", ""), "Отдел": G_people.nodes[node].get("dept", ""),
                    "Кросс-функц.": round(metrics_people['dept_diversity'].get(node, 0), 3),
                    "Доверие (IDD)": round(hr_metrics['idd'].get(node, 0), 3),
                    "Интеграция (CII)": round(hr_metrics['cii'].get(node, 0), 2),
                    "Отправлено": round(metrics_people['out_strength'].get(node, 0), 1),
                })
            st.dataframe(pd.DataFrame(top_data).sort_values("Кросс-функц.", ascending=False).head(20),
                         use_container_width=True, hide_index=True)

        with tab_t4:
            st.markdown("**Топ поддержки** — наставничество, надёжное плечо")
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
                            "ФИО": G_people.nodes[node].get("label", ""), "Отдел": G_people.nodes[node].get("dept", ""),
                            "Мериты за поддержку": int(s_recv), "Доля (MSI)": round(msi, 2), "Всего получено": int(t_recv),
                        })
                if top_data:
                    st.dataframe(pd.DataFrame(top_data).sort_values("Мериты за поддержку", ascending=False).head(20),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("Нет данных")
            else:
                st.info("Нет благодарностей за ценности поддержки")

        # ============================================================
        # ПРОДВИНУТАЯ СТАТИСТИКА
        # ============================================================
        st.markdown("---")
        st.header("📊 Продвинутая статистика")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🎯 Описание метрик")
            st.markdown("""<div class='metric-card'>
            <strong>Индекс влиятельности</strong> — признание + широта + устойчивость + стаж<br>
            <strong>Индекс посредничества</strong> — кто соединяет группы<br>
            <strong>Индекс доступности</strong> — скорость распространения информации<br>
            <strong>Плотность окружения</strong> — связанность соседей<br>
            <strong>Глубина интеграции</strong> — ядро сети (K-core)<br>
            <strong>Признание (GF)</strong> — мериты / среднее<br>
            <strong>Устойчивость (SI)</strong> — доля активных месяцев<br>
            <strong>Интеграция (CII)</strong> — доля внешних связей<br>
            <strong>Концентрация (CI)</strong> — зависимость от топ-3<br>
            <strong>Кросс-функциональность</strong> — разнообразие отделов
            </div>""", unsafe_allow_html=True)

        with col_b:
            st.subheader("📈 Сетевые показатели")
            nodes_list = list(G_people.nodes())
            avg_cl = np.mean([metrics_people["clustering"].get(n, 0) for n in nodes_list])
            avg_con = np.mean([metrics_people["constraint"].get(n, 0) for n in nodes_list])
            n_br = sum(1 for n in nodes_list if metrics_people["is_bridge"].get(n, 0) == 1)
            max_core = max([metrics_people["core_number"].get(n, 0) for n in nodes_list]) if nodes_list else 0
            in_vals = [metrics_people["in_strength"].get(n, 0) for n in nodes_list]
            out_vals = [metrics_people["out_strength"].get(n, 0) for n in nodes_list]
            evr_r = calculate_evenness(in_vals)
            evr_s = calculate_evenness(out_vals)
            nn = G_people.number_of_nodes()
            ne = G_people.number_of_edges()
            dens = ne / (nn * (nn - 1)) if nn > 1 else 0
            st.markdown(f"""<div class='metric-card'>
            <strong>Плотность сети:</strong> {dens:.4f}<br><br>
            <strong>Равномерность получения:</strong> {evr_r:.3f} {'✅' if evr_r >= 0.6 else '⚠️' if evr_r >= 0.4 else '🔴'}<br>
            <strong>Равномерность отправки:</strong> {evr_s:.3f} {'✅' if evr_s >= 0.6 else '⚠️' if evr_s >= 0.4 else '🔴'}<br><br>
            <strong>Плотность окружения (ср.):</strong> {avg_cl:.3f}<br>
            <strong>Constraint (ср.):</strong> {avg_con:.3f}<br>
            <strong>Мостов:</strong> {n_br} &nbsp; <strong>K-core (макс.):</strong> {max_core}
            </div>""", unsafe_allow_html=True)

        # Отделы
        st.markdown("### 🏢 Статистика по отделам")
        dept_stats = []
        for dept, members in dept_members.items():
            if not members: continue
            senders = set(m for m in members if metrics_people['out_strength'].get(m, 0) > 0)
            er = len(senders) / len(members) if members else 0
            cii_avg = np.mean([hr_metrics['cii'].get(m, 0) for m in members])
            dept_in = [metrics_people['in_strength'].get(m, 0) for m in members]
            dept_evr = calculate_evenness(dept_in)
            total_ext = 0; ext_pm = {}
            for m in members:
                ec = sum(1 for nb in G_people.neighbors(m) if G_people.nodes[nb].get('dept', '') != dept)
                ext_pm[m] = ec; total_ext += ec
            bdi = sum(sorted(ext_pm.values(), reverse=True)[:2]) / total_ext if total_ext > 0 else 0.0
            dept_stats.append({
                "Отдел": dept, "Сотрудников": len(members),
                "Вовлечённость (ER)": round(er, 2), "Интеграция (CII)": round(cii_avg, 2),
                "Равномерность (EvR)": round(dept_evr, 2), "Хрупкость (BDI)": round(bdi, 2),
                "Входящие": round(metrics_depts['in_strength'].get(dept, 0), 1),
                "Исходящие": round(metrics_depts['out_strength'].get(dept, 0), 1),
            })
        st.dataframe(pd.DataFrame(dept_stats).sort_values("Сотрудников", ascending=False),
                     use_container_width=True, hide_index=True)

        # Полная таблица
        st.markdown("### 📋 Полная таблица метрик")
        full_metrics = []
        for node in G_people.nodes():
            nd = G_people.nodes[node]
            rk = social_roles.get(node, "")
            ri = SOCIAL_ROLES.get(rk, {"name": "—", "icon": ""})
            full_metrics.append({
                "ФИО": nd.get("label", ""), "Отдел": nd.get("dept", ""),
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
    # ЭКСПОРТ
    # ============================================================
    st.markdown("---")
    st.subheader("💾 Экспорт данных")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if cfg["show_social_stats"]:
            csv = df_full.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Метрики (CSV)", csv, "sociograph_metrics_v7.csv", "text/csv")
    with col_e2:
        graph_data = {
            "nodes": [{"id": str(n), "label": G_people.nodes[n].get("label", ""),
                       "dept": G_people.nodes[n].get("dept", ""),
                       "influence": float(hr_metrics.get('influence_index', {}).get(n, 0)),
                       "role": SOCIAL_ROLES.get(social_roles.get(n, ""), {}).get("name", ""),
                       "community": int(metrics_people.get("communities", {}).get(n, 0))}
                      for n in G_people.nodes()],
            "edges": [{"source": str(u), "target": str(v), "weight": float(d.get("weight", 1))}
                      for u, v, d in G_people.edges(data=True)],
            "stats": {"modularity": float(metrics_people.get("modularity", 0)),
                      "reciprocity": float(metrics_people.get("reciprocity", 0)),
                      "n_communities": len(set(metrics_people.get("communities", {}).values()))}
        }
        st.download_button("📥 Граф (JSON)", json.dumps(graph_data, indent=2, ensure_ascii=False),
                           "network_graph_v7.json", "application/json")


if __name__ == "__main__":
    main()