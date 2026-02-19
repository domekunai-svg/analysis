# -*- coding: utf-8 -*-
"""
🕸️ СоциоГраф 5.0
==========================================================
Объединяет лучшее из V3 и V4:
- Иерархическая интерактивная визуализация (из V3)
- Глубокая социальная статистика (из V4)
- Диапазон меритов вместо минимума
- Множественные визуализации

Запуск: streamlit run streamlit_app_v5.py
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
from sklearn.manifold import MDS

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

def build_hierarchical_graph(df: pd.DataFrame, merit_range: tuple = (1, 100), allow_self: bool = False):
    """Построение иерархического графа (отделы + люди) с фильтром по диапазону меритов"""
    if not allow_self:
        df = df[df[COLS["sender_id"]] != df[COLS["receiver_id"]]].copy()
    
    # Граф людей
    person_agg = (
        df.groupby([
            COLS["sender_id"], COLS["receiver_id"],
            COLS["sender"], COLS["receiver"],
            COLS["sender_dept"], COLS["receiver_dept"]
        ], dropna=False)
        .agg(total_merits=(COLS["merits"], "sum"), n_msgs=("dt", "count"))
        .reset_index()
    )
    
    # Фильтруем по диапазону меритов
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
    
    # Граф отделов
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

# ========================= СОЦИАЛЬНЫЕ МЕТРИКИ =========================

def calculate_advanced_metrics(G: nx.DiGraph):
    """Расчет продвинутых социологических метрик"""
    
    if G.number_of_nodes() == 0:
        return {}
    
    metrics = {}
    
    # Базовые метрики
    metrics['in_strength'] = dict(G.in_degree(weight="weight"))
    metrics['out_strength'] = dict(G.out_degree(weight="weight"))
    
    try:
        metrics['pagerank'] = nx.pagerank(G, weight="weight", max_iter=100)
    except:
        metrics['pagerank'] = {n: 1.0/G.number_of_nodes() for n in G.nodes()}
    
    # Неориентированный граф
    UG = G.to_undirected()
    
    # 1. BETWEENNESS - посредничество
    try:
        metrics['betweenness'] = nx.betweenness_centrality(UG, weight='length', normalized=True)
    except:
        metrics['betweenness'] = {n: 0.0 for n in G.nodes()}
    
    # 2. CLOSENESS - близость
    try:
        metrics['closeness'] = nx.closeness_centrality(UG, distance='length')
    except:
        metrics['closeness'] = {n: 0.0 for n in G.nodes()}
    
    # 3. CLUSTERING COEFFICIENT
    try:
        metrics['clustering'] = nx.clustering(UG, weight='weight')
    except:
        metrics['clustering'] = {n: 0.0 for n in G.nodes()}
    
    # 4. EIGENVECTOR CENTRALITY
    try:
        metrics['eigenvector'] = nx.eigenvector_centrality(UG, weight='weight', max_iter=200)
    except:
        metrics['eigenvector'] = {n: 0.0 for n in G.nodes()}
    
    # 5. CONSTRAINT (Burt's structural holes)
    try:
        metrics['constraint'] = nx.constraint(UG, weight='weight')
    except:
        metrics['constraint'] = {n: 0.0 for n in G.nodes()}
    
    # 6. K-CORE decomposition
    try:
        metrics['core_number'] = nx.core_number(UG)
    except:
        metrics['core_number'] = {n: 0 for n in G.nodes()}
    
    # 7. LOAD CENTRALITY
    try:
        metrics['load'] = nx.load_centrality(UG, weight='length')
    except:
        metrics['load'] = {n: 0.0 for n in G.nodes()}
    
    # 8. BRIDGES
    try:
        bridges = list(nx.bridges(UG))
        bridge_nodes = set()
        for u, v in bridges:
            bridge_nodes.add(u)
            bridge_nodes.add(v)
        metrics['is_bridge'] = {n: 1 if n in bridge_nodes else 0 for n in G.nodes()}
    except:
        metrics['is_bridge'] = {n: 0 for n in G.nodes()}
    
    # 9. TRIADIC CLOSURE
    triadic_closure = {}
    for node in G.nodes():
        neighbors = set(G.neighbors(node)) | set(G.predecessors(node))
        if len(neighbors) < 2:
            triadic_closure[node] = 0.0
        else:
            edges_between = 0
            possible_edges = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 < n2:
                        possible_edges += 1
                        if UG.has_edge(n1, n2):
                            edges_between += 1
            triadic_closure[node] = edges_between / possible_edges if possible_edges > 0 else 0.0
    metrics['triadic_closure'] = triadic_closure
    
    # 10. DEPARTMENT DIVERSITY
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
            dept_diversity[node] = len(depts) / len(neighbors)
    metrics['dept_diversity'] = dept_diversity
    
    # СЕТЕВЫЕ МЕТРИКИ
    try:
        part = community_louvain.best_partition(UG, weight="weight")
        mod = community_louvain.modularity(part, UG, weight="weight")
        metrics['communities'] = part
        metrics['modularity'] = mod
    except:
        metrics['communities'] = {n: 0 for n in G.nodes()}
        metrics['modularity'] = 0.0
    
    metrics['reciprocity'] = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0
    
    return metrics

# ========================= ИЕРАРХИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ =========================

def create_hierarchical_d3_viz(G_depts, G_people, dept_members, metrics_depts, metrics_people):
    """
    D3.js визуализация с иерархией:
    - Стартуем с отделов
    - Клик на отдел -> раскрываем в людей
    - Double click -> сворачиваем обратно
    """
    
    # Подготовка данных отделов
    dept_nodes = []
    for node in G_depts.nodes():
        node_data = G_depts.nodes[node]
        dept_nodes.append({
            "id": f"dept_{node}",
            "original_id": node,
            "label": node_data.get("label", str(node)),
            "type": "dept",
            "size": node_data.get("size", 1),
            "members": node_data.get("members", []),
            "in_strength": metrics_depts.get("in_strength", {}).get(node, 0),
            "out_strength": metrics_depts.get("out_strength", {}).get(node, 0),
        })
    
    dept_edges = []
    for u, v, data in G_depts.edges(data=True):
        dept_edges.append({
            "source": f"dept_{u}",
            "target": f"dept_{v}",
            "weight": data.get("weight", 1),
            "people": data.get("people", 0),
        })
    
    # Подготовка данных людей
    people_nodes = []
    for node in G_people.nodes():
        node_data = G_people.nodes[node]
        people_nodes.append({
            "id": f"person_{node}",
            "original_id": node,
            "label": node_data.get("label", str(node)),
            "dept": node_data.get("dept", ""),
            "type": "person",
            "in_strength": metrics_people.get("in_strength", {}).get(node, 0),
            "out_strength": metrics_people.get("out_strength", {}).get(node, 0),
            "pagerank": metrics_people.get("pagerank", {}).get(node, 0),
        })
    
    people_edges = []
    for u, v, data in G_people.edges(data=True):
        people_edges.append({
            "source": f"person_{u}",
            "target": f"person_{v}",
            "weight": data.get("weight", 1),
            "msgs": data.get("msgs", 0),
        })
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #0a0e27;
                font-family: 'Segoe UI', Tahoma, sans-serif;
                overflow: hidden;
            }}
            #viz {{
                width: 100%;
                height: 100vh;
            }}
            .controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
            }}
            .btn {{
                background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
                color: white;
                border: none;
                padding: 8px 15px;
                margin: 2px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: 600;
                font-size: 12px;
            }}
            .btn:hover {{
                opacity: 0.8;
            }}
            .node {{
                cursor: pointer;
                stroke: #fff;
                stroke-width: 2px;
            }}
            .node.dept {{
                fill: #7b2cbf;
            }}
            .node.person {{
                fill: #00d4ff;
            }}
            .link {{
                stroke: #999;
                stroke-opacity: 0.4;
            }}
            .label {{
                fill: white;
                font-size: 11px;
                pointer-events: none;
                text-anchor: middle;
                text-shadow: 0 0 3px #000;
            }}
            #breadcrumb {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: #00d4ff;
                font-size: 16px;
                font-weight: bold;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
            }}
            #info {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                color: white;
                font-size: 12px;
                background: rgba(0, 0, 0, 0.7);
                padding: 10px;
                border-radius: 5px;
                max-width: 300px;
            }}
        </style>
    </head>
    <body>
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
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            const deptNodesData = {json.dumps(dept_nodes)};
            const deptLinksData = {json.dumps(dept_edges)};
            const peopleNodesData = {json.dumps(people_nodes)};
            const peopleLinksData = {json.dumps(people_edges)};
            
            let nodes = [...deptNodesData];
            let links = [...deptLinksData];
            let currentLevel = "depts";
            let expandedDept = null;
            
            const svg = d3.select("#viz")
                .attr("width", width)
                .attr("height", height);
            
            const g = svg.append("g");
            
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => {{
                    g.attr("transform", event.transform);
                }});
            
            svg.call(zoom);
            
            let linkElements, nodeElements, labels, simulation;
            
            function initSimulation() {{
                g.selectAll("*").remove();
                
                linkElements = g.append("g")
                    .selectAll("line")
                    .data(links)
                    .join("line")
                    .attr("class", "link")
                    .attr("stroke-width", d => Math.sqrt(d.weight) / 2);
                
                nodeElements = g.append("g")
                    .selectAll("circle")
                    .data(nodes)
                    .join("circle")
                    .attr("class", d => `node ${{d.type}}`)
                    .attr("r", d => {{
                        if (d.type === "dept") return Math.sqrt(d.size) * 5 + 10;
                        return 6;
                    }})
                    .on("click", handleNodeClick)
                    .on("dblclick", handleNodeDoubleClick)
                    .on("mouseover", showNodeInfo)
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                labels = g.append("g")
                    .selectAll("text")
                    .data(nodes)
                    .join("text")
                    .attr("class", "label")
                    .attr("dy", -10)
                    .text(d => d.label.length > 20 ? d.label.slice(0, 20) + "..." : d.label);
                
                simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links)
                        .id(d => d.id)
                        .distance(d => currentLevel === "depts" ? 150 : 80))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collision", d3.forceCollide().radius(d => {{
                        if (d.type === "dept") return Math.sqrt(d.size) * 5 + 15;
                        return 10;
                    }}))
                    .on("tick", ticked);
            }}
            
            function ticked() {{
                linkElements
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                nodeElements
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                
                labels
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            }}
            
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
            
            function handleNodeClick(event, d) {{
                event.stopPropagation();
                if (currentLevel === "depts" && d.type === "dept") {{
                    expandDept(d);
                }}
            }}
            
            function handleNodeDoubleClick(event, d) {{
                event.stopPropagation();
                if (currentLevel === "people") {{
                    collapseToDepts();
                }}
            }}
            
            function expandDept(deptNode) {{
                currentLevel = "people";
                expandedDept = deptNode.original_id;
                
                const members = deptNode.members || [];
                nodes = peopleNodesData.filter(n => members.includes(n.original_id));
                
                const memberIds = new Set(nodes.map(n => n.id));
                links = peopleLinksData.filter(l => 
                    memberIds.has(l.source) && memberIds.has(l.target)
                );
                
                document.getElementById("breadcrumb").textContent = 
                    `Уровень: ${{deptNode.label}} (double-click для возврата)`;
                
                simulation.stop();
                initSimulation();
            }}
            
            function collapseToDepts() {{
                currentLevel = "depts";
                expandedDept = null;
                nodes = [...deptNodesData];
                links = [...deptLinksData];
                
                document.getElementById("breadcrumb").textContent = "Уровень: Отделы";
                
                simulation.stop();
                initSimulation();
            }}
            
            function resetView() {{
                collapseToDepts();
            }}
            
            function resetZoom() {{
                svg.transition().duration(750).call(
                    zoom.transform, d3.zoomIdentity
                );
            }}
            
            let labelsVisible = true;
            function toggleLabels() {{
                labelsVisible = !labelsVisible;
                labels.style("opacity", labelsVisible ? 1 : 0);
            }}
            
            let physicsEnabled = true;
            function togglePhysics() {{
                physicsEnabled = !physicsEnabled;
                if (physicsEnabled) {{
                    simulation.alpha(0.3).restart();
                }} else {{
                    simulation.stop();
                }}
            }}
            
            function showNodeInfo(event, d) {{
                let info = `<strong>${{d.label}}</strong><br>`;
                if (d.type === "dept") {{
                    info += `Тип: Отдел<br>`;
                    info += `Сотрудников: ${{d.size}}<br>`;
                    info += `Входящие: ${{d.in_strength.toFixed(1)}}<br>`;
                    info += `Исходящие: ${{d.out_strength.toFixed(1)}}`;
                }} else {{
                    info += `Отдел: ${{d.dept}}<br>`;
                    info += `PageRank: ${{d.pagerank.toFixed(4)}}<br>`;
                    info += `Входящие: ${{d.in_strength.toFixed(1)}}<br>`;
                    info += `Исходящие: ${{d.out_strength.toFixed(1)}}`;
                }}
                document.getElementById("info").innerHTML = info;
            }}
            
            initSimulation();
        </script>
    </body>
    </html>
    """
    
    return html

# ========================= FORCE-DIRECTED ВИЗУАЛИЗАЦИЯ =========================

def create_force_d3_viz(G, metrics):
    """Force-directed визуализация для графа людей"""
    
    nodes_data = []
    for node in G.nodes():
        node_data = G.nodes[node]
        comm = metrics.get("communities", {}).get(node, 0)
        nodes_data.append({
            "id": str(node),
            "label": node_data.get("label", str(node)),
            "dept": node_data.get("dept", ""),
            "community": comm,
            "pagerank": metrics.get("pagerank", {}).get(node, 0),
            "in_strength": metrics.get("in_strength", {}).get(node, 0),
            "out_strength": metrics.get("out_strength", {}).get(node, 0),
        })
    
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "source": str(u),
            "target": str(v),
            "weight": data.get("weight", 1),
        })
    
    # Цветовая палитра для сообществ
    n_communities = len(set(metrics.get("communities", {}).values()))
    colors = ["#00d4ff", "#7b2cbf", "#ff006e", "#ffbe0b", "#8ac926", 
              "#ff006e", "#3a86ff", "#fb5607", "#06ffa5", "#8338ec"]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #0a0e27;
                font-family: sans-serif;
                overflow: hidden;
            }}
            #viz {{ width: 100%; height: 100vh; }}
            .node {{ cursor: pointer; stroke: #fff; stroke-width: 1.5px; }}
            .link {{ stroke: #999; stroke-opacity: 0.3; }}
            .label {{
                fill: white;
                font-size: 10px;
                pointer-events: none;
                text-anchor: middle;
                text-shadow: 0 0 3px #000;
            }}
            .controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
            }}
            .btn {{
                background: linear-gradient(90deg, #00d4ff, #7b2cbf);
                color: white;
                border: none;
                padding: 8px 15px;
                margin: 2px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 12px;
            }}
            #info {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                color: white;
                background: rgba(0,0,0,0.7);
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                max-width: 300px;
            }}
        </style>
    </head>
    <body>
        <div class="controls">
            <button class="btn" onclick="resetZoom()">🔍 Сбросить зум</button>
            <button class="btn" onclick="toggleLabels()">🏷️ Метки</button>
            <button class="btn" onclick="togglePhysics()">⚡ Физика</button>
        </div>
        <div id="info">Наведите на узел для информации</div>
        <svg id="viz"></svg>
        
        <script>
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            const nodes = {json.dumps(nodes_data)};
            const links = {json.dumps(edges_data)};
            const colors = {json.dumps(colors[:n_communities])};
            
            const svg = d3.select("#viz")
                .attr("width", width)
                .attr("height", height);
            
            const g = svg.append("g");
            
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => g.attr("transform", event.transform));
            
            svg.call(zoom);
            
            const linkElements = g.append("g")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.sqrt(d.weight) / 2);
            
            const nodeElements = g.append("g")
                .selectAll("circle")
                .data(nodes)
                .join("circle")
                .attr("class", "node")
                .attr("r", d => 3 + Math.sqrt(d.pagerank * 1000))
                .attr("fill", d => colors[d.community % colors.length])
                .on("mouseover", showInfo)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            const labels = g.append("g")
                .selectAll("text")
                .data(nodes)
                .join("text")
                .attr("class", "label")
                .attr("dy", -8)
                .text(d => d.label.length > 15 ? d.label.slice(0, 15) + "..." : d.label);
            
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(70))
                .force("charge", d3.forceManyBody().strength(-200))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(15))
                .on("tick", ticked);
            
            function ticked() {{
                linkElements
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                nodeElements
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                
                labels
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            }}
            
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
            
            function showInfo(event, d) {{
                document.getElementById("info").innerHTML = 
                    `<strong>${{d.label}}</strong><br>
                    Отдел: ${{d.dept}}<br>
                    PageRank: ${{d.pagerank.toFixed(4)}}<br>
                    Сообщество: ${{d.community}}<br>
                    Входящие: ${{d.in_strength.toFixed(1)}}<br>
                    Исходящие: ${{d.out_strength.toFixed(1)}}`;
            }}
            
            function resetZoom() {{
                svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
            }}
            
            let labelsVisible = true;
            function toggleLabels() {{
                labelsVisible = !labelsVisible;
                labels.style("opacity", labelsVisible ? 1 : 0);
            }}
            
            let physicsEnabled = true;
            function togglePhysics() {{
                physicsEnabled = !physicsEnabled;
                if (physicsEnabled) {{
                    simulation.alpha(0.3).restart();
                }} else {{
                    simulation.stop();
                }}
            }}
        </script>
    </body>
    </html>
    """
    
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
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    
    if isinstance(period, tuple):
        start_date, end_date = period
    else:
        start_date, end_date = period, period
    
    st.sidebar.markdown("---")
    
    values_list = sorted(df[COLS["value"]].dropna().unique().tolist())
    selected_values = st.sidebar.multiselect(
        "Ценности", options=values_list, default=values_list
    )
    
    # НОВОЕ: Диапазон меритов вместо минимума
    st.sidebar.markdown("### 💎 Мериты на связь")
    max_merits_possible = 1000
    
    merit_range = st.sidebar.slider(
        "Диапазон меритов",
        min_value=1,
        max_value=max_merits_possible,
        value=(1, max_merits_possible),
        step=1,
        help="Выберите минимальное и максимальное количество меритов для фильтрации связей"
    )
    
    st.sidebar.markdown(f"*Связи с {merit_range[0]} по {merit_range[1]} меритов*")
    
    allow_self = st.sidebar.checkbox("Самонаграждения", value=False)
    
    st.sidebar.markdown("---")
    show_social_stats = st.sidebar.checkbox(
        "📊 Показать продвинутую статистику", 
        value=True,
        help="Включает расчет и отображение глубоких социологических метрик"
    )
    
    return {
        "start": pd.to_datetime(start_date),
        "end": pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        "values": set(selected_values),
        "merit_range": merit_range,
        "allow_self": allow_self,
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
            <h1 style='font-size: 3rem;'>🕸️ СоциоГраф 5.0</h1>
            <p style='font-size: 1.2rem; color: #00d4ff;'>
                Иерархическая визуализация + Глубокая социальная аналитика
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Загружаем dataset.xlsx, который вшит/лежит рядом со скриптом
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
    
    # Метрики
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
    
    # Построение графов
    with st.spinner("🔄 Строим иерархическую структуру..."):
        G_people, G_depts, dept_members = build_hierarchical_graph(
            df_filtered, cfg["merit_range"], cfg["allow_self"]
        )
        
        if G_depts.number_of_nodes() == 0 or G_people.number_of_nodes() == 0:
            st.warning("⚠️ Граф пуст после применения фильтров")
            st.stop()
        
        metrics_depts = calculate_advanced_metrics(G_depts)
        metrics_people = calculate_advanced_metrics(G_people)
    
    st.markdown(f"""
        <div class='metric-card'>
            <strong>Граф:</strong> {G_depts.number_of_nodes()} отделов, 
            {G_people.number_of_nodes()} сотрудников, 
            {G_people.number_of_edges()} связей | 
            <strong>Модулярность:</strong> {metrics_people.get('modularity', 0):.3f} | 
            <strong>Взаимность:</strong> {metrics_people.get('reciprocity', 0):.3f}
        </div>
    """, unsafe_allow_html=True)
    
    # ВИЗУАЛИЗАЦИИ
    st.markdown("---")
    st.header("🗺️ Визуализации")
    
    tab1, tab2 = st.tabs(["🌐 Иерархическая сеть", "🌀 Force-Directed"])
    
    with tab1:
        st.markdown("""
        <div class='info-box'>
            <strong>🌐 Иерархическая интерактивная сеть</strong><br><br>
            <strong>Как использовать:</strong><br>
            🖱️ <strong>Клик на отдел</strong> - раскрывает его в людей этого отдела<br>
            🖱️ <strong>Double-click на человека</strong> - сворачивает обратно в отделы<br>
            🔍 <strong>Scroll</strong> - зум<br>
            ✋ <strong>Drag</strong> - перетащить узел<br>
            🏠 <strong>Кнопка "Домой"</strong> - вернуться к отделам<br><br>
            <strong>Особенности:</strong><br>
            ✅ Начальный вид - отделы компании<br>
            ✅ Размер отдела = количество сотрудников<br>
            ✅ При раскрытии показываются только люди выбранного отдела<br>
            ✅ Толщина связи = сила взаимодействия
        </div>
        """, unsafe_allow_html=True)
        
        html_hierarchical = create_hierarchical_d3_viz(
            G_depts, G_people, dept_members,
            metrics_depts, metrics_people
        )
        components.html(html_hierarchical, height=800, scrolling=False)
    
    with tab2:
        st.markdown("""
        <div class='info-box'>
            <strong>🌀 Force-Directed Layout</strong><br><br>
            Классическая физическая симуляция всех сотрудников:<br>
            • Узлы отталкиваются друг от друга<br>
            • Рёбра притягивают связанные узлы<br>
            • Цвет узла = сообщество (по алгоритму Louvain)<br>
            • Размер узла = PageRank (влиятельность)
        </div>
        """, unsafe_allow_html=True)
        
        html_force = create_force_d3_viz(G_people, metrics_people)
        components.html(html_force, height=700, scrolling=False)
    
    # СОЦИАЛЬНАЯ СТАТИСТИКА
    if cfg["show_social_stats"]:
        st.markdown("---")
        st.header("📊 Продвинутая социальная статистика")
        
        # Создаем датафрейм с метриками
        nodes_metrics = []
        for node in G_people.nodes():
            node_data = G_people.nodes[node]
            nodes_metrics.append({
                "id": node,
                "ФИО": node_data.get("label", ""),
                "Отдел": node_data.get("dept", ""),
                "PageRank": metrics_people["pagerank"].get(node, 0),
                "Betweenness": metrics_people["betweenness"].get(node, 0),
                "Closeness": metrics_people["closeness"].get(node, 0),
                "Clustering": metrics_people["clustering"].get(node, 0),
                "Eigenvector": metrics_people["eigenvector"].get(node, 0),
                "Constraint": metrics_people["constraint"].get(node, 0),
                "Core": metrics_people["core_number"].get(node, 0),
                "Bridge": metrics_people["is_bridge"].get(node, 0),
                "Load": metrics_people["load"].get(node, 0),
                "Triadic": metrics_people["triadic_closure"].get(node, 0),
                "DeptDiv": metrics_people["dept_diversity"].get(node, 0),
                "In": metrics_people["in_strength"].get(node, 0),
                "Out": metrics_people["out_strength"].get(node, 0),
            })
        df_metrics = pd.DataFrame(nodes_metrics)
        
        # Интерпретация метрик
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Интерпретация метрик")
            st.markdown("""
            <div class='metric-card'>
            <strong>PageRank</strong> - Влиятельность (кто важен в сети)<br>
            <strong>Betweenness</strong> - Посредничество (кто соединяет группы)<br>
            <strong>Closeness</strong> - Близость к центру (насколько быстро достичь других)<br>
            <strong>Clustering</strong> - Кластеризация (насколько связаны соседи)<br>
            <strong>Eigenvector</strong> - Влияние через связи (связан с влиятельными)<br>
            <strong>Constraint</strong> - Ограниченность (низкий = больше структурных дыр)<br>
            <strong>Core</strong> - K-core (к какому ядру принадлежит)<br>
            <strong>Bridge</strong> - Мост (соединяет сообщества)<br>
            <strong>Load</strong> - Нагрузка (через кого проходит много путей)<br>
            <strong>Triadic</strong> - Замыкание триад (% друзей, знакомых друг с другом)<br>
            <strong>DeptDiv</strong> - Разнообразие отделов в связях
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📈 Сетевые показатели")
            
            avg_clustering = df_metrics["Clustering"].mean()
            avg_constraint = df_metrics["Constraint"].mean()
            avg_triadic = df_metrics["Triadic"].mean()
            n_bridges = df_metrics["Bridge"].sum()
            max_core = df_metrics["Core"].max()
            
            st.markdown(f"""
            <div class='metric-card'>
            <strong>Средняя кластеризация:</strong> {avg_clustering:.3f}<br>
            {'✅ Высокая (> 0.3) - много триад' if avg_clustering > 0.3 else '⚠️ Низкая - разреженная сеть'}<br><br>
            
            <strong>Средний Constraint:</strong> {avg_constraint:.3f}<br>
            {'✅ Низкий (< 0.5) - много структурных дыр' if avg_constraint < 0.5 else '⚠️ Высокий - мало возможностей'}<br><br>
            
            <strong>Среднее триадное замыкание:</strong> {avg_triadic:.3f}<br>
            {'✅ Высокое (> 0.3) - плотные группы' if avg_triadic > 0.3 else '⚠️ Низкое - слабые группы'}<br><br>
            
            <strong>Мостов в сети:</strong> {int(n_bridges)}<br>
            <strong>Максимальный k-core:</strong> {int(max_core)}
            </div>
            """, unsafe_allow_html=True)
        
        # Топы
        st.markdown("### 🏆 Топ сотрудников по метрикам")
        
        tab_tops = st.tabs([
            "Opinion Leaders", "Brokers", "Gatekeepers", 
            "Influencers", "Diverse Networks", "Core Members"
        ])
        
        with tab_tops[0]:
            st.markdown("**Opinion Leaders (высокий PageRank)** - лидеры мнений")
            top = df_metrics.nlargest(15, "PageRank")[["ФИО", "Отдел", "PageRank", "In", "Out"]]
            st.dataframe(top, use_container_width=True, hide_index=True)
        
        with tab_tops[1]:
            st.markdown("**Brokers (высокий Betweenness)** - посредники между группами")
            top = df_metrics.nlargest(15, "Betweenness")[["ФИО", "Отдел", "Betweenness", "Bridge"]]
            st.dataframe(top, use_container_width=True, hide_index=True)
        
        with tab_tops[2]:
            st.markdown("**Gatekeepers (низкий Constraint)** - контроль структурных дыр")
            top = df_metrics.nsmallest(15, "Constraint")[["ФИО", "Отдел", "Constraint", "Betweenness"]]
            st.dataframe(top, use_container_width=True, hide_index=True)
        
        with tab_tops[3]:
            st.markdown("**Influencers (высокий Eigenvector)** - связаны с влиятельными")
            top = df_metrics.nlargest(15, "Eigenvector")[["ФИО", "Отдел", "Eigenvector", "PageRank"]]
            st.dataframe(top, use_container_width=True, hide_index=True)
        
        with tab_tops[4]:
            st.markdown("**Diverse Networks (высокий DeptDiv)** - разнообразие связей")
            top = df_metrics.nlargest(15, "DeptDiv")[["ФИО", "Отдел", "DeptDiv", "Out"]]
            st.dataframe(top, use_container_width=True, hide_index=True)
        
        with tab_tops[5]:
            st.markdown("**Core Members (высокий Core)** - ядро сети")
            top = df_metrics.nlargest(15, "Core")[["ФИО", "Отдел", "Core", "PageRank"]]
            st.dataframe(top, use_container_width=True, hide_index=True)
        
        # Статистика по отделам
        st.markdown("### 🏢 Статистика по отделам")
        
        dept_stats = []
        for node in G_depts.nodes():
            dept_stats.append({
                "Отдел": G_depts.nodes[node].get("label", ""),
                "Сотрудников": G_depts.nodes[node].get("size", 0),
                "Входящие": metrics_depts["in_strength"].get(node, 0),
                "Исходящие": metrics_depts["out_strength"].get(node, 0),
                "PageRank": metrics_depts["pagerank"].get(node, 0),
            })
        df_dept_stats = pd.DataFrame(dept_stats).sort_values("Сотрудников", ascending=False)
        st.dataframe(df_dept_stats, use_container_width=True, hide_index=True)
        
        # Полная таблица метрик
        st.markdown("### 📋 Полная таблица метрик")
        st.dataframe(df_metrics.sort_values("PageRank", ascending=False), 
                    use_container_width=True, hide_index=True, height=400)
    
    # Экспорт
    st.markdown("---")
    st.subheader("💾 Экспорт данных")
    
    if cfg["show_social_stats"]:
        col1, col2 = st.columns(2)
        with col1:
            csv = df_metrics.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 Скачать все метрики (CSV)",
                csv,
                "social_network_metrics.csv",
                "text/csv"
            )
        
        with col2:
            # Экспорт структуры графа
            graph_data = {
                "nodes": [
                    {
                        "id": str(n),
                        "label": G_people.nodes[n].get("label", ""),
                        "dept": G_people.nodes[n].get("dept", ""),
                        "pagerank": float(metrics_people["pagerank"].get(n, 0)),
                        "community": int(metrics_people["communities"].get(n, 0))
                    }
                    for n in G_people.nodes()
                ],
                "edges": [
                    {
                        "source": str(u),
                        "target": str(v),
                        "weight": float(data.get("weight", 1))
                    }
                    for u, v, data in G_people.edges(data=True)
                ],
                "stats": {
                    "modularity": float(metrics_people.get("modularity", 0)),
                    "reciprocity": float(metrics_people.get("reciprocity", 0)),
                    "n_communities": len(set(metrics_people.get("communities", {}).values()))
                }
            }
            json_str = json.dumps(graph_data, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Скачать граф (JSON)",
                json_str,
                "network_graph.json",
                "application/json"
            )

if __name__ == "__main__":
    main()