# -*- coding: utf-8 -*-
"""
Визуализации графа (D3.js) для СоциоГраф 7.0.
Вынесено из app.py отдельным модулем. Чистые функции: получают граф и метрики,
возвращают HTML-строку для streamlit.components.v1.html.
"""
import json

PALETTE = ["#58a6ff","#f78166","#3fb950","#d2a8ff","#ffa657","#79c0ff","#ff7b72","#56d364",
           "#bc8cff","#e3b341","#39d353","#ff9b8c","#a5d6ff","#cfba7c","#7ee787","#f0883e",
           "#a371f7","#2ea043","#db61a2","#6cb6ff","#c69026","#8ddb88","#ec6547","#b083f0"]


def social_graph_html(G, mt):
    """Force-directed граф всех сотрудников. Цвет — сообщество, размер — охват, зелёные рёбра — взаимные."""
    comm = mt.get("communities", {})
    pr = mt.get("pagerank", {})
    nodes = [dict(id=str(n), label=G.nodes[n].get("label", str(n)), dept=G.nodes[n].get("dept", ""),
                  company=G.nodes[n].get("company", ""), position=G.nodes[n].get("position", ""),
                  community=comm.get(n, 0), pagerank=pr.get(n, 0),
                  ins=mt.get("in_strength", {}).get(n, 0), outs=mt.get("out_strength", {}).get(n, 0))
             for n in G.nodes()]
    edges = [dict(source=str(u), target=str(v), weight=d.get("weight", 1), mutual=bool(d.get("mutual", False)))
             for u, v, d in G.edges(data=True)]
    n_comm = max(1, len(set(comm.values())))
    colors = (PALETTE * (n_comm // len(PALETTE) + 1))[:max(n_comm, 1)]
    return _social_template(nodes, edges, colors)


def _social_template(nodes, edges, colors):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
 body{{margin:0;background:#0d1117;font-family:'IBM Plex Sans',sans-serif;overflow:hidden}}
 #viz{{width:100%;height:100vh}}
 .controls{{position:absolute;top:12px;right:12px;z-index:1000;display:flex;gap:6px}}
 .btn{{background:#161b22;color:#58a6ff;border:1px solid #30363d;padding:6px 14px;border-radius:5px;cursor:pointer;font-size:12px;font-family:'IBM Plex Mono',monospace}}
 .btn:hover{{background:#21262d}}
 .label{{fill:#8b949e;font-size:10px;pointer-events:none;text-anchor:middle}}
 #tip{{position:absolute;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:10px 14px;border-radius:6px;font-size:12px;pointer-events:none;opacity:0;transition:opacity .2s;max-width:230px;line-height:1.6}}
</style></head><body>
<div class="controls">
 <button class="btn" onclick="rz()">↺ Сброс</button>
 <button class="btn" onclick="tl()">Метки</button>
 <button class="btn" onclick="tm()">Только взаимные</button>
 <button class="btn" onclick="tp()">Физика</button>
</div>
<div id="tip"></div><svg id="viz"></svg>
<script>
const W=innerWidth,H=innerHeight;
const nodes={json.dumps(nodes)},allLinks={json.dumps(edges)},colors={json.dumps(colors)};
let links=allLinks.slice();
const svg=d3.select("#viz").attr("width",W).attr("height",H);
const g=svg.append("g");
const zoom=d3.zoom().scaleExtent([.05,12]).on("zoom",e=>g.attr("transform",e.transform));
svg.call(zoom);
let link=g.append("g");
const pr=nodes.map(n=>n.pagerank),mn=Math.min(...pr),mx=Math.max(...pr),rg=(mx-mn)||1;
function draw(){{
 link.selectAll("line").remove();
 window._le=link.selectAll("line").data(links).join("line")
   .attr("stroke",d=>d.mutual?"#3fb950":"#30363d").attr("stroke-opacity",d=>d.mutual?.7:.45)
   .attr("stroke-width",d=>Math.sqrt(d.weight)*.6+.3);
}}
draw();
const node=g.append("g").selectAll("circle").data(nodes).join("circle")
 .attr("r",d=>4+12*(d.pagerank-mn)/rg)
 .attr("fill",d=>colors[d.community%colors.length]).attr("stroke","#0d1117").attr("stroke-width",1.5)
 .attr("cursor","pointer")
 .on("mouseover",(e,d)=>{{const t=document.getElementById("tip");
   t.innerHTML=`<strong>${{d.label}}</strong><br>${{d.position}}<br><span style="color:#58a6ff">${{d.company}}</span> / ${{d.dept}}<hr style="border-color:#30363d;margin:6px 0">Входящих: ${{d.ins.toFixed(0)}} · Исходящих: ${{d.outs.toFixed(0)}}`;
   t.style.opacity=1;t.style.left=(e.pageX+12)+"px";t.style.top=(e.pageY-10)+"px";}})
 .on("mouseout",()=>document.getElementById("tip").style.opacity=0)
 .call(d3.drag().on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}})
   .on("drag",(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
   .on("end",(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));
const labels=g.append("g").selectAll("text").data(nodes).join("text").attr("class","label").attr("dy",-8)
 .text(d=>d.label.length>18?d.label.slice(0,18)+"…":d.label);
const sim=d3.forceSimulation(nodes)
 .force("link",d3.forceLink(links).id(d=>d.id).distance(70))
 .force("charge",d3.forceManyBody().strength(-180))
 .force("center",d3.forceCenter(W/2,H/2))
 .force("collision",d3.forceCollide().radius(14))
 .on("tick",()=>{{window._le.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
   node.attr("cx",d=>d.x).attr("cy",d=>d.y);labels.attr("x",d=>d.x).attr("y",d=>d.y);}});
function rz(){{svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity);}}
let lo=true;function tl(){{lo=!lo;labels.style("opacity",lo?1:0);}}
let mo=false;function tm(){{mo=!mo;links=mo?allLinks.filter(l=>l.mutual):allLinks.slice();
  sim.force("link",d3.forceLink(links).id(d=>d.id).distance(70));draw();sim.alpha(.3).restart();}}
let po=true;function tp(){{po=!po;po?sim.alpha(.3).restart():sim.stop();}}
</script></body></html>"""


def functional_html(Gg, G, gmembers, mtg, mtp):
    """Иерархия: группы → люди. Клик на группу раскрывает её сотрудников, двойной клик — назад."""
    gn = [dict(id=f"g_{x}", original_id=x, label=Gg.nodes[x].get("label", str(x)), type="group",
               size=Gg.nodes[x].get("size", 1), members=Gg.nodes[x].get("members", []),
               ins=mtg.get("in_strength", {}).get(x, 0), outs=mtg.get("out_strength", {}).get(x, 0)) for x in Gg.nodes()]
    ge = [dict(source=f"g_{u}", target=f"g_{v}", weight=d.get("weight", 1)) for u, v, d in Gg.edges(data=True)]
    pn = [dict(id=f"p_{x}", original_id=x, label=G.nodes[x].get("label", str(x)), dept=G.nodes[x].get("dept", ""),
               company=G.nodes[x].get("company", ""), position=G.nodes[x].get("position", ""), type="person",
               ins=mtp.get("in_strength", {}).get(x, 0), outs=mtp.get("out_strength", {}).get(x, 0)) for x in G.nodes()]
    pe = [dict(source=f"p_{u}", target=f"p_{v}", weight=d.get("weight", 1)) for u, v, d in G.edges(data=True)]
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
 body{{margin:0;background:#0d1117;font-family:'IBM Plex Sans',sans-serif;overflow:hidden}}
 #viz{{width:100%;height:100vh}}
 .controls{{position:absolute;top:12px;right:12px;display:flex;gap:6px;z-index:1000}}
 .btn{{background:#161b22;color:#58a6ff;border:1px solid #30363d;padding:6px 14px;border-radius:5px;cursor:pointer;font-size:12px;font-family:'IBM Plex Mono',monospace}}
 .btn:hover{{background:#21262d}}
 #bc{{position:absolute;top:14px;left:14px;color:#58a6ff;font-family:'IBM Plex Mono',monospace;font-size:13px}}
 #tip{{position:absolute;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:10px 14px;border-radius:6px;font-size:12px;pointer-events:none;opacity:0;max-width:230px;line-height:1.6}}
</style></head><body>
<div id="bc">Уровень: Группы</div>
<div class="controls"><button class="btn" onclick="home()">↺ Домой</button>
<button class="btn" onclick="rz()">⊕ Сброс</button><button class="btn" onclick="tp()">Физика</button></div>
<div id="tip"></div><svg id="viz"></svg>
<script>
const W=innerWidth,H=innerHeight;
const GN={json.dumps(gn)},GE={json.dumps(ge)},PN={json.dumps(pn)},PE={json.dumps(pe)};
let nodes=[...GN],links=[...GE],level="g",sim;
const svg=d3.select("#viz").attr("width",W).attr("height",H);const g=svg.append("g");
const zoom=d3.zoom().scaleExtent([.05,12]).on("zoom",e=>g.attr("transform",e.transform));svg.call(zoom);
let le,ne,la;
function init(){{
 g.selectAll("*").remove();
 le=g.append("g").selectAll("line").data(links).join("line").attr("stroke","#30363d").attr("stroke-opacity",.7).attr("stroke-width",d=>Math.sqrt(d.weight)*.5+.5);
 ne=g.append("g").selectAll("circle").data(nodes).join("circle")
  .attr("r",d=>d.type==="group"?Math.sqrt(d.size)*4+10:6)
  .attr("fill",d=>d.type==="group"?"#58a6ff":"#3fb950").attr("stroke","#0d1117").attr("stroke-width",2).attr("cursor","pointer")
  .on("click",(e,d)=>{{if(level==="g"&&d.type==="group")expand(d);}})
  .on("dblclick",()=>{{if(level==="p")home();}})
  .on("mouseover",(e,d)=>{{const t=document.getElementById("tip");
    t.innerHTML=d.type==="group"?`<strong>${{d.label}}</strong><br>Участников: ${{d.size}}<br>Входящих: ${{d.ins.toFixed(0)}} · Исходящих: ${{d.outs.toFixed(0)}}<br><em style="color:#8b949e">Клик — раскрыть</em>`:`<strong>${{d.label}}</strong><br>${{d.position}}<br><span style="color:#58a6ff">${{d.company}}</span> / ${{d.dept}}<br><em style="color:#8b949e">Двойной клик — назад</em>`;
    t.style.opacity=1;t.style.left=(e.pageX+12)+"px";t.style.top=(e.pageY-10)+"px";}})
  .on("mouseout",()=>document.getElementById("tip").style.opacity=0)
  .call(d3.drag().on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}})
   .on("drag",(e,d)=>{{d.fx=e.x;d.fy=e.y;}}).on("end",(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));
 la=g.append("g").selectAll("text").data(nodes).join("text").attr("fill","#8b949e").attr("font-size","10px").attr("text-anchor","middle").attr("dy",-10).attr("pointer-events","none").text(d=>d.label&&d.label.length>20?d.label.slice(0,20)+"…":d.label);
 if(sim)sim.stop();
 sim=d3.forceSimulation(nodes).force("link",d3.forceLink(links).id(d=>d.id).distance(level==="g"?160:80))
  .force("charge",d3.forceManyBody().strength(-280)).force("center",d3.forceCenter(W/2,H/2))
  .force("collision",d3.forceCollide().radius(d=>d.type==="group"?Math.sqrt(d.size)*4+15:12))
  .on("tick",()=>{{le.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
    ne.attr("cx",d=>d.x).attr("cy",d=>d.y);la.attr("x",d=>d.x).attr("y",d=>d.y);}});
}}
function expand(gnode){{level="p";const mem=gnode.members||[];nodes=PN.filter(n=>mem.includes(n.original_id));
 const ids=new Set(nodes.map(n=>n.id));links=PE.filter(l=>ids.has(l.source)&&ids.has(l.target));
 document.getElementById("bc").textContent=`Уровень: ${{gnode.label}} (двойной клик — назад)`;sim.stop();init();}}
function home(){{level="g";nodes=[...GN];links=[...GE];document.getElementById("bc").textContent="Уровень: Группы";sim.stop();init();}}
function rz(){{svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity);}}
let po=true;function tp(){{po=!po;po?sim.alpha(.3).restart():sim.stop();}}
init();
</script></body></html>"""
