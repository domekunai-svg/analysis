# -*- coding: utf-8 -*-
"""
Визуализации графа (D3.js) для СоциоГраф.
  • Социальный граф: направленные рёбра (стрелки) по умолчанию; кнопка «Только взаимные»;
    клик по узлу подсвечивает его связи; усиленный контраст размеров узлов.
  • Иерархия: Компании → Отделы → Люди (+ режим «Все отделы ГК»).
"""
import json

PALETTE = ["#58a6ff","#f78166","#3fb950","#d2a8ff","#ffa657","#79c0ff","#ff7b72","#56d364",
           "#bc8cff","#e3b341","#39d353","#ff9b8c","#a5d6ff","#cfba7c","#7ee787","#f0883e",
           "#a371f7","#2ea043","#db61a2","#6cb6ff","#c69026","#8ddb88","#ec6547","#b083f0"]


def social_graph_html(G, mt):
    comm = mt.get("communities", {}); pr = mt.get("pagerank", {})
    nodes = [dict(id=str(n), label=G.nodes[n].get("label", str(n)), dept=G.nodes[n].get("dept", ""),
                  company=G.nodes[n].get("company", ""), position=G.nodes[n].get("position", ""),
                  community=comm.get(n, 0), pagerank=pr.get(n, 0),
                  ins=mt.get("in_strength", {}).get(n, 0), outs=mt.get("out_strength", {}).get(n, 0))
             for n in G.nodes()]
    edges = [dict(source=str(u), target=str(v), weight=d.get("weight", 1), mutual=bool(d.get("mutual", False)))
             for u, v, d in G.edges(data=True)]
    n_comm = max(1, len(set(comm.values())))
    colors = (PALETTE * (n_comm // len(PALETTE) + 1))[:max(n_comm, 1)]
    return _SOCIAL.replace("__NODES__", json.dumps(nodes)).replace("__LINKS__", json.dumps(edges)).replace("__COLORS__", json.dumps(colors))


def hierarchy_html(G_people, mt_people, emp_lookup):
    """Компании → Отделы → Люди. emp_lookup не нужен — берём атрибуты узлов графа людей."""
    pr = mt_people.get("pagerank", {})
    people = [dict(id=str(n), label=G_people.nodes[n].get("label", str(n)),
                   dept=G_people.nodes[n].get("dept", ""), company=G_people.nodes[n].get("company", ""),
                   deptkey=(str(G_people.nodes[n].get("company", "")) + " / " + str(G_people.nodes[n].get("dept", ""))),
                   position=G_people.nodes[n].get("position", ""), pagerank=pr.get(n, 0),
                   ins=mt_people.get("in_strength", {}).get(n, 0), outs=mt_people.get("out_strength", {}).get(n, 0))
              for n in G_people.nodes()]
    pe = [dict(source=str(u), target=str(v), weight=d.get("weight", 1)) for u, v, d in G_people.edges(data=True)]
    return _HIER.replace("__PEOPLE__", json.dumps(people)).replace("__PEDGES__", json.dumps(pe))


# ─────────────────────────── СОЦИАЛЬНЫЙ ГРАФ ───────────────────────────
_SOCIAL = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
 body{margin:0;background:#0d1117;font-family:'IBM Plex Sans',sans-serif;overflow:hidden}
 #viz{width:100%;height:100vh}
 .controls{position:absolute;top:12px;right:12px;z-index:1000;display:flex;gap:6px}
 .btn{background:#161b22;color:#58a6ff;border:1px solid #30363d;padding:6px 14px;border-radius:5px;cursor:pointer;font-size:12px;font-family:'IBM Plex Mono',monospace}
 .btn:hover{background:#21262d} .btn.on{background:#58a6ff;color:#0d1117}
 .label{fill:#8b949e;font-size:10px;pointer-events:none;text-anchor:middle}
 #tip{position:absolute;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:10px 14px;border-radius:6px;font-size:12px;pointer-events:none;opacity:0;transition:opacity .2s;max-width:230px;line-height:1.6}
</style></head><body>
<div class="controls">
 <button class="btn" onclick="rz()">↺ Сброс</button>
 <button class="btn" onclick="tl()">Метки</button>
 <button class="btn" id="mb" onclick="tm()">Только взаимные</button>
 <button class="btn" onclick="tp()">Физика</button>
</div>
<div id="tip"></div><svg id="viz">
<defs>
 <marker id="arr" viewBox="0 -5 10 10" refX="18" refY="0" markerWidth="6" markerHeight="6" orient="auto">
  <path d="M0,-4L8,0L0,4" fill="#5a6573"></path></marker>
 <marker id="arrm" viewBox="0 -5 10 10" refX="18" refY="0" markerWidth="6" markerHeight="6" orient="auto">
  <path d="M0,-4L8,0L0,4" fill="#3fb950"></path></marker>
</defs></svg>
<script>
const W=innerWidth,H=innerHeight;
const nodes=__NODES__,allLinks=__LINKS__,colors=__COLORS__;
let links=allLinks.slice(), mutualOnly=false, selected=null;
const svg=d3.select("#viz").attr("width",W).attr("height",H);
const g=svg.append("g");
const zoom=d3.zoom().scaleExtent([.05,12]).on("zoom",e=>g.attr("transform",e.transform));
svg.call(zoom);
svg.on("click",()=>{selected=null;refresh();});
// степень узла для связности при подсветке
const adj={}; nodes.forEach(n=>adj[n.id]=new Set());
allLinks.forEach(l=>{adj[l.source].add(l.target);adj[l.target].add(l.source);});
const pr=nodes.map(n=>n.pagerank),mn=Math.min(...pr),mx=Math.max(...pr),rg=(mx-mn)||1;
const R=d=>3+27*Math.pow((d.pagerank-mn)/rg,0.7);   // усиленный контраст
let link=g.append("g"), node, labels;
function drawLinks(){
 link.selectAll("line").remove();
 window._le=link.selectAll("line").data(links).join("line")
   .attr("stroke",d=>d.mutual?"#3fb950":"#5a6573").attr("stroke-opacity",.5)
   .attr("stroke-width",d=>Math.sqrt(d.weight)*.6+.4)
   .attr("marker-end",d=>d.mutual?"url(#arrm)":"url(#arr)");
}
drawLinks();
node=g.append("g").selectAll("circle").data(nodes).join("circle")
 .attr("r",R).attr("fill",d=>colors[d.community%colors.length]).attr("stroke","#0d1117").attr("stroke-width",1.5).attr("cursor","pointer")
 .on("click",(e,d)=>{e.stopPropagation();selected=(selected===d.id?null:d.id);refresh();})
 .on("mouseover",(e,d)=>{const t=document.getElementById("tip");
   t.innerHTML=`<strong>${d.label}</strong><br>${d.position}<br><span style="color:#58a6ff">${d.company}</span> / ${d.dept}<hr style="border-color:#30363d;margin:6px 0">Входящих: ${d.ins.toFixed(0)} · Исходящих: ${d.outs.toFixed(0)}<br><em style="color:#8b949e">клик — подсветить связи</em>`;
   t.style.opacity=1;t.style.left=(e.pageX+12)+"px";t.style.top=(e.pageY-10)+"px";})
 .on("mouseout",()=>document.getElementById("tip").style.opacity=0)
 .call(d3.drag().on("start",(e,d)=>{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;})
   .on("drag",(e,d)=>{d.fx=e.x;d.fy=e.y;}).on("end",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));
labels=g.append("g").selectAll("text").data(nodes).join("text").attr("class","label").attr("dy",d=>-R(d)-3)
 .text(d=>d.label.length>18?d.label.slice(0,18)+"…":d.label);
function refresh(){
 if(!selected){node.attr("opacity",1);labels.style("opacity",labelsOn?1:0);window._le.attr("stroke-opacity",.5);return;}
 const near=adj[selected];
 node.attr("opacity",d=>(d.id===selected||near.has(d.id))?1:0.12);
 labels.style("opacity",d=>(labelsOn&&(d.id===selected||near.has(d.id)))?1:0);
 window._le.attr("stroke-opacity",l=>(l.source.id===selected||l.target.id===selected)?0.95:0.05)
           .attr("stroke",l=>(l.source.id===selected||l.target.id===selected)?"#e3b341":(l.mutual?"#3fb950":"#5a6573"));
}
const sim=d3.forceSimulation(nodes)
 .force("link",d3.forceLink(links).id(d=>d.id).distance(70))
 .force("charge",d3.forceManyBody().strength(-200))
 .force("center",d3.forceCenter(W/2,H/2))
 .force("collision",d3.forceCollide().radius(d=>R(d)+3))
 .on("tick",()=>{window._le.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
   node.attr("cx",d=>d.x).attr("cy",d=>d.y);labels.attr("x",d=>d.x).attr("y",d=>d.y);});
function rz(){svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity);}
let labelsOn=true;function tl(){labelsOn=!labelsOn;refresh();}
function tm(){mutualOnly=!mutualOnly;document.getElementById("mb").classList.toggle("on",mutualOnly);
  links=mutualOnly?allLinks.filter(l=>l.mutual):allLinks.slice();
  sim.force("link",d3.forceLink(links).id(d=>d.id).distance(70));drawLinks();refresh();sim.alpha(.3).restart();}
let po=true;function tp(){po=!po;po?sim.alpha(.3).restart():sim.stop();}
</script></body></html>"""


# ─────────────────────────── ИЕРАРХИЯ Компании→Отделы→Люди ───────────────────────────
_HIER = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
 body{margin:0;background:#0d1117;font-family:'IBM Plex Sans',sans-serif;overflow:hidden}
 #viz{width:100%;height:100vh}
 .controls{position:absolute;top:12px;right:12px;display:flex;gap:6px;z-index:1000}
 .btn{background:#161b22;color:#58a6ff;border:1px solid #30363d;padding:6px 14px;border-radius:5px;cursor:pointer;font-size:12px;font-family:'IBM Plex Mono',monospace}
 .btn:hover{background:#21262d}
 #bc{position:absolute;top:14px;left:14px;color:#58a6ff;font-family:'IBM Plex Mono',monospace;font-size:13px}
 #tip{position:absolute;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:10px 14px;border-radius:6px;font-size:12px;pointer-events:none;opacity:0;max-width:240px;line-height:1.6}
</style></head><body>
<div id="bc">Уровень: Компании</div>
<div class="controls">
 <button class="btn" onclick="goCompanies()">↺ Компании</button>
 <button class="btn" onclick="allDepts()">Все отделы ГК</button>
 <button class="btn" onclick="rz()">⊕ Сброс</button>
 <button class="btn" onclick="tp()">Физика</button>
</div>
<div id="tip"></div><svg id="viz"></svg>
<script>
const W=innerWidth,H=innerHeight;
const PEOPLE=__PEOPLE__, PE=__PEDGES__;
// агрегаты
function aggregate(keyFn){
 const groups={}; PEOPLE.forEach(p=>{const k=keyFn(p);(groups[k]=groups[k]||[]).push(p);});
 const nodes=Object.entries(groups).map(([k,mem])=>({id:k,label:k,size:mem.length,members:new Set(mem.map(m=>m.id))}));
 const idx={}; nodes.forEach(n=>idx[n.id]=n);
 const ew={};
 PE.forEach(l=>{const a=byId[l.source],b=byId[l.target];if(!a||!b)return;const ka=keyFn(a),kb=keyFn(b);if(ka===kb)return;
   const key=ka+"||"+kb; ew[key]=(ew[key]||0)+l.weight;});
 const edges=Object.entries(ew).map(([k,w])=>{const[s,t]=k.split("||");return{source:s,target:t,weight:w};});
 return {nodes,edges};
}
const byId={}; PEOPLE.forEach(p=>byId[p.id]=p);
let nodes=[],links=[],level="companies",sim;
const svg=d3.select("#viz").attr("width",W).attr("height",H);const g=svg.append("g");
const zoom=d3.zoom().scaleExtent([.05,12]).on("zoom",e=>g.attr("transform",e.transform));svg.call(zoom);
let le,ne,la;
function setData(nd,lk,lvl,crumb){nodes=nd;links=lk;level=lvl;document.getElementById("bc").textContent=crumb;sim&&sim.stop();init();}
function goCompanies(){const a=aggregate(p=>p.company||"—");setData(a.nodes,a.edges,"companies","Уровень: Компании (клик — отделы)");}
function allDepts(){const a=aggregate(p=>p.deptkey||"—");setData(a.nodes,a.edges,"alldepts","Уровень: Все отделы ГК");}
function deptsOf(company){
 const sub=PEOPLE.filter(p=>(p.company||"—")===company);const ids=new Set(sub.map(p=>p.id));
 const groups={};sub.forEach(p=>{(groups[p.deptkey]=groups[p.deptkey]||[]).push(p);});
 const nd=Object.entries(groups).map(([k,mem])=>({id:k,label:k.split(" / ").pop(),size:mem.length,members:new Set(mem.map(m=>m.id)),company:company}));
 const ew={};PE.forEach(l=>{if(!ids.has(l.source)||!ids.has(l.target))return;const a=byId[l.source],b=byId[l.target];if(a.deptkey===b.deptkey)return;const key=a.deptkey+"||"+b.deptkey;ew[key]=(ew[key]||0)+l.weight;});
 const ed=Object.entries(ew).map(([k,w])=>{const[s,t]=k.split("||");return{source:s,target:t,weight:w};});
 setData(nd,ed,"depts","Компания: "+company+" → отделы (клик — люди · двойной клик — назад)");
}
function peopleOf(deptkey){
 const sub=PEOPLE.filter(p=>p.deptkey===deptkey);const ids=new Set(sub.map(p=>p.id));
 const nd=sub.map(p=>({id:p.id,label:p.label,size:1,person:true,position:p.position,company:p.company,dept:p.dept,ins:p.ins,outs:p.outs}));
 const ed=PE.filter(l=>ids.has(l.source)&&ids.has(l.target));
 setData(nd,ed,"people","Отдел: "+deptkey+" (двойной клик — назад)");
}
function init(){
 g.selectAll("*").remove();
 le=g.append("g").selectAll("line").data(links).join("line").attr("stroke","#30363d").attr("stroke-opacity",.6).attr("stroke-width",d=>Math.sqrt(d.weight)*.4+.5);
 const mx=Math.max(1,...nodes.map(n=>n.size));
 ne=g.append("g").selectAll("circle").data(nodes).join("circle")
  .attr("r",d=>d.person?6:8+22*Math.sqrt(d.size/mx))
  .attr("fill",d=>d.person?"#3fb950":(level==="companies"?"#58a6ff":"#a371f7")).attr("stroke","#0d1117").attr("stroke-width",2).attr("cursor","pointer")
  .on("click",(e,d)=>{if(level==="companies")deptsOf(d.id);else if(level==="depts"||level==="alldepts")peopleOf(d.id);})
  .on("dblclick",()=>{if(level==="people")goCompanies();else if(level==="depts")goCompanies();})
  .on("mouseover",(e,d)=>{const t=document.getElementById("tip");
    t.innerHTML=d.person?`<strong>${d.label}</strong><br>${d.position}<br><span style="color:#58a6ff">${d.company}</span> / ${d.dept}<br>Входящих: ${d.ins.toFixed(0)} · Исходящих: ${d.outs.toFixed(0)}`:`<strong>${d.label}</strong><br>Участников: ${d.size}<br><em style="color:#8b949e">клик — раскрыть</em>`;
    t.style.opacity=1;t.style.left=(e.pageX+12)+"px";t.style.top=(e.pageY-10)+"px";})
  .on("mouseout",()=>document.getElementById("tip").style.opacity=0)
  .call(d3.drag().on("start",(e,d)=>{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;})
   .on("drag",(e,d)=>{d.fx=e.x;d.fy=e.y;}).on("end",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));
 la=g.append("g").selectAll("text").data(nodes).join("text").attr("fill","#8b949e").attr("font-size","10px").attr("text-anchor","middle").attr("dy",-12).attr("pointer-events","none").text(d=>d.label&&d.label.length>22?d.label.slice(0,22)+"…":d.label);
 sim=d3.forceSimulation(nodes).force("link",d3.forceLink(links).id(d=>d.id).distance(level==="people"?70:150))
  .force("charge",d3.forceManyBody().strength(level==="people"?-160:-360)).force("center",d3.forceCenter(W/2,H/2))
  .force("collision",d3.forceCollide().radius(d=>(d.person?12:8+22*Math.sqrt(d.size/mx))+6))
  .on("tick",()=>{le.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
    ne.attr("cx",d=>d.x).attr("cy",d=>d.y);la.attr("x",d=>d.x).attr("y",d=>d.y);});
}
function rz(){svg.transition().duration(600).call(zoom.transform,d3.zoomIdentity);}
let po=true;function tp(){po=!po;po?sim.alpha(.3).restart():sim.stop();}
goCompanies();
</script></body></html>"""
