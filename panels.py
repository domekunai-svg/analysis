# -*- coding: utf-8 -*-
"""
Панели дашборда: рейтинг охвата, меритпаспорт, дневной пульс с ФИО,
drill «ценность → кто её сформировал». Тёплая палитра, тема светлая/тёмная (theme.py).
"""
import pandas as pd
import streamlit as st
import theme
try:
    import plotly.graph_objects as go
except Exception:
    go = None

EMP = {"last": "Фамилия", "first": "Имя", "id": "Персональный номер",
       "pos": "Должность", "company": "Компания", "dept": "Отдел"}
TX = {"date": "Дата", "sid": "Номер отправителя", "rid": "Номер получателя",
      "value": "Ценность", "merits": "Мериты", "comment": "Комментарий"}

CORAL = "#e95f3e"; OLIVE = "#6b8e23"; AMBER = "#cf8b22"
WARM = ["#e95f3e", "#5e7d16", "#c9871f", "#c0492f", "#8a9a3f", "#b5743a",
        "#6b8e23", "#d98a5a", "#9aa83f", "#a85d3a", "#e0a23e", "#90ae3c", "#cf8b22", "#8a6d3b"]


def _light(fig, title, height):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=44, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=theme.INK, family="Golos Text, sans-serif"), title=title)
    fig.update_xaxes(gridcolor=theme.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False)
    return fig


def _emap(emp):
    m = emp.set_index(EMP["id"])
    return m[~m.index.duplicated(keep="first")]


def _fio(m, pid):
    if pid in m.index:
        r = m.loc[pid]
        return f"{r.get(EMP['last'],'')} {r.get(EMP['first'],'')}".strip()
    return str(pid)


# ─────────────────────────── ДНЕВНОЙ ПУЛЬС с ФИО ───────────────────────────
def daily_pulse_fig(fd, emp):
    if go is None:
        return None
    m = _emap(emp)
    a = fd.dropna(subset=["dt"]).copy()
    a["d"] = a["dt"].dt.normalize()
    daily = a.groupby("d").agg(acts=(TX["merits"], "size")).reset_index()
    names_by_day = {}
    for d, sub in a.groupby("d"):
        top = sub.groupby(TX["sid"]).size().sort_values(ascending=False).head(6)
        names = [f"{_fio(m, pid)} ({c})" for pid, c in top.items()]
        extra = sub[TX["sid"]].nunique() - len(names)
        names_by_day[d] = "<br>".join(names) + (f"<br>…и ещё {extra}" if extra > 0 else "")
    daily["names"] = daily["d"].map(names_by_day)
    fig = go.Figure(go.Bar(
        x=daily["d"], y=daily["acts"], customdata=daily["names"], marker_color=CORAL,
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Благодарностей: %{y}<br><br>"
                      "<b>Кто благодарил:</b><br>%{customdata}<extra></extra>"))
    _light(fig, "Пульс по дням — наведите на столбик, чтобы увидеть, кто благодарил в этот день", 320)
    fig.update_layout(bargap=0.1)
    return fig


# ─────────────────────────── ЦЕННОСТЬ → КТО ЕЁ СФОРМИРОВАЛ ───────────────────────────
def render_value_people(fd, emp):
    vals = sorted(fd[TX["value"]].dropna().unique().tolist())
    if not vals:
        return
    v = st.selectbox("Ценность — кто её формирует (отправители)", vals, index=None, placeholder="выберите ценность…")
    if not v:
        return
    m = _emap(emp)
    sub = fd[fd[TX["value"]] == v]
    top = (sub.groupby(TX["sid"]).agg(Голосов=(TX["merits"], "sum"), Карточек=(TX["merits"], "size"))
           .reset_index().sort_values("Голосов", ascending=False).head(25))
    rows = [{"ФИО": _fio(m, r[TX["sid"]]), "Отдел": m[EMP["dept"]].get(r[TX["sid"]], ""),
             "Голосов": int(r["Голосов"]), "Карточек": int(r["Карточек"])} for _, r in top.iterrows()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────── РЕЙТИНГ ОХВАТА ───────────────────────────
def render_rating(fd, emp, N=20):
    m = _emap(emp)

    def table(kind):
        if kind == "merits_in":
            s = fd.groupby(TX["rid"])[TX["merits"]].sum()
        elif kind == "merits_out":
            s = fd.groupby(TX["sid"])[TX["merits"]].sum()
        elif kind == "reach_in":
            s = fd.groupby(TX["rid"])[TX["sid"]].nunique()
        else:
            s = fd.groupby(TX["sid"])[TX["rid"]].nunique()
        s = s.sort_values(ascending=False).head(N)
        col = {"merits_in": "Получено голосов", "merits_out": "Отдано голосов",
               "reach_in": "От скольких получил", "reach_out": "Скольких поблагодарил"}[kind]
        return pd.DataFrame([{"#": i, "ФИО": _fio(m, pid), "Должность": m[EMP["pos"]].get(pid, ""),
                              "Отдел": m[EMP["dept"]].get(pid, ""), col: int(val)}
                             for i, (pid, val) in enumerate(s.items(), 1)])

    t1, t2, t3, t4 = st.tabs(["🏆 Получено голосов", "🚀 Отдано голосов",
                              "👥 Охват — от скольких получил", "📣 Охват — скольких поблагодарил"])
    for tab, kind in zip((t1, t2, t3, t4), ("merits_in", "merits_out", "reach_in", "reach_out")):
        with tab:
            st.dataframe(table(kind), use_container_width=True, hide_index=True)


# ─────────────────────────── МЕРИТПАСПОРТ ───────────────────────────
def _donut(title, series):
    fig = go.Figure(go.Pie(labels=series.index.tolist(), values=series.values.tolist(), hole=0.55,
                           marker=dict(colors=WARM), textinfo="percent", insidetextorientation="radial"))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=44, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=theme.INK, family="Golos Text, sans-serif"),
                      title=title, showlegend=False)
    return fig


def render_meritpassport(fd, tx_all, emp):
    names = sorted(emp["full_name"].dropna().unique().tolist())
    who = st.selectbox("Сотрудник", names, index=None, placeholder="выберите сотрудника…")
    if not who:
        st.info("Выберите сотрудника, чтобы увидеть его меритпаспорт.")
        return
    row = emp[emp["full_name"] == who].iloc[0]
    pid = row[EMP["id"]]
    st.markdown(f'<div class="card"><span style="font-size:1.15rem"><strong>{who}</strong></span><br>'
                f'<span>{row.get(EMP["pos"],"")}</span> · '
                f'{row.get(EMP["company"],"")} / {row.get(EMP["dept"],"")}</div>', unsafe_allow_html=True)

    sent = fd[fd[TX["sid"]] == pid]
    recv = fd[fd[TX["rid"]] == pid]
    reach_in = recv[TX["sid"]].nunique()
    bal = ("сбалансированный" if abs(len(sent) - len(recv)) <= max(3, 0.3 * max(len(sent), len(recv), 1))
           else "преимущественно отдающий" if len(sent) > len(recv) else "преимущественно получающий")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Получено признаний", f"{len(recv):,}")
    c2.metric("Отдано благодарностей", f"{len(sent):,}")
    c3.metric("Признан коллегами", f"{reach_in}", help="скольких разных людей он(а) поблагодарили — охват признания")
    c4.metric("Стиль участия", bal)

    d1, d2 = st.columns(2)
    with d1:
        if go is not None and len(recv):
            st.plotly_chart(_donut("За что признают",
                                   recv.groupby(TX["value"])[TX["merits"]].sum().sort_values(ascending=False)),
                            use_container_width=True)
    with d2:
        if go is not None and len(sent):
            st.plotly_chart(_donut("За что благодарит других",
                                   sent.groupby(TX["value"])[TX["merits"]].sum().sort_values(ascending=False)),
                            use_container_width=True)

    if go is not None and (len(recv) or len(sent)):
        rin = recv.dropna(subset=["dt"]).groupby("ym").size()
        rout = sent.dropna(subset=["dt"]).groupby("ym").size()
        idx = sorted(set(rin.index) | set(rout.index))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=[rin.get(i, 0) for i in idx], name="Получено",
                                 mode="lines+markers", line=dict(color=CORAL)))
        fig.add_trace(go.Scatter(x=idx, y=[rout.get(i, 0) for i in idx], name="Отдано",
                                 mode="lines+markers", line=dict(color=OLIVE)))
        _light(fig, "Личная динамика по месяцам", 260)
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    m = _emap(emp)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Полученные комментарии**")
        rc = recv[recv[TX["comment"]].notna()].copy()
        if len(rc):
            rc = rc.sort_values("dt", ascending=False) if "dt" in rc.columns else rc
            rc["От кого"] = rc[TX["sid"]].map(lambda p: _fio(m, p))
            show = rc[["От кого", TX["value"], TX["comment"]]].rename(
                columns={TX["value"]: "Ценность", TX["comment"]: "Комментарий"})
            st.caption(f"Всего: {len(show)}")
            st.dataframe(show, use_container_width=True, hide_index=True, height=320)
        else:
            st.caption("Пока нет полученных комментариев.")
    with cc2:
        st.markdown("**Отправленные комментарии**")
        sc = sent[sent[TX["comment"]].notna()].copy()
        if len(sc):
            sc = sc.sort_values("dt", ascending=False) if "dt" in sc.columns else sc
            sc["Кому"] = sc[TX["rid"]].map(lambda p: _fio(m, p))
            show = sc[["Кому", TX["value"], TX["comment"]]].rename(
                columns={TX["value"]: "Ценность", TX["comment"]: "Комментарий"})
            st.caption(f"Всего: {len(show)}")
            st.dataframe(show, use_container_width=True, hide_index=True, height=320)
        else:
            st.caption("Пока нет отправленных комментариев.")

    st.caption("Меритпаспорт — зеркало участия в признании, не оценка. Характеристики описывают позицию в этот период.")
