# -*- coding: utf-8 -*-
"""
Панели дашборда: рейтинг охвата, меритпаспорт сотрудника, дневной пульс с ФИО,
drill «ценность → кто её сформировал». Вынесено из app.py.
"""
import pandas as pd
import streamlit as st
try:
    import plotly.graph_objects as go
except Exception:
    go = None

EMP = {"last": "Фамилия", "first": "Имя", "id": "Персональный номер",
       "pos": "Должность", "company": "Компания", "dept": "Отдел"}
TX = {"date": "Дата", "sid": "Номер отправителя", "rid": "Номер получателя",
      "value": "Ценность", "merits": "Мериты", "comment": "Комментарий"}


def _emap(emp):
    m = emp.set_index(EMP["id"])
    return m[~m.index.duplicated(keep="first")]


def _fio(m, pid):
    if pid in m.index:
        r = m.loc[pid]
        return f"{r.get(EMP['last'],'')} {r.get(EMP['first'],'')}".strip()
    return str(pid)


# ─────────────────────────────────────────────────────────────────────────────
#  ДНЕВНОЙ ПУЛЬС с ФИО в подсказке
# ─────────────────────────────────────────────────────────────────────────────
def daily_pulse_fig(fd, emp):
    if go is None:
        return None
    m = _emap(emp)
    a = fd.dropna(subset=["dt"]).copy()
    a["d"] = a["dt"].dt.normalize()
    daily = a.groupby("d").agg(acts=(TX["merits"], "size")).reset_index()
    # топ-благодарящие за день — в подсказку
    names_by_day = {}
    for d, sub in a.groupby("d"):
        top = sub.groupby(TX["sid"]).size().sort_values(ascending=False).head(6)
        names = [f"{_fio(m, pid)} ({c})" for pid, c in top.items()]
        extra = sub[TX["sid"]].nunique() - len(names)
        names_by_day[d] = "<br>".join(names) + (f"<br>…и ещё {extra}" if extra > 0 else "")
    daily["names"] = daily["d"].map(names_by_day)
    fig = go.Figure(go.Bar(
        x=daily["d"], y=daily["acts"], customdata=daily["names"], marker_color="#58a6ff",
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Благодарностей: %{y}<br><br>"
                      "<b>Кто благодарил:</b><br>%{customdata}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      title="Пульс по дням — наведите на столбик, чтобы увидеть, кто благодарил в этот день",
                      bargap=0.1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  ЦЕННОСТЬ → КТО ЕЁ СФОРМИРОВАЛ (ФИО)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  РЕЙТИНГ ОХВАТА (мериты + число сотрудников)
# ─────────────────────────────────────────────────────────────────────────────
def render_rating(fd, emp, N=20):
    m = _emap(emp)

    def table(group_col, agg_kind):
        if agg_kind == "merits_in":
            s = fd.groupby(TX["rid"])[TX["merits"]].sum()
        elif agg_kind == "merits_out":
            s = fd.groupby(TX["sid"])[TX["merits"]].sum()
        elif agg_kind == "reach_in":
            s = fd.groupby(TX["rid"])[TX["sid"]].nunique()
        else:
            s = fd.groupby(TX["sid"])[TX["rid"]].nunique()
        s = s.sort_values(ascending=False).head(N)
        col = {"merits_in": "Получено голосов", "merits_out": "Отдано голосов",
               "reach_in": "От скольких получил", "reach_out": "Скольких поблагодарил"}[agg_kind]
        return pd.DataFrame([{"#": i, "ФИО": _fio(m, pid), "Должность": m[EMP["pos"]].get(pid, ""),
                              "Отдел": m[EMP["dept"]].get(pid, ""), col: int(val)}
                             for i, (pid, val) in enumerate(s.items(), 1)])

    t1, t2, t3, t4 = st.tabs(["🏆 Получено голосов", "🚀 Отдано голосов",
                              "👥 Охват — от скольких получил", "📣 Охват — скольких поблагодарил"])
    for tab, kind in zip((t1, t2, t3, t4), ("merits_in", "merits_out", "reach_in", "reach_out")):
        with tab:
            st.dataframe(table(None, kind), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  МЕРИТПАСПОРТ
# ─────────────────────────────────────────────────────────────────────────────
def _donut(title, series):
    fig = go.Figure(go.Pie(labels=series.index.tolist(), values=series.values.tolist(), hole=0.55,
                           textinfo="percent", insidetextorientation="radial"))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="#0d1117", title=title, showlegend=False)
    return fig


def render_meritpassport(fd, tx_all, emp):
    names = sorted(emp["full_name"].dropna().unique().tolist())
    who = st.selectbox("Сотрудник", names, index=None, placeholder="выберите сотрудника…")
    if not who:
        st.info("Выберите сотрудника, чтобы увидеть его меритпаспорт.")
        return
    row = emp[emp["full_name"] == who].iloc[0]
    pid = row[EMP["id"]]
    # заголовок: должность и отдел читаются чётко
    st.markdown(f'<div class="card"><span style="font-size:1.15rem"><strong>{who}</strong></span><br>'
                f'<span style="color:#c9d1d9">{row.get(EMP["pos"],"")}</span> · '
                f'{row.get(EMP["company"],"")} / {row.get(EMP["dept"],"")}</div>', unsafe_allow_html=True)

    sent = fd[fd[TX["sid"]] == pid]
    recv = fd[fd[TX["rid"]] == pid]
    reach_in = recv[TX["sid"]].nunique()
    depts_in = recv[recv[TX["sid"]].isin(emp[EMP["id"]])].copy()
    n_dep_in = depts_in.merge(emp[[EMP["id"], EMP["dept"]]], left_on=TX["sid"], right_on=EMP["id"], how="left")[EMP["dept"]].nunique()
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
            st.plotly_chart(_donut("За что признают (полученные)",
                                   recv.groupby(TX["value"])[TX["merits"]].sum().sort_values(ascending=False)),
                            use_container_width=True)
    with d2:
        if go is not None and len(sent):
            st.plotly_chart(_donut("За что благодарит других (отданные)",
                                   sent.groupby(TX["value"])[TX["merits"]].sum().sort_values(ascending=False)),
                            use_container_width=True)

    # личная динамика по месяцам
    if go is not None and (len(recv) or len(sent)):
        rin = recv.dropna(subset=["dt"]).groupby("ym").size()
        rout = sent.dropna(subset=["dt"]).groupby("ym").size()
        idx = sorted(set(rin.index) | set(rout.index))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=[rin.get(i, 0) for i in idx], name="Получено",
                                 mode="lines+markers", line=dict(color="#58a6ff")))
        fig.add_trace(go.Scatter(x=idx, y=[rout.get(i, 0) for i in idx], name="Отдано",
                                 mode="lines+markers", line=dict(color="#f78166")))
        fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", title="Личная динамика по месяцам",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    m = _emap(emp)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Полученные комментарии**")
        rc = recv[recv[TX["comment"]].notna()].copy()
        rc["От кого"] = rc[TX["sid"]].map(lambda p: _fio(m, p))
        show = rc[["От кого", TX["value"], TX["comment"]]].rename(columns={TX["value"]: "Ценность", TX["comment"]: "Комментарий"}).tail(15)
        st.dataframe(show, use_container_width=True, hide_index=True) if len(show) else st.caption("Пока нет комментариев.")
    with cc2:
        st.markdown("**Отправленные комментарии**")
        sc = sent[sent[TX["comment"]].notna()].copy()
        sc["Кому"] = sc[TX["rid"]].map(lambda p: _fio(m, p))
        show = sc[["Кому", TX["value"], TX["comment"]]].rename(columns={TX["value"]: "Ценность", TX["comment"]: "Комментарий"}).tail(15)
        st.dataframe(show, use_container_width=True, hide_index=True) if len(show) else st.caption("Пока нет комментариев.")

    st.caption("Меритпаспорт — зеркало участия в признании, не оценка. Характеристики описывают позицию в этот период.")
