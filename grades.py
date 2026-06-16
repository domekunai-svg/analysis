# -*- coding: utf-8 -*-
"""
Маппинг ценностей Термекса → грады Болтански–Тевено и темпоральная динамика
градовой структуры. Маппинг ПРОВИЗОРНЫЙ (демо до лингвистической валидации) —
правится в одном месте: словарь GRADE_MAP ниже.
"""
import re
import pandas as pd

# Порядок и цвета градов (как в эталонном примере)
GRADE_ORDER = ["Индустриальный", "Рыночный", "Гражданский", "Патриархальный",
               "Вдохновения", "Репутации", "Проектный"]
GRADE_COLORS = {
    "Индустриальный": "#1f9e94", "Рыночный": "#e08a3c", "Гражданский": "#4f7a4f",
    "Патриархальный": "#c0508f", "Вдохновения": "#e3c324", "Репутации": "#8a7152",
    "Проектный": "#2f4b7c",
}

# Провизорный маппинг 14 ценностей → доминирующий град (ключи нормализованы)
GRADE_MAP = {
    "командная работа": "Проектный",
    "надежное плечо": "Патриархальный",
    "профессионализм": "Индустриальный",
    "ответственность": "Индустриальный",
    "результативность": "Рыночный",
    "инициативность": "Вдохновения",
    "открытость": "Проектный",
    "наставничество": "Патриархальный",
    "активное участие в жизни корпорации": "Гражданский",
    "исполнительность": "Индустриальный",
    "лидерство": "Репутации",
    "организованность": "Индустриальный",
    "выше ожиданий": "Рыночный",
    "гибкость": "Проектный",
}


def _norm(v):
    return re.sub(r"\s+", " ", str(v).strip().lower().replace("ё", "е"))


def grade_of(value):
    key = _norm(value)
    if key in GRADE_MAP:
        return GRADE_MAP[key]
    for k, g in GRADE_MAP.items():           # запасной поиск по началу строки
        if key.startswith(k[:12]):
            return g
    return None


def attach_grade(df, value_col="Ценность"):
    out = df.copy()
    out["__grade"] = out[value_col].map(grade_of)
    return out


def grade_share_by_year(acts, value_col="Ценность", date_col="dt"):
    """Доля каждого града по годам (в %). acts — подвыборка актов."""
    a = acts.dropna(subset=[date_col]).copy()
    a["__grade"] = a[value_col].map(grade_of)
    a = a[a["__grade"].notna()]
    a["__year"] = a[date_col].dt.year
    if len(a) == 0:
        return pd.DataFrame(), {}
    pivot = a.groupby(["__year", "__grade"]).size().unstack(fill_value=0)
    share = pivot.div(pivot.sum(axis=1), axis=0) * 100
    counts = a.groupby("__year").size().to_dict()
    for g in GRADE_ORDER:
        if g not in share.columns:
            share[g] = 0.0
    return share[GRADE_ORDER], counts


def grade_dynamics_figure(go, make_subplots, tx, unit_col, unit_value):
    """3 панели: Я-концепция (исходящие), Персона (входящие), Внутренние (внутри юнита)."""
    s_col = "s_" + unit_col if unit_col != "dept_key" else "s_dept_key"
    r_col = "r_" + unit_col if unit_col != "dept_key" else "r_dept_key"
    out_acts = tx[tx[s_col] == unit_value]                                   # что благодарят сотрудники юнита
    in_acts = tx[tx[r_col] == unit_value]                                    # за что благодарят сотрудников юнита
    within = tx[(tx[s_col] == unit_value) & (tx[r_col] == unit_value)]       # внутри юнита

    panels = [("Я-концепция", "что и за что благодарят сотрудники", out_acts),
              ("Персона", "за что благодарят сотрудников", in_acts),
              ("Внутренние", "как благодарят друг друга", within)]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{t}<br><sub>{s}</sub>" for t, s, _ in panels],
                        shared_yaxes=True)
    for i, (_, _, acts) in enumerate(panels, start=1):
        share, counts = grade_share_by_year(acts)
        if len(share) == 0:
            continue
        for g in GRADE_ORDER:
            fig.add_trace(go.Scatter(x=share.index.astype(str), y=share[g], name=g,
                                     mode="lines+markers", line=dict(color=GRADE_COLORS[g], width=2),
                                     marker=dict(size=7), legendgroup=g, showlegend=(i == 1)),
                          row=1, col=i)
    fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      margin=dict(l=10, r=10, t=70, b=10), legend=dict(orientation="h", y=-0.12),
                      title=f"Темпоральная динамика градовой структуры — {unit_value}")
    fig.update_yaxes(title_text="Доля града, %", row=1, col=1)
    return fig
