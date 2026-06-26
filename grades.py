# -*- coding: utf-8 -*-
"""
Грады Болтански–Тевено и темпоральная динамика градовой структуры.
ДОМ МАППИНГА — лист 3_lineage_registry, колонка grade_primary (читается приложением и
передаётся сюда через set_grade_map). GRADE_MAP ниже — запасной/демо-словарь.
Тёплая палитра, тема — theme.py.
"""
import re
import pandas as pd
import theme

GRADE_ORDER = ["Индустриальный", "Рыночный", "Гражданский", "Патриархальный",
               "Вдохновения", "Репутации", "Проектный"]
GRADE_COLORS = {
    "Индустриальный": "#5e7d16", "Рыночный": "#e95f3e", "Гражданский": "#90ae3c",
    "Патриархальный": "#c0492f", "Вдохновения": "#e0a23e", "Репутации": "#8a6d3b",
    "Проектный": "#c9871f",
}

# запасной словарь (если в реестре grade_primary пуст). Ключи нормализованы при чтении.
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


def set_grade_map(mapping):
    """Влить маппинг из реестра (имя ценности → град). Перетирает запасной словарь."""
    for k, v in (mapping or {}).items():
        if v and str(v).strip() and str(v).strip().lower() != "nan":
            GRADE_MAP[_norm(k)] = str(v).strip()


def grade_of(value):
    key = _norm(value)
    if key in GRADE_MAP:
        return GRADE_MAP[key]
    for k, g in GRADE_MAP.items():
        if key.startswith(k[:12]):
            return g
    return None


def grade_share_by_year(acts, value_col="Ценность", date_col="dt"):
    a = acts.dropna(subset=[date_col]).copy()
    a["__grade"] = a[value_col].map(grade_of)
    a = a[a["__grade"].notna()]
    a["__year"] = a[date_col].dt.year
    if len(a) == 0:
        return pd.DataFrame()
    pivot = a.groupby(["__year", "__grade"]).size().unstack(fill_value=0)
    share = pivot.div(pivot.sum(axis=1), axis=0) * 100
    for g in GRADE_ORDER:
        if g not in share.columns:
            share[g] = 0.0
    return share[GRADE_ORDER]


def grade_dynamics_figure(go, make_subplots, tx, unit_col, unit_value):
    """4 панели: Исходящие, Входящие, Внутренние, Внешние."""
    s_col = "s_" + unit_col
    r_col = "r_" + unit_col
    in_unit_s = tx[s_col] == unit_value
    in_unit_r = tx[r_col] == unit_value
    panels = [
        ("Исходящие", "за что благодарят сотрудники подразделения", tx[in_unit_s]),
        ("Входящие", "за что благодарят сотрудников подразделения", tx[in_unit_r]),
        ("Внутренние", "признание внутри подразделения", tx[in_unit_s & in_unit_r]),
        ("Внешние", "признание с другими подразделениями", tx[in_unit_s ^ in_unit_r]),
    ]
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[f"<b>{t}</b><br><span style='font-size:11px'>{s}</span>" for t, s, _ in panels])
    for i, (_, _, acts) in enumerate(panels, start=1):
        share = grade_share_by_year(acts)
        if len(share) == 0:
            continue
        for g in GRADE_ORDER:
            fig.add_trace(go.Scatter(x=share.index.astype(str), y=share[g], name=g,
                                     mode="lines+markers", line=dict(color=GRADE_COLORS[g], width=2),
                                     marker=dict(size=6), legendgroup=g, showlegend=(i == 1)),
                          row=1, col=i)
    fig.update_layout(template="plotly_white", height=430, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=theme.INK, family="Golos Text, sans-serif"),
                      margin=dict(l=46, r=10, t=64, b=10), legend=dict(orientation="h", y=-0.14),
                      dragmode="pan", modebar=dict(remove=["zoom2d", "select2d", "lasso2d", "autoScale2d"]))
    fig.update_annotations(font_size=13)
    fig.update_xaxes(gridcolor=theme.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False)
    fig.update_yaxes(title_text="Доля града, %", row=1, col=1)
    return fig
