# -*- coding: utf-8 -*-
"""Общая тема (светлая/тёмная) для графиков. app.py переключает её через set_dark().
panels.py и grades.py читают INK/GRID в момент отрисовки."""

INK = "#37312c"
GRID = "rgba(55,49,44,0.08)"
DARK = False


def set_dark(dark):
    global INK, GRID, DARK
    DARK = bool(dark)
    if dark:
        INK = "#f1e8e2"
        GRID = "rgba(241,232,226,0.10)"
    else:
        INK = "#37312c"
        GRID = "rgba(55,49,44,0.08)"
