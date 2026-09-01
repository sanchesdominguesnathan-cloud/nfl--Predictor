"""
Utilidades compartilhadas pelas páginas do dashboard Streamlit (Módulo 6).

Centraliza:
    - cache de recursos pesados (GamePredictor, dados de schedule)
    - mapeamento de logos dos times (CDN pública da ESPN)
    - paleta de cores do tema (config/settings.py)
    - injeção de CSS leve para o "tema escuro inspirado em ESPN"
"""

from __future__ import annotations

import streamlit as st

from config.settings import (
    COLOR_AWAY,
    COLOR_EV_NEGATIVE,
    COLOR_EV_POSITIVE,
    COLOR_HOME,
    ENABLE_ODDS,
    THEME_CARD_BACKGROUND,
)
from predict import GamePredictor
from utils.data_loader import NFLDataLoader

# --------------------------------------------------------------------------- #
# Logos (CDN pública da ESPN — mesmo padrão usado por vários dashboards de NFL)
# --------------------------------------------------------------------------- #
_ESPN_SLUG_OVERRIDES = {"LA": "lar", "WAS": "wsh"}


def team_logo_url(team: str) -> str:
    slug = _ESPN_SLUG_OVERRIDES.get(team, team).lower()
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"


# --------------------------------------------------------------------------- #
# Recursos cacheados (carregados uma única vez por processo do Streamlit)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Carregando modelos e Elo...")
def get_predictor() -> GamePredictor:
    predictor = GamePredictor()
    predictor._ensure_elo()  # aquece o cache de Elo já na primeira carga
    return predictor


@st.cache_data(show_spinner="Carregando calendário...", ttl=3600)
def get_schedules(seasons: tuple[int, ...]):
    loader = NFLDataLoader()
    return loader.load_schedules(list(seasons))


@st.cache_data(show_spinner=False)
def get_current_elo_ratings(_predictor: GamePredictor) -> dict[str, float]:
    return _predictor._ensure_elo().current_ratings()


# --------------------------------------------------------------------------- #
# CSS — tema escuro inspirado em ESPN
# --------------------------------------------------------------------------- #
def inject_theme_css() -> None:
    st.markdown(
        f"""
        <style>
        div[data-testid="stMetric"] {{
            background-color: {THEME_CARD_BACKGROUND};
            border: 1px solid #262B36;
            border-radius: 10px;
            padding: 14px 16px;
        }}
        .game-card {{
            background-color: {THEME_CARD_BACKGROUND};
            border: 1px solid #262B36;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 10px;
        }}
        .favorite-tag {{
            display: inline-block;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 999px;
            background-color: {COLOR_HOME}22;
            color: {COLOR_HOME};
            font-weight: 600;
        }}
        .ev-positive {{ color: {COLOR_EV_POSITIVE}; font-weight: 600; }}
        .ev-negative {{ color: {COLOR_EV_NEGATIVE}; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def confidence_label(score: int) -> str:
    if score >= 70:
        return "Alta confiança"
    if score >= 45:
        return "Confiança moderada"
    return "Jogo imprevisível"


def odds_status_badge() -> str:
    return "🟢 Odds ao vivo ativas" if ENABLE_ODDS else "⚪ Odds ao vivo em standby (sem chave configurada)"
