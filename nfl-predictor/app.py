"""
app.py — NFL Predictor 3.0 · Página inicial (Visão Geral da rodada)

Rodar com:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from config.settings import ENABLE_ODDS
from utils.dashboard_common import (
    confidence_label,
    get_predictor,
    get_schedules,
    inject_theme_css,
    odds_status_badge,
    team_logo_url,
)
from utils.data_loader import NFLDataLoader

st.set_page_config(
    page_title="NFL Predictor 3.0",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme_css()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
current_year = dt.date.today().year

with st.sidebar:
    st.markdown("## 🏈 NFL Predictor 3.0")
    st.caption("Previsões de vitória via ML, Elo dinâmico e simulação Monte Carlo.")

    season = st.selectbox(
        "Temporada",
        options=list(range(current_year, current_year - 10, -1)),
        index=0,
    )

    st.divider()
    st.caption(odds_status_badge())
    if not ENABLE_ODDS:
        st.caption(
            "Para ativar, defina `ODDS_API_KEY` (variável de ambiente) e "
            "`ENABLE_ODDS=True` — nenhum outro código precisa mudar."
        )

    st.divider()
    if st.button("🔄 Atualizar banco de dados", width='stretch'):
        with st.spinner("Baixando dados mais recentes do nflverse..."):
            loader = NFLDataLoader()
            summary = loader.refresh_all([season])
        st.success(
            f"Atualizado: {summary['schedules']} jogos, {summary['pbp']} jogadas, "
            f"{summary['rosters']} atletas."
        )
        get_schedules.clear()
        st.rerun()

# --------------------------------------------------------------------------- #
# Dados da temporada selecionada
# --------------------------------------------------------------------------- #
schedules = get_schedules((season,))
if schedules.empty:
    st.warning(f"Nenhum dado encontrado para a temporada {season} ainda.")
    st.stop()

date_col = "gameday" if "gameday" in schedules.columns else "game_date"
schedules[date_col] = pd.to_datetime(schedules[date_col], errors="coerce")

available_weeks = sorted(schedules["week"].dropna().unique().tolist())
# Sugere a próxima rodada com jogos ainda não disputados, senão a última.
upcoming = schedules[schedules["home_score"].isna()]
default_week = int(upcoming["week"].min()) if not upcoming.empty else int(available_weeks[-1])

st.markdown(f"### Temporada {season}")
week = st.select_slider("Rodada", options=available_weeks, value=default_week)

week_games = schedules[schedules["week"] == week].sort_values(date_col)

st.markdown(f"## Rodada {week}")
if week_games.empty:
    st.info("Sem jogos cadastrados para esta rodada.")
    st.stop()

# --------------------------------------------------------------------------- #
# Probabilidades rápidas via Elo (sem rodar o pipeline completo de ML +
# Monte Carlo para cada jogo — isso é reservado para a página de detalhe,
# por performance: calcular só o Elo para uma rodada inteira é instantâneo).
# --------------------------------------------------------------------------- #
predictor = get_predictor()
elo_system = predictor._ensure_elo()

# --------------------------------------------------------------------------- #
# Grid de cards dos jogos
# --------------------------------------------------------------------------- #
cols_per_row = 3
rows = [week_games.iloc[i : i + cols_per_row] for i in range(0, len(week_games), cols_per_row)]

for row_games in rows:
    cols = st.columns(cols_per_row)
    for col, (_, game) in zip(cols, row_games.iterrows()):
        home, away = game["home_team"], game["away_team"]
        home_prob = elo_system.predict_win_probability(home, away)
        favorite = home if home_prob >= 0.5 else away
        favorite_prob = max(home_prob, 1 - home_prob)

        kickoff = game[date_col]
        kickoff_str = kickoff.strftime("%a %d/%m · %H:%M") if pd.notna(kickoff) else "Horário a definir"

        played = pd.notna(game.get("home_score"))
        score_str = (
            f"{int(game['home_score'])} x {int(game['away_score'])}" if played else None
        )

        with col:
            with st.container(border=True):
                logo_col1, vs_col, logo_col2 = st.columns([1, 0.4, 1])
                with logo_col1:
                    st.image(team_logo_url(home), width=56)
                    st.markdown(f"**{home}**")
                with vs_col:
                    st.markdown("<div style='text-align:center; padding-top:18px;'>@</div>", unsafe_allow_html=True)
                with logo_col2:
                    st.image(team_logo_url(away), width=56)
                    st.markdown(f"**{away}**")

                st.caption(kickoff_str)
                if score_str:
                    st.markdown(f"**Placar final: {score_str}**")

                st.progress(home_prob, text=f"{home} {home_prob:.0%} · {away} {1 - home_prob:.0%}")
                st.markdown(
                    f"<span class='favorite-tag'>Favorito: {favorite} ({favorite_prob:.0%})</span>",
                    unsafe_allow_html=True,
                )

                if st.button("Ver detalhes →", key=f"detail_{game['game_id']}", width='stretch'):
                    st.session_state["selected_game"] = {
                        "home_team": home,
                        "away_team": away,
                        "season": int(season),
                        "week": int(week),
                    }
                    st.switch_page("pages/1_📊_Detalhe_do_Jogo.py")

st.divider()
st.caption(
    "Probabilidades da visão geral usam apenas o Elo dinâmico (Módulo 3) para "
    "carregar a rodada inteira instantaneamente. Abra os detalhes de um jogo "
    "para ver a previsão completa (ML + Elo + Monte Carlo + SHAP)."
)

# --------------------------------------------------------------------------- #
# Bônus: exportar previsões da rodada para CSV
# --------------------------------------------------------------------------- #
export_rows = []
for _, game in week_games.iterrows():
    home, away = game["home_team"], game["away_team"]
    home_prob = elo_system.predict_win_probability(home, away)
    export_rows.append({
        "season": season,
        "week": week,
        "home_team": home,
        "away_team": away,
        "home_win_prob_elo": round(home_prob, 4),
        "favorite": home if home_prob >= 0.5 else away,
    })
export_df = pd.DataFrame(export_rows)
st.download_button(
    "⬇️ Exportar previsões da rodada (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"previsoes_temporada{season}_rodada{week}.csv",
    mime="text/csv",
)
