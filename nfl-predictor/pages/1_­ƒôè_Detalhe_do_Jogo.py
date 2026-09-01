"""
pages/1_📊_Detalhe_do_Jogo.py — Página de detalhe de um confronto específico.

Mostra: probabilidade (ML + Elo), score de confiança, comparação
ataque/defesa, EPA, Elo, descanso, QB, simulação Monte Carlo (10.000
partidas), placares mais prováveis, explicação SHAP e a calculadora de
Value Betting (Módulo 8 — funciona com odd manual mesmo com o Módulo 7 em
standby).
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config.settings import COLOR_AWAY, COLOR_EV_NEGATIVE, COLOR_EV_POSITIVE, COLOR_HOME, ENABLE_ODDS
from utils.dashboard_common import (
    confidence_label,
    get_predictor,
    get_schedules,
    inject_theme_css,
    team_logo_url,
)

st.set_page_config(page_title="Detalhe do Jogo · NFL Predictor 3.0", page_icon="📊", layout="wide")
inject_theme_css()

predictor = get_predictor()

# --------------------------------------------------------------------------- #
# Seleção do jogo (vindo da Home via session_state, com fallback manual)
# --------------------------------------------------------------------------- #
selected = st.session_state.get("selected_game")

with st.sidebar:
    st.markdown("## 🔍 Selecionar jogo")
    teams = sorted(set(get_schedules((2025,))["home_team"]).union(get_schedules((2025,))["away_team"]))
    default_home = selected["home_team"] if selected else teams[0]
    default_away = selected["away_team"] if selected else teams[1]

    home_team = st.selectbox("Mandante", options=teams, index=teams.index(default_home))
    away_options = [t for t in teams if t != home_team]
    away_default_index = away_options.index(default_away) if default_away in away_options else 0
    away_team = st.selectbox("Visitante", options=away_options, index=away_default_index)

    st.divider()
    st.markdown("### 💰 Value Betting (Módulo 8)")
    st.caption("Funciona com odd manual mesmo sem uma API de odds conectada.")
    use_manual_odds = st.checkbox("Informar odd manualmente")
    manual_odds = None
    bankroll = 100.0
    if use_manual_odds:
        manual_odds = st.number_input("Odd decimal (mandante)", min_value=1.01, value=2.20, step=0.01)
        bankroll = st.number_input("Banca (R$)", min_value=0.0, value=1000.0, step=100.0)

# --------------------------------------------------------------------------- #
# Previsão completa
# --------------------------------------------------------------------------- #
with st.spinner("Calculando previsão (ML + Elo + Monte Carlo)..."):
    prediction = predictor.predict_game(home_team, away_team, manual_odds=manual_odds, bankroll=bankroll)

# --------------------------------------------------------------------------- #
# Cabeçalho
# --------------------------------------------------------------------------- #
header_l, header_c, header_r = st.columns([1, 1.4, 1])
with header_l:
    st.image(team_logo_url(home_team), width=110)
    st.markdown(f"## {home_team}")
    st.caption("Mandante")
with header_c:
    st.markdown(
        f"<div style='text-align:center; padding-top:30px; font-size:2rem;'>"
        f"{prediction.blended_home_win_prob:.0%} <span style='font-size:1rem; color:gray;'>vs</span> "
        f"{1 - prediction.blended_home_win_prob:.0%}</div>",
        unsafe_allow_html=True,
    )
    st.progress(prediction.blended_home_win_prob)
    st.caption(
        f"ML: {prediction.ml_home_win_prob:.0%}  ·  Elo: {prediction.elo_home_win_prob:.0%}  ·  "
        f"Probabilidade final: média dos dois modelos"
    )
with header_r:
    st.image(team_logo_url(away_team), width=110)
    st.markdown(f"## {away_team}")
    st.caption("Visitante")

conf = prediction.confidence_score
st.metric("Score de confiança", f"{conf}/100", confidence_label(conf))

st.divider()

# --------------------------------------------------------------------------- #
# Comparação de ataque e defesa
# --------------------------------------------------------------------------- #
st.markdown("### Ataque x Defesa")

metric_labels = {
    "off_epa_per_play": "EPA/jogada (ataque)",
    "def_epa_allowed": "EPA/jogada permitido (defesa)",
    "off_success_rate": "Success Rate (ataque)",
    "third_down_pct": "3ª descida %",
    "redzone_td_pct": "Red Zone TD %",
    "pressure_rate": "Pressure Rate (defesa)",
    "turnovers_forced": "Turnovers forçados",
}

home_vals = [prediction.home_form.get(k, 0) or 0 for k in metric_labels]
away_vals = [prediction.away_form.get(k, 0) or 0 for k in metric_labels]

fig = go.Figure()
fig.add_bar(name=home_team, x=list(metric_labels.values()), y=home_vals, marker_color=COLOR_HOME)
fig.add_bar(name=away_team, x=list(metric_labels.values()), y=away_vals, marker_color=COLOR_AWAY)
fig.update_layout(
    barmode="group", template="plotly_dark", height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------------- #
# EPA, Elo, Descanso, QB
# --------------------------------------------------------------------------- #
st.markdown("### EPA · Elo · Descanso · QB")
c1, c2, c3, c4 = st.columns(4)
elo_ratings = predictor._ensure_elo().current_ratings()
c1.metric(f"Elo {home_team}", f"{elo_ratings.get(home_team, 1500):.0f}")
c1.metric(f"Elo {away_team}", f"{elo_ratings.get(away_team, 1500):.0f}")
c2.metric(f"EPA/play {home_team}", f"{prediction.home_form.get('off_epa_per_play', 0):.3f}")
c2.metric(f"EPA/play {away_team}", f"{prediction.away_form.get('off_epa_per_play', 0):.3f}")
c3.metric(f"QB EPA {home_team}", f"{prediction.home_form.get('qb_epa', 0):.3f}")
c3.metric(f"QB EPA {away_team}", f"{prediction.away_form.get('qb_epa', 0):.3f}")
c4.metric(f"CPOE {home_team}", f"{prediction.home_form.get('cpoe', 0):.2f}")
c4.metric(f"CPOE {away_team}", f"{prediction.away_form.get('cpoe', 0):.2f}")

st.divider()

# --------------------------------------------------------------------------- #
# Simulação Monte Carlo
# --------------------------------------------------------------------------- #
st.markdown("### 🎲 Simulação Monte Carlo (10.000 partidas)")

mc = prediction.monte_carlo
mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
mc_c1.metric("Placar médio", f"{mc['home_avg_score']:.1f} x {mc['away_avg_score']:.1f}")
mc_c2.metric("Chance de vitória (mandante)", f"{mc['home_win_prob']:.1%}")
mc_c3.metric("Chance de overtime", f"{mc['overtime_prob']:.2%}")
mc_c4.metric(
    f"{home_team} vencer por +7",
    f"{mc['home_win_margin_probs']['+7']:.1%}",
)

margin = prediction.mc_result.raw_home_scores - prediction.mc_result.raw_away_scores
hist = go.Figure()
hist.add_histogram(x=margin, nbinsx=40, marker_color=COLOR_HOME, opacity=0.85)
hist.add_vline(x=0, line_dash="dash", line_color="white")
hist.update_layout(
    template="plotly_dark", height=320,
    title=f"Distribuição da margem ({home_team} − {away_team})",
    margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(hist, width='stretch')

margin_col1, margin_col2 = st.columns(2)
with margin_col1:
    st.markdown(f"**{home_team} vencer por:**")
    for margin_label, prob in mc["home_win_margin_probs"].items():
        st.write(f"{margin_label}: {prob:.1%}")
with margin_col2:
    st.markdown(f"**{away_team} vencer por:**")
    for margin_label, prob in mc["away_win_margin_probs"].items():
        st.write(f"{margin_label}: {prob:.1%}")

st.markdown("**Placares mais prováveis**")
scores_df_data = [
    {"Placar": f"{h} x {a}", "Probabilidade": f"{p:.2%}"} for (h, a), p in mc["most_likely_scores"]
]
st.table(scores_df_data)

st.divider()

# --------------------------------------------------------------------------- #
# Explicação SHAP (bônus)
# --------------------------------------------------------------------------- #
if prediction.shap_contributions:
    st.markdown("### 🧠 O que mais influenciou esta previsão (SHAP)")
    st.caption(
        f"Valores positivos empurram a probabilidade a favor do {home_team} "
        f"(mandante); valores negativos, a favor do {away_team}."
    )
    items = sorted(prediction.shap_contributions.items(), key=lambda kv: kv[1])
    shap_fig = go.Figure()
    shap_fig.add_bar(
        x=[v for _, v in items],
        y=[k for k, _ in items],
        orientation="h",
        marker_color=[COLOR_HOME if v >= 0 else COLOR_AWAY for _, v in items],
    )
    shap_fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(shap_fig, width='stretch')
    st.divider()

# --------------------------------------------------------------------------- #
# Modo What-If (bônus)
# --------------------------------------------------------------------------- #
with st.expander("🔀 Modo What-If — e se as condições fossem outras?"):
    st.caption("Ajuste o descanso ou simule um QB reserva e recalcule a probabilidade.")
    wi_col1, wi_col2, wi_col3 = st.columns(3)
    with wi_col1:
        wi_home_rest = st.slider(f"Dias de descanso — {home_team}", 3, 14, 7)
    with wi_col2:
        wi_away_rest = st.slider(f"Dias de descanso — {away_team}", 3, 14, 7)
    with wi_col3:
        wi_backup_qb_home = st.checkbox(f"QB reserva — {home_team}")
        wi_backup_qb_away = st.checkbox(f"QB reserva — {away_team}")

    if st.button("Recalcular com esses ajustes"):
        home_form_wi = dict(prediction.home_form)
        away_form_wi = dict(prediction.away_form)
        # Penalidade aproximada de QB reserva: reduz EPA de passe e QB em ~60%
        # (heurística simples — QBs reservas historicamente jogam bem abaixo do titular).
        if wi_backup_qb_home:
            home_form_wi["qb_epa"] = home_form_wi.get("qb_epa", 0) * 0.4 - 0.05
            home_form_wi["off_pass_epa"] = home_form_wi.get("off_pass_epa", 0) * 0.4 - 0.05
        if wi_backup_qb_away:
            away_form_wi["qb_epa"] = away_form_wi.get("qb_epa", 0) * 0.4 - 0.05
            away_form_wi["off_pass_epa"] = away_form_wi.get("off_pass_epa", 0) * 0.4 - 0.05

        ml_prob_wi = predictor._predict_ml(home_team, away_team, home_form_wi, away_form_wi)
        st.info(
            f"Nova probabilidade estimada (ML) para {home_team}: **{ml_prob_wi:.1%}** "
            f"(original: {prediction.ml_home_win_prob:.1%})"
        )

st.divider()

# --------------------------------------------------------------------------- #
# Odds ao vivo (Módulo 7 — em standby) e Value Betting (Módulo 8)
# --------------------------------------------------------------------------- #
st.markdown("### 💰 Odds & Value Betting")

if ENABLE_ODDS and prediction.odds:
    st.json(prediction.odds)
else:
    st.info(
        "Odds ao vivo estão em standby (Módulo 7) — nenhuma chamada de API paga é "
        "feita. A calculadora abaixo funciona com uma odd digitada manualmente."
    )

if prediction.value_bet:
    vb = prediction.value_bet
    ev_class = "ev-positive" if vb["expected_value_pct"] >= 0 else "ev-negative"
    vb_c1, vb_c2, vb_c3 = st.columns(3)
    vb_c1.metric("Probabilidade implícita da odd", f"{vb['implied_probability']:.1%}")
    vb_c2.markdown(
        f"**Expected Value**<br><span class='{ev_class}'>{vb['expected_value_pct']:+.1%}</span>",
        unsafe_allow_html=True,
    )
    vb_c3.metric("Sugestão de stake (Kelly)", f"R$ {vb['kelly_stake_suggestion']:.2f}")
else:
    st.caption("Marque \"Informar odd manualmente\" na barra lateral para calcular EV e Kelly.")
