"""
predict.py — Orquestrador de previsões do NFL Predictor 3.0.

Junta:
    - Módulo 4 (ML)          -> probabilidade de vitória via XGBoost
    - Módulo 3 (Elo)         -> probabilidade de vitória via Elo (comparação)
    - Módulo 5 (Monte Carlo) -> distribuição de placares, overtime, margens
    - Score de confiança     -> 0-100
    - Módulo 7 (Odds)        -> EM STANDBY (arquitetura pronta, sem chamadas)
    - Módulo 8 (Value Betting) -> funções prontas, aceitam odd manual

Uso típico:
    from predict import GamePredictor
    predictor = GamePredictor()
    prediction = predictor.predict_game("KC", "BUF")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import ENABLE_ODDS, MODELS_DIR, ODDS_API_KEY
from utils.data_loader import NFLDataLoader
from utils.elo import EloRatingSystem
from utils.feature_engineering import PBP_FEATURE_COLUMNS, FeatureEngineer
from utils.monte_carlo import MonteCarloResult, MonteCarloSimulator, TeamGameInputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Módulo 7 — Odds (EM STANDBY)
# =============================================================================
class OddsProvider:
    """
    Arquitetura pronta para odds em tempo real. Enquanto `ENABLE_ODDS` for
    False (config/settings.py), nenhuma chamada de rede é feita e todos os
    métodos retornam None. Quando uma API key for configurada, basta
    implementar o corpo de `get_live_odds` — nenhum outro módulo precisa
    mudar.
    """

    def __init__(self, api_key: str | None = ODDS_API_KEY, enabled: bool = ENABLE_ODDS) -> None:
        self.api_key = api_key
        self.enabled = enabled

    def get_live_odds(self, home: str, away: str) -> dict | None:
        """Retorna odds ao vivo para o confronto, ou None se ENABLE_ODDS=False."""
        if not self.enabled:
            return None
        # TODO: quando houver uma chave de API de odds, implementar a chamada
        # HTTP aqui e retornar um dict, por exemplo:
        # {"home_moneyline": -150, "away_moneyline": +130,
        #  "home_decimal_odds": 1.67, "away_decimal_odds": 2.30}
        raise NotImplementedError("Nenhum provedor de odds configurado ainda.")


# =============================================================================
# Módulo 8 — Value Betting (funções prontas, aguardando odds manuais ou da API)
# =============================================================================
def implied_probability(decimal_odds: float) -> float:
    """Converte odds decimais (ex.: 2.20) na probabilidade implícita (ex.: 0.4545)."""
    if decimal_odds <= 1.0:
        raise ValueError("Odds decimais devem ser > 1.0")
    return 1.0 / decimal_odds


def expected_value(model_probability: float, decimal_odds: float, stake: float = 1.0) -> float:
    """
    EV esperado de uma aposta, em unidades de `stake` (ex.: stake=1.0 -> EV
    fracionário; multiplique pela banca para EV em dinheiro).

        EV = (p_modelo * (odds - 1)) - (1 - p_modelo)
    """
    return (model_probability * (decimal_odds - 1) * stake) - ((1 - model_probability) * stake)


def kelly_criterion(model_probability: float, decimal_odds: float, fraction: float = 1.0) -> float:
    """
    Fração da banca sugerida pelo critério de Kelly (0 se não há valor).
    `fraction` permite usar "Kelly fracionário" (ex.: 0.5 para meio-Kelly,
    mais conservador — recomendado na prática para reduzir variância).
    """
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - model_probability
    kelly = (b * model_probability - q) / b
    return max(0.0, kelly * fraction)


def closing_line_value(bet_odds: float, closing_odds: float) -> float:
    """
    CLV (Closing Line Value): quanto sua odd era melhor que a odd de
    fechamento do mercado. Positivo = você "bateu" o mercado.
        CLV = (odd_apostada / odd_fechamento) - 1
    """
    return (bet_odds / closing_odds) - 1.0


def value_bet_summary(model_probability: float, decimal_odds: float, bankroll: float = 100.0) -> dict:
    """Atalho que devolve o pacote completo do exemplo do Módulo 8."""
    implied = implied_probability(decimal_odds)
    ev = expected_value(model_probability, decimal_odds)
    kelly = kelly_criterion(model_probability, decimal_odds)
    return {
        "model_probability": model_probability,
        "decimal_odds": decimal_odds,
        "implied_probability": implied,
        "expected_value_pct": ev,  # EV por unidade apostada (ex.: 0.342 = +34.2%)
        "kelly_fraction": kelly,
        "kelly_stake_suggestion": kelly * bankroll,
    }


# =============================================================================
# Score de confiança (0-100)
# =============================================================================
def compute_confidence_score(
    elo_diff: float,
    off_epa_stability_home: float,
    off_epa_stability_away: float,
    monte_carlo_std: float,
) -> int:
    """
    Combina 4 sinais em um score de 0 a 100:
        - |diferença de Elo|: quanto maior, mais lopsided é o confronto
          (mais "previsível").
        - estabilidade ofensiva de cada time (inverso do desvio-padrão do
          EPA/play entre jogos — times inconsistentes reduzem a confiança).
        - variância da simulação Monte Carlo (distribuição de margem mais
          espalhada = jogo mais imprevisível).

    83/100 -> alta confiança | 42/100 -> jogo imprevisível (exemplos do spec).
    """
    elo_component = np.clip(abs(elo_diff) / 400, 0, 1) * 40  # até 40 pontos

    stability = np.clip(1 - (off_epa_stability_home + off_epa_stability_away), 0, 1) * 30  # até 30 pontos

    variance_component = np.clip(1 - (monte_carlo_std / 20), 0, 1) * 30  # até 30 pontos

    score = elo_component + stability + variance_component
    return int(round(np.clip(score, 0, 100)))


# =============================================================================
# GamePredictor — orquestrador principal
# =============================================================================
@dataclass
class GamePrediction:
    home_team: str
    away_team: str
    ml_home_win_prob: float
    elo_home_win_prob: float
    blended_home_win_prob: float
    confidence_score: int
    monte_carlo: dict
    home_form: dict
    away_form: dict
    odds: dict | None = None
    value_bet: dict | None = None
    mc_result: object = field(default=None, repr=False)  # MonteCarloResult bruto (para gráficos)
    shap_contributions: dict | None = None  # {feature: valor_shap} do jogo previsto


class GamePredictor:
    """
    Orquestrador de previsão de um jogo: carrega os artefatos treinados
    (Módulo 4), calcula a forma atual dos times (Módulo 2), o Elo atual
    (Módulo 3), roda a simulação Monte Carlo (Módulo 5) e, se uma odd
    manual for informada, calcula EV/Kelly (Módulo 8).
    """

    def __init__(self, models_dir: Path = MODELS_DIR, seasons_for_form: int = 8) -> None:
        self.models_dir = Path(models_dir)
        self.seasons_for_form = seasons_for_form

        self.xgb_model = joblib.load(self.models_dir / "xgb_model.pkl")
        self.imputer = joblib.load(self.models_dir / "imputer.pkl")
        with open(self.models_dir / "feature_columns.json", encoding="utf-8") as f:
            self.feature_columns: list[str] = json.load(f)

        self.loader = NFLDataLoader()
        self.fe = FeatureEngineer()
        self.mc_simulator = MonteCarloSimulator()
        self.odds_provider = OddsProvider()

        self._team_game_stats: pd.DataFrame | None = None
        self._elo_system: EloRatingSystem | None = None
        self._elo_history: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Preparação (lazy, cacheada em memória na instância)
    # ------------------------------------------------------------------ #
    def _ensure_team_game_stats(self) -> pd.DataFrame:
        if self._team_game_stats is None:
            import datetime

            current_year = datetime.date.today().year
            seasons = list(range(current_year - self.seasons_for_form, current_year + 1))
            pbp = self.loader.load_pbp(seasons, columns=PBP_FEATURE_COLUMNS, skip_missing=True)
            self._team_game_stats = self.fe.build_team_game_stats(pbp)
        return self._team_game_stats

    def _ensure_elo(self) -> EloRatingSystem:
        if self._elo_system is None:
            import datetime

            current_year = datetime.date.today().year
            schedules = self.loader.load_schedules(list(range(2002, current_year + 1)))
            self._elo_system = EloRatingSystem()
            self._elo_history = self._elo_system.run(schedules)
        return self._elo_system

    # ------------------------------------------------------------------ #
    # Forma atual de um time
    # ------------------------------------------------------------------ #
    def get_team_current_form(self, team: str, last_n_games: int = 8) -> dict:
        """
        Média das últimas `last_n_games` partidas OBSERVADAS do time
        (across temporadas, mais recente primeiro) — representa "como o
        time está jogando agora", usado tanto como feature de ML quanto
        como insumo para o Monte Carlo.
        """
        stats = self._ensure_team_game_stats()
        team_rows = stats[stats["team"] == team].sort_values(["season", "week"])
        if team_rows.empty:
            logger.warning("Sem histórico recente para o time %s — usando médias da liga.", team)
            return {}
        recent = team_rows.tail(last_n_games)
        numeric_cols = [
            c for c in recent.columns
            if c not in {"game_id", "season", "week", "team", "opponent", "is_home"}
        ]
        return recent[numeric_cols].mean(numeric_only=True).to_dict()

    # ------------------------------------------------------------------ #
    # Previsão completa de um jogo
    # ------------------------------------------------------------------ #
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        manual_odds: float | None = None,
        bankroll: float = 100.0,
    ) -> GamePrediction:
        home_form = self.get_team_current_form(home_team, self.seasons_for_form_games())
        away_form = self.get_team_current_form(away_team, self.seasons_for_form_games())

        elo_system = self._ensure_elo()
        elo_home_win_prob = elo_system.predict_win_probability(home_team, away_team)
        ratings = elo_system.current_ratings()
        elo_diff = ratings.get(home_team, 1500) - ratings.get(away_team, 1500)

        ml_home_win_prob = self._predict_ml(home_team, away_team, home_form, away_form)

        # Média simples entre ML e Elo como probabilidade "oficial" exibida —
        # dá robustez quando um dos dois modelos está com sinal fraco (ex.:
        # time novo na liga, poucos jogos de amostra).
        blended = float(np.mean([ml_home_win_prob, elo_home_win_prob]))

        mc_result = self._run_monte_carlo(home_team, away_team, home_form, away_form)

        home_epa_std = self._offense_stability(home_team)
        away_epa_std = self._offense_stability(away_team)
        margin_std = float(np.std(mc_result.raw_home_scores - mc_result.raw_away_scores))
        confidence = compute_confidence_score(elo_diff, home_epa_std, away_epa_std, margin_std)

        odds = self.odds_provider.get_live_odds(home_team, away_team)
        value_bet = None
        if manual_odds is not None:
            value_bet = value_bet_summary(blended, manual_odds, bankroll)

        shap_contributions = self._explain_with_shap(home_team, away_team, home_form, away_form)

        return GamePrediction(
            home_team=home_team,
            away_team=away_team,
            ml_home_win_prob=ml_home_win_prob,
            elo_home_win_prob=elo_home_win_prob,
            blended_home_win_prob=blended,
            confidence_score=confidence,
            monte_carlo=mc_result.to_summary_dict(),
            home_form=home_form,
            away_form=away_form,
            odds=odds,
            value_bet=value_bet,
            mc_result=mc_result,
            shap_contributions=shap_contributions,
        )

    def seasons_for_form_games(self) -> int:
        """Quantos jogos recentes usar na 'forma atual' (heurística simples)."""
        return 8

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _build_feature_row(self, home_team: str, away_team: str, home_form: dict, away_form: dict) -> pd.DataFrame:
        elo_system = self._ensure_elo()
        ratings = elo_system.current_ratings()
        home_elo = ratings.get(home_team, 1500.0)
        away_elo = ratings.get(away_team, 1500.0)
        elo_win_prob = elo_system.predict_win_probability(home_team, away_team)

        row = {}
        for col in self.feature_columns:
            if col.startswith("home_pregame_"):
                key = col.replace("home_pregame_", "")
                row[col] = home_form.get(key, np.nan)
            elif col.startswith("away_pregame_"):
                key = col.replace("away_pregame_", "")
                row[col] = away_form.get(key, np.nan)
            elif col == "home_rest_days" or col == "away_rest_days":
                row[col] = 7.0
            elif col == "home_bye_week" or col == "away_bye_week":
                row[col] = 0
            elif col == "home_elo_pre":
                row[col] = home_elo
            elif col == "away_elo_pre":
                row[col] = away_elo
            elif col == "home_win_prob_elo":
                row[col] = elo_win_prob
            elif col == "elo_diff_pre":
                row[col] = home_elo - away_elo
            else:
                row[col] = np.nan
        return pd.DataFrame([row])[self.feature_columns]

    def _predict_ml(self, home_team: str, away_team: str, home_form: dict, away_form: dict) -> float:
        X = self._build_feature_row(home_team, away_team, home_form, away_form)
        X_imputed = self.imputer.transform(X)
        proba = self.xgb_model.predict_proba(X_imputed)[0, 1]
        return float(proba)

    def _explain_with_shap(
        self, home_team: str, away_team: str, home_form: dict, away_form: dict, top_n: int = 8
    ) -> dict | None:
        """
        Bônus: usa SHAP (TreeExplainer) para explicar quais features mais
        empurraram a previsão para o mandante vencer (valor positivo) ou
        para o visitante vencer (valor negativo). Retorna as `top_n`
        features de maior impacto absoluto.
        """
        try:
            import shap  # import local: dependência pesada, só carregada quando necessário
        except ImportError:
            logger.warning("shap não instalado — pulando explicação (bônus opcional).")
            return None

        X = self._build_feature_row(home_team, away_team, home_form, away_form)
        X_imputed = self.imputer.transform(X)

        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_imputed)
        if isinstance(shap_values, list):  # algumas versões retornam lista por classe
            shap_values = shap_values[-1]

        contributions = dict(zip(self.feature_columns, shap_values[0].tolist()))
        top = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n])
        return top



    def _run_monte_carlo(
        self, home_team: str, away_team: str, home_form: dict, away_form: dict
    ) -> MonteCarloResult:
        home_inputs = TeamGameInputs(
            team=home_team,
            off_epa_per_play=home_form.get("off_epa_per_play", 0.0),
            def_epa_allowed=home_form.get("def_epa_allowed", 0.0),
            redzone_td_pct=home_form.get("redzone_td_pct", np.nan),
            int_rate=home_form.get("int_rate", np.nan),
            sack_rate=home_form.get("sack_rate", np.nan),
            pressure_rate_allowed=away_form.get("pressure_rate", np.nan),
            is_home=True,
        )
        away_inputs = TeamGameInputs(
            team=away_team,
            off_epa_per_play=away_form.get("off_epa_per_play", 0.0),
            def_epa_allowed=away_form.get("def_epa_allowed", 0.0),
            redzone_td_pct=away_form.get("redzone_td_pct", np.nan),
            int_rate=away_form.get("int_rate", np.nan),
            sack_rate=away_form.get("sack_rate", np.nan),
            pressure_rate_allowed=home_form.get("pressure_rate", np.nan),
            is_home=False,
        )
        return self.mc_simulator.simulate(home_inputs, away_inputs)

    def _offense_stability(self, team: str) -> float:
        """Desvio-padrão do EPA/play do time nos últimos jogos (proxy de consistência)."""
        stats = self._ensure_team_game_stats()
        team_rows = stats[stats["team"] == team].sort_values(["season", "week"]).tail(8)
        if len(team_rows) < 2:
            return 0.15  # neutro
        return float(team_rows["off_epa_per_play"].std())


if __name__ == "__main__":
    # Teste rápido manual: python predict.py
    predictor = GamePredictor()
    prediction = predictor.predict_game("KC", "BUF", manual_odds=2.20, bankroll=1000)

    print(f"\n{prediction.home_team} (casa) vs {prediction.away_team} (fora)")
    print(f"Probabilidade ML:   {prediction.ml_home_win_prob:.1%}")
    print(f"Probabilidade Elo:  {prediction.elo_home_win_prob:.1%}")
    print(f"Probabilidade final:{prediction.blended_home_win_prob:.1%}")
    print(f"Confiança: {prediction.confidence_score}/100")
    print(f"\nMonte Carlo: {prediction.monte_carlo['home_avg_score']:.1f} x {prediction.monte_carlo['away_avg_score']:.1f}")
    print(f"Chance de overtime: {prediction.monte_carlo['overtime_prob']:.2%}")
    if prediction.value_bet:
        print(f"\nValue bet (odd manual 2.20): {prediction.value_bet}")
