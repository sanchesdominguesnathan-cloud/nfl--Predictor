"""
Módulo 3 — Elo Dinâmico.

Implementação própria (não usa nenhuma lib externa de Elo) do sistema de
rating Elo para times de NFL, com:
    - Rating inicial = 1500
    - Atualização após cada jogo
    - Vantagem de mando ajustável (soma-se ao rating do mandante antes de
      calcular a probabilidade esperada)
    - K-factor configurável
    - Regressão à média entre temporadas (evita que um time carregue um
      rating extremo de uma temporada para a outra)
    - Histórico salvo em CSV

IMPORTANTE sobre vazamento de dados: `run()` retorna, para cada jogo, o
rating de CADA time ANTES daquele jogo (`home_elo_pre` / `away_elo_pre`) —
essas são as colunas que devem ser usadas como feature de treino. O rating
"depois" do jogo (`home_elo_post` / `away_elo_post`) só é conhecido
retroativamente e nunca deve virar feature de um modelo que tenta prever
esse mesmo jogo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import (
    ELO_HISTORY_DIR,
    ELO_HOME_FIELD_ADVANTAGE,
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ELO_SEASON_REGRESSION,
)

logger = logging.getLogger(__name__)


class EloRatingSystem:
    """
    Sistema de rating Elo dinâmico para times de NFL.

    Exemplo:
        >>> elo = EloRatingSystem()
        >>> history = elo.run(schedules)  # DataFrame com 1 linha por jogo
        >>> elo.current_ratings()          # dict {time: rating atual}
        >>> elo.save_history(history)
    """

    def __init__(
        self,
        initial_rating: float = ELO_INITIAL_RATING,
        k_factor: float = ELO_K_FACTOR,
        home_field_advantage: float = ELO_HOME_FIELD_ADVANTAGE,
        season_regression: float = ELO_SEASON_REGRESSION,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_field_advantage = home_field_advantage
        self.season_regression = season_regression
        self._ratings: dict[str, float] = {}
        self._current_season: int | None = None

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def current_ratings(self) -> dict[str, float]:
        """Retorna uma cópia do dicionário {time: rating} no estado atual."""
        return dict(self._ratings)

    def run(self, schedules: pd.DataFrame) -> pd.DataFrame:
        """
        Processa todos os jogos do schedule em ordem cronológica, atualizando
        os ratings a cada resultado, e retorna o histórico completo.

        Espera colunas: season, week, gameday (ou game_date), home_team,
        away_team, home_score, away_score, game_id.
        """
        sched = schedules.copy()
        date_col = "gameday" if "gameday" in sched.columns else "game_date"
        sched[date_col] = pd.to_datetime(sched[date_col], errors="coerce")

        # Só processamos jogos já disputados (com placar) e temporada regular,
        # exceto se o schedule não tiver season_type (assume tudo relevante).
        if "game_type" in sched.columns:
            sched = sched[sched["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
        played = sched.dropna(subset=["home_score", "away_score"]).copy()
        played = played.sort_values(["season", date_col, "week"]).reset_index(drop=True)

        records = []
        for row in played.itertuples(index=False):
            season = row.season
            self._maybe_regress_to_mean(season)

            home = row.home_team
            away = row.away_team
            home_pre = self._get_rating(home)
            away_pre = self._get_rating(away)

            home_win_prob = self._expected_score(home_pre, away_pre)

            if row.home_score > row.away_score:
                actual_home = 1.0
            elif row.home_score < row.away_score:
                actual_home = 0.0
            else:
                actual_home = 0.5  # empate

            margin = abs(row.home_score - row.away_score)
            mov_multiplier = self._margin_of_victory_multiplier(margin, home_pre, away_pre, actual_home)

            delta = self.k_factor * mov_multiplier * (actual_home - home_win_prob)
            home_post = home_pre + delta
            away_post = away_pre - delta

            self._ratings[home] = home_post
            self._ratings[away] = away_post

            records.append({
                "game_id": row.game_id,
                "season": season,
                "week": row.week,
                "home_team": home,
                "away_team": away,
                "home_elo_pre": home_pre,
                "away_elo_pre": away_pre,
                "home_win_prob_elo": home_win_prob,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "home_elo_post": home_post,
                "away_elo_post": away_post,
                "elo_diff_pre": (home_pre + self.home_field_advantage) - away_pre,
            })

        return pd.DataFrame(records)

    def predict_win_probability(self, home_team: str, away_team: str) -> float:
        """Probabilidade de vitória do mandante com os ratings atuais (uso em `predict.py`)."""
        home_rating = self._get_rating(home_team)
        away_rating = self._get_rating(away_team)
        return self._expected_score(home_rating, away_rating)

    def save_history(self, history: pd.DataFrame, path: Path | None = None) -> Path:
        """Salva o histórico de Elo em CSV (Módulo 3: 'Salvar histórico em CSV')."""
        out_path = path or (ELO_HISTORY_DIR / "elo_history.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(out_path, index=False)
        logger.info("Histórico de Elo salvo em %s (%d jogos)", out_path, len(history))
        return out_path

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _get_rating(self, team: str) -> float:
        return self._ratings.setdefault(team, self.initial_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probabilidade esperada de A vencer, já somando a vantagem de mando a A."""
        adjusted_a = rating_a + self.home_field_advantage
        return 1.0 / (1.0 + 10 ** ((rating_b - adjusted_a) / 400.0))

    def _margin_of_victory_multiplier(
        self, margin: float, home_pre: float, away_pre: float, actual_home: float
    ) -> float:
        """
        Multiplicador de margem de vitória (inspirado no Elo do 538/FiveThirtyEight):
        vitórias por goleada pesam mais, mas o efeito é amortecido quando o
        favorito já era muito mais forte (evita "chutar" o rating para longe
        em blowouts esperados).
        """
        winner_pre = home_pre if actual_home >= 0.5 else away_pre
        loser_pre = away_pre if actual_home >= 0.5 else home_pre
        elo_diff_winner = winner_pre - loser_pre
        return ((margin + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)

    def _maybe_regress_to_mean(self, season: int) -> None:
        """No início de cada nova temporada, puxa todos os ratings em direção a 1500."""
        if self._current_season is None:
            self._current_season = season
            return
        if season == self._current_season:
            return

        for team, rating in self._ratings.items():
            self._ratings[team] = (
                rating * (1 - self.season_regression) + self.initial_rating * self.season_regression
            )
        self._current_season = season
        logger.info("Regressão à média aplicada para a temporada %s", season)


if __name__ == "__main__":
    # Teste rápido manual: python -m utils.elo
    from utils.data_loader import NFLDataLoader

    loader = NFLDataLoader()
    schedules = loader.load_schedules(seasons=list(range(2010, 2026)))

    elo = EloRatingSystem()
    history = elo.run(schedules)
    elo.save_history(history)

    print("Histórico:", history.shape)
    print(history.tail(10)[["season", "week", "home_team", "away_team", "home_elo_pre", "away_elo_pre", "home_win_prob_elo"]])

    print("\nTop 10 ratings atuais:")
    ratings = elo.current_ratings()
    for team, rating in sorted(ratings.items(), key=lambda x: -x[1])[:10]:
        print(f"  {team}: {rating:.1f}")
