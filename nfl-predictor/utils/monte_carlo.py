"""
Módulo 5 — Simulação Monte Carlo.

Depois que o modelo de ML estima a força relativa dos dois times (via EPA
ofensivo/defensivo pré-jogo), este módulo simula 10.000 partidas jogada-a-
-jogada em nível de "drives de pontuação" (não só um sorteio de vencedor),
para produzir uma distribuição real de placares — incluindo probabilidade
de overtime e de vencer por margens específicas (+3, +7, +14).

Como o placar é simulado:
    1. Convertemos EPA/play pré-jogo em pontos esperados por time (a EPA é,
       por construção, medida em "pontos esperados adicionados", então
       EPA/play * jogadas/jogo aproxima a diferença de pontos esperada).
    2. Pontos esperados viram um número esperado de "drives de pontuação"
       (posses que terminam em TD ou FG).
    3. Para CADA simulação, sorteamos:
         - um multiplicador de eficiência ofensiva (variável "eficiência
           ofensiva" do enunciado) que escala o número de drives de
           pontuação daquela partida inteira (correlaciona os eventos
           dentro do mesmo jogo simulado, mais realista que sortear cada
           posse de forma independente);
         - turnovers (variável "turnovers"), que removem drives de
           pontuação do time que erra;
         - pressão do adversário (variável "pressão"), que reduz a
           eficiência ofensiva quando o adversário tem alta pressure_rate;
         - big plays (variável "big plays"), uma chance extra de um drive
           de pontuação bônus.
    4. Cada drive de pontuação simulado sorteia entre TD (7/6/8 pontos,
       conforme a taxa de conversão em 2 pontos e extra-point) e FG (3
       pontos), usando a red-zone TD% pré-jogo do time quando disponível.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import MONTE_CARLO_RANDOM_STATE, MONTE_CARLO_SIMULATIONS

# --------------------------------------------------------------------------- #
# Constantes da liga (médias históricas aproximadas da NFL moderna)
# --------------------------------------------------------------------------- #
LEAGUE_AVG_POINTS_PER_TEAM: float = 22.5
LEAGUE_AVG_PLAYS_PER_TEAM: float = 63.0
LEAGUE_AVG_SCORE_VALUE: float = 5.3       # pontos médios por "drive de pontuação" (mix TD/FG)
LEAGUE_AVG_REDZONE_TD_PCT: float = 0.58   # taxa média de TD em red zone
HOME_POINTS_BUMP: float = 1.0             # pequeno bônus de mando de campo, em pontos

_TD_POINT_VALUES = np.array([7, 6, 8])
_TD_POINT_WEIGHTS = np.array([0.94, 0.04, 0.02])  # XP bom / XP errado / 2pt convertido


@dataclass
class TeamGameInputs:
    """Insumos pré-jogo de um time para a simulação (vindos do Módulo 2/3)."""

    team: str
    off_epa_per_play: float = 0.0
    def_epa_allowed: float = 0.0
    redzone_td_pct: float = LEAGUE_AVG_REDZONE_TD_PCT
    int_rate: float = 0.025
    sack_rate: float = 0.06
    pressure_rate_allowed: float = 0.22  # pressure_rate que a DEFESA do adversário costuma gerar
    is_home: bool = False

    def with_defaults(self) -> "TeamGameInputs":
        """Substitui NaN por valores neutros (liga média) para não quebrar a simulação."""
        def clean(x, default):
            return default if (x is None or (isinstance(x, float) and np.isnan(x))) else x
        return TeamGameInputs(
            team=self.team,
            off_epa_per_play=clean(self.off_epa_per_play, 0.0),
            def_epa_allowed=clean(self.def_epa_allowed, 0.0),
            redzone_td_pct=clean(self.redzone_td_pct, LEAGUE_AVG_REDZONE_TD_PCT),
            int_rate=clean(self.int_rate, 0.025),
            sack_rate=clean(self.sack_rate, 0.06),
            pressure_rate_allowed=clean(self.pressure_rate_allowed, 0.22),
            is_home=self.is_home,
        )


@dataclass
class MonteCarloResult:
    home_team: str
    away_team: str
    n_simulations: int
    home_win_prob: float
    away_win_prob: float
    overtime_prob: float
    home_avg_score: float
    away_avg_score: float
    home_median_score: float
    away_median_score: float
    most_likely_scores: list[tuple[tuple[int, int], float]]  # [((home,away), prob), ...]
    home_win_margin_probs: dict[str, float]  # {"+3": p, "+7": p, "+14": p}
    away_win_margin_probs: dict[str, float]
    raw_home_scores: np.ndarray = field(repr=False)
    raw_away_scores: np.ndarray = field(repr=False)

    def to_summary_dict(self) -> dict:
        """Versão serializável (sem os arrays brutos) para exibir no dashboard/JSON."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "n_simulations": self.n_simulations,
            "home_win_prob": self.home_win_prob,
            "away_win_prob": self.away_win_prob,
            "overtime_prob": self.overtime_prob,
            "home_avg_score": self.home_avg_score,
            "away_avg_score": self.away_avg_score,
            "home_median_score": self.home_median_score,
            "away_median_score": self.away_median_score,
            "most_likely_scores": self.most_likely_scores,
            "home_win_margin_probs": self.home_win_margin_probs,
            "away_win_margin_probs": self.away_win_margin_probs,
        }


class MonteCarloSimulator:
    """
    Simulador Monte Carlo de jogos de NFL.

    Exemplo:
        >>> home = TeamGameInputs(team="KC", off_epa_per_play=0.12, def_epa_allowed=-0.03, is_home=True)
        >>> away = TeamGameInputs(team="BUF", off_epa_per_play=0.09, def_epa_allowed=0.01)
        >>> sim = MonteCarloSimulator()
        >>> result = sim.simulate(home, away)
        >>> result.home_win_prob
    """

    def __init__(
        self,
        n_simulations: int = MONTE_CARLO_SIMULATIONS,
        random_state: int | None = MONTE_CARLO_RANDOM_STATE,
    ) -> None:
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_state)

    def simulate(self, home: TeamGameInputs, away: TeamGameInputs) -> MonteCarloResult:
        home = home.with_defaults()
        away = away.with_defaults()
        n = self.n_simulations

        expected_home_pts, expected_away_pts = self._expected_points(home, away)
        home_scores, away_scores = self._simulate_scores(
            home, away, expected_home_pts, expected_away_pts, n
        )

        home_win = home_scores > away_scores
        tie = home_scores == away_scores
        away_win = home_scores < away_scores

        overtime_prob = float(np.mean(tie))
        # Em caso de empate na simulação, distribuímos a "vitória" 50/50 para
        # reportar uma probabilidade de vitória bem definida (na vida real,
        # o jogo iria para a prorrogação e um dos dois venceria).
        home_win_prob = float(np.mean(home_win) + 0.5 * np.mean(tie))
        away_win_prob = float(np.mean(away_win) + 0.5 * np.mean(tie))

        margin = home_scores - away_scores
        home_margin_probs = {
            "+3": float(np.mean(margin >= 3)),
            "+7": float(np.mean(margin >= 7)),
            "+14": float(np.mean(margin >= 14)),
        }
        away_margin_probs = {
            "+3": float(np.mean(-margin >= 3)),
            "+7": float(np.mean(-margin >= 7)),
            "+14": float(np.mean(-margin >= 14)),
        }

        most_likely = self._most_likely_scores(home_scores, away_scores, top_n=5)

        return MonteCarloResult(
            home_team=home.team,
            away_team=away.team,
            n_simulations=n,
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            overtime_prob=overtime_prob,
            home_avg_score=float(np.mean(home_scores)),
            away_avg_score=float(np.mean(away_scores)),
            home_median_score=float(np.median(home_scores)),
            away_median_score=float(np.median(away_scores)),
            most_likely_scores=most_likely,
            home_win_margin_probs=home_margin_probs,
            away_win_margin_probs=away_margin_probs,
            raw_home_scores=home_scores,
            raw_away_scores=away_scores,
        )

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _expected_points(self, home: TeamGameInputs, away: TeamGameInputs) -> tuple[float, float]:
        """Pontos esperados por time, combinando EPA ofensivo próprio e EPA defensivo do rival."""
        home_epa_per_play = 0.5 * home.off_epa_per_play + 0.5 * away.def_epa_allowed
        away_epa_per_play = 0.5 * away.off_epa_per_play + 0.5 * home.def_epa_allowed

        expected_home = (
            LEAGUE_AVG_POINTS_PER_TEAM
            + home_epa_per_play * LEAGUE_AVG_PLAYS_PER_TEAM
            + HOME_POINTS_BUMP
        )
        expected_away = (
            LEAGUE_AVG_POINTS_PER_TEAM
            + away_epa_per_play * LEAGUE_AVG_PLAYS_PER_TEAM
        )
        return max(expected_home, 3.0), max(expected_away, 3.0)

    def _simulate_scores(
        self,
        home: TeamGameInputs,
        away: TeamGameInputs,
        expected_home_pts: float,
        expected_away_pts: float,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        home_scores = np.zeros(n, dtype=int)
        away_scores = np.zeros(n, dtype=int)

        base_home_drives = expected_home_pts / LEAGUE_AVG_SCORE_VALUE
        base_away_drives = expected_away_pts / LEAGUE_AVG_SCORE_VALUE

        # Variável "eficiência ofensiva": um multiplicador por jogo (correlaciona
        # todos os drives daquela simulação, em vez de sortear cada um isolado).
        home_efficiency = self.rng.normal(1.0, 0.20, size=n)
        away_efficiency = self.rng.normal(1.0, 0.20, size=n)

        # Variável "pressão": defesa com pressure_rate acima da média reduz a
        # eficiência ofensiva esperada do adversário.
        home_efficiency *= 1.0 - 0.35 * (away.pressure_rate_allowed - 0.22)
        away_efficiency *= 1.0 - 0.35 * (home.pressure_rate_allowed - 0.22)

        # Variável "turnovers": cada turnover custa, em média, ~1 drive de
        # pontuação a menos para quem erra.
        home_turnovers = self.rng.poisson(lam=max(home.int_rate * 10 + home.sack_rate * 3, 0.05), size=n)
        away_turnovers = self.rng.poisson(lam=max(away.int_rate * 10 + away.sack_rate * 3, 0.05), size=n)

        home_drives = np.clip(base_home_drives * home_efficiency - 0.4 * home_turnovers, 0.3, None)
        away_drives = np.clip(base_away_drives * away_efficiency - 0.4 * away_turnovers, 0.3, None)

        n_home_drives = self.rng.poisson(lam=home_drives)
        n_away_drives = self.rng.poisson(lam=away_drives)

        # Variável "big plays": ~15% de chance de um drive de pontuação bônus.
        n_home_drives += (self.rng.random(n) < 0.15).astype(int)
        n_away_drives += (self.rng.random(n) < 0.15).astype(int)

        max_drives = int(max(n_home_drives.max(), n_away_drives.max(), 1))
        home_td_prob = np.clip(home.redzone_td_pct, 0.2, 0.9)
        away_td_prob = np.clip(away.redzone_td_pct, 0.2, 0.9)

        for team_scores, n_drives, td_prob in (
            (home_scores, n_home_drives, home_td_prob),
            (away_scores, n_away_drives, away_td_prob),
        ):
            for drive_count in range(1, max_drives + 1):
                mask = n_drives >= drive_count
                if not mask.any():
                    continue
                n_active = int(mask.sum())
                is_td = self.rng.random(n_active) < td_prob
                points = np.where(
                    is_td,
                    self.rng.choice(_TD_POINT_VALUES, size=n_active, p=_TD_POINT_WEIGHTS),
                    3,
                )
                team_scores[mask] += points

        return home_scores, away_scores

    def _most_likely_scores(
        self, home_scores: np.ndarray, away_scores: np.ndarray, top_n: int = 5
    ) -> list[tuple[tuple[int, int], float]]:
        pairs = pd.Series(list(zip(home_scores.tolist(), away_scores.tolist())))
        counts = pairs.value_counts(normalize=True).head(top_n)
        return [(idx, float(prob)) for idx, prob in counts.items()]


if __name__ == "__main__":
    # Teste rápido manual: python -m utils.monte_carlo
    home = TeamGameInputs(
        team="KC", off_epa_per_play=0.12, def_epa_allowed=-0.05,
        redzone_td_pct=0.62, int_rate=0.018, sack_rate=0.05,
        pressure_rate_allowed=0.19, is_home=True,
    )
    away = TeamGameInputs(
        team="BUF", off_epa_per_play=0.10, def_epa_allowed=0.02,
        redzone_td_pct=0.58, int_rate=0.022, sack_rate=0.07,
        pressure_rate_allowed=0.25, is_home=False,
    )

    sim = MonteCarloSimulator()
    result = sim.simulate(home, away)

    print(f"{result.home_team} vs {result.away_team} — {result.n_simulations} simulações")
    print(f"Chance de vitória: {result.home_team} {result.home_win_prob:.1%} | {result.away_team} {result.away_win_prob:.1%}")
    print(f"Placar médio: {result.home_team} {result.home_avg_score:.1f} x {result.away_avg_score:.1f} {result.away_team}")
    print(f"Chance de overtime: {result.overtime_prob:.2%}")
    print(f"{result.home_team} vencer por +3/+7/+14: {result.home_win_margin_probs}")
    print(f"{result.away_team} vencer por +3/+7/+14: {result.away_win_margin_probs}")
    print("Placares mais prováveis:")
    for (h, a), prob in result.most_likely_scores:
        print(f"  {h} x {a}: {prob:.2%}")
