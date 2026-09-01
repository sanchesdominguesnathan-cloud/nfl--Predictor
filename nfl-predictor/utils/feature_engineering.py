"""
Módulo 2 — Engenharia de Features.

Transforma play-by-play bruto em métricas por time/jogo (ataque, defesa e
QB) e depois em features "pré-jogo": médias móveis (expanding, sem
vazamento de dados) do que cada time vinha fazendo ANTES de cada partida.

Todas as features são documentadas na docstring de cada função e no
dicionário `FEATURE_DEFINITIONS` ao final do arquivo.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tipos de jogada considerados "scrimmage plays" (exclui kickoff, punt, PAT etc.)
_SCRIMMAGE_PLAY_TYPES = {"pass", "run"}

# Colunas de pbp efetivamente usadas por este módulo. Passar isso para
# `NFLDataLoader.load_pbp(..., columns=PBP_FEATURE_COLUMNS)` reduz o uso de
# memória em ~15x ao processar várias temporadas de uma vez (pbp completo
# tem ~370 colunas; usamos ~20).
PBP_FEATURE_COLUMNS: list[str] = [
    "game_id", "season", "season_type", "week", "home_team", "away_team",
    "posteam", "defteam", "play_type", "epa", "success", "yards_gained",
    "down", "third_down_converted", "yardline_100", "fixed_drive", "touchdown",
    "sack", "pass_attempt", "qb_hit", "interception", "fumble_lost",
    "qb_epa", "cpoe",
]


class FeatureEngineer:
    """
    Constrói a tabela de features pré-jogo a partir de play-by-play e schedules.

    Fluxo:
        1. `build_team_game_stats(pbp)`   -> uma linha por (time, jogo) com
           estatísticas OBSERVADAS naquele jogo (ataque, defesa, QB).
        2. `build_pregame_features(team_game_stats)` -> mesma granularidade,
           mas cada valor é a média móvel (expanding) de todos os jogos
           ANTERIORES do time na temporada — ou seja, o que se sabia sobre o
           time ANTES daquele jogo começar. Evita vazamento de dados (o
           modelo nunca "vê" o resultado do jogo que está tentando prever).
        3. `build_game_feature_table(...)` -> junta home/away lado a lado,
           uma linha por jogo, pronta para treino do modelo.
    """

    # ------------------------------------------------------------------ #
    # Passo 1: estatísticas observadas por time/jogo
    # ------------------------------------------------------------------ #
    def build_team_game_stats(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega o play-by-play em uma linha por (game_id, team), contendo
        as métricas de ATAQUE, DEFESA e QB observadas naquele jogo.

        Métricas de ataque (quando `team` estava de posse):
            off_epa_per_play   - EPA médio por jogada
            off_success_rate   - % de jogadas "bem sucedidas" (success=1)
            off_pass_epa       - EPA médio em jogadas de passe
            off_rush_epa       - EPA médio em jogadas de corrida
            off_yards_per_play - jazidas médias por jogada (pass+rush)
            redzone_td_pct     - % de possessões em red zone (yardline<=20)
                                  que terminaram em touchdown
            third_down_pct     - % de conversões de 3ª descida

        Métricas de defesa (quando `team` era o defteam):
            def_epa_allowed         - EPA médio permitido por jogada
            def_success_rate_allowed- success rate permitido
            sacks                   - sacks realizados pela defesa
            pressure_rate           - % de dropbacks do adversário com QB hit ou sack
            turnovers_forced        - interceptações + fumbles recuperados pela defesa

        Métricas de QB (do passador principal do time naquele jogo):
            qb_epa    - EPA médio do passador em jogadas de passe
            cpoe      - completion % over expected médio
            int_rate  - % de tentativas de passe que resultaram em interceptação
            sack_rate - % de dropbacks (pass_attempt + sack) que terminaram em sack
        """
        pbp = pbp[pbp["season_type"] == "REG"].copy() if "season_type" in pbp.columns else pbp.copy()
        scrimmage = pbp[pbp["play_type"].isin(_SCRIMMAGE_PLAY_TYPES)].copy()

        offense = self._offense_stats(scrimmage)
        defense = self._defense_stats(scrimmage, pbp)
        qb = self._qb_stats(scrimmage)

        stats = offense.merge(defense, on=["game_id", "team"], how="outer")
        stats = stats.merge(qb, on=["game_id", "team"], how="outer")

        # Reanexa contexto do jogo (season, week, opponent, home/away)
        context = self._game_context(pbp)
        stats = stats.merge(context, on=["game_id", "team"], how="left")

        return stats.sort_values(["season", "week", "team"]).reset_index(drop=True)

    def _offense_stats(self, scrimmage: pd.DataFrame) -> pd.DataFrame:
        off = scrimmage.groupby(["game_id", "posteam"], dropna=True).agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            off_yards_per_play=("yards_gained", "mean"),
        ).reset_index().rename(columns={"posteam": "team"})

        pass_epa = (
            scrimmage[scrimmage["play_type"] == "pass"]
            .groupby(["game_id", "posteam"])["epa"].mean()
            .reset_index().rename(columns={"posteam": "team", "epa": "off_pass_epa"})
        )
        rush_epa = (
            scrimmage[scrimmage["play_type"] == "run"]
            .groupby(["game_id", "posteam"])["epa"].mean()
            .reset_index().rename(columns={"posteam": "team", "epa": "off_rush_epa"})
        )

        third_down = scrimmage[scrimmage["down"] == 3]
        third_down_pct = (
            third_down.groupby(["game_id", "posteam"])["third_down_converted"].mean()
            .reset_index().rename(columns={"posteam": "team", "third_down_converted": "third_down_pct"})
        )

        redzone = scrimmage[scrimmage["yardline_100"] <= 20]
        # Uma linha por (game, team, drive) para saber se a red-zone trip terminou em TD
        if "fixed_drive" in scrimmage.columns:
            rz_drives = redzone.groupby(["game_id", "posteam", "fixed_drive"])["touchdown"].max().reset_index()
            redzone_td_pct = (
                rz_drives.groupby(["game_id", "posteam"])["touchdown"].mean()
                .reset_index().rename(columns={"posteam": "team", "touchdown": "redzone_td_pct"})
            )
        else:
            redzone_td_pct = pd.DataFrame(columns=["game_id", "team", "redzone_td_pct"])

        out = off.merge(pass_epa, on=["game_id", "team"], how="left")
        out = out.merge(rush_epa, on=["game_id", "team"], how="left")
        out = out.merge(third_down_pct, on=["game_id", "team"], how="left")
        out = out.merge(redzone_td_pct, on=["game_id", "team"], how="left")
        return out

    def _defense_stats(self, scrimmage: pd.DataFrame, pbp_full: pd.DataFrame) -> pd.DataFrame:
        de = scrimmage.groupby(["game_id", "defteam"], dropna=True).agg(
            def_epa_allowed=("epa", "mean"),
            def_success_rate_allowed=("success", "mean"),
        ).reset_index().rename(columns={"defteam": "team"})

        sacks = (
            scrimmage.groupby(["game_id", "defteam"])["sack"].sum()
            .reset_index().rename(columns={"defteam": "team", "sack": "sacks"})
        )

        dropbacks = scrimmage[(scrimmage["pass_attempt"] == 1) | (scrimmage["sack"] == 1)].copy()
        dropbacks["pressured"] = ((dropbacks["qb_hit"] == 1) | (dropbacks["sack"] == 1)).astype(int)
        pressure = (
            dropbacks.groupby(["game_id", "defteam"])["pressured"].mean()
            .reset_index().rename(columns={"defteam": "team", "pressured": "pressure_rate"})
        )

        turnovers = scrimmage.copy()
        turnovers["turnover"] = (
            (turnovers["interception"] == 1) | (turnovers["fumble_lost"] == 1)
        ).astype(int)
        turnovers_forced = (
            turnovers.groupby(["game_id", "defteam"])["turnover"].sum()
            .reset_index().rename(columns={"defteam": "team", "turnover": "turnovers_forced"})
        )

        out = de.merge(sacks, on=["game_id", "team"], how="left")
        out = out.merge(pressure, on=["game_id", "team"], how="left")
        out = out.merge(turnovers_forced, on=["game_id", "team"], how="left")
        return out

    def _qb_stats(self, scrimmage: pd.DataFrame) -> pd.DataFrame:
        passes = scrimmage[scrimmage["play_type"] == "pass"].copy()

        qb_epa = (
            passes.groupby(["game_id", "posteam"])["qb_epa"].mean()
            .reset_index().rename(columns={"posteam": "team"})
        )
        cpoe = (
            passes.groupby(["game_id", "posteam"])["cpoe"].mean()
            .reset_index().rename(columns={"posteam": "team"})
        )
        int_rate = (
            passes.groupby(["game_id", "posteam"])["interception"].mean()
            .reset_index().rename(columns={"posteam": "team", "interception": "int_rate"})
        )

        dropbacks = scrimmage[(scrimmage["pass_attempt"] == 1) | (scrimmage["sack"] == 1)]
        sack_rate = (
            dropbacks.groupby(["game_id", "posteam"])["sack"].mean()
            .reset_index().rename(columns={"posteam": "team", "sack": "sack_rate"})
        )

        out = qb_epa.merge(cpoe, on=["game_id", "team"], how="outer")
        out = out.merge(int_rate, on=["game_id", "team"], how="outer")
        out = out.merge(sack_rate, on=["game_id", "team"], how="outer")
        return out

    def _game_context(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Uma linha por (game_id, team) com season/week/opponent/is_home."""
        base_cols = ["game_id", "season", "week", "home_team", "away_team"]
        games = pbp[base_cols].drop_duplicates()

        home = games.rename(columns={"home_team": "team", "away_team": "opponent"}).copy()
        home["is_home"] = 1
        away = games.rename(columns={"away_team": "team", "home_team": "opponent"}).copy()
        away["is_home"] = 0

        return pd.concat([home, away], ignore_index=True)[
            ["game_id", "season", "week", "team", "opponent", "is_home"]
        ]

    # ------------------------------------------------------------------ #
    # Passo 2: features pré-jogo (sem vazamento de dados)
    # ------------------------------------------------------------------ #
    def build_pregame_features(self, team_game_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada (time, jogo), calcula a média expanding de todas as
        métricas usando SOMENTE jogos anteriores na mesma temporada
        (`shift(1)` antes do `expanding().mean()`). O primeiro jogo de
        cada time na temporada fica com NaN (ainda não há histórico) —
        o `train_model.py` decide como tratar isso (ex.: imputar com a
        média da liga ou o rating final da temporada anterior).
        """
        metric_cols = [
            c for c in team_game_stats.columns
            if c not in {"game_id", "season", "week", "team", "opponent", "is_home"}
        ]

        df = team_game_stats.sort_values(["team", "season", "week"]).copy()
        grouped = df.groupby(["team", "season"], group_keys=False)

        pregame = df[["game_id", "season", "week", "team", "opponent", "is_home"]].copy()
        for col in metric_cols:
            pregame[col] = grouped[col].apply(lambda s: s.shift(1).expanding().mean())

        pregame = pregame.rename(columns={c: f"pregame_{c}" for c in metric_cols})
        return pregame.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Passo 3: tabela final por jogo (home vs away lado a lado)
    # ------------------------------------------------------------------ #
    def build_game_feature_table(
        self,
        pregame_features: pd.DataFrame,
        schedules: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Junta as features pré-jogo do mandante e do visitante em uma única
        linha por jogo, mais o contexto (descanso, distância de viagem,
        bye week) calculado a partir do `schedules`.
        """
        home_feats = pregame_features[pregame_features["is_home"] == 1].add_prefix("home_")
        away_feats = pregame_features[pregame_features["is_home"] == 0].add_prefix("away_")

        merged = home_feats.merge(
            away_feats,
            left_on="home_game_id",
            right_on="away_game_id",
            how="inner",
        )
        merged = merged.rename(columns={
            "home_game_id": "game_id",
            "home_season": "season",
            "home_week": "week",
            "home_team": "home_team",
            "away_team": "away_team",
        })
        merged = merged.drop(columns=["away_game_id", "away_season", "away_week"], errors="ignore")

        context = self._context_features(schedules)
        merged = merged.merge(context, on="game_id", how="left")

        return merged.sort_values(["season", "week"]).reset_index(drop=True)

    def _context_features(self, schedules: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features de contexto a partir do schedule:
            home_rest_days / away_rest_days - dias desde o último jogo
            home_bye_week / away_bye_week   - 1 se o time descansou uma semana a mais
        (Distância de viagem e Elo são calculados em módulos dedicados e
        mesclados posteriormente por `game_id`.)
        """
        sched = schedules.copy()
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")

        rest_records = []
        for _, row in sched.iterrows():
            rest_records.append({"game_id": row["game_id"], "team": row["home_team"], "gameday": row["gameday"], "season": row["season"]})
            rest_records.append({"game_id": row["game_id"], "team": row["away_team"], "gameday": row["gameday"], "season": row["season"]})
        rest_df = pd.DataFrame(rest_records).sort_values(["team", "season", "gameday"])
        rest_df["prev_gameday"] = rest_df.groupby(["team", "season"])["gameday"].shift(1)
        rest_df["rest_days"] = (rest_df["gameday"] - rest_df["prev_gameday"]).dt.days
        rest_df["bye_week"] = (rest_df["rest_days"] > 10).astype("Int64")

        home_rest = rest_df.merge(sched[["game_id", "home_team"]], left_on=["game_id", "team"], right_on=["game_id", "home_team"])
        home_rest = home_rest[["game_id", "rest_days", "bye_week"]].rename(
            columns={"rest_days": "home_rest_days", "bye_week": "home_bye_week"}
        )
        away_rest = rest_df.merge(sched[["game_id", "away_team"]], left_on=["game_id", "team"], right_on=["game_id", "away_team"])
        away_rest = away_rest[["game_id", "rest_days", "bye_week"]].rename(
            columns={"rest_days": "away_rest_days", "bye_week": "away_bye_week"}
        )

        return home_rest.merge(away_rest, on="game_id", how="outer")


# --------------------------------------------------------------------------- #
# Documentação centralizada de todas as features geradas
# --------------------------------------------------------------------------- #
FEATURE_DEFINITIONS: dict[str, str] = {
    "off_epa_per_play": "EPA médio por jogada de scrimmage (ataque).",
    "off_success_rate": "% de jogadas de ataque consideradas bem sucedidas (coluna `success` do nflverse).",
    "off_pass_epa": "EPA médio em jogadas de passe.",
    "off_rush_epa": "EPA médio em jogadas de corrida.",
    "off_yards_per_play": "Jardas médias por jogada de scrimmage.",
    "third_down_pct": "% de conversões de 3ª descida.",
    "redzone_td_pct": "% de possessões em red zone (yardline<=20) que terminaram em touchdown.",
    "def_epa_allowed": "EPA médio permitido por jogada (defesa).",
    "def_success_rate_allowed": "% de jogadas do adversário bem sucedidas contra esta defesa.",
    "sacks": "Sacks realizados pela defesa no jogo.",
    "pressure_rate": "% de dropbacks do adversário com QB hit ou sack.",
    "turnovers_forced": "Interceptações + fumbles recuperados pela defesa.",
    "qb_epa": "EPA médio do passador principal em jogadas de passe.",
    "cpoe": "Completion % Over Expected médio do passador.",
    "int_rate": "% de tentativas de passe que resultaram em interceptação.",
    "sack_rate": "% de dropbacks que terminaram em sack.",
    "rest_days": "Dias corridos desde o jogo anterior do time.",
    "bye_week": "1 se o time teve bye week (rest_days > 10) antes deste jogo.",
    "is_home": "1 se o time era o mandante.",
}


if __name__ == "__main__":
    # Teste rápido manual: python -m utils.feature_engineering
    from utils.data_loader import NFLDataLoader

    loader = NFLDataLoader()
    pbp = loader.load_pbp([2023, 2024])
    schedules = loader.load_schedules([2023, 2024])

    fe = FeatureEngineer()
    team_game_stats = fe.build_team_game_stats(pbp)
    print("team_game_stats:", team_game_stats.shape)
    print(team_game_stats.head(3))

    pregame = fe.build_pregame_features(team_game_stats)
    print("\npregame_features:", pregame.shape)
    print(pregame.dropna().head(3))

    game_table = fe.build_game_feature_table(pregame, schedules)
    print("\ngame_feature_table:", game_table.shape)
    print(game_table.columns.tolist())
