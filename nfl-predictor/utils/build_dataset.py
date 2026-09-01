"""
Pipeline de construção do dataset final de treino.

Une:
    - Módulo 1 (NFLDataLoader)     -> pbp, schedules
    - Módulo 2 (FeatureEngineer)   -> features de EPA/success/QB pré-jogo
    - Módulo 3 (EloRatingSystem)   -> elo_pre de cada time, home_win_prob_elo

... em uma única tabela (`game_id` como chave), pronta para `train_model.py`
consumir e para o dashboard exibir "quais fatores mais influenciaram".
"""

from __future__ import annotations

import logging

import pandas as pd

from utils.data_loader import NFLDataLoader
from utils.elo import EloRatingSystem
from utils.feature_engineering import PBP_FEATURE_COLUMNS, FeatureEngineer

logger = logging.getLogger(__name__)


def build_full_dataset(season_start: int, season_end: int) -> pd.DataFrame:
    """
    Constrói o dataset completo de features pré-jogo para as temporadas
    [season_start, season_end] (inclusive).

    Retorna um DataFrame com uma linha por jogo já disputado, contendo:
        - features de ataque/defesa/QB pré-jogo (home_pregame_* / away_pregame_*)
        - contexto (rest_days, bye_week)
        - elo pré-jogo (home_elo_pre, away_elo_pre, home_win_prob_elo, elo_diff_pre)
        - o alvo de treino: home_win (1 se o mandante venceu)
    """
    seasons = list(range(season_start, season_end + 1))
    loader = NFLDataLoader()

    logger.info("Carregando play-by-play de %s a %s...", season_start, season_end)
    pbp = loader.load_pbp(seasons, columns=PBP_FEATURE_COLUMNS)

    logger.info("Carregando schedules...")
    schedules = loader.load_schedules(seasons)

    logger.info("Calculando features de ataque/defesa/QB (Módulo 2)...")
    fe = FeatureEngineer()
    team_game_stats = fe.build_team_game_stats(pbp)
    pregame_features = fe.build_pregame_features(team_game_stats)
    game_table = fe.build_game_feature_table(pregame_features, schedules)

    logger.info("Calculando Elo dinâmico (Módulo 3)...")
    # O Elo precisa do histórico completo (não só das temporadas de interesse)
    # para os ratings já chegarem "aquecidos" na primeira temporada pedida.
    elo_seasons = list(range(min(2002, season_start), season_end + 1))
    full_schedules = loader.load_schedules(elo_seasons)
    elo = EloRatingSystem()
    elo_history = elo.run(full_schedules)
    elo.save_history(elo_history)

    elo_cols = [
        "game_id", "home_elo_pre", "away_elo_pre", "home_win_prob_elo", "elo_diff_pre",
    ]
    dataset = game_table.merge(elo_history[elo_cols], on="game_id", how="left")

    # Alvo de treino
    scores = schedules[["game_id", "home_score", "away_score"]].dropna(
        subset=["home_score", "away_score"]
    )
    dataset = dataset.merge(scores, on="game_id", how="inner")
    dataset["home_win"] = (dataset["home_score"] > dataset["away_score"]).astype(int)

    logger.info("Dataset final: %d jogos, %d colunas", *dataset.shape)
    return dataset


if __name__ == "__main__":
    from config.settings import ML_TRAIN_SEASON_END, ML_TRAIN_SEASON_START, DATA_DIR

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset = build_full_dataset(ML_TRAIN_SEASON_START, ML_TRAIN_SEASON_END)
    out_path = DATA_DIR / "training_dataset.parquet"
    dataset.to_parquet(out_path, index=False)
    print(f"\nSalvo em {out_path}")
    print(dataset.shape)
    print("\nColunas nulas (esperado: primeiro jogo de cada time na temporada):")
    print(dataset.isna().sum()[dataset.isna().sum() > 0].sort_values(ascending=False).head(10))
    print("\nDistribuição do alvo (home_win):")
    print(dataset["home_win"].value_counts(normalize=True))
