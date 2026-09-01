"""
Configurações centrais do NFL Predictor 3.0.

Este módulo concentra todos os parâmetros ajustáveis do sistema: caminhos de
dados, parâmetros do Elo, parâmetros de treino do modelo, parâmetros da
simulação Monte Carlo e a flag de Odds (Módulo 7 - em standby).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos do projeto
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
MODELS_DIR: Path = BASE_DIR / "models"
ELO_HISTORY_DIR: Path = BASE_DIR / "elo_history"
DATABASE_DIR: Path = BASE_DIR / "database"

for _dir in (DATA_DIR, CACHE_DIR, MODELS_DIR, ELO_HISTORY_DIR, DATABASE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Fonte de dados (nflverse-data releases no GitHub)
# --------------------------------------------------------------------------- #
# nfl_data_py exige pandas<2.0, incompatível com o ambiente do projeto.
# Por isso o NFL Predictor 3.0 baixa os arquivos parquet/csv diretamente dos
# releases do nflverse-data, a mesma fonte usada pelo pacote nfl_data_py.
NFLVERSE_RELEASES_BASE_URL: str = (
    "https://github.com/nflverse/nflverse-data/releases/download"
)

# Temporadas cobertas pelo treino/coleta (conforme especificação do projeto)
SEASON_START: int = 2010
SEASON_END: int = 2025  # inclusive; ajustar a cada nova temporada concluída

# --------------------------------------------------------------------------- #
# Banco de dados
# --------------------------------------------------------------------------- #
# Por padrão usa SQLite local (zero-config). Para usar PostgreSQL, defina a
# variável de ambiente DATABASE_URL, ex:
#   postgresql+psycopg2://usuario:senha@localhost:5432/nfl_predictor
import os

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", f"sqlite:///{DATABASE_DIR / 'nfl_predictor.db'}"
)

# --------------------------------------------------------------------------- #
# Elo Dinâmico (Módulo 3)
# --------------------------------------------------------------------------- #
ELO_INITIAL_RATING: float = 1500.0
ELO_K_FACTOR: float = 20.0
ELO_HOME_FIELD_ADVANTAGE: float = 55.0  # pontos de Elo somados ao mandante
ELO_SEASON_REGRESSION: float = 0.33  # regressão à média entre temporadas (0-1)

# --------------------------------------------------------------------------- #
# Machine Learning (Módulo 4)
# --------------------------------------------------------------------------- #
ML_TRAIN_SEASON_START: int = 2010
ML_TRAIN_SEASON_END: int = 2025
ML_TEST_SIZE: float = 0.2
ML_RANDOM_STATE: int = 42

XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": ML_RANDOM_STATE,
}

# --------------------------------------------------------------------------- #
# Simulação Monte Carlo (Módulo 5)
# --------------------------------------------------------------------------- #
MONTE_CARLO_SIMULATIONS: int = 10_000
MONTE_CARLO_RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Módulo 7 — Odds (EM STANDBY)
# --------------------------------------------------------------------------- #
# Nenhuma chamada a API paga é feita enquanto ENABLE_ODDS for False.
# Quando o usuário obtiver uma chave, basta definir a variável de ambiente
# ODDS_API_KEY e trocar ENABLE_ODDS para True — nenhum outro trecho do
# código precisa ser alterado.
ODDS_API_KEY: str | None = os.environ.get("ODDS_API_KEY", None)
ENABLE_ODDS: bool = bool(ODDS_API_KEY)  # só liga se houver chave definida

# --------------------------------------------------------------------------- #
# Visual
# --------------------------------------------------------------------------- #
COLOR_EV_POSITIVE: str = "#00C46A"
COLOR_EV_NEGATIVE: str = "#FF4B4B"
COLOR_HOME: str = "#1F77FF"
COLOR_AWAY: str = "#FF8C1F"
THEME_BACKGROUND: str = "#0E1117"
THEME_CARD_BACKGROUND: str = "#161B22"
