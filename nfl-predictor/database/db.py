"""
Conexão e operações de escrita/leitura do banco de dados.

Usa SQLite por padrão (arquivo em database/nfl_predictor.db); troque para
PostgreSQL definindo a variável de ambiente DATABASE_URL (ver
database/models.py para o formato da connection string).
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from config.settings import DATABASE_URL
from database.models import Base, Game, Prediction, TeamMetric

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)


def init_db() -> None:
    """Cria as tabelas (games, team_metrics, predictions) se ainda não existirem."""
    Base.metadata.create_all(engine)
    logger.info("Banco de dados inicializado em %s", DATABASE_URL)


def upsert_games(schedules: pd.DataFrame) -> int:
    """Insere/atualiza jogos a partir de um DataFrame de schedules do nflverse."""
    cols = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]
    rows = schedules[cols].dropna(subset=["game_id"]).to_dict("records")

    with SessionLocal() as session:
        for row in rows:
            stmt = sqlite_insert(Game).values(**row)
            stmt = stmt.on_conflict_do_update(index_elements=["game_id"], set_=row)
            session.execute(stmt)
        session.commit()
    logger.info("Upsert de %d jogos concluído.", len(rows))
    return len(rows)


def upsert_team_metrics(team_game_stats: pd.DataFrame) -> int:
    """Insere/atualiza métricas de time a partir da saída do FeatureEngineer."""
    with SessionLocal() as session:
        n = 0
        for row in team_game_stats.itertuples(index=False):
            payload = {
                "team": row.team,
                "season": row.season,
                "week": row.week,
                "off_epa": getattr(row, "off_epa_per_play", None),
                "def_epa": getattr(row, "def_epa_allowed", None),
            }
            stmt = sqlite_insert(TeamMetric).values(**payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=["team", "season", "week"], set_=payload
            )
            session.execute(stmt)
            n += 1
        session.commit()
    logger.info("Upsert de %d métricas de time concluído.", n)
    return n


def save_prediction(
    game_id: str,
    win_probability: float,
    confidence: int,
    predicted_home_score: float | None = None,
    predicted_away_score: float | None = None,
) -> None:
    """Salva uma previsão no histórico (tabela `predictions`)."""
    with SessionLocal() as session:
        pred = Prediction(
            game_id=game_id,
            win_probability=win_probability,
            confidence=confidence,
            predicted_home_score=predicted_home_score,
            predicted_away_score=predicted_away_score,
        )
        session.add(pred)
        session.commit()


def get_session() -> Session:
    return SessionLocal()
