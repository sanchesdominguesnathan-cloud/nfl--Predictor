"""
Modelos SQLAlchemy do NFL Predictor 3.0.

Por padrão usa SQLite local (zero-config, ver config/settings.DATABASE_URL).
Para usar PostgreSQL (conforme a stack original do projeto), defina a
variável de ambiente DATABASE_URL antes de rodar, por exemplo:

    export DATABASE_URL="postgresql+psycopg2://usuario:senha@localhost:5432/nfl_predictor"

Nenhum outro código do projeto precisa mudar — o `database/db.py` lê
`DATABASE_URL` de `config/settings.py`.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Game(Base):
    """Um jogo da NFL (resultado real, quando já disputado)."""

    __tablename__ = "games"
    __table_args__ = (UniqueConstraint("season", "week", "home_team", "away_team", name="uq_game"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer, index=True)
    home_team: Mapped[str] = mapped_column(String(4), index=True)
    away_team: Mapped[str] = mapped_column(String(4), index=True)
    home_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="game")


class TeamMetric(Base):
    """Métricas agregadas de um time em uma temporada/semana (snapshot pré-jogo)."""

    __tablename__ = "team_metrics"
    __table_args__ = (UniqueConstraint("team", "season", "week", name="uq_team_metric"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team: Mapped[str] = mapped_column(String(4), index=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer, index=True)
    off_epa: Mapped[float | None] = mapped_column(Float, nullable=True)
    def_epa: Mapped[float | None] = mapped_column(Float, nullable=True)
    elo: Mapped[float | None] = mapped_column(Float, nullable=True)
    rest_days: Mapped[float | None] = mapped_column(Float, nullable=True)


class Prediction(Base):
    """Uma previsão gerada pelo sistema para um jogo (histórico de previsões)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), index=True)
    win_probability: Mapped[float] = mapped_column(Float)
    confidence: Mapped[int] = mapped_column(Integer)
    predicted_home_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_away_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)

    game: Mapped["Game"] = relationship(back_populates="predictions")
