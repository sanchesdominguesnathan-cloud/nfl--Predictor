"""
Módulo 4 — Machine Learning.

Treina o modelo principal (XGBoost Classifier) para prever a probabilidade
de vitória do mandante, compara com uma Regressão Logística como baseline,
reporta métricas (Accuracy, Precision, Recall, ROC-AUC, Brier Score) e
salva os artefatos treinados em `models/`.

Uso:
    python train_model.py
    python train_model.py --season-start 2015 --season-end 2025
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import (
    DATA_DIR,
    ML_RANDOM_STATE,
    ML_TRAIN_SEASON_END,
    ML_TRAIN_SEASON_START,
    MODELS_DIR,
    XGB_PARAMS,
)
from utils.build_dataset import build_full_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "home_win"

# Colunas que NUNCA entram como feature (identificadores, alvo, ou dados
# só conhecidos depois do jogo terminar — evita vazamento de dados).
_NON_FEATURE_COLS = {
    "game_id", "season", "week", "home_team", "away_team",
    "home_opponent", "away_opponent", "home_is_home", "away_is_home",
    "home_score", "away_score", "home_win",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Todas as colunas numéricas exceto identificadores/alvo/pós-jogo."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def chronological_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide em treino/teste respeitando a ordem cronológica (treina no
    passado, testa no "futuro"). Mais realista que um split aleatório
    para um problema de série temporal como jogos de temporada.
    """
    df_sorted = df.sort_values(["season", "week"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Calcula Accuracy, Precision, Recall, ROC-AUC e Brier Score."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
    }
    logger.info(
        "%-20s | Acc=%.3f  Prec=%.3f  Rec=%.3f  ROC-AUC=%.3f  Brier=%.4f",
        name, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["roc_auc"], metrics["brier_score"],
    )
    return metrics


def train(season_start: int, season_end: int, test_size: float = 0.2) -> dict:
    dataset_path = DATA_DIR / "training_dataset.parquet"
    if dataset_path.exists():
        logger.info("Usando dataset já construído em %s", dataset_path)
        df = pd.read_parquet(dataset_path)
        if df["season"].min() > season_start or df["season"].max() < season_end:
            logger.info("Dataset em cache não cobre o range pedido, reconstruindo...")
            df = build_full_dataset(season_start, season_end)
            df.to_parquet(dataset_path, index=False)
    else:
        df = build_full_dataset(season_start, season_end)
        df.to_parquet(dataset_path, index=False)

    feature_cols = get_feature_columns(df)
    logger.info("Usando %d features", len(feature_cols))

    train_df, test_df = chronological_split(df, test_size)
    logger.info(
        "Treino: %d jogos (temporadas %s-%s) | Teste: %d jogos (temporadas %s-%s)",
        len(train_df), train_df["season"].min(), train_df["season"].max(),
        len(test_df), test_df["season"].min(), test_df["season"].max(),
    )

    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # Imputação: primeiro jogo de cada time na temporada não tem histórico
    # (features pregame_* = NaN). Preenchemos com a média do conjunto de
    # treino (nunca do teste, para não vazar informação do futuro).
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    # Padronização (usada pela Regressão Logística; XGBoost é invariante a
    # escala mas reaproveitamos o mesmo scaler para manter um único artefato).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # --- Modelo principal: XGBoost ---
    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train_imp, y_train)
    xgb_metrics = evaluate(xgb_model, X_test_imp, y_test, "XGBoost")

    # --- Baseline: Regressão Logística ---
    logreg_model = LogisticRegression(max_iter=1000, random_state=ML_RANDOM_STATE)
    logreg_model.fit(X_train_scaled, y_train)
    logreg_metrics = evaluate(logreg_model, X_test_scaled, y_test, "Logistic Regression")

    # --- Salvar artefatos ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(logreg_model, MODELS_DIR / "logreg_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    metrics_summary = {"xgboost": xgb_metrics, "logistic_regression": logreg_metrics}
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    logger.info("Artefatos salvos em %s", MODELS_DIR)
    return metrics_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo do NFL Predictor 3.0")
    parser.add_argument("--season-start", type=int, default=ML_TRAIN_SEASON_START)
    parser.add_argument("--season-end", type=int, default=ML_TRAIN_SEASON_END)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.season_start, args.season_end, args.test_size)
