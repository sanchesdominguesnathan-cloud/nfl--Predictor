"""
Módulo 1 — Coleta automática de dados.

Baixa play-by-play, weekly stats, rosters e schedules diretamente dos
releases do nflverse-data (GitHub) e mantém um cache local em parquet para
evitar downloads repetidos. Também oferece uma rotina de atualização
incremental que baixa apenas as temporadas faltantes.

Por que não usamos o pacote `nfl_data_py`:
    A versão publicada no PyPI fixa `pandas<2.0`, o que conflita com o
    restante do stack deste projeto (pandas>=2.2). Para evitar downgrade
    forçado do pandas em todo o projeto, este módulo replica exatamente o
    que o `nfl_data_py` faz internamente: buscar os arquivos parquet/csv
    publicados nos releases do repositório `nflverse/nflverse-data`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import CACHE_DIR, NFLVERSE_RELEASES_BASE_URL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class NFLDataLoaderError(RuntimeError):
    """Erro ao baixar ou ler um dataset do nflverse."""


@dataclass(frozen=True)
class _Endpoint:
    """Descreve como montar a URL de um asset de release do nflverse."""

    tag: str
    filename_template: str  # pode conter "{season}"
    per_season: bool


# Tags e nomes de arquivo confirmados nos releases do nflverse-data.
_ENDPOINTS: dict[str, _Endpoint] = {
    "pbp": _Endpoint("pbp", "play_by_play_{season}.parquet", per_season=True),
    "weekly": _Endpoint("player_stats", "player_stats.parquet", per_season=False),
    "rosters": _Endpoint("rosters", "roster_{season}.parquet", per_season=True),
    "schedules": _Endpoint("schedules", "games.parquet", per_season=False),
}


class NFLDataLoader:
    """
    Carrega datasets do nflverse com cache local em parquet.

    Exemplo:
        >>> loader = NFLDataLoader()
        >>> pbp = loader.load_pbp([2023, 2024])
        >>> schedules = loader.load_schedules()
    """

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def load_pbp(
        self, seasons: list[int], force_refresh: bool = False, columns: list[str] | None = None,
        skip_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Carrega play-by-play para as temporadas informadas (uma requisição por temporada).

        `columns`, quando informado, restringe as colunas lidas do parquet
        (o pbp completo tem ~370 colunas; para features usamos ~20). Isso
        reduz drasticamente o uso de memória ao concatenar várias
        temporadas de uma vez.

        `skip_missing=True` ignora silenciosamente temporadas ainda não
        publicadas pelo nflverse (ex.: a temporada corrente antes do
        primeiro jogo), em vez de lançar erro.
        """
        frames = []
        for season in seasons:
            try:
                frames.append(
                    self._load_season_dataset("pbp", season, force_refresh=force_refresh, columns=columns)
                )
            except NFLDataLoaderError:
                if skip_missing:
                    logger.warning("Temporada %s ainda sem dados de pbp publicados — ignorando.", season)
                    continue
                raise
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_weekly_stats(self, seasons: list[int] | None = None, force_refresh: bool = False) -> pd.DataFrame:
        """Carrega estatísticas semanais de jogadores (dataset único, filtrado por temporada)."""
        df = self._load_flat_dataset("weekly", force_refresh=force_refresh)
        if seasons is not None and "season" in df.columns:
            df = df[df["season"].isin(seasons)].reset_index(drop=True)
        return df

    def load_rosters(self, seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
        """Carrega rosters por temporada."""
        frames = [
            self._load_season_dataset("rosters", season, force_refresh=force_refresh)
            for season in seasons
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_schedules(self, seasons: list[int] | None = None, force_refresh: bool = False) -> pd.DataFrame:
        """Carrega o calendário/resultados de jogos (schedules)."""
        df = self._load_flat_dataset("schedules", force_refresh=force_refresh)
        if seasons is not None and "season" in df.columns:
            df = df[df["season"].isin(seasons)].reset_index(drop=True)
        return df

    def refresh_all(self, seasons: list[int]) -> dict[str, int]:
        """
        Rotina usada pelo botão "Atualizar banco" do Streamlit.

        Baixa (ou reaproveita do cache) todos os datasets para as temporadas
        informadas e retorna um resumo de quantas linhas cada dataset possui.
        """
        summary: dict[str, int] = {}
        summary["pbp"] = len(self.load_pbp(seasons))
        summary["weekly_stats"] = len(self.load_weekly_stats(seasons))
        summary["rosters"] = len(self.load_rosters(seasons))
        summary["schedules"] = len(self.load_schedules(seasons))
        return summary

    def missing_seasons(self, seasons: list[int], dataset: str = "pbp") -> list[int]:
        """Retorna quais temporadas ainda não têm cache local para o dataset informado."""
        endpoint = _ENDPOINTS[dataset]
        if not endpoint.per_season:
            return [] if self._cache_path(dataset).exists() else list(seasons)
        return [s for s in seasons if not self._cache_path(dataset, s).exists()]

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _cache_path(self, dataset: str, season: int | None = None) -> Path:
        if season is not None:
            return self.cache_dir / f"{dataset}_{season}.parquet"
        return self.cache_dir / f"{dataset}.parquet"

    def _remote_url(self, dataset: str, season: int | None = None) -> str:
        endpoint = _ENDPOINTS[dataset]
        filename = endpoint.filename_template.format(season=season)
        return f"{NFLVERSE_RELEASES_BASE_URL}/{endpoint.tag}/{filename}"

    def _load_season_dataset(
        self, dataset: str, season: int, force_refresh: bool, columns: list[str] | None = None
    ) -> pd.DataFrame:
        cache_path = self._cache_path(dataset, season)
        if cache_path.exists() and not force_refresh:
            logger.info("Cache HIT: %s (temporada %s)", dataset, season)
            df = pd.read_parquet(cache_path, columns=columns)
            return df

        url = self._remote_url(dataset, season)
        logger.info("Baixando %s da temporada %s de %s", dataset, season, url)
        try:
            df = pd.read_parquet(url)
        except Exception as exc:  # noqa: BLE001 - queremos reportar qualquer falha de rede/arquivo
            raise NFLDataLoaderError(
                f"Falha ao baixar '{dataset}' para a temporada {season} em {url}: {exc}"
            ) from exc

        df.to_parquet(cache_path, index=False)
        return df[columns] if columns else df

    def _load_flat_dataset(self, dataset: str, force_refresh: bool) -> pd.DataFrame:
        cache_path = self._cache_path(dataset)
        if cache_path.exists() and not force_refresh:
            logger.info("Cache HIT: %s", dataset)
            return pd.read_parquet(cache_path)

        url = self._remote_url(dataset)
        logger.info("Baixando %s de %s", dataset, url)
        try:
            df = pd.read_parquet(url)
        except Exception as exc:  # noqa: BLE001
            raise NFLDataLoaderError(f"Falha ao baixar '{dataset}' em {url}: {exc}") from exc

        df.to_parquet(cache_path, index=False)
        return df


if __name__ == "__main__":
    # Teste rápido manual: python -m utils.data_loader
    loader = NFLDataLoader()
    sched = loader.load_schedules(seasons=[2024, 2025])
    print(f"Schedules carregados: {sched.shape}")
    print(sched[["season", "week", "home_team", "away_team", "home_score", "away_score"]].tail())
