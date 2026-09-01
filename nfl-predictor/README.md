# 🏈 NFL Predictor 3.0

Sistema completo de previsão de jogos da NFL usando machine learning,
estatísticas avançadas (EPA, success rate, red zone %), Elo dinâmico e
simulação Monte Carlo — com dashboard interativo em Streamlit.

## ⚠️ Uma decisão de arquitetura importante

O pacote `nfl_data_py` (PyPI) trava a dependência em `pandas<2.0`, o que é
incompatível com o restante do stack deste projeto (`pandas>=2.2`,
`numpy>=1.26`, necessários para scikit-learn/xgboost modernos).

Em vez de forçar um downgrade que quebraria o resto do stack, o
`utils/data_loader.py` baixa os **mesmos arquivos parquet diretamente dos
releases do [`nflverse/nflverse-data`](https://github.com/nflverse/nflverse-data/releases)
no GitHub** — a fonte de dados real usada por baixo dos panos pelo
`nfl_data_py`. Resultado: os mesmos dados, sem conflito de dependências,
com cache local automático.

## Instalação

```bash
python3 -m pip install -r requirements.txt
```

> Se você usar um ambiente com Python gerenciado externamente (Debian/Ubuntu
> recentes), pode ser necessário `pip install -r requirements.txt --break-system-packages`.

## Uso

### 1. Treinar o modelo

```bash
python train_model.py
# ou, para um range de temporadas específico:
python train_model.py --season-start 2015 --season-end 2025
```

Isso baixa (ou reaproveita do cache) o play-by-play e o calendário de
2010–2025, calcula as features de EPA/Elo, treina o XGBoost e a Regressão
Logística, e salva os artefatos em `models/`:

- `xgb_model.pkl` — modelo principal
- `logreg_model.pkl` — baseline de comparação
- `scaler.pkl` / `imputer.pkl` — pré-processamento
- `feature_columns.json` — ordem das features esperadas pelo modelo
- `metrics.json` — Accuracy, Precision, Recall, ROC-AUC, Brier Score

### 2. Rodar o dashboard

```bash
streamlit run app.py
```

Abre em `http://localhost:8501`. A página inicial mostra todos os jogos da
rodada selecionada (probabilidade via Elo, carregamento instantâneo);
clicar em "Ver detalhes →" abre a previsão completa (ML + Elo + Monte Carlo
+ SHAP) para aquele confronto.

### 3. Previsão via linha de comando

```bash
python predict.py
```

(edite o `if __name__ == "__main__"` no fim do arquivo para trocar os
times, ou importe `GamePredictor` em outro script).

## Estrutura do projeto

```
nfl-predictor/
├── app.py                          # Dashboard Streamlit — página inicial
├── train_model.py                  # Treino do modelo (Módulo 4)
├── predict.py                      # Orquestrador de previsão (ML+Elo+MC+Odds+Value Betting)
│
├── config/
│   └── settings.py                 # Todos os parâmetros ajustáveis do sistema
│
├── pages/
│   └── 1_📊_Detalhe_do_Jogo.py     # Dashboard Streamlit — detalhe do jogo
│
├── utils/
│   ├── data_loader.py              # Módulo 1 — coleta de dados (cache parquet)
│   ├── feature_engineering.py      # Módulo 2 — features de EPA/success/QB pré-jogo
│   ├── elo.py                      # Módulo 3 — Elo dinâmico
│   ├── monte_carlo.py              # Módulo 5 — simulação Monte Carlo
│   ├── build_dataset.py            # Pipeline que une tudo em um dataset de treino
│   └── dashboard_common.py         # Utilidades compartilhadas do Streamlit (cache, CSS, logos)
│
├── database/
│   ├── models.py                   # Tabelas SQLAlchemy: games, team_metrics, predictions
│   └── db.py                       # Conexão + upsert/leitura
│
├── data/
│   ├── cache/                      # Cache local dos parquets baixados do nflverse
│   └── training_dataset.parquet    # Dataset final de treino já processado
│
├── models/                         # Artefatos treinados (gerados por train_model.py)
├── elo_history/                    # Histórico de Elo em CSV (gerado por utils/elo.py)
│
└── requirements.txt
```

## Módulos implementados

| # | Módulo | Onde |
|---|--------|------|
| 1 | Coleta automática de dados (cache parquet) | `utils/data_loader.py` |
| 2 | Engenharia de features (EPA, success rate, QB, contexto) | `utils/feature_engineering.py` |
| 3 | Elo dinâmico | `utils/elo.py` |
| 4 | Machine Learning (XGBoost + Regressão Logística) | `train_model.py` |
| 5 | Simulação Monte Carlo (10.000 partidas) | `utils/monte_carlo.py` |
| 6 | Dashboard Streamlit | `app.py` + `pages/1_📊_Detalhe_do_Jogo.py` |
| 7 | Odds — **em standby** (arquitetura pronta, sem chave obrigatória) | `predict.py::OddsProvider` |
| 8 | Value Betting (funciona com odd manual) | `predict.py` (`implied_probability`, `expected_value`, `kelly_criterion`, `closing_line_value`) |
| — | Banco de dados (games / team_metrics / predictions) | `database/models.py`, `database/db.py` |
| — | Score de confiança (0–100) | `predict.py::compute_confidence_score` |

### Bônus implementados

- ✅ **SHAP** — explica quais features mais influenciaram cada previsão (aba
  "🧠 O que mais influenciou esta previsão" na página de detalhe).
- ✅ **Exportar previsões para CSV** — botão na página inicial.
- ✅ **Modo "What-If"** — expander na página de detalhe para simular
  descanso diferente ou QB reserva e recalcular a probabilidade.
- ⏳ **Atualização automática toda terça-feira** — não implementado (exigiria
  um scheduler/cron externo ao processo do Streamlit); o botão manual
  "🔄 Atualizar banco de dados" na sidebar cobre a mesma necessidade sob
  demanda.

## Ativando odds em tempo real (Módulo 7)

O sistema nasce com a arquitetura pronta, mas **nenhuma chamada a API paga é
feita por padrão**. Para ativar:

1. Defina a variável de ambiente `ODDS_API_KEY` com sua chave.
2. Isso automaticamente liga `ENABLE_ODDS=True` em `config/settings.py`.
3. Implemente o corpo de `OddsProvider.get_live_odds()` em `predict.py` com
   a chamada HTTP ao provedor escolhido.

Nenhum outro módulo precisa ser alterado — o dashboard já verifica
`ENABLE_ODDS` para mostrar/esconder as seções de odds ao vivo.

## Usando PostgreSQL em vez de SQLite

Por padrão o projeto usa SQLite (zero-config). Para usar PostgreSQL:

```bash
export DATABASE_URL="postgresql+psycopg2://usuario:senha@localhost:5432/nfl_predictor"
```

(requer `pip install psycopg2-binary`, não incluso no requirements.txt para
manter a instalação padrão leve).

## Limitações conhecidas

- As métricas do modelo (Accuracy ~62-64%, ROC-AUC ~0.66-0.69 no teste
  2022-2025) estão na faixa esperada para previsão de resultados da NFL —
  esse é um esporte com alta variância, e mesmo os melhores modelos
  públicos raramente passam de 65-68% de acerto.
- A "forma atual" de um time (usada no dashboard e em `predict.py`) é a
  média dos últimos 8 jogos observados — não há dados de lesões, clima ou
  transações recentes de elenco.
- Distância de viagem (citada no briefing original) ainda não está
  implementada como feature — seria necessário um dicionário de
  coordenadas de estádios, hoje ausente.
