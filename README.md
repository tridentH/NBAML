NBAML — NBA Championship Odds Machine Learning Pipeline

Predicting championship probabilities using real NBA game data (2018–2024)
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" /> <img src="https://img.shields.io/badge/Build-Passing-brightgreen" /> <img src="https://img.shields.io/badge/ML-Logistic%20Regression-orange" /> <img src="https://img.shields.io/badge/Data-NBA_API-red?logo=basketball" /> </p>

NBAML is a full end-to-end machine learning pipeline that:

Pulls real NBA game logs using nba_api

Engineers rolling 10-game performance features

Computes team strength metrics

Labels each season with the actual NBA Champion

Trains a Logistic Regression classifier

Predicts championship probabilities for any season

Project Structure
"""
NBAML/
├── run_pipeline.py
├── fetch_games.py
├── notebooks/
│   └── eda_features.ipynb
├── src/
│   ├── features/
│   │   ├── team_game_stats.py
│   │   ├── rolling_stats.py
│   │   ├── merge_labels.py
│   │   ├── build_training_table.py
│   ├── models/
│   │   ├── simple_strength.py
│   │   ├── train_logreg.py
│   │   └── predict_season.py
│   ├── utils/
│       └── paths.py
├── .data/                # (ignored by Git)
├── artifacts/
│   └── logreg_champion_model.joblib
└── requirements.txt
"""
