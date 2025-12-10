
<strong>NBAML — NBA Championship Odds Machine Learning Pipeline <strong/>

Predicting championship probabilities using real NBA game data (2018–2024)
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" /> <img src="https://img.shields.io/badge/Build-Passing-brightgreen" /> <img src="https://img.shields.io/badge/ML-Logistic%20Regression-orange" /> <img src="https://img.shields.io/badge/Data-NBA_API-red?logo=basketball" /> </p>

NBAML is a full end-to-end machine learning pipeline that:

Pulls real NBA game logs using nba_api

Engineers rolling 10-game performance features

Computes team strength metrics

Labels each season with the actual NBA Champion

Trains a Logistic Regression classifier

Predicts championship probabilities for any season

<strong> Project Structure <strong/>
```text
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
```

## 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Full Pipeline

This pipeline:

- Downloads all game logs  
- Builds team-level and rolling features  
- Adds champion labels  
- Computes strength scores  
- Generates a multi-season training dataset  

Run it with:

```bash
python run_pipeline.py
```

---

## Train the Championship Prediction Model

```bash
python src/models/train_logreg.py
```

Outputs include:

-  Accuracy  
-  Log loss  
-  ROC AUC  
-  Top predicted champions (test set)  
-  Saved model: `artifacts/logreg_champion_model.joblib`

---

## Predict Odds for Any Season

Example: predict odds for **2023–24**:

```bash
python src/models/predict_season.py
```

Produces:

```text
.data/features/champion_odds_202324.csv
```
