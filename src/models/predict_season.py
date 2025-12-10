from pathlib import Path

import joblib
import pandas as pd


def build_team_season_features(root: Path, season: str) -> pd.DataFrame:
    """
    Build team-season feature table for a given season, matching the features
    used in training (mean rolling stats + strength_score).
    """
    season_key = season.replace("-", "")
    features_dir = root / ".data" / "features"

    feat_path = features_dir / f"features_{season_key}.csv"
    strength_path = features_dir / f"team_strength_{season_key}.csv"

    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")

    df = pd.read_csv(feat_path)

    # Drop rows where rolling stats aren't ready yet
    df = df.dropna(subset=["pt_diff_roll10"])

    grouped = (
        df.groupby("team_abbreviation")
        .agg(
            mean_pt_diff_roll10=("pt_diff_roll10", "mean"),
            mean_pts_roll10=("pts_roll10", "mean"),
            mean_opp_pts_roll10=("opp_pts_roll10", "mean"),
        )
        .reset_index()
    )
    grouped["season"] = season

    # Merge strength_score
    if strength_path.exists():
        strength = pd.read_csv(strength_path)
        grouped = grouped.merge(
            strength,
            on="team_abbreviation",
            how="left",
        )
    else:
        grouped["strength_score"] = None

    return grouped


def predict_season_chances(season: str = "2023-24") -> str:
    """
    Load trained model and produce champion probabilities for all teams
    in a given season.
    """
    print(f"=== NBAML: predicting champion odds for {season} ===")

    root = Path(__file__).resolve().parents[2]

    # Load model artifact
    model_path = root / "artifacts" / "logreg_champion_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Build team-season features
    features = build_team_season_features(root, season)

    # Ensure feature order matches training
    X = features[feature_cols].values

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]
    features["champion_prob"] = probs

    # Sort best to worst
    ranked = features.sort_values("champion_prob", ascending=False).reset_index(drop=True)

    # Show top 10
    print("\nTop 10 teams by predicted championship probability:")
    print(
        ranked[["team_abbreviation", "season", "champion_prob"]]
        .head(10)
        .to_string(index=False)
    )

    # Save full table
    season_key = season.replace("-", "")
    out_path = root / ".data" / "features" / f"champion_odds_{season_key}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)

    print(f"\nSaved full odds table to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    predict_season_chances("2023-24")
