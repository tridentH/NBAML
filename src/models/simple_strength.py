from pathlib import Path
import pandas as pd


def compute_strength(season: str = "2023-24") -> str:
    """
    Compute simple strength ranking for a season
    based on average pt_diff_roll10.
    """
    print(f"=== NBAML: compute_strength({season}) ===")

    root = Path(__file__).resolve().parents[2]
    season_key = season.replace("-", "")

    features_path = root / ".data" / "features" / f"features_{season_key}.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"Loading features from: {features_path}")
    df = pd.read_csv(features_path)

    required_cols = ["team_abbreviation", "pt_diff_roll10"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    clean = df.dropna(subset=["pt_diff_roll10"]).copy()

    grouped = (
        clean
        .groupby("team_abbreviation", as_index=False)["pt_diff_roll10"]
        .mean()
        .rename(columns={"pt_diff_roll10": "strength_score"})
    )

    ranked = grouped.sort_values("strength_score", ascending=False).reset_index(drop=True)

    print("\nTop 10 teams by strength_score:")
    print(ranked.head(10))

    out_path = root / ".data" / "features" / f"team_strength_{season_key}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving team strength table to: {out_path}")
    ranked.to_csv(out_path, index=False)
    print("Done!")
    return str(out_path)


def main():
    compute_strength("2023-24")


if __name__ == "__main__":
    main()
