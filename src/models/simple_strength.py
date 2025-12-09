from pathlib import Path

import pandas as pd


def main():
    print("=== NBAML: computing simple team strength for 2023-24 ===")

    root = Path(__file__).resolve().parents[2]
    features_path = root / ".data" / "features" / "features_202324.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"Loading features from: {features_path}")
    df = pd.read_csv(features_path)

    # Basic sanity
    required_cols = ["team_abbreviation", "pt_diff_roll10"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Drop rows where rolling window isn't ready yet (NaNs at season start)
    clean = df.dropna(subset=["pt_diff_roll10"]).copy()

    # Compute average rolling pt diff per team as a simple strength metric
    grouped = (
        clean
        .groupby("team_abbreviation", as_index=False)["pt_diff_roll10"]
        .mean()
        .rename(columns={"pt_diff_roll10": "strength_score"})
    )

    # Sort best to worst
    ranked = grouped.sort_values("strength_score", ascending=False).reset_index(drop=True)

    # Print top 10
    print("\nTop 10 teams by simple strength_score (avg pt_diff_roll10):")
    print(ranked.head(10))

    # Save full ranking for later use
    out_path = root / ".data" / "features" / "team_strength_202324.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving full team strength table to: {out_path}")
    ranked.to_csv(out_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
