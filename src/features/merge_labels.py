from pathlib import Path
import pandas as pd


def merge_labels(season: str = "2023-24") -> str:
    print(f"=== NBAML: merge_labels({season}) ===")

    root = Path(__file__).resolve().parents[2]
    season_key = season.replace("-", "")

    # Load features
    feat_path = root / ".data" / "features" / f"features_{season_key}.csv"
    df = pd.read_csv(feat_path)

    # Load champions table
    labels_path = root / ".data" / "labels" / "champions.csv"
    labels = pd.read_csv(labels_path)

    # Add season column to df
    df["season"] = season

    # Merge labels
    merged = df.merge(labels, on="season", how="left")

    # Add binary label
    merged["is_champion"] = (merged["team_abbreviation"] == merged["champion"]).astype(int)

    # Save file
    out_path = root / ".data" / "features" / f"features_labeled_{season_key}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged labels to: {out_path}")
    merged.to_csv(out_path, index=False)
    print("Done!")

    return str(out_path)


if __name__ == "__main__":
    merge_labels("2023-24")
