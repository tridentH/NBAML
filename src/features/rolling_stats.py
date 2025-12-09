from pathlib import Path

import pandas as pd


def main():
    print("=== NBAML: building rolling features for 2023-24 ===")

    root = Path(__file__).resolve().parents[2]
    in_path = root / ".data" / "interim" / "team_games_202324.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Interim team_games file not found: {in_path}")

    print(f"Loading team games from: {in_path}")
    df = pd.read_csv(in_path)

    # Ensure proper dtypes
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    # Group by team for rolling calculations
    group = df.groupby("team_id", group_keys=False)

    # We'll compute rolling means over the LAST 10 games, not including current
    def rolling_last_10(s: pd.Series) -> pd.Series:
        # shift by 1 so current game isn't included (no leakage)
        return s.shift(1).rolling(window=10, min_periods=3).mean()

    df["pt_diff_roll10"] = group["pt_diff"].apply(rolling_last_10)
    df["pts_roll10"] = group["pts"].apply(rolling_last_10)
    df["opp_pts_roll10"] = group["opp_pts"].apply(rolling_last_10)

    out_path = root / ".data" / "features" / "features_202324.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving rolling features to: {out_path}")
    df.to_csv(out_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
