from pathlib import Path
import pandas as pd


def build_rolling_features(season: str = "2023-24") -> str:
    """
    Build rolling last-10-game stats for a given season.
    """
    print(f"=== NBAML: build_rolling_features({season}) ===")

    root = Path(__file__).resolve().parents[2]
    season_key = season.replace("-", "")

    in_path = root / ".data" / "interim" / f"team_games_{season_key}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Interim team_games file not found: {in_path}")

    print(f"Loading team games from: {in_path}")
    df = pd.read_csv(in_path)

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    group = df.groupby("team_id", group_keys=False)

    def rolling_last_10(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=10, min_periods=3).mean()

    df["pt_diff_roll10"] = group["pt_diff"].apply(rolling_last_10)
    df["pts_roll10"] = group["pts"].apply(rolling_last_10)
    df["opp_pts_roll10"] = group["opp_pts"].apply(rolling_last_10)

    out_path = root / ".data" / "features" / f"features_{season_key}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving rolling features to: {out_path}")
    df.to_csv(out_path, index=False)
    print("Done!")
    return str(out_path)


def main():
    build_rolling_features("2023-24")


if __name__ == "__main__":
    main()