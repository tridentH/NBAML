from pathlib import Path

import pandas as pd


def main():
    print("=== NBAML: building team_game_stats for 2023-24 ===")

    # Project root = .../NBAML
    root = Path(__file__).resolve().parents[2]

    raw_path = root / ".data" / "raw" / "games_202324.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    print(f"Loading raw games from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Normalize column names in case they aren't already lowercase
    df.columns = [c.lower() for c in df.columns]

    # Basic sanity check
    required_cols = ["game_id", "team_id", "team_name", "team_abbreviation",
                     "game_date", "matchup", "wl", "pts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # There should be 2 rows per game_id (one per team)
    print(f"Total rows: {len(df)}, unique games: {df['game_id'].nunique()}")

    # Build opponent table: for each game, find the other team's pts and id
    opp = df[["game_id", "team_id", "pts"]].rename(
        columns={"team_id": "opp_team_id", "pts": "opp_pts"}
    )
    merged = df.merge(opp, on="game_id", how="inner")

    # Drop rows where we matched a team with itself
    merged = merged[merged["team_id"] != merged["opp_team_id"]]

    # Home/away flag based on matchup like "LAL vs. DEN" or "LAL @ DEN"
    # 'vs.' => home, '@' => away
    merged["is_home"] = merged["matchup"].str.contains("vs\.", regex=True)

    # Point differential
    merged["pt_diff"] = merged["pts"] - merged["opp_pts"]

    # Parse game_date to datetime for sorting
    merged["game_date"] = pd.to_datetime(merged["game_date"])

    # Keep a clean subset of columns we care about
    cols = [
        "game_id",
        "game_date",
        "team_id",
        "team_abbreviation",
        "team_name",
        "opp_team_id",
        "matchup",
        "is_home",
        "wl",
        "pts",
        "opp_pts",
        "pt_diff",
    ]
    team_games = merged[cols].sort_values(["team_id", "game_date"]).reset_index(drop=True)

    out_path = root / ".data" / "interim" / "team_games_202324.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving team-game stats to: {out_path}")
    team_games.to_csv(out_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
