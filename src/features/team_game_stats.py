from pathlib import Path
import pandas as pd


def build_team_games(season: str = "2023-24") -> str:
    """
    Build team-per-game stats (with opponent + pt_diff) for a given season.
    """
    print(f"=== NBAML: build_team_games({season}) ===")

    root = Path(__file__).resolve().parents[2]
    season_key = season.replace("-", "")

    raw_path = root / ".data" / "raw" / f"games_{season_key}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    print(f"Loading raw games from: {raw_path}")
    df = pd.read_csv(raw_path)
    df.columns = [c.lower() for c in df.columns]

    required_cols = ["game_id", "team_id", "team_name", "team_abbreviation",
                     "game_date", "matchup", "wl", "pts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    opp = df[["game_id", "team_id", "pts"]].rename(
        columns={"team_id": "opp_team_id", "pts": "opp_pts"}
    )
    merged = df.merge(opp, on="game_id", how="inner")
    merged = merged[merged["team_id"] != merged["opp_team_id"]]

    merged["is_home"] = merged["matchup"].str.contains("vs\.", regex=True)
    merged["pt_diff"] = merged["pts"] - merged["opp_pts"]
    merged["game_date"] = pd.to_datetime(merged["game_date"])

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

    out_path = root / ".data" / "interim" / f"team_games_{season_key}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving team-game stats to: {out_path}")
    team_games.to_csv(out_path, index=False)
    print("Done!")
    return str(out_path)


def main():
    build_team_games("2023-24")


if __name__ == "__main__":
    main()
