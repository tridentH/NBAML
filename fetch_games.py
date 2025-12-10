from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}


def fetch_games_for_season(season: str = "2023-24") -> str:
    """
    Download regular season game logs for a given season and save to .data/raw.
    season format: 'YYYY-YY', e.g. '2023-24'
    """
    print(f"=== NBAML: fetch_games_for_season({season}) ===")

    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        headers=HEADERS,
        timeout=10,
    )
    df = resp.get_data_frames()[0]
    df.columns = [c.lower() for c in df.columns]

    root = Path(__file__).resolve().parent
    out_dir = root / ".data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    season_key = season.replace("-", "")  # "2023-24" -> "202324"
    out_path = out_dir / f"games_{season_key}.csv"

    print(f"Saving to: {out_path}")
    df.to_csv(out_path, index=False)
    print("Done!")
    return str(out_path)


def main():
    # keep default behavior: single season for now
    fetch_games_for_season("2023-24")


if __name__ == "__main__":
    main()