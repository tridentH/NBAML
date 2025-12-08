from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


def main():
    print("=== NBAML: fetch_games starting ===")

    season = "2023-24"
    print(f"Fetching game logs for season {season}...")

    # Custom headers to avoid being blocked by NBA stats
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
    }

    try:
        resp = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            headers=headers,
            timeout=10,  # prevent hanging forever
        )
    except Exception as e:
        print("❌ NBA API error:", repr(e))
        return

    # Convert response to DataFrame
    dfs = resp.get_data_frames()
    if not dfs:
        print("❌ No data frames returned from NBA API.")
        return

    df = dfs[0]
    print(f"Downloaded {len(df)} rows")

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Project root = directory of this script
    root = Path(__file__).resolve().parent
    out_dir = root / ".data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use CSV to keep it simple
    out_path = out_dir / "games_202324.csv"

    print(f"Saving to: {out_path}")
    df.to_csv(out_path, index=False)

    print("Done!")


if __name__ == "__main__":
    main()