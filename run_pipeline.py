from fetch_games import fetch_games_for_season
from src.features.team_game_stats import build_team_games
from src.features.rolling_stats import build_rolling_features
from src.features.merge_labels import merge_labels
from src.models.simple_strength import compute_strength


SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]


def main():
    print("=== NBAML: Running full pipeline ===")

    for season in SEASONS:
        print(f"\n--- Processing season {season} ---")

        fetch_games_for_season(season)
        build_team_games(season)
        build_rolling_features(season)
        merge_labels(season)       # <- adds champion / is_champion
        compute_strength(season)   # <- simple season-level ranking

    print("\nPipeline complete! 🎉")


if __name__ == "__main__":
    main()
