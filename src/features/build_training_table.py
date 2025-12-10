from pathlib import Path

import pandas as pd


SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]


def build_training_table() -> str:
    print("=== NBAML: building training table ===")

    root = Path(__file__).resolve().parents[2]
    features_dir = root / ".data" / "features"

    all_rows = []

    for season in SEASONS:
        season_key = season.replace("-", "")
        print(f"\nProcessing season {season} ({season_key})")

        # Per-game labeled features
        feat_path = features_dir / f"features_labeled_{season_key}.csv"
        if not feat_path.exists():
            print(f"  !!! Missing features file: {feat_path}, skipping season.")
            continue

        df = pd.read_csv(feat_path)

        # Drop early games where rolling stats aren't ready
        df = df.dropna(subset=["pt_diff_roll10"])

        # Aggregate to team-season level
        grouped = (
            df.groupby("team_abbreviation")
            .agg(
                mean_pt_diff_roll10=("pt_diff_roll10", "mean"),
                mean_pts_roll10=("pts_roll10", "mean"),
                mean_opp_pts_roll10=("opp_pts_roll10", "mean"),
                is_champion=("is_champion", "max"),  # 1 for champion, 0 otherwise
            )
            .reset_index()
        )

        # Add season column
        grouped["season"] = season

        # Merge in strength_score from team_strength_YYYYYY.csv
        strength_path = features_dir / f"team_strength_{season_key}.csv"
        if strength_path.exists():
            strength = pd.read_csv(strength_path)
            grouped = grouped.merge(
                strength,
                on="team_abbreviation",
                how="left",
            )
        else:
            grouped["strength_score"] = None

        all_rows.append(grouped)

    if not all_rows:
        raise RuntimeError("No seasons produced data; check inputs.")

    full = pd.concat(all_rows, ignore_index=True)

    # Filter out seasons where champion is unknown (e.g., 2023-24 UNK)
    # by using is_champion column: if all zeros for a season, we drop it
    mask_known = full.groupby("season")["is_champion"].transform("max") == 1
    labeled = full[mask_known].reset_index(drop=True)

    out_path = features_dir / "training_table.csv"
    labeled.to_csv(out_path, index=False)

    print(f"\nSaved training table to: {out_path}")
    print(f"Rows: {len(labeled)}, columns: {list(labeled.columns)}")
    return str(out_path)


if __name__ == "__main__":
    build_training_table()
