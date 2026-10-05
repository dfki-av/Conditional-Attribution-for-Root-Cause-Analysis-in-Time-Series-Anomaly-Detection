"""Export SWaT sensor rows outside and inside timed attacks as CSVs."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/SWaT_Dataset_Attack_v0.xlsx"
DEFAULT_NORMAL = PROJECT_ROOT / "data/processed/SWAT_Non_Attack_Instances.csv"
DEFAULT_ATTACKS = PROJECT_ROOT / "data/processed/SWAT_Attacks"


@dataclass(frozen=True)
class AttackInterval:
    name: str
    start: str
    end: str


# Dates come from 10_Isolate_Attack_Instances. Its reversed dates for attacks
# 32–36 (December instead of January) and 37–41 (2015 instead of 2016) are fixed.
ATTACK_INTERVALS = (
    AttackInterval("1_MV-101", "2015-12-28 10:29:14", "2015-12-28 10:44:53"),
    AttackInterval("2_P-102", "2015-12-28 10:51:08", "2015-12-28 10:58:30"),
    AttackInterval("3_LIT-101", "2015-12-28 11:22:00", "2015-12-28 11:28:22"),
    AttackInterval("4_MV-504", "2015-12-28 11:47:39", "2015-12-28 11:54:08"),
    AttackInterval("6_AIT-202", "2015-12-28 12:00:55", "2015-12-28 12:04:10"),
    AttackInterval("7_LIT-301", "2015-12-28 12:08:25", "2015-12-28 12:15:33"),
    AttackInterval("8_DPIT-301", "2015-12-28 13:10:10", "2015-12-28 13:26:13"),
    AttackInterval("10_FIT-401", "2015-12-28 14:16:20", "2015-12-28 14:19:00"),
    AttackInterval("11_FIT-401", "2015-12-28 14:19:00", "2015-12-28 14:28:20"),
    AttackInterval("13_MV-304", "2015-12-29 11:11:25", "2015-12-29 11:15:17"),
    AttackInterval("14_MV-303", "2015-12-29 11:35:40", "2015-12-29 11:42:50"),
    AttackInterval("16_LIT-301", "2015-12-29 11:57:25", "2015-12-29 12:02:00"),
    AttackInterval("17_MV-303", "2015-12-29 14:38:12", "2015-12-29 14:50:08"),
    AttackInterval("19_AIT-504", "2015-12-29 18:10:43", "2015-12-29 18:15:01"),
    AttackInterval("20_AIT-504", "2015-12-29 18:15:43", "2015-12-29 18:22:17"),
    AttackInterval("21_MV-101_LIT-101", "2015-12-29 18:30:00", "2015-12-29 18:42:00"),
    AttackInterval("22_UV-401_AIT-502_P-501", "2015-12-29 22:55:18", "2015-12-29 23:03:00"),
    AttackInterval("23_P-602_DIT-301_MV-302", "2015-12-30 01:42:34", "2015-12-30 01:54:10"),
    AttackInterval("24_P-203_P-205", "2015-12-30 09:51:08", "2015-12-30 09:56:28"),
    AttackInterval("25_LIT-401_P-401", "2015-12-30 10:01:50", "2015-12-30 10:12:01"),
    AttackInterval("26_P-101_LIT-301", "2015-12-30 17:04:56", "2015-12-30 17:29:00"),
    AttackInterval("27_P-302_LIT-401", "2015-12-31 01:17:08", "2015-12-31 01:45:18"),
    AttackInterval("28_P-302", "2015-12-31 01:45:19", "2015-12-31 11:15:27"),
    AttackInterval("29_P-201_P-203_P-205", "2015-12-31 15:32:00", "2015-12-31 15:34:00"),
    AttackInterval("30_LIT-101_P-101_MV-201", "2015-12-31 15:47:40", "2015-12-31 16:07:10"),
    AttackInterval("31_LIT-401", "2015-12-31 22:05:34", "2015-12-31 22:11:40"),
    AttackInterval("32_LIT-301", "2016-01-01 10:36:00", "2016-01-01 10:46:00"),
    AttackInterval("33_LIT-101", "2016-01-01 14:21:12", "2016-01-01 14:28:35"),
    AttackInterval("34_P-101", "2016-01-01 17:12:40", "2016-01-01 17:14:20"),
    AttackInterval("35_P-101_P-102", "2016-01-01 17:18:56", "2016-01-01 17:26:56"),
    AttackInterval("36_LIT-101", "2016-01-01 22:16:01", "2016-01-01 22:25:00"),
    AttackInterval("37_P-501_FIT-502", "2016-01-02 11:17:02", "2016-01-02 11:24:50"),
    AttackInterval("38_AIT-402_AIT-502", "2016-01-02 11:31:38", "2016-01-02 11:36:18"),
    AttackInterval("39_FIT-401_AIT-502", "2016-01-02 11:43:48", "2016-01-02 11:50:28"),
    AttackInterval("40_FIT-401", "2016-01-02 11:51:42", "2016-01-02 11:56:38"),
    AttackInterval("41_LIT-301", "2016-01-02 13:13:02", "2016-01-02 13:40:56"),
)


def isolate_swat_attacks(
    input_path: Path = DEFAULT_INPUT,
    normal_output: Path = DEFAULT_NORMAL,
    attacks_dir: Path = DEFAULT_ATTACKS,
    intervals: tuple[AttackInterval, ...] = ATTACK_INTERVALS,
) -> None:
    """Save complete sensor rows outside attacks and each timed attack as CSV."""
    input_path = Path(input_path)
    normal_output = Path(normal_output)
    attacks_dir = Path(attacks_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"SWaT workbook not found: {input_path}")

    frame = pd.read_excel(input_path, sheet_name="Combined Data", header=1)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Timestamp"}
    if not required.issubset(frame.columns):
        raise ValueError(f"SWaT workbook is missing columns: {sorted(required - set(frame.columns))}")

    raw_timestamps = frame["Timestamp"]
    if pd.api.types.is_datetime64_any_dtype(raw_timestamps):
        timestamps = pd.to_datetime(raw_timestamps, errors="raise")
    else:
        timestamps = pd.to_datetime(
            raw_timestamps.astype(str).str.strip(),
            format="%d/%m/%Y %I:%M:%S %p",
            errors="raise",
        )
    if timestamps.isna().any():
        raise ValueError("SWaT workbook contains missing timestamps")

    feature_columns = [
        column for column in frame.columns
        if column not in {"Timestamp", "Normal/Attack", "Index"}
        and not column.startswith("Unnamed:")
    ]
    if not feature_columns:
        raise ValueError("SWaT workbook contains no sensor columns")

    # Validate all intervals before creating or overwriting output files.
    attack_rows = []
    in_attack = pd.Series(False, index=frame.index)
    for interval in intervals:
        start, end = pd.Timestamp(interval.start), pd.Timestamp(interval.end)
        if start > end:
            raise ValueError(f"Attack {interval.name} ends before it starts")
        interval_mask = timestamps.between(start, end)
        rows = frame.loc[interval_mask, feature_columns]
        if rows.empty:
            raise ValueError(f"No rows found for attack {interval.name} ({start} to {end})")
        attack_rows.append((interval.name, rows))
        in_attack |= interval_mask

    normal_rows = frame.loc[~in_attack, feature_columns].dropna()
    if normal_rows.empty:
        raise ValueError("No complete sensor rows found outside attack intervals")

    normal_output.parent.mkdir(parents=True, exist_ok=True)
    attacks_dir.mkdir(parents=True, exist_ok=True)
    normal_rows.to_csv(normal_output, index=False)
    print(f"Saved {len(normal_rows)} normal rows: {normal_output}")
    for name, rows in attack_rows:
        output_path = attacks_dir / f"{name}.csv"
        rows.to_csv(output_path, index=False)
        print(f"Saved {len(rows)} attack rows: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="SWaT attack workbook")
    parser.add_argument("--normal-output", type=Path, default=DEFAULT_NORMAL, help="Normal CSV path")
    parser.add_argument("--attacks-dir", type=Path, default=DEFAULT_ATTACKS, help="Attack CSV directory")
    args = parser.parse_args()
    isolate_swat_attacks(args.input, args.normal_output, args.attacks_dir)


if __name__ == "__main__":
    main()
