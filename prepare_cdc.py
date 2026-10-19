
from pathlib import Path
import pandas as pd

# Change these paths to where you extracted your CDC files
FILES = {
    "2023": Path("LLCP2023.XPT"),
    "2024": Path("LLCP2024.XPT"),
}

# Select relevant health and diabetes features.
# Some variable names differ between CDC survey years.
FEATURES = [
    "_AGEG5YR",   # Age group
    "_BMI5",      # BMI (scaled by 100)
    "SEXVAR",     # Sex
    "_RFHYPE6",   # High blood pressure indicator
    "DIABETE4",   # Diabetes status
    "_TOTINDA",   # Physical activity
    "_SMOKER3",   # Smoking status
]

SAMPLE_SIZE = 10000
CHUNK_SIZE = 5000
RANDOM_SEED = 42


def prepare_dataset(year, input_path):
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    # Reservoir sampling across the entire dataset.
    # Each record has an equal chance of being selected.
    reservoir = []
    seen = 0

    import random
    rng = random.Random(RANDOM_SEED)

    for chunk in pd.read_sas(
        input_path,
        format="xport",
        chunksize=CHUNK_SIZE
    ):
        available = [c for c in FEATURES if c in chunk.columns]
        chunk = chunk[available]

        for record in chunk.to_dict("records"):
            seen += 1

            if len(reservoir) < SAMPLE_SIZE:
                reservoir.append(record)
            else:
                index = rng.randrange(seen)
                if index < SAMPLE_SIZE:
                    reservoir[index] = record

        print(f"{year}: processed {seen:,} rows", end="\r")

    if not reservoir:
        raise ValueError(f"No records found in {input_path}")

    df = pd.DataFrame(reservoir)

    # Retain only features present in both years later.
    # These mappings are field-specific, not global:
    # a value of 7 or 9 can be legitimate in other columns.
    missing_codes = {
        "_AGEG5YR": [14],
        "_BMI5": [9999],
        "_RFHYPE6": [7, 9],
        "DIABETE4": [7, 9],
        "_TOTINDA": [9],
        "_SMOKER3": [9],
    }

    for column, codes in missing_codes.items():
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            ).replace(codes, float("nan"))

    # Convert BMI from CDC's scaled representation
    if "_BMI5" in df.columns:
        df["_BMI5"] = df["_BMI5"] / 100

    output = f"cdc_{year}_diabetes_sample.csv"
    df.to_csv(output, index=False)

    print(
        f"\nSaved {len(df):,} records to {output}"
    )


for year, path in FILES.items():
    prepare_dataset(year, path)
