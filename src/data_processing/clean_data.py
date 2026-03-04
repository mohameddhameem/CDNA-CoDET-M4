"""
Clean real_experiment_data.csv: drop failed generations and parsing errors.
"""

import pandas as pd
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    input_path = base / "real_experiment_data.csv"
    output_path = base / "final_clean_data.csv"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Drop failed generations (Token_Count < 5)
    df = df[df["Token_Count"] >= 5]

    # Drop parsing errors (Max_Depth == 0)
    df = df[df["Max_Depth"] > 0]

    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
