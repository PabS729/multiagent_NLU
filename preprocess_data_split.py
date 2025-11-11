import pandas as pd
import json
from sklearn.model_selection import train_test_split
# import datasets

def split_imdb_dataset(csv_path: str,
                       jsonl_path: str = "IMDB_train.jsonl",
                       train_size: float = 0.8,
                       random_state: int = 42) -> None:
    """
    Randomly split the IMDB csv dataset into train/val and save the training
    portion as a JSONL file.

    Parameters
    ----------
    csv_path : str
        Path to the IMDB_Dataset.csv file.
    jsonl_path : str, optional
        Output JSONL file name/path for the training split.  Default is
        "train.jsonl".
    train_size : float, optional
        Fraction of data to use for training (0–1).  Default is 0.8.
    random_state : int, optional
        Seed for reproducible splits.  Default is 42.
    """
    # 1. Load the CSV
    df = pd.read_csv(csv_path)

    # 2. Split into train and validation (we only keep the train part here)
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=True
    )

    # 3. Save training rows as JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for _, row in train_df.iterrows():
            json_record = json.dumps(
                {"review": row["review"], "sentiment": row["sentiment"]},
                ensure_ascii=False
            )
            f_out.write(json_record + "\n")

    with open("IMDB_test.jsonl", "w", encoding="utf-8") as f_out:
        for _, row in test_df.iterrows():
            json_record = json.dumps(
                {"review": row["review"], "sentiment": row["sentiment"]},
                ensure_ascii=False
            )
            f_out.write(json_record + "\n")

    print(f"Saved {len(train_df)} training samples to {jsonl_path}")
    print(f"Saved {len(test_df)} testing samples to IMDB_test.jsonl")

"""
Pre-processing script for three datasets:
1. curaihealth/medical_questions_pairs  (Hugging Face)
2. climate_fever                      (Hugging Face)
3. IMDB_Dataset.csv                   (local CSV)

Each dataset is split 80/20 (train/test) and **both** splits are saved as
JSONL files with one JSON object per line.  All original columns are kept.
"""

import os
import json
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20 % test split

def process_ethos(csv_path: str, out_dir: str = "data", normal_offensive_ratio: int = 1, random_state: int = 42) -> None:
    """
    Pre-process the ETHOS dataset (en_dataset_with_stop_words.csv).
    1. Keep rows where 'target' == 'origin'
    2. Keep rows where 'sentiment' in ['normal', 'offensive']
    3. Down-sample so that normal:offensive == normal_offensive_ratio (default 1:3)
    4. Split 80/20 train/test with identical ratio in both splits
    5. Save train & test JSONL files
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)

    # 1 & 2: filter
    df = df[(df["target"] == "origin") & (df["sentiment"].isin(["normal", "offensive"]))]

    # 3: balance classes to desired ratio
    normal_df = df[df["sentiment"] == "normal"]
    offensive_df = df[df["sentiment"] == "offensive"]
    n_normal = len(normal_df)
    n_offensive = len(offensive_df)
    # Determine minority class
    desired_normal = int((normal_offensive_ratio * n_offensive) / 1) if (normal_offensive_ratio / 1) < (n_normal / n_offensive) else n_normal
    desired_offensive = int(n_normal / normal_offensive_ratio) if (n_normal / normal_offensive_ratio) < n_offensive else n_offensive
    # Sample
    normal_df = normal_df.sample(n=desired_normal, random_state=random_state)
    offensive_df = offensive_df.sample(n=desired_offensive, random_state=random_state)
    balanced_df = pd.concat([normal_df, offensive_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 4: stratified split
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=0.2,
        stratify=balanced_df["sentiment"],
        random_state=random_state
    )

    # 5: save
    os.makedirs(out_dir, exist_ok=True)
    for split, split_df in [("train", train_df), ("test", test_df)]:
        out_file = os.path.join(out_dir, f"ethos_{split}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                f.write(row.to_json(force_ascii=False) + "\n")
        print(f"Saved {len(split_df)} rows -> {out_file}")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _write_jsonl(dataset: Dataset, outfile: str) -> None:
    """Write a Hugging-Face Dataset to JSONL."""
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} rows -> {outfile}")


def _train_test_split(dataset: Dataset, test_size: float = TEST_SIZE) -> tuple[Dataset, Dataset]:
    """Return (train, test) datasets."""
    train_df, test_df = train_test_split(
        dataset.to_pandas(),
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return (
        Dataset.from_pandas(train_df, preserve_index=False),
        Dataset.from_pandas(test_df, preserve_index=False),
    )


# ------------------------------------------------------------------
# Dataset-specific processors
# ------------------------------------------------------------------
def process_medical_questions_pairs(out_dir: str = "data") -> None:
    ds = load_dataset("aps/super_glue","axg", split="test")
    train_ds, test_ds = _train_test_split(ds)
    _write_jsonl(train_ds, os.path.join(out_dir, "axg_train.jsonl"))
    _write_jsonl(test_ds, os.path.join(out_dir, "axg_test.jsonl"))


def process_climate_fever(out_dir: str = "data") -> None:
    ds = load_dataset("aps/super_glue","axb", split="test")  # only split available
    train_ds, test_ds = _train_test_split(ds)
    _write_jsonl(train_ds, os.path.join(out_dir, "axb_train.jsonl"))
    _write_jsonl(test_ds, os.path.join(out_dir, "axb_test.jsonl"))


def process_imdb_csv(csv_path: str, out_dir: str = "data") -> None:
    ds = load_dataset("csv", data_files=csv_path, split="train")
    train_ds, test_ds = _train_test_split(ds)
    _write_jsonl(train_ds, os.path.join(out_dir, "imdb_train.jsonl"))
    _write_jsonl(test_ds, os.path.join(out_dir, "imdb_test.jsonl"))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_medical_questions_pairs(OUTPUT_DIR + "/axg")
    process_climate_fever(OUTPUT_DIR + "/axb")
    # Uncomment and supply your local CSV path when ready
    # process_imdb_csv("IMDB_Dataset.csv", OUTPUT_DIR + "/IMDB")
    # process_ethos("en_dataset_with_stop_words.csv", "data/ethos-origin", 0.3)
# Example usage (uncomment to run):
# split_imdb_dataset("IMDB_Dataset.csv", "train.jsonl")