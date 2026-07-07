"""Build the Financial PhraseBank (75agree) fine-tuning splits.

Downloads the original FinancialPhraseBank-v1.0 archive, keeps the 75%-agreement
subset, encodes each sentiment with *this project's* label ids, and writes
stratified 80/10/10 train/valid/test parquet files under data/processed/.

Label ids follow the project convention in ``src/models/inference.py``:
``LABELS = ("neutral", "positive", "negative")`` -> 0=neutral, 1=positive,
2=negative. The dataset stores the sentiment as a name per line, so we map that
name straight to the project id, and ``validate_splits`` asserts at runtime that
``LABELS`` here stays in sync with ``inference.LABELS``.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PHRASEBANK_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/"
    "resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
RAW_ZIP_PATH = RAW_DIR / "FinancialPhraseBank-v1.0.zip"          # cached download
CONFIG_MEMBER = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"  # the 75agree filter

SPLIT_PATHS = {
    split: PROCESSED_DIR / f"phrasebank_75agree_{split}.parquet"
    for split in ("train", "valid", "test")
}

RANDOM_SEED = 42
TEST_FRAC = 0.10                          # peeled off first (stratified)
VALID_FRAC_OF_REMAINDER = TEST_FRAC / (1 - TEST_FRAC)  # 0.10/0.90 -> valid ~= 10% of total

# Project label convention (matches inference.LABELS; drift-guarded at runtime).
# Order is load-bearing: the index IS the label id the fine-tuned head learns.
LABELS = ("neutral", "positive", "negative")            # 0=neutral, 1=positive, 2=negative
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def download_phrasebank() -> Path:
    """Download the FinancialPhraseBank archive to RAW_ZIP_PATH if not already cached."""
    if RAW_ZIP_PATH.exists():
        print(f"Using cached archive at {RAW_ZIP_PATH}")
        return RAW_ZIP_PATH
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {PHRASEBANK_URL} ...")
    with urllib.request.urlopen(PHRASEBANK_URL) as resp:
        RAW_ZIP_PATH.write_bytes(resp.read())
    print(f"Wrote {RAW_ZIP_PATH.stat().st_size:,} bytes to {RAW_ZIP_PATH}")
    return RAW_ZIP_PATH


def load_75agree(zip_path: Path) -> pd.DataFrame:
    """Parse the 75agree member into a frame of (text, label, label_name).

    Each line is ``sentence@sentiment`` in latin-1; the sentiment name is encoded
    with the project ``LABEL_TO_ID``. Exact-duplicate sentences are dropped so an
    identical sentence can never straddle the train/valid/test boundary (the
    75agree duplicates all carry a consistent label, so keeping the first is lossless).
    """
    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        text = zf.read(CONFIG_MEMBER).decode("latin-1")  # not valid UTF-8 (£, ñ, ...)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on the LAST '@': a sentence may itself contain '@'.
        sentence, sentiment = line.rsplit("@", 1)
        sentiment = sentiment.strip()
        rows.append((sentence.strip(), LABEL_TO_ID[sentiment], sentiment))

    df = pd.DataFrame(rows, columns=["text", "label", "label_name"])
    print(f"Loaded {len(df)} sentences from {CONFIG_MEMBER}")

    deduped = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(deduped) != len(df):
        print(f"Dropped {len(df) - len(deduped)} duplicate sentences -> {len(deduped)} unique")
    return deduped


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 train/valid/test split on ``label``, seeded for reproducibility.

    Two stages: peel off the test set (10% of all), then split the 90% remainder
    into valid (~10% of all) and train (~80% of all). Stratifying on ``label``
    preserves the class ratios in every split despite the ~12% negative imbalance.
    """
    remainder, test = train_test_split(
        df,
        test_size=TEST_FRAC,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    train, valid = train_test_split(
        remainder,
        test_size=VALID_FRAC_OF_REMAINDER,
        stratify=remainder["label"],
        random_state=RANDOM_SEED,
    )
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def assert_label_convention() -> None:
    """Fail if our label ids drift from the project convention in inference.

    ``label`` here is coupled to ``inference.LABELS`` by position; if that
    tuple is ever reordered, this training data would be silently mislabeled
    relative to the model head. Read the tuple from source (via ``ast``) rather
    than importing the module, which would drag in torch/transformers just to
    read a constant.
    """
    import ast

    source = (ROOT / "src" / "models" / "inference.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "LABELS" for t in node.targets
        ):
            baseline_labels = tuple(ast.literal_eval(node.value))
            break
    else:
        raise ValueError("could not find LABELS in inference.py")

    if LABELS != baseline_labels:
        raise ValueError(
            f"label convention drift: load_phrasebank.LABELS={LABELS} != "
            f"inference.LABELS={baseline_labels}"
        )


def validate_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    full: pd.DataFrame,
) -> None:
    """Assert the label ids match the project convention and the splits partition cleanly."""
    assert_label_convention()
    splits = {"train": train, "valid": valid, "test": test}

    total = sum(len(s) for s in splits.values())
    if total != len(full):
        raise ValueError(f"split rows sum to {total}, expected {len(full)}")

    # Disjoint on text: no sentence leaks across splits.
    texts = [set(s["text"]) for s in splits.values()]
    if len(set().union(*texts)) != total:
        raise ValueError("splits overlap on text; a sentence appears in more than one split")

    for name, s in splits.items():
        if set(s["label"]) != {0, 1, 2}:
            raise ValueError(f"{name} does not contain all three classes")


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Write each split to its parquet and log per-split class distributions."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in SPLIT_PATHS.items():
        out = splits[name]
        out.to_parquet(path, index=False)
        dist = out["label_name"].value_counts()
        dist_str = "  ".join(
            f"{label}={dist.get(label, 0)} ({dist.get(label, 0) / len(out):.1%})"
            for label in LABELS
        )
        print(f"Wrote {len(out):5d} rows to {path.name}  |  {dist_str}")


def main() -> None:
    """Download, parse, split, validate, and save the 75agree splits."""
    zip_path = download_phrasebank()
    df = load_75agree(zip_path)
    train, valid, test = split_dataset(df)
    validate_splits(train, valid, test, df)
    save_splits({"train": train, "valid": valid, "test": test})


if __name__ == "__main__":
    main()
