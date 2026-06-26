"""05: Extract per-speaker prepared remarks from earnings transcripts.

Slices the "Prepared Remarks" section out of each raw transcript, walks it line
by line, and buckets every speaker turn into CEO, CFO, or other-executive based
on the attribution line (``Name -- Title``) that introduces it. Non-executives
(analysts, IR, operator, ...) are dropped. Adds four columns to the master frame:

  text_prepared_ceo        — concatenated CEO turns (str or None)
  text_prepared_cfo        — concatenated CFO turns (str or None)
  text_prepared_other_exec — concatenated other-executive turns (str or None)
  section_parse_ok         — False when no CEO or CFO turn was found.

The check order in ``classify_role`` is load-bearing: CFO runs before the
EVP/SVP exclusion so "EVP and Chief Financial Officer" resolves to CFO.
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MASTER_PATH = PROCESSED_DIR / "master.parquet"
MASTER_CLEAN_PATH = PROCESSED_DIR / "master_clean.parquet"

PREPARED_HEADER = "Prepared Remarks"
QA_HEADER = "Questions and Answers"

# Corporate-name tails that mark the title side as a firm affiliation (an
# analyst line), not an executive title — e.g. "... -- Morgan Stanley & Co."
_ATTR_TAIL = re.compile(r"(\band company|& co\.?|& associates)\s*$", re.IGNORECASE)

# Title tokens that mark a speaker as a non-executive to be dropped entirely.
_EXCLUDE_TITLE = re.compile(
    r"analyst|investor relations|securities|capital|research|partners|moderator"
    r"|operator|counsel|communications|board|reuters|cnbc|bloomberg|reporter"
    r"|correspondent|vice president|svp|evp",
    re.IGNORECASE,
)

_CEO_TITLE = re.compile(
    r"chief executive officer|president and ceo|president & ceo", re.IGNORECASE
)
_CHAIRMAN = re.compile(r"\bchairman\b", re.IGNORECASE)
_OTHER_EXEC_TITLE = re.compile(r"\bchief\b|\bpresident\b", re.IGNORECASE)

_OUTPUT_KEYS = (
    "text_prepared_ceo",
    "text_prepared_cfo",
    "text_prepared_other_exec",
)

_ROLES = ("ceo", "cfo", "other_exec")


def normalize_transcript(text: str) -> str:
    """Replace non-breaking spaces with regular spaces.

    Attribution dashes are padded with ``\\xa0`` in the source; normalizing
    lets the rest of the pipeline match on `` -- ``.
    """
    return text.replace("\xa0", " ")


def extract_prepared_remarks(text: str) -> str | None:
    """Return the prepared-remarks slice, or None if there is no header.

    Slices from the "Prepared Remarks" header to the "Questions and Answers"
    header, or to the end of the string when no Q&A header is present.
    """
    start = text.find(PREPARED_HEADER)
    if start == -1:
        return None
    end = text.find(QA_HEADER, start)
    if end == -1:
        return text[start:]
    return text[start:end]


def is_attribution_line(line: str) -> bool:
    """True when ``line`` looks like a ``Name -- Title`` speaker attribution.

    Rejects prose: requires a `` -- `` separator, length < 100, no trailing
    sentence punctuation or dangling ``--``, a capitalized name side, and a
    title side that is not a corporate-name affiliation (see ``_ATTR_TAIL``).
    """
    if " -- " not in line:
        return False
    if len(line) >= 100:
        return False
    stripped = line.rstrip()
    if stripped.endswith((".", "?", "!", ",")) or stripped.endswith("--"):
        return False
    left, _, right = line.partition(" -- ")
    if not left[:1].isupper():
        return False
    if _ATTR_TAIL.search(right):
        return False
    return True


def classify_role(line: str) -> str:
    """Map an attribution line to 'ceo', 'cfo', 'other_exec', or 'exclude'.

    The checks run in a fixed, load-bearing order:
      1. two or more ``--`` separators -> malformed, exclude
      2. an explicit CEO title -> ceo
      3. chairman (but not vice chairman) -> ceo
      4. CFO title -> cfo (must precede the EVP/SVP exclusion below so a title
         like "EVP and Chief Financial Officer" lands in cfo, not exclude)
      5. a non-executive token -> exclude
      6. any other chief/president title -> other_exec
      7. default -> exclude
    """
    lowered = line.lower()
    if line.count("--") >= 2:
        return "exclude"
    if _CEO_TITLE.search(lowered):
        return "ceo"
    if _CHAIRMAN.search(lowered) and "vice chairman" not in lowered:
        return "ceo"
    if "chief financial officer" in lowered:
        return "cfo"
    if _EXCLUDE_TITLE.search(lowered):
        return "exclude"
    if _OTHER_EXEC_TITLE.search(lowered):
        return "other_exec"
    return "exclude"


def extract_speaker_blocks(prepared_text: str) -> dict[str, str | None]:
    """Walk the prepared-remarks text and bucket each turn by speaker role.

    Each attribution line flushes the buffered turn into the current role's
    bucket and re-points to the new speaker (excluded speakers buffer nothing).
    Returns each role's turns newline-joined, or None when that role never spoke.
    """
    blocks: dict[str, list[str]] = {role: [] for role in _ROLES}
    current_role: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if current_role is not None and buffer:
            turn = "\n".join(buffer).strip()
            if turn:
                blocks[current_role].append(turn)
        buffer = []

    for line in prepared_text.split("\n"):
        if is_attribution_line(line):
            flush()
            role = classify_role(line)
            current_role = role if role != "exclude" else None
        elif current_role is not None:
            buffer.append(line)
    flush()

    return {role: ("\n".join(turns).strip() or None) for role, turns in blocks.items()}


def parse_transcript(text: object) -> dict[str, object]:
    """Parse one transcript into the four output columns.

    Non-string input (e.g. a null transcript) yields all-None text columns and
    section_parse_ok=False. section_parse_ok is also False when prepared remarks
    are missing or when neither a CEO nor a CFO turn was extracted.
    """
    null_result = {key: None for key in _OUTPUT_KEYS} | {"section_parse_ok": False}
    if not isinstance(text, str):
        return dict(null_result)

    prepared = extract_prepared_remarks(normalize_transcript(text))
    if prepared is None:
        return dict(null_result)

    blocks = extract_speaker_blocks(prepared)
    ceo = blocks["ceo"]
    cfo = blocks["cfo"]
    return {
        "text_prepared_ceo": ceo,
        "text_prepared_cfo": cfo,
        "text_prepared_other_exec": blocks["other_exec"],
        "section_parse_ok": not (ceo is None and cfo is None),
    }


def log_stats(out: pd.DataFrame) -> None:
    """Print row count, section_parse_ok rate, and CEO/CFO coverage breakdown."""
    n = len(out)
    ok = int(out["section_parse_ok"].sum())
    has_ceo = out["text_prepared_ceo"].notna()
    has_cfo = out["text_prepared_cfo"].notna()
    print(f"Rows: {n}")
    print(f"section_parse_ok: {ok} ({ok / n:.2%})" if n else "section_parse_ok: 0")
    print(f"  CEO present:        {int(has_ceo.sum())} ({has_ceo.mean():.2%})")
    print(f"  CFO present:        {int(has_cfo.sum())} ({has_cfo.mean():.2%})")
    print(f"  CEO and CFO:        {int((has_ceo & has_cfo).sum())} ({(has_ceo & has_cfo).mean():.2%})")
    print(f"  CEO or CFO:         {int((has_ceo | has_cfo).sum())} ({(has_ceo | has_cfo).mean():.2%})")
    print(f"  neither (parse_ok False): {int((~(has_ceo | has_cfo)).sum())}")


def main() -> None:
    """Parse every transcript in master.parquet and write master_clean.parquet."""
    df = pd.read_parquet(MASTER_PATH)
    parsed = pd.DataFrame(
        list(df["transcript"].apply(parse_transcript)), index=df.index
    )
    out = df.copy()
    for col in (*_OUTPUT_KEYS, "section_parse_ok"):
        out[col] = parsed[col]
    out["section_parse_ok"] = out["section_parse_ok"].astype(bool)

    log_stats(out)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MASTER_CLEAN_PATH, index=False)
    print(f"Wrote {len(out)} rows to {MASTER_CLEAN_PATH}")


if __name__ == "__main__":
    main()
