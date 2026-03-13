from __future__ import annotations

from pathlib import Path
import ast

import pandas as pd

from bizintel.config.settings import (
    STANDARD_COLUMNS,
    MIN_DESCRIPTION_LENGTH,
    DEDUP_SUBSET,
    YC_COLUMN_MAP,
    CRUNCHBASE_COLUMN_MAP,
    CRUNCHBASE_ENTITY_FILTER,
    OUTPUT_DIR,
    YC_OUTPUT_FILENAME,
    CRUNCHBASE_OUTPUT_FILENAME,
    UNIFIED_OUTPUT_FILENAME,
)


# ── Text helpers ──────────────────────────────────────────────────────────


def _clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
    )


def _parse_tags(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())

    if not isinstance(value, str):
        return str(value).strip()

    candidate = value.strip()

    if candidate.startswith("[") and candidate.endswith("]"):
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list):
                return ", ".join(str(v).strip() for v in parsed if str(v).strip())
        except Exception:
            pass

    return candidate


def _extract_first_tag(tags: str) -> str:
    if not tags:
        return ""
    return tags.split(",", 1)[0].strip()


# ── Finalize ──────────────────────────────────────────────────────────────


def _finalize_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:

    df = df.copy()

    df["name"] = _clean_text_series(df["name"])
    df["description"] = _clean_text_series(df["description"])
    df["industry"] = _clean_text_series(df["industry"])
    df["tags"] = _clean_text_series(df["tags"])
    df["country"] = _clean_text_series(df["country"])

    # Remove rows without name
    df = df[df["name"] != ""]

    # Remove extremely weak descriptions
    df = df[df["description"].str.len() > MIN_DESCRIPTION_LENGTH]

    df["source"] = source
    df["founded_year"] = pd.to_numeric(df["founded_year"], errors="coerce").astype("Int64")

    # Deduplicate
    df["name_lower"] = df["name"].str.lower()
    df = df.drop_duplicates(subset=DEDUP_SUBSET)
    df = df.drop(columns=["name_lower"])

    return df[STANDARD_COLUMNS]


# ── Dataset loaders ───────────────────────────────────────────────────────


def load_yc_companies(paths: list[Path]) -> pd.DataFrame:

    frames = [pd.read_csv(p, encoding_errors="replace") for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns=YC_COLUMN_MAP)

    df["short_description"] = _clean_text_series(df.get("short_description"))
    df["long_description"] = _clean_text_series(df.get("long_description"))

    # Better fallback chain
    df["description"] = (
        df["long_description"]
        .replace("", pd.NA)
        .fillna(df["short_description"])
    )

    df["tags"] = df.get("tags").apply(_parse_tags)
    df["industry"] = df["tags"].apply(_extract_first_tag)

    return _finalize_dataframe(df, source="YC")


def load_crunchbase_companies(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path, encoding_errors="replace")

    df = df[df.get("entity_type") == CRUNCHBASE_ENTITY_FILTER].copy()

    df = df.rename(columns=CRUNCHBASE_COLUMN_MAP)

    df["short_description"] = _clean_text_series(df.get("short_description"))
    df["description"] = _clean_text_series(df.get("description"))
    df["overview"] = _clean_text_series(df.get("overview"))
    df["industry"] = _clean_text_series(df.get("category_code"))
    df["tags"] = _clean_text_series(df.get("tag_list"))

    # Improved description fallback chain
    df["description"] = (
        df["description"]
        .replace("", pd.NA)
        .fillna(df["overview"])
        .fillna(df["short_description"])
    )

    # If still empty build minimal text
    fallback_text = (
        "Startup in "
        + df["industry"].fillna("")
        + " industry. Tags: "
        + df["tags"].fillna("")
    )

    df["description"] = df["description"].fillna(fallback_text)

    df["founded_year"] = pd.to_datetime(df.get("founded_at"), errors="coerce").dt.year

    return _finalize_dataframe(df, source="Crunchbase")


# ── I/O helpers ───────────────────────────────────────────────────────────


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_dataset(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


# ── Pipeline entry point ─────────────────────────────────────────────────


def run_preprocessing(
    yc_paths: list[Path],
    crunchbase_path: Path,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    yc_df = load_yc_companies(yc_paths)
    cb_df = load_crunchbase_companies(crunchbase_path)

    ensure_output_dir(output_dir)

    write_dataset(yc_df, output_dir / YC_OUTPUT_FILENAME)
    write_dataset(cb_df, output_dir / CRUNCHBASE_OUTPUT_FILENAME)

    # Merge both into a single unified CSV
    unified_df = pd.concat([yc_df, cb_df], ignore_index=True)
    write_dataset(unified_df, output_dir / UNIFIED_OUTPUT_FILENAME)

    return yc_df, cb_df, unified_df
