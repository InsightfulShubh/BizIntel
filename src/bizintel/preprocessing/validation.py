from __future__ import annotations

from datetime import datetime
import re

import pandas as pd

from bizintel.config.settings import (
    SUSPICIOUS_NAME_MIN_LENGTH,
    SUSPICIOUS_DESC_MIN_LENGTH,
    SUSPICIOUS_YEAR_MIN,
    SUSPICIOUS_PLACEHOLDER_PATTERN,
)


_PLACEHOLDER_NAME = re.compile(SUSPICIOUS_PLACEHOLDER_PATTERN, re.IGNORECASE)


def add_suspicious_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    name_len = df["name"].fillna("").str.len()
    desc_len = df["description"].fillna("").str.len()
    founded_year = pd.to_numeric(df["founded_year"], errors="coerce")

    current_year = datetime.utcnow().year

    suspicious = (
        (name_len < SUSPICIOUS_NAME_MIN_LENGTH)
        | (desc_len < SUSPICIOUS_DESC_MIN_LENGTH)
        | (df["name"].fillna("").str.match(_PLACEHOLDER_NAME))
        | (founded_year < SUSPICIOUS_YEAR_MIN)
        | (founded_year > current_year + 1)
    )

    df["is_suspicious"] = suspicious.fillna(False)
    return df
