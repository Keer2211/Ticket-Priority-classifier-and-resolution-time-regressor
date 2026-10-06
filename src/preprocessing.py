"""Data loading and preprocessing utilities for customer support tickets."""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "customer_support_tickets.csv"


def load_tickets(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the ticket dataset from a CSV file."""
    return pd.read_csv(path)


def clean_text(value: object) -> str:
    """Normalize a ticket text value for downstream NLP tasks."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())
