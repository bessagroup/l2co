import re


def normalize_key(key: str) -> str:
    """Removes all non-alphanumeric characters and lowercases the string."""
    return re.sub(r"[^a-zA-Z0-9]", "", key).lower()
