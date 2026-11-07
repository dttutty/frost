"""Official candidate layouts for the supported TGB link-prediction datasets."""

TGB_NEGATIVE_COUNTS = {
    "TGBL_WIKI": 999,
    "TGBL_COIN": 20,
    "TGBL_COMMENT": 20,
    "TGBL_FLIGHT": 20,
    "TGBL_REVIEW": 100,
}


def tgb_negative_layout(dataset: str) -> tuple[int, bool]:
    """Return candidate capacity and whether shorter queries need padding."""
    normalized = dataset.strip().upper().replace("-", "_")
    if normalized not in TGB_NEGATIVE_COUNTS:
        raise ValueError(
            f"Unsupported official TGB dataset {dataset!r}; expected one of "
            + ", ".join(TGB_NEGATIVE_COUNTS)
            + ". Classic datasets such as WIKI are not official TGB datasets."
        )
    return TGB_NEGATIVE_COUNTS[normalized], normalized == "TGBL_WIKI"
