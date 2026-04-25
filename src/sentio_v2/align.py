import pandas as pd


def align_to_millisecond(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx["timestamp_ms"] = dfx["timestamp_ms"].round().astype("int64")
    dfx = dfx.drop_duplicates("timestamp_ms").set_index("timestamp_ms")
    full_index = pd.RangeIndex(dfx.index.min(), dfx.index.max() + 1, step=1)
    aligned = dfx.reindex(full_index).interpolate(method="linear")
    aligned.index.name = "timestamp_ms"
    return aligned.reset_index()

