import ast
import numpy as np
import pandas as pd


def _parse_tags(x) -> set:
    if isinstance(x, list):
        return set(h.lower() for h in x if isinstance(h, str))
    if isinstance(x, str):
        try:
            return set(h.lower() for h in ast.literal_eval(x) if isinstance(h, str))
        except Exception:
            return set(t.strip().lstrip("#").lower() for t in x.split(",") if t.strip())
    return set()


def _parse_coords(x):
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return tuple(x)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return tuple(float(v) for v in val)
        except Exception:
            pass
    return None


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Step 0: Load CSV, clean text, and engineer temporal, spatial, and engagement features.
    """
    print("\n[STEP 0] Loading data ...")
    df = pd.read_csv(csv_path)
    print(f"  Raw rows : {len(df)}")

    df["timestamp"] = pd.to_datetime(df["date"], utc=True).astype("int64") // 10**9
    df["hashtag_set"] = df["hashtags"].apply(_parse_tags)
    df["coords"] = df["coordinates"].apply(_parse_coords)

    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    df["rt_count"] = pd.to_numeric(df["rt_count"], errors="coerce").fillna(0)
    df["followers_count"] = pd.to_numeric(df["followers_count"], errors="coerce").fillna(0)
    df["isVerified"] = df["isVerified"].astype(bool)

    # Influence-weighted engagement & virality
    verify_boost = df["isVerified"].apply(lambda v: 1.5 if v else 1.0)
    df["engagement"] = (df["like_count"] + df["rt_count"]) * verify_boost
    df["virality"] = np.log1p(df["rt_count"]) / np.log1p(df["followers_count"] + 1)

    # Clean text
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.strip() != ""].reset_index(drop=True)

    if "language" in df.columns:
        print(f"  Language distribution :\n{df['language'].value_counts().to_string()}")

    print(f"  Rows after cleaning : {len(df)}")
    return df