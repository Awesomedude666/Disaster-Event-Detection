import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import great_circle
from src.config import (
    TEMPORAL_DECAY,
    MAX_DIST_KM,
    W_SEMANTIC,
    W_HASHTAG,
    W_TEMPORAL,
    W_SPATIAL,
)


def hashtag_similarity(tags1: set, tags2: set) -> float:
    if not tags1 and not tags2:
        return 0.0
    union = tags1 | tags2
    return len(tags1 & tags2) / len(union)


def temporal_similarity(t1: float, t2: float, decay: float = TEMPORAL_DECAY) -> float:
    return float(np.exp(-abs(t1 - t2) / decay))


def spatial_similarity(c1, c2, max_km: float = MAX_DIST_KM) -> float:
    if c1 is None or c2 is None:
        return 0.0
    try:
        dist_km = great_circle(c1, c2).km
        return 1.0 / (1.0 + dist_km / max_km)
    except Exception:
        return 0.0


def build_similarity_matrix(df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    """
    Step 2: Build NxN weighted multi-attribute similarity matrix.
    """
    print("\n[STEP 2] Building multi-attribute similarity matrix ...")

    N = len(df)
    timestamps = df["timestamp"].values
    hashtag_sets = df["hashtag_set"].tolist()
    coords = df["coords"].tolist()

    semantic_matrix = cosine_similarity(embeddings)
    similarity_matrix = np.zeros((N, N), dtype=np.float32)

    for i in tqdm(range(N), desc="  Computing pairwise similarities"):
        for j in range(i, N):
            sem = float(semantic_matrix[i, j])
            htag = hashtag_similarity(hashtag_sets[i], hashtag_sets[j])
            temp = temporal_similarity(timestamps[i], timestamps[j])
            spat = spatial_similarity(coords[i], coords[j])

            sim = (
                W_SEMANTIC * sem
                + W_HASHTAG * htag
                + W_TEMPORAL * temp
                + W_SPATIAL * spat
            )

            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    print(f"  Similarity matrix shape : {similarity_matrix.shape}")
    print(f"  Non-zero entries        : {np.count_nonzero(similarity_matrix)}")
    print(f"  Mean similarity         : {similarity_matrix.mean():.4f}")
    print(f"  Max  similarity         : {similarity_matrix.max():.4f}")
    return similarity_matrix