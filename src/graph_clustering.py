import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from src.config import LAMBDA, SEED


def build_adaptive_graph(
    similarity_matrix: np.ndarray,
    df: pd.DataFrame,
    lamda: float = LAMBDA,
) -> nx.Graph:
    """
    Step 3: Construct symmetric graph using per-node adaptive thresholds.
    """
    print(f"\n[STEP 3] Building adaptive graph (lambda={lamda}) ...")
    N = similarity_matrix.shape[0]

    means = np.mean(similarity_matrix, axis=1)
    stds = np.std(similarity_matrix, axis=1)
    thresholds = means + lamda * stds

    G = nx.Graph()

    for i in range(N):
        G.add_node(
            i,
            threshold=float(thresholds[i]),
            timestamp=int(df["timestamp"].iloc[i]),
            language=str(df["language"].iloc[i]),
            verified=bool(df["isVerified"].iloc[i]),
            engagement=float(df["engagement"].iloc[i]),
        )

    for i in range(N):
        for j in range(i + 1, N):
            sim = float(similarity_matrix[i, j])
            if sim > thresholds[i] and sim > thresholds[j]:
                G.add_edge(i, j, weight=sim)

    density = nx.density(G)
    n_comp = nx.number_connected_components(G)
    avg_deg = sum(d for _, d in G.degree()) / N if N > 0 else 0

    print(f"  Nodes              : {G.number_of_nodes()}")
    print(f"  Edges              : {G.number_of_edges()}")
    print(f"  Graph density      : {density:.6f}")
    print(f"  Connected comps    : {n_comp}")
    print(f"  Avg degree         : {avg_deg:.2f}")

    return G


def detect_communities(G: nx.Graph, df: pd.DataFrame, seed: int = SEED):
    """
    Step 4: Louvain weighted community detection.
    """
    print("\n[STEP 4] Louvain community detection ...")

    partition = community_louvain.best_partition(G, weight="weight", random_state=seed)

    max_cluster = max(partition.values()) if partition else -1
    for node in G.nodes():
        if node not in partition:
            max_cluster += 1
            partition[node] = max_cluster

    df["cluster"] = df.index.map(partition)

    modularity = community_louvain.modularity(partition, G, weight="weight")
    sizes = df["cluster"].value_counts()

    print(f"  Total clusters       : {df['cluster'].nunique()}")
    print(f"  Modularity score     : {modularity:.4f}")
    print(f"  Largest cluster size : {sizes.max()}")
    print(f"  Smallest cluster size: {sizes.min()}")
    print(f"  Singleton clusters   : {(sizes == 1).sum()}")

    return df, modularity