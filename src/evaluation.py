import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from src.config import DEFAULT_OUTPUT_DIR, SEED


def evaluate(df: pd.DataFrame, modularity: float) -> None:
    """
    Step 7a: Compute supervised clustering evaluation metrics against ground-truth labels.
    """
    print("\n[STEP 7] Evaluation ...")

    if "label" not in df.columns:
        print("  No ground-truth `label` column found. Skipping supervised metrics.")
        return

    y_true = df["label"].values
    y_pred = df["cluster"].values

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    hom, comp, vmeasure = homogeneity_completeness_v_measure(y_true, y_pred)

    print(f"  Modularity   : {modularity:.4f}  (higher = better community structure)")
    print(f"  NMI          : {nmi:.4f}")
    print(f"  ARI          : {ari:.4f}")
    print(f"  Homogeneity  : {hom:.4f}")
    print(f"  Completeness : {comp:.4f}")
    print(f"  V-measure    : {vmeasure:.4f}")


def visualise(
    df: pd.DataFrame,
    G: nx.Graph,
    similarity_matrix: np.ndarray,
    scored_events: pd.DataFrame,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Step 7b: Generate and save diagnostic and event importance visualisations.
    """
    print("\n  Generating visualisations ...")
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Similarity score distribution
    flat_sim = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat_sim, bins=80, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.set_title("Pairwise Similarity Score Distribution (XLM-RoBERTa)", fontsize=14)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "similarity_distribution.png"), dpi=150)
    plt.close()

    # 2. Cluster size distribution
    sizes = df["cluster"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(min(30, len(sizes))), sizes.values[:30], color="#ff7f0e")
    ax.set_title("Top-30 Cluster Sizes", fontsize=14)
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Tweet count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cluster_sizes.png"), dpi=150)
    plt.close()

    # 3. Event importance bar chart
    top = scored_events.head(15).copy()
    top["label"] = top["cluster_id"].astype(str) + " (" + top["tweet_count"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top["label"][::-1], top["importance_score"][::-1], color="#2ca02c")
    ax.set_title("Top Events by Importance Score", fontsize=14)
    ax.set_xlabel("Importance Score (normalised)")
    for bar, score in zip(bars, top["importance_score"][::-1]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "event_importance.png"), dpi=150)
    plt.close()

    # 4. Graph visualisation
    MAX_NODES = 300
    sub_G = G.subgraph(list(G.nodes())[:MAX_NODES]) if G.number_of_nodes() > MAX_NODES else G
    cluster_map = df["cluster"].to_dict()
    node_colors = [cluster_map.get(n, 0) for n in sub_G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(sub_G, seed=SEED)
    nx.draw_networkx(
        sub_G,
        pos,
        ax=ax,
        node_color=node_colors,
        cmap=cm.tab20,
        node_size=40,
        with_labels=False,
        edge_color="#cccccc",
        alpha=0.7,
        width=0.4,
    )
    ax.set_title("Adaptive Similarity Graph — coloured by cluster (XLM-RoBERTa)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "graph_clusters.png"), dpi=150)
    plt.close()

    # 5. Language distribution per top event
    if "languages" in scored_events.columns:
        top5 = scored_events.head(5)
        fig, axes = plt.subplots(1, min(5, len(top5)), figsize=(14, 4))
        if len(top5) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, top5.iterrows()):
            lang_dist = row["languages"]
            if isinstance(lang_dist, dict) and lang_dist:
                ax.pie(lang_dist.values(), labels=lang_dist.keys(), autopct="%1.0f%%")
                ax.set_title(f"Cluster {row['cluster_id']}", fontsize=10)
        fig.suptitle("Language Distribution in Top 5 Events", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "language_distribution.png"), dpi=150)
        plt.close()

    print(f"  Plots saved to `{output_dir}/`")