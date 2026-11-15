import argparse
import pickle
import warnings
import numpy as np

from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    XLM_MODEL,
    LAMBDA,
    MIN_CLUSTER_SIZE,
)
from src.preprocessing import load_and_preprocess
from src.embeddings import get_xlm_roberta_embeddings
from src.similarity import build_similarity_matrix
from src.graph_clustering import build_adaptive_graph, detect_communities
from src.event_analysis import detect_events, compute_event_scores
from src.evaluation import evaluate, visualise

warnings.filterwarnings("ignore")


def run_pipeline(csv_path: str, save_artifacts: bool = True) -> dict:
    """End-to-end multilingual disaster event detection pipeline."""

    # Step 0: Load & preprocess
    df = load_and_preprocess(csv_path)

    # Step 1: XLM-RoBERTa embeddings
    embeddings = get_xlm_roberta_embeddings(df["content"].tolist(), model_name=XLM_MODEL)
    if save_artifacts:
        np.save("embeddings.npy", embeddings)
        print("  Embeddings saved -> embeddings.npy")

    # Step 2: Similarity matrix
    sim_matrix = build_similarity_matrix(df, embeddings)
    if save_artifacts:
        np.save("similarity_matrix.npy", sim_matrix)
        print("  Similarity matrix saved -> similarity_matrix.npy")

    # Step 3: Adaptive graph
    G = build_adaptive_graph(sim_matrix, df, lamda=LAMBDA)
    if save_artifacts:
        with open("graph.gpickle", "wb") as f:
            pickle.dump(G, f)
        print("  Graph saved -> graph.gpickle")

    # Step 4: Louvain clustering
    df, modularity = detect_communities(G, df)

    # Step 5: Event detection
    event_df = detect_events(df, min_cluster_size=MIN_CLUSTER_SIZE)

    # Step 6: Event scoring
    scored_events = compute_event_scores(df, event_df)

    # Step 7: Evaluation & visualisation
    evaluate(df, modularity)
    visualise(df, G, sim_matrix, scored_events, output_dir=DEFAULT_OUTPUT_DIR)

    if save_artifacts:
        df.drop(columns=["hashtag_set", "coords"], errors="ignore").to_csv(
            "tweets_with_clusters.csv", index=False
        )
        scored_events.to_csv("events_scored.csv", index=False)
        print("\n  Output CSVs saved.")

    print("\n[DONE] Pipeline complete.")
    return {
        "df": df,
        "embeddings": embeddings,
        "sim_matrix": sim_matrix,
        "graph": G,
        "modularity": modularity,
        "events": scored_events,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster Event Detection Pipeline")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to input CSV")
    parser.add_argument("--no-save", action="store_true", help="Run in-memory without writing local artifacts")
    args = parser.parse_args()

    run_pipeline(csv_path=args.data, save_artifacts=not args.no_save)