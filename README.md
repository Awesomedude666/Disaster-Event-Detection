# Multilingual Disaster Event Detection from Social Media

An unsupervised, end-to-end disaster event detection pipeline that discovers, clusters, and ranks real-world crisis events from multilingual social media streams (**English, Hindi (Devanagari), and code-mixed Hinglish**).

The system combines **XLM-RoBERTa** contextual embeddings, **multi-attribute similarity fusion** (semantic, hashtag, temporal, and geospatial), **adaptive graph thresholding**, and **Louvain community detection** to surface high-severity events without requiring predefined event counts.

---

## Architecture & Pipeline

```text
Raw Tweets CSV
      │
      ▼
[Step 0] Preprocessing & Feature Engineering (src/preprocessing.py)
      │  ├── Unix timestamp extraction & hashtag/coordinate parsing
      │  └── Influence-weighted engagement & log-normalized virality
      ▼
[Step 1] Multilingual Embeddings (src/embeddings.py)
      │  └── xlm-roberta-base + attention-mask weighted mean pooling (768-d)
      ▼
[Step 2] Multi-Attribute Similarity Matrix (src/similarity.py)
      │  └── 0.60 Semantic + 0.20 Hashtag (Jaccard) + 0.10 Temporal + 0.10 Spatial
      ▼
[Step 3] Adaptive Graph Construction (src/graph_clustering.py)
      │  └── Symmetric edges via per-node threshold: τ(i) = μ(i) + λ·σ(i)
      ▼
[Step 4] Louvain Community Detection (src/graph_clustering.py)
      │  └── Weighted modularity optimization → candidate event clusters
      ▼
[Step 5 & 6] Event Extraction & Importance Scoring (src/event_analysis.py)
      │  ├── Language-agnostic TF-IDF keywords, top hashtags & geo bounding boxes
      │  └── Min-Max weighted ranking: Volume (0.50) + Severity (0.30) + Engagement (0.20) + Verified (0.05)
      ▼
[Step 7] Evaluation & Visualisation (src/evaluation.py)
         └── Modularity, NMI, ARI, Homogeneity, Completeness, V-Measure & diagnostic plots




Disaster-Event-Detection/
├── src/
│   ├── __init__.py
│   ├── config.py               # Hyperparameters, similarity weights & scoring constants
│   ├── preprocessing.py        # Step 0: Data cleaning, parsing & engagement metrics
│   ├── embeddings.py           # Step 1: XLM-RoBERTa tokenization & mean pooling
│   ├── similarity.py           # Step 2: Pairwise multi-attribute similarity computation
│   ├── graph_clustering.py     # Step 3 & 4: Adaptive threshold graph & Louvain clustering
│   ├── event_analysis.py       # Step 5 & 6: Event summarization & severity-based ranking
│   └── evaluation.py           # Step 7: Supervised metrics & Matplotlib visualisations
├── main.py                     # CLI entry point & pipeline orchestrator
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes local datasets, binaries, and output plots
└── README.md