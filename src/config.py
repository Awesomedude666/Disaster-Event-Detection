# Default runtime paths (created automatically at runtime, ignored by git)
DEFAULT_DATA_PATH = "sample_tweets.csv"
DEFAULT_OUTPUT_DIR = "output_plots"

# Model Configuration
XLM_MODEL = "xlm-roberta-base"
BATCH_SIZE = 16
MAX_SEQ_LEN = 128

# Graph Configuration
LAMBDA = 0.5
TEMPORAL_DECAY = 3600       # 1-hour decay window (seconds)
MAX_DIST_KM = 500           # Spatial normalization cap
MIN_CLUSTER_SIZE = 3

# Multi-Attribute Similarity Weights
W_SEMANTIC = 0.60
W_HASHTAG = 0.20
W_TEMPORAL = 0.10
W_SPATIAL = 0.10

# Event Importance Scoring Weights
SCORE_VOLUME = 0.50
SCORE_SEVERITY = 0.30
SCORE_ENGAGEMENT = 0.20
SCORE_VERIFIED = 0.05

SEED = 42