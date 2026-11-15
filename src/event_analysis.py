from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.config import (
    MIN_CLUSTER_SIZE,
    SCORE_VOLUME,
    SCORE_SEVERITY,
    SCORE_ENGAGEMENT,
    SCORE_VERIFIED,
)


def detect_events(df: pd.DataFrame, min_cluster_size: int = MIN_CLUSTER_SIZE) -> pd.DataFrame:
    """
    Step 5: Extract candidate events from clusters with keywords, hashtags, and geo-bounds.
    """
    print("\n[STEP 5] Detecting events from clusters ...")

    events = []
    cluster_ids = df["cluster"].unique()
    tfidf = TfidfVectorizer(max_features=10)

    for cid in cluster_ids:
        sub = df[df["cluster"] == cid]

        if len(sub) < min_cluster_size:
            continue

        t_min = pd.to_datetime(sub["date"]).min()
        t_max = pd.to_datetime(sub["date"]).max()
        span_hours = (t_max - t_min).total_seconds() / 3600

        try:
            tfidf.fit(sub["content"].tolist())
            keywords = tfidf.get_feature_names_out().tolist()
        except Exception:
            keywords = []

        all_tags = [tag for s in sub["hashtag_set"] for tag in s]
        top_tags = [t for t, _ in Counter(all_tags).most_common(5)]

        valid_coords = [c for c in sub["coords"] if c is not None]
        if valid_coords:
            lats = [c[0] for c in valid_coords]
            lons = [c[1] for c in valid_coords]
            geo_box = {
                "lat_min": min(lats),
                "lat_max": max(lats),
                "lon_min": min(lons),
                "lon_max": max(lons),
            }
        else:
            geo_box = None

        events.append({
            "cluster_id": cid,
            "tweet_count": len(sub),
            "start_time": str(t_min),
            "end_time": str(t_max),
            "span_hours": round(span_hours, 2),
            "keywords": keywords,
            "top_hashtags": top_tags,
            "geo_box": geo_box,
            "verified_tweets": int(sub["isVerified"].sum()),
            "languages": sub["language"].value_counts().to_dict(),
        })

    event_df = pd.DataFrame(events)
    print(f"  Events detected (>={min_cluster_size} tweets) : {len(event_df)}")
    return event_df


def compute_event_scores(df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Rank events by normalized volume, negative sentiment severity, and engagement.
    """
    print("\n[STEP 6] Computing event importance scores ...")
    print("  [Note] TextBlob + VADER are English-based. Hindi sentiment may be weaker.")
    print("         Consider: cardiffnlp/twitter-xlm-roberta-base-sentiment for full coverage.")

    vader = SentimentIntensityAnalyzer()

    def get_sentiment(text: str) -> dict:
        tb = TextBlob(text).sentiment.polarity
        vader_compound = vader.polarity_scores(text)["compound"]
        return {"textblob": tb, "vader": vader_compound}

    print("  Computing sentiments ...")
    sentiments = df["content"].apply(get_sentiment)
    df["sent_tb"] = sentiments.apply(lambda x: x["textblob"])
    df["sent_vader"] = sentiments.apply(lambda x: x["vader"])

    scores_list = []

    for _, row in event_df.iterrows():
        cid = row["cluster_id"]
        sub = df[df["cluster"] == cid]

        volume = len(sub)
        avg_sent = (sub["sent_tb"].mean() + sub["sent_vader"].mean()) / 2
        severity = -avg_sent

        avg_engagement = sub["engagement"].mean()
        avg_virality = sub["virality"].mean()
        engagement_score = avg_engagement + 10 * avg_virality
        verified_ratio = sub["isVerified"].mean()

        scores_list.append({
            "cluster_id": cid,
            "volume": volume,
            "severity": severity,
            "engagement_score": engagement_score,
            "verified_ratio": verified_ratio,
        })

    score_df = pd.DataFrame(scores_list)

    scaler = MinMaxScaler()
    cols_to_norm = ["volume", "severity", "engagement_score"]
    score_df[cols_to_norm] = scaler.fit_transform(score_df[cols_to_norm])

    score_df["importance_score"] = (
        SCORE_VOLUME * score_df["volume"]
        + SCORE_SEVERITY * score_df["severity"]
        + SCORE_ENGAGEMENT * score_df["engagement_score"]
        + SCORE_VERIFIED * score_df["verified_ratio"]
    )

    result = (
        event_df.merge(score_df, on="cluster_id")
        .sort_values("importance_score", ascending=False)
        .reset_index(drop=True)
    )

    print("  Top 5 events:")
    print(result[["cluster_id", "tweet_count", "keywords", "importance_score"]].head())
    return result