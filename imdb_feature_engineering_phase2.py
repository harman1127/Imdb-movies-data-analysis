"""
Phase 2: Feature Engineering
Trending Movies Data Analysis and Popularity Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load cleaned dataset generated in Phase 1
df = pd.read_csv("cleaned_imdb_movies.csv")

# --------------------------------------------------
# 1. Profit
# --------------------------------------------------
df["profit"] = df["gross"] - df["budget"]

# --------------------------------------------------
# 2. Return on Investment (ROI)
# --------------------------------------------------
df["roi"] = np.where(
    df["budget"] != 0,
    (df["gross"] - df["budget"]) / df["budget"],
    0
)

# --------------------------------------------------
# 3. Genre Count
# --------------------------------------------------
df["genre_count"] = df["genres"].astype(str).apply(lambda x: len(x.split("|")))

# --------------------------------------------------
# 4. Movie Age
# --------------------------------------------------
CURRENT_YEAR = 2026
df["movie_age"] = CURRENT_YEAR - df["title_year"]

# --------------------------------------------------
# 5. Total Facebook Popularity
# --------------------------------------------------
df["total_facebook_popularity"] = (
    df["movie_facebook_likes"] +
    df["director_facebook_likes"] +
    df["actor_1_facebook_likes"] +
    df["actor_2_facebook_likes"] +
    df["actor_3_facebook_likes"]
)

# --------------------------------------------------
# 6. Total Reviews
# --------------------------------------------------
df["total_reviews"] = (
    df["num_critic_for_reviews"] +
    df["num_user_for_reviews"]
)

# --------------------------------------------------
# 7. Log Votes
# --------------------------------------------------
df["log_votes"] = np.log1p(df["num_voted_users"])

# --------------------------------------------------
# 8. Budget Category
# --------------------------------------------------
df["budget_category"] = pd.cut(
    df["budget"],
    bins=[-np.inf, -0.5, 0.5, np.inf],
    labels=["Low", "Medium", "High"]
)

# --------------------------------------------------
# 9. Rating Category
# --------------------------------------------------
df["rating_category"] = pd.cut(
    df["imdb_score"],
    bins=[0,5,7,10],
    labels=["Low","Average","High"]
)

# --------------------------------------------------
# 10. Encode Categories
# --------------------------------------------------
le = LabelEncoder()
df["budget_category"] = le.fit_transform(df["budget_category"])
df["rating_category"] = le.fit_transform(df["rating_category"])

# --------------------------------------------------
# 11. Optional PCA
# --------------------------------------------------
numeric_df = df.select_dtypes(include=["int64","float64"])
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(numeric_df)

print("Original Numeric Features:", numeric_df.shape[1])
print("PCA Components:", pca_features.shape[1])

# Save engineered dataset
df.to_csv("feature_engineered_imdb_movies.csv", index=False)

print("\nFeature Engineering Completed Successfully.")
print("Dataset saved as feature_engineered_imdb_movies.csv")
