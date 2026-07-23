"""
Phase 3: Machine Learning Model Building and Evaluation
Trending Movies Data Analysis and Popularity Prediction
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# --------------------------------------------------
# Load Feature Engineered Dataset
# --------------------------------------------------
df = pd.read_csv("feature_engineered_imdb_movies.csv")

# --------------------------------------------------
# Prepare Features and Target
# --------------------------------------------------

# Remove non-numeric/object columns not suitable for regression
drop_columns = [
    "imdb_score",
    "movie_title",
    "director_name",
    "actor_1_name",
    "actor_2_name",
    "actor_3_name",
    "movie_imdb_link",
    "plot_keywords",
    "genres",
]

X = df.drop(columns=[c for c in drop_columns if c in df.columns])

# Convert remaining object columns using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Fill any remaining missing values
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

y = df["imdb_score"]

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

print("\n================ MODEL RESULTS ================\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(name)
    print("-"*40)
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print()

    results.append({
        "Model": name,
        "R2 Score": round(r2,4),
        "MAE": round(mae,4),
        "MSE": round(mse,4),
        "RMSE": round(rmse,4)
    })

# --------------------------------------------------
# Compare Results
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("=========== MODEL COMPARISON ===========")
print(results_df)

# --------------------------------------------------
# Save Comparison
# --------------------------------------------------
results_df.to_csv("model_comparison_results.csv", index=False)

print("\nModel comparison saved as model_comparison_results.csv")

best_model = results_df.iloc[0]["Model"]
print(f"\nBest Performing Model: {best_model}")
