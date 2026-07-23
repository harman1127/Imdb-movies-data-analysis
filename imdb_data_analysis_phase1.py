"""
Trending Movies Data Analysis and Popularity Prediction
Phase 1: Project Definition, Data Preprocessing, and Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# 1. DEFINE PROJECT GOAL
# -------------------------------
print("="*70)
print("PROJECT: Trending Movies Data Analysis and Popularity Prediction")
print("Machine Learning Type : Regression")
print("Target Variable       : imdb_score")
print("="*70)

# -------------------------------
# 2. COLLECT DATA
# -------------------------------
# Change the filename if required
df = pd.read_csv("IMDB_Movies_Dataset.csv")

print("\nDataset Shape:", df.shape)

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe(include="all"))

# -------------------------------
# 3. DATA PREPROCESSING
# -------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicate rows
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)
df.drop_duplicates(inplace=True)

# Convert numeric columns if necessary
if "num_user_for_reviews" in df.columns:
    df["num_user_for_reviews"] = pd.to_numeric(
        df["num_user_for_reviews"],
        errors="coerce"
    )

# Fill missing values
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode selected categorical variables
encoder = LabelEncoder()

for col in ["language", "country", "content_rating"]:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])

# Scale selected numerical features
scale_columns = [
    "duration",
    "budget",
    "gross",
    "num_voted_users",
    "movie_facebook_likes"
]

available_columns = [c for c in scale_columns if c in df.columns]

scaler = StandardScaler()
df[available_columns] = scaler.fit_transform(df[available_columns])

# Remove outliers using IQR
Q1 = df[available_columns].quantile(0.25)
Q3 = df[available_columns].quantile(0.75)
IQR = Q3 - Q1

df = df[
    ~(
        (
            df[available_columns] < (Q1 - 1.5 * IQR)
        ) |
        (
            df[available_columns] > (Q3 + 1.5 * IQR)
        )
    ).any(axis=1)
]

print("\nDataset Shape After Cleaning:", df.shape)

# -------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# -------------------------------

sns.set(style="whitegrid")

# IMDb Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["imdb_score"], bins=25, kde=True)
plt.title("Distribution of IMDb Scores")
plt.tight_layout()
plt.show()

# Top Genres
genres = (
    df["genres"]
    .str.split("|")
    .explode()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,5))
sns.barplot(x=genres.values, y=genres.index)
plt.title("Top 10 Movie Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14,10))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Budget vs Gross
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="budget", y="gross")
plt.title("Budget vs Gross Earnings")
plt.tight_layout()
plt.show()

# IMDb Score by Content Rating
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="content_rating", y="imdb_score")
plt.title("IMDb Score by Content Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Movies Released Per Year
plt.figure(figsize=(14,6))
order = sorted(df["title_year"].dropna().unique())
sns.countplot(data=df, x="title_year", order=order)
plt.xticks(rotation=90)
plt.title("Movies Released Per Year")
plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully!")

# Save cleaned dataset
df.to_csv("cleaned_imdb_movies.csv", index=False)
print("Cleaned dataset saved as cleaned_imdb_movies.csv")
