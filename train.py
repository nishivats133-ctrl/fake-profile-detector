 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("🔹 Loading dataset...")
df = pd.read_csv("example_dataset.csv")

# Split data
X = df.drop("label", axis=1)
y = df["label"]

# Define features
text_features = ["bio", "post_text"]
numeric_features = ["followers", "following", "account_age_days", "total_posts"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("bio", TfidfVectorizer(analyzer='word', stop_words='english'), "bio"),
        ("post", TfidfVectorizer(analyzer='word', stop_words='english'), "post_text"),
        ("num", StandardScaler(), numeric_features)
    ],
    remainder='drop'
)

# Model pipeline
model = Pipeline([
    ("features", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_pipeline.pkl")
print("✅ Model saved as models/rf_pipeline.pkl")
