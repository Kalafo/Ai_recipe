from tensorflow.keras.models import load_model
import joblib
import pandas as pd
from nltk.stem import WordNetLemmatizer
import re

# Load model, label encoder, and precomputed recipe embeddings
model = load_model("recipe_model.keras")
label_encoder = joblib.load("label_encoder.pkl")
recipes_df = pd.read_pickle("recipe_embeddings.pkl")

# Ingredient normalization function
lemmatizer = WordNetLemmatizer()
def normalize_ingredient(word):
    word = word.lower()
    word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
    return lemmatizer.lemmatize(word)

# Precompute normalized ingredients ONCE
def precompute_normalized_ingredients(df):
    return df.assign(
        normalized_ingredients=df['ner_labeled'].apply(
            lambda s: set(normalize_ingredient(w) for w in s.split())
        )
    )

recipes_df = precompute_normalized_ingredients(recipes_df)

def recommend_by_ingredient_overlap(user_ingredients, recipes_df, top_n=6):
    user_set = set(normalize_ingredient(i.strip()) for i in user_ingredients)
    # Vectorized overlap count
    recipes_df = recipes_df.copy()
    recipes_df['overlap_count'] = recipes_df['normalized_ingredients'].apply(lambda s: len(s & user_set))
    recipes_df['matched'] = recipes_df['normalized_ingredients'].apply(lambda s: s & user_set)
    recipes_df['not_matched'] = recipes_df['normalized_ingredients'].apply(lambda s: s - user_set)
    top = recipes_df.sort_values('overlap_count', ascending=False).head(top_n)
    return top

# User input
user_ingredients = ["milk", "egg"]  # Example

top_matches = recommend_by_ingredient_overlap(user_ingredients, recipes_df, top_n=6)
print("Top 3 recipes by ingredient overlap:")
for _, row in top_matches.iterrows():
    print(f"{row['title']} ({row['dish_type']}) - {row['overlap_count']} matches: {', '.join(row['matched'])}")
    print(f"notmatched but in recipe: {', '.join(row['not_matched'])}\n")