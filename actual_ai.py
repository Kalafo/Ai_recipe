from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

# Load model, label encoder, and precomputed recipe embeddings
model = load_model("recipe_model.h5")
label_encoder = joblib.load("label_encoder.pkl")
recipes_df = pd.read_pickle("recipe_embeddings.pkl")

# Load GloVe
def load_glove_embeddings(glove_path):
    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    return embeddings_index

glove_path = "glove.6B.300d.txt"
embeddings_index = load_glove_embeddings(glove_path)

# Preprocessing config
custom_stopwords = set(ENGLISH_STOP_WORDS).union({'all-purpose', 'sugar', 'salt', 'flour', 'water', 'milk', 'eggs', "powder", "baking", "soda"})
lemmatizer = WordNetLemmatizer()

def preprocess_ingredients(ingredients):
    words = [w for w in ingredients.lower().split() if w not in custom_stopwords]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def get_embedding(text, embeddings_index):
    words = text.split()
    embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(next(iter(embeddings_index.values())).shape)

# User input
new_ingredients = ["beef", "onion", "garlic", "tomato", "eggs", "flour", "chicken"]
new_input_proc = preprocess_ingredients(' '.join(new_ingredients))
new_vec = get_embedding(new_input_proc, embeddings_index).reshape(1, -1)

# Predict dish type
predictions = model.predict(new_vec)
top_indices = np.argsort(predictions[0])[::-1][:3]
top_types = label_encoder.inverse_transform(top_indices)

# Load the original test dataset for NER info
test_df = pd.read_csv("test_dataset.csv", usecols=['title', 'NER'])

print("Rekomendowane typy potraw:")
for i, dish_type in enumerate(top_types):
    print(f"{i + 1}. {dish_type}")
    recipes = recipes_df[recipes_df['dish_type'] == dish_type]
    if not recipes.empty:
        recipe_embs = np.vstack(recipes['embedding'].values)
        sim = np.dot(recipe_embs, new_vec[0])
        best_idx = np.argmax(sim)
        best_recipe = recipes.iloc[best_idx]
        print(f"   Najlepszy przepis: {best_recipe['title']}")
        # Find the original NER ingredients for this recipe title
        ner_row = test_df[test_df['title'] == best_recipe['title']]
        if not ner_row.empty:
            print(f"   Składniki: {ner_row.iloc[0]['NER']}")
        else:
            print("   Składniki: (brak danych)")
        print()
    else:
        print("   Brak przepisu dla tego typu potrawy.\n")