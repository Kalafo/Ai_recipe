from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
import re
import pickle

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Step 1: Load the CSV file with only the required columns
df = pd.read_csv("test_dataset.csv", usecols=['title', 'NER'])

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Extract the NER column and preprocess
df['ner_labeled'] = df['NER'].apply(lambda x: ' '.join(ast.literal_eval(x)))  # Convert NER list to a single string

# Split ingredients like "chicken/turkey" into separate words
df['ner_labeled'] = df['ner_labeled'].str.replace(r'/', ' ', regex=True)

# Remove unwanted characters from ingredients
df['ner_labeled'] = df['ner_labeled'].str.replace(r'[()\[\]\"*,\+\-]', '', regex=True)

# Remove recipes with only 1 or 2 ingredients
df = df[df['ner_labeled'].apply(lambda x: len(x.split()) > 2)].reset_index(drop=True)

# Custom stopword list
custom_stopwords = set(ENGLISH_STOP_WORDS).union({'l','all-purpose', 'salt', 'water', "powder", "baking", "soda","brown","beat","dark","yellow","white","cold","oleo","green","red","juice","fresh","hot","large","small","medium","whole","ground","dried","canned","frozen","raw","cooked"})

# Remove custom stopwords
df['ner_labeled'] = df['ner_labeled'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in custom_stopwords])
)

# Remove words with less than 3 characters
df['ner_labeled'] = df['ner_labeled'].apply(
    lambda x: ' '.join([word for word in x.split() if len(word) >= 3])
)

# Apply lemmatization
lemmatizer = WordNetLemmatizer()
df['ner_labeled'] = df['ner_labeled'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
)

print("Unique entries in ner_labeled:", df['ner_labeled'].nunique())

# Flatten the list of words and count frequencies
word_counts = Counter(" ".join(df['ner_labeled']).split())
print(word_counts.most_common(20))  # Print the 20 most common terms

# Load GloVe embeddings into a dictionary
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

# Example: beef is more important than pepper
ingredient_weights = {
    'beef': 13.0,
    'chicken': 14.0,
    'fish': 7.0,
    'pasta': 2.0,
    'cake': 2.0,
    'pepper': 1.0,
    'onion': 0.7,
    'garlic': 1.5,
    'tomato': 1.5,
    'sugar': 0.8,
    'flour': 1.0,
    'water': 0.5,
    'milk': 1.0,
    'eggs': 1.0,
    'yeast': 1.0,
    'butter': 1.1,
    'oil': 1.1,
    'salt': 0.5,
}
default_weight = 0.9  # Weight for ingredients not in the dictionary

# Function to compute the average GloVe embedding for a text
def get_weighted_embedding(text):
    words = text.split()
    embeddings = []
    weights = []
    for word in words:
        if word in embeddings_index:
            weight = ingredient_weights.get(word, default_weight)
            embeddings.append(embeddings_index[word] * weight)
            weights.append(weight)
    if embeddings:
        # Weighted average
        return np.sum(embeddings, axis=0) / np.sum(weights)
    else:
        return np.zeros(next(iter(embeddings_index.values())).shape)

# Apply GloVe embeddings to the 'ner_labeled' column
df['embedding'] = df['ner_labeled'].apply(get_weighted_embedding)
X = np.vstack(df['embedding'].values)

# Define keywords for dish types (expand as needed)
dish_keywords = [
    'cake','pie','bread','cookie','casserole','roll','soup','salad','pasta','chicken','beef','fish',
    'pudding','muffin','brownie','bar','sandwich','biscuits','pancake','waffle','pizza','tart',
    'crisp','cobbler','loaf','dough','mayonnaise'
]

def categorize_by_title_or_ingredient(row):
    # Try to extract from title first
    for kw in dish_keywords:
        if re.search(rf'\b{kw}\b', row['title'], re.IGNORECASE):
            return kw
    # If not found in title, check ingredients
    for kw in dish_keywords:
        if re.search(rf'\b{kw}\b', row['ner_labeled'], re.IGNORECASE):
            return kw
    return 'other'

df['dish_type'] = df.apply(categorize_by_title_or_ingredient, axis=1)

# Print category counts
print("Dish type counts:")
print(df['dish_type'].value_counts())
print("\n")

# Encode the dish types
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['dish_type'])

# One-hot encode the labels
y_cat = to_categorical(y)

# Combine the processed features and labels into a DataFrame
processed_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
processed_data['label'] = y  # Add the labels as a new column

# Save the processed data to a new CSV file
processed_data.to_csv("processed_data.csv", index=False)

# Save the intermediate processed data for inspection
df[['title', 'ner_labeled']].to_csv("processed_title_ner.csv", index=False)

# Print the number of unique titles after filtering
print("Number of unique titles after filtering:", df['title'].nunique())

# Save all recipe info and embeddings for fast inference
df[['title', 'dish_type', 'ner_labeled', 'embedding']].to_pickle("recipe_embeddings.pkl")

# After you have your vocabulary (e.g., all words in df['ner_labeled'])
used_words = set(" ".join(df['ner_labeled']).split())
small_embeddings = {w: v for w, v in embeddings_index.items() if w in used_words}

with open("small_glove.pkl", "wb") as f:
    pickle.dump(small_embeddings, f)

# Save the list of unique ingredients used in the model to a CSV
unique_ingredients = sorted(used_words)
pd.DataFrame({'ingredient': unique_ingredients}).to_csv("model_ingredients.csv", index=False)

# Step 3: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Step 4: Build the model
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(y_cat.shape[1], activation='softmax')
])

# Compile the model
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,  # You can increase epochs with early stopping
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stop]
)

# Save the model and preprocessing objects
model.save("recipe_model.keras", save_format="keras")
joblib.dump(label_encoder, "label_encoder.pkl")








