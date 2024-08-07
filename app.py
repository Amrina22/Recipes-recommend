from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from bahan_parser import bahan_parser

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
        sorted_doc = sorted(doc)
        corpus_sorted.append(sorted_doc)
    return corpus_sorted

# Load the Word2Vec model within the Flask route
model = Word2Vec.load("models/recom.model")
if model:
    print("Successfully loaded model")

def get_average_word_vector(words, model):
    vector_size = model.vector_size
    feature_vector = np.zeros((vector_size,), dtype="float32")
    n_words = 0
    for word in words:
        if word in model.wv:
            n_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if n_words > 0:
        # Calculate the average by dividing by the number of words
        feature_vector = np.divide(feature_vector, n_words)

    return feature_vector

# Read the data with explicit encoding
data = pd.read_csv("data/resep_dataset2.csv", encoding="utf-8")
data["parsed"] = data["Bahan"].apply(bahan_parser)
corpus = get_and_sort_corpus(data)

# Calculate the average word vector for each recipe in the corpus
recipe_vectors = [get_average_word_vector(recipe, model) for recipe in corpus]

# Define the Flask route
@app.route('/recommend', methods=['POST'])
def recommend_recipe():
    try:
        # Get input from the form
        input_str = request.form['Bahan']
        input_par = input_str.strip().split(', ')

        # Get the average word vector for the input
        input_vector = get_average_word_vector(input_par, model)

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity([input_vector], recipe_vectors)

        # Get the indices of the top 5 similarities
        top_indices = cosine_similarities.argsort()[0][::-1][:5]

        # Retrieve recipe information for the top recommendations
        recipes = []
        for idx in top_indices:
            recipe_title = data.loc[idx, 'Judul']
            recipe = corpus[idx]
            similarity = cosine_similarities[0, idx]

            # Find matching ingredients
            matching_ingredients = [ing for ing in input_par if ing in recipe]

            # Calculate matching ratio
            matching_ratio = len(matching_ingredients) / len(input_par) if len(input_par) > 0 else 0

            recipe_info = {
                'judul': recipe_title,
                'bahan': recipe,
                'url': data.loc[idx, 'Url'],
                'similarity': similarity,
                'matching_ratio': matching_ratio,
                'matching_ingredients': matching_ingredients
            }
            recipes.append(recipe_info)

        return render_template('rekomendasi.html', recipes=recipes)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
