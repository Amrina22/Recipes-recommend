import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from bahan_parser import bahan_parser

def get_recommendations(N, scores):
    # load in recipe dataset
    df_recipes = pd.read_csv('data/train_clean.csv')
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["Judul", "Bahan", "score"])
    count = 0
    for i in top:
        recommendation.at[count, "Judul"] = df_recipes["Judul"][i]
        recommendation.at[count, "Bahan"] = df_recipes["Bahan"][i]
        recommendation.at[count, "Step"] = df_recipes["Step"][i]
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    return recommendation

def RecSys(Bahan, N=5):
    with open("models/tfidf_encoding.pkl", "rb") as f:
        tfidf_encodings = pickle.load(f)

    with open("models/tfidf_model.pkl", "rb") as f:
        tfidf = pickle.load(f)
    # parse the ingredients using my ingredient_parser
    try:
        bahan_parsed = bahan_parser(str(Bahan))
    except:
        bahan_parsed = bahan_parser([Bahan])

    # use our pretrained tfidf model to encode our input ingredients
    bahan_parsed = " ".join(bahan_parsed)
    bahan_tfidf = tfidf.transform([bahan_parsed])

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(bahan_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations

if __name__ == "__main__":
    # test ingredients
    test_ingredients = "bawang merah,bawang putih, lada"
    recs = RecSys(test_ingredients)
    print(recs.score)