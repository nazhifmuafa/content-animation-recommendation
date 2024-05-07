import pickle
import string
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies = pd.read_csv("filtered_movies2.csv")

# Set index to movie titles
movies.set_index('title', inplace=True)
indices = pd.Series(movies.index)

# Function to load pickled data
def load_pickled_data(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Load necessary data from pickle file
with open('cosine_similarity_matrix.pkl', 'rb') as f:
    cos_sim = pickle.load(f)

# Load necessary data from pickle file
with open('recommendation_keyword_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
tfidf_vectorizer = data['tfidf_vectorizer']
svd = data['svd']
svd_matrix = data['svd_matrix']

# Function to get recommendations based on movie title
def get_recommendations_by_title(movie_title, num_recommendations, cosine=cos_sim):
    recommended_films = []

    movie_title = movie_title.lower()

    # Check if movie title exists
    if movie_title not in indices.str.lower().values:
        return recommended_films  # Return empty list if not found

    idx = indices[indices.str.lower() == movie_title].index[0]

    score_series = pd.Series(cosine[idx]).sort_values(ascending=False)  # Sort scores

    top_indexes = list(score_series.iloc[1:num_recommendations+1].index)  # Top indexes

    for i in top_indexes:
        recommended_film = {
            'title': movies.index[i],
            'poster_path': movies['poster_path'].iloc[i],  # Assuming poster_path exists
            'Similarity Score': score_series[i]
        }
        recommended_films.append(recommended_film)

    return recommended_films

# Function to get recommendations based on user input keyword
def get_recommendations_by_keyword(user_input, num_recommendations):

    user_input = user_input.lower()
    user_input = re.sub(r"\d+", "", user_input)
    user_input = user_input.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|~-=_+-"})
    user_input = user_input.translate(str.maketrans("", "", string.punctuation))
    user_input = user_input.strip()
    user_input = re.sub('\s+', ' ', user_input)
    user_input = re.sub(r"\b[a-zA-Z]\b", "", user_input)
    user_input = word_tokenize(user_input)
    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    user_input = [lemmatizer.lemmatize(word) for word in user_input]

    user_input = ' '.join(map(str, user_input))

    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    user_input_svd = svd.transform(user_input_tfidf)
    user_cosine_sim = cosine_similarity(user_input_svd, svd_matrix)

    user_sim_scores = list(enumerate(user_cosine_sim[0]))
    user_sim_scores = sorted(user_sim_scores, key=lambda x: x[1], reverse=True)
    user_sim_scores = user_sim_scores[:num_recommendations]

    recommended_movie_indices = [i[0] for i in user_sim_scores]

    recommended_movies = []
    for idx in recommended_movie_indices:
        recommended_movie = {
            'title': movies.index[idx],  # Consistent title access
            'poster_path': movies.iloc[idx]['poster_path'],  # Consistent poster path access
            'Similarity Score': user_sim_scores[recommended_movie_indices.index(idx)][1]
        }
        recommended_movies.append(recommended_movie)

    return recommended_movies