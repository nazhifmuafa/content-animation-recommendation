from flask import Flask, render_template, request
import pandas as pd
import numpy as np
# from sqlalchemy import create_engine
from animation_data import get_animation_data
from recommendation2 import get_recommendations_by_title, get_recommendations_by_keyword
# import joblib

app = Flask(__name__)

# # Buat koneksi ke MySQL menggunakan SQLAlchemy
# engine = create_engine('mysql+mysqlconnector://root:@localhost/animation_data')

# Load movie data (assuming data is already loaded)
movies = pd.read_csv("filtered_movies2.csv")

# Load pre-computed data from pickle files
# cosine_sim = load_pickled_data('cosine_similarity_matrix.pkl')
movies_list = movies['title'].tolist()

@app.route('/')
def home():
    return render_template('index.html')

# Jumlah data yang ingin ditampilkan per halaman
PER_PAGE = 24

@app.route('/data-animasi')
def data_animasi():
    # Ambil nomor halaman dari query parameter
    page = request.args.get('page', default=1, type=int)
    
    # Ambil data film animasi dari fungsi get_animation_data()
    all_films = get_animation_data()

    # Hitung total halaman berdasarkan jumlah data per halaman
    total_pages = (len(all_films) - 1) // PER_PAGE + 1

    # Atur nilai min_page dan max_page
    min_page = page - 3 if page > 3 else 1
    max_page = page + 2 if page < total_pages - 2 else total_pages

    # Render template HTML dan lewatkan data film dan informasi halaman ke dalamnya
    return render_template("data-animasi.html", films=all_films[(page-1)*PER_PAGE:page*PER_PAGE],
                           total_pages=total_pages, current_page=page,
                           min_page=min_page, max_page=max_page)

@app.route('/rekomendasi-judul')
def rekomendasi_judul():
    # Get list of movie titles
    movie_titles = [movies.loc[index]['title'] for index in movies.index]
    return render_template("rekomendasi-judul.html", movie_titles=movie_titles)

@app.route('/recommend-by-title', methods=['POST'])
def recommend_by_title():
    movie_title = request.form['movie_title']
    num_recommendations = int(request.form['num_recommendations'])

    if movie_title in movies_list:
        similar_movies = get_recommendations_by_title(movie_title, num_recommendations)
        return render_template("rekomendasi.html", movies=similar_movies, recommendation_type="Title")
    else:
        return render_template("home.html", error_message="Movie not found!")

@app.route('/rekomendasi-keywords')
def rekomendasi_keywords():
    return render_template("rekomendasi-keywords.html")

@app.route('/recommend-by-keyword', methods=['POST'])
def recommend_by_keyword():
    user_input = request.form['user_input']
    num_recommendations = int(request.form['num_recommendations'])  # Get number of recommendations

    similar_movies = get_recommendations_by_keyword(user_input, num_recommendations)  # Call function to get recommendations

    if similar_movies:  # Check if recommendations were found
        return render_template("rekomendasi.html", movies=similar_movies, recommendation_type="Keyword")
    else:
        return render_template("home.html", error_message="No recommendations found!")  # Handle no recommendations case
    
if __name__ == '__main__':
    app.run(debug=True)
