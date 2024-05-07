import pandas as pd

def get_animation_data():
    # Read animation data from the CSV file
    animation_data = pd.read_csv('filtered_movies2.csv')

    # Convert the DataFrame into a list of dictionaries
    films = animation_data.to_dict(orient='records')

    return films
