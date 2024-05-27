import pandas as pd
import joblib
import os
import json

def clean_text(text):
    # Función para limpiar el texto
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_movie_genre(title, year):
    # Carga los modelos y transformadores
    model = joblib.load(os.path.join(os.path.dirname(__file__), 'model.pkl'))
    vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), 'vectorizer.pkl'))
    mlb = joblib.load(os.path.join(os.path.dirname(__file__), 'mlb.pkl'))
    
    # Búsqueda de la película por título y año
    movie_data = dataTesting[(dataTesting['title'] == title) & (dataTesting['year'] == year)]
    
    if movie_data.empty:
        return "No se encontró la película en el conjunto de datos."
    
    # Extracción de la trama de la película
    plot = movie_data.iloc[0]['plot']
    
    # Limpieza y vectorización del texto
    plot_clean = clean_text(plot)
    plot_vectorized = vectorizer.transform([plot_clean])
    
    # Predicción de géneros
    genre_predictions = model.predict(plot_vectorized)
    genres_formatted = mlb.inverse_transform(genre_predictions)
    
    return genres_formatted

if __name__ == "__main__":
    # Carga del conjunto de datos de prueba
    dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip')
    
    # Opción 1: Entrada de usuario en formato JSON
    user_input = input("Ingrese el título y el año de la película en formato JSON: ")
    user_input_json = json.loads(user_input)
    title = user_input_json.get('title')
    year = user_input_json.get('year')

