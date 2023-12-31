from collections import Counter
from datetime import datetime
import gzip
import json
import numpy as np
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



app = FastAPI()

# Ruta a tu archivo JSON comprimido
archivo = 'output_steam_games_limpio_reducido.json.gz'

# Lista para guardar cada fila
lista = []

with gzip.open(archivo, 'r') as file:
    data = file.read().decode('utf-8')
    if data[0] == '[':
        # Los datos están en formato de array
        lista = json.loads(data)
    else:
        # Los datos están separados por nuevas líneas
        for line in data.splitlines():
            lista.append(json.loads(line))

# Crear un DataFrame a partir de la lista
df1 = pd.DataFrame(lista)

# Verificar si 'release_date' existe en df1
if 'release_date' in df1.columns:
    # Asegúrate de que 'release_date' sea un objeto datetime
    df1['release_date'] = pd.to_datetime(df1['release_date'], errors='coerce', format='%Y-%m-%d')

    # Calcular las horas desde el lanzamiento hasta la fecha actual
    now = datetime.now()
    df1['hours_since_release'] = (now - df1['release_date']).dt.total_seconds() / 3600

# Supongamos que 'df' es tu DataFrame
df2 = pd.read_csv('Funcion2.csv')

# Limpia los valores NaN en la columna de géneros
df2['genres'] = df2['genres'].replace(np.nan, '')

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero: str) -> Dict[str, int]:
    # Asegúrate de que todos los valores en 'genres' sean listas
    df1['genres'] = df1['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Filtrar el DataFrame por el género especificado
    df_genre = df1[df1['genres'].apply(lambda x: genero in x)]
    
    # Agrupar por año de lanzamiento y sumar las horas desde el lanzamiento
    df_genre_year = df_genre.groupby(df_genre['release_date'].dt.year)['hours_since_release'].sum()
    
    # Encontrar el año de lanzamiento con más horas desde el lanzamiento
    max_year = df_genre_year.idxmax()
    
    return {f"Año de lanzamiento con más horas jugadas para {genero}" : max_year}

@app.get("/UserForGenre/{genero}")
async def read_user_for_genre(genero: str):
    # Ruta a tu archivo JSON comprimido
    archivo = 'output_steam_games_limpio_reducido.json.gz'

    # Lista para guardar cada fila
    lista = []

    with gzip.open(archivo, 'r') as file:
        data = file.read().decode('utf-8')
        if data[0] == '[':
            # Los datos están en formato de array
            lista = json.loads(data)
        else:
            # Los datos están separados por nuevas líneas
            for line in data.splitlines():
                lista.append(json.loads(line))

    # Crear un DataFrame a partir de la lista
    df1 = pd.DataFrame(lista)

    # Asegúrate de que 'genres' es una lista
    df1['genres'] = df1['genres'].apply(lambda x: x if isinstance(x, list) else [x])

    # Crea una nueva columna con el primer género
    df1['primer_genero'] = df1['genres'].str[0]

    # Elimina las filas con valores nulos en la columna 'primer_genero'
    df1 = df1.dropna(subset=['primer_genero'])

    # Ahora puedes filtrar usando 'primer_genero'
    df_genre = df1[df1['primer_genero'].str.contains(genero)]
   
    # Agrupamos por usuario y sumamos las horas jugadas
    df_grouped = df2.groupby('user_id')['playtime_forever'].sum().reset_index()
    
    # Encontramos el usuario con más horas jugadas
    max_playtime_user = df_grouped[df_grouped['playtime_forever'] == df_grouped['playtime_forever'].max()]['user_id'].values[0]
    
    # Creamos un dataframe con las horas jugadas por año
    df2['year'] = pd.to_datetime(df2['posted_date']).dt.year
    playtime_per_year = df2.groupby('year')['playtime_forever'].sum().reset_index().to_dict('records')
    
    return {"Usuario con más horas jugadas para Género {}".format(genero) : max_playtime_user, "Horas jugadas": playtime_per_year}


# Descomprimir el archivo csv
with gzip.open('Funcion3_final.csv.gz', 'rt') as f:
    df_funcion3 = pd.read_csv(f)


@app.get("/UsersRecommend/{year}")
async def get_recommend(year: int):
    # Crear un contador para los juegos
    game_counter = Counter()

    # Convertir la columna 'posted' a datetime
    df_funcion3['posted_date'] = pd.to_datetime(df_funcion3['posted'].str[7:-1], errors='coerce')

    # Filtrar las filas donde 'posted' es del año dado y 'recommend' es True
    df_filtered = df_funcion3[(df_funcion3['posted_date'].dt.year == year) & df_funcion3['recommend']]

    # Contar las ocurrencias de cada juego
    game_counter = Counter(df_filtered['item_name'])

    # Obtener los 3 juegos más comunes
    most_common_games = game_counter.most_common(3)

    # Devolver los resultados
    return [{"Puesto " + str(i+1) : most_common_games[i]} if len(most_common_games) > i else {"Puesto " + str(i+1) : None} for i in range(3)]

@app.get("/UsersNotRecommend/{year}")
async def get_not_recommend(year: int):
    # Crear un contador para los juegos
    game_counter = Counter()

    # Convertir la columna 'posted' a datetime
    df_funcion3['posted_date'] = pd.to_datetime(df_funcion3['posted'].str[7:-1], errors='coerce')

    # Filtrar las filas donde 'posted' es del año dado y 'recommend' es False
    df_filtered = df_funcion3[(df_funcion3['posted_date'].dt.year == year) & (df_funcion3['recommend'] == False)]

    # Contar las ocurrencias de cada juego
    game_counter = Counter(df_filtered['item_name'])

    # Obtener los 3 juegos más comunes
    most_common_games = game_counter.most_common(3)

    # Devolver los resultados
    return [{"Puesto " + str(i+1) : most_common_games[i]} if len(most_common_games) > i else {"Puesto " + str(i+1) : None} for i in range(3)]


# Cargar el DataFrame desde el archivo CSV
df_userid_recommend = pd.read_csv('Funcion5.csv')

@app.get("/Sentiment_analysis/{year}")
def read_sentiment_analysis(year: int):
    df_year = df_userid_recommend[df_userid_recommend['year'] == year]
    sentiment_counts = df_year['sentiment_analysis'].value_counts().to_dict()
    sentiment_dict = {'Negative': sentiment_counts.get(0, 0), 'Neutral': sentiment_counts.get(1, 0), 'Positive': sentiment_counts.get(2, 0)}
    return sentiment_dict

# Elimina las filas con valores NaN en la columna de juegos
df1 = df1.dropna(subset=['title'])

# Usa solo el 10% de los datos
df1 = df1.sample(frac=0.1, random_state=1)

# Crea una matriz de características utilizando TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df1['title'])

# Entrena un modelo NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)

@app.get("/recomendacion_titulo/{titulo}")
def get_recomendacion_titulo(titulo: str):
    # Transforma el título en un vector
    titulo_vec = vectorizer.transform([titulo])

    # Obtiene los índices de los 5 juegos más cercanos
    _, indices = nbrs.kneighbors(titulo_vec)

    # Devuelve los juegos recomendados
    return {"Juegos recomendados para el título {}: {}".format(titulo, list(df1['title'].iloc[indices[0][1:]]))}

# Carga los datos para el segundo modelo de recomendación
df_usuario = pd.read_csv('Funcion2_final.csv')

# Reducción de datos al 10%
df_usuario = df_usuario.sample(frac=0.1, random_state=42)

# Utilidad
utilidad = df_usuario.pivot_table(index='user_id', columns='title', values='playtime_forever').fillna(0)

# Similitud del coseno
similitud_usuario = cosine_similarity(utilidad)

# Mapeo de ID de usuario a índice de matriz
user_id_to_index = {user_id: index for index, user_id in enumerate(utilidad.index)}

# Función de recomendación de usuario
def recomendacion_usuario(user_id):
    user_index = user_id_to_index.get(user_id, None)
    if user_index is not None:
        indices_similares = np.argsort(similitud_usuario[user_index])[:-7:-1]
        juegos_recomendados = utilidad.iloc[indices_similares].mean().sort_values(ascending=False).index[1:7]
        return juegos_recomendados
    else:
        return None

# Ruta de la API
@app.get("/recomendacion_usuario/{user_id}")
def get_recomendacion_usuario(user_id: str):
    juegos_recomendados = recomendacion_usuario(user_id)
    if juegos_recomendados is not None:
        return {"Juegos recomendados para el usuario {}: {}".format(user_id, list(juegos_recomendados))}
    else:
        return {"error": "El ID de usuario {} no se encuentra en los datos.".format(user_id)}
    
    

# http://localhost:8000/UserForGenre/Action
# http://localhost:8000/PlayTimeGenre/Casual
# uvicorn apiapp:app --reload
# http://127.0.0.1:8000/docs#/


# https://test-pwmj.onrender.com/
# https://test-pwmj.onrender.com/PlayTimeGenre/Casual
# https://test-pwmj.onrender.com/UserForGenre/Action