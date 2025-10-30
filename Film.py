import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'film': ['Inception', 'Interstellar', 'The Dark Knight', 'Tenet', 'The Matrix', 'John Wick', 'Avengers'],
    'tür': ['sci-fi thriller', 'sci-fi drama', 'action thriller', 'sci-fi action', 'sci-fi action', 'action',
            'superhero action']
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df['tür'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)


def film_oner(film_adi):
    if film_adi not in df['film'].values:
        return "Film veri tabanında yok."

    film_index = df[df['film'] == film_adi].index[0]
    benzerlikler = list(enumerate(cosine_sim[film_index]))
    benzerlikler = sorted(benzerlikler, key=lambda x: x[1], reverse=True)

    öneriler = []
    for i, skor in benzerlikler[1:4]:
        öneriler.append(df.iloc[i]['film'])

    return öneriler


film = input("Bir film adı girin: ")
print("Benzer filmler:", film_oner(film))
