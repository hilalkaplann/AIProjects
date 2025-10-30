import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

data = {
    'film': ['Inception', 'Interstellar', 'The Dark Knight', 'Tenet', 'The Matrix', 'John Wick', 'Avengers'],
    'tür': ['sci-fi thriller', 'sci-fi drama', 'action thriller', 'sci-fi action', 'sci-fi action', 'action',
            'superhero action']
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df['tür'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


def film_bul(film_adi):
    film_listesi = df['film'].tolist()
    en_yakin, skor, _ = process.extractOne(film_adi, film_listesi, scorer=fuzz.token_sort_ratio)
    if skor > 60:
        return en_yakin
    else:
        return None


def film_oner(film_adi):
    duzeltilmis = film_bul(film_adi)
    if not duzeltilmis:
        return "Film veri tabanında bulunamadı."

    film_index = df[df['film'] == duzeltilmis].index[0]
    benzerlikler = list(enumerate(cosine_sim[film_index]))
    benzerlikler = sorted(benzerlikler, key=lambda x: x[1], reverse=True)

    öneriler = []
    for i, skor in benzerlikler[1:4]:
        öneriler.append(df.iloc[i]['film'])

    return duzeltilmis, öneriler


film_girdisi = input("Bir film adı girin: ")

sonuc = film_oner(film_girdisi)
if isinstance(sonuc, tuple):
    duzeltilmis, öneriler = sonuc
    print(f"Aradığınız film: {duzeltilmis}")
    print("Benzer filmler:", öneriler)
else:
    print(sonuc)
