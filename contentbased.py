from tools import *
import numpy as np
import pandas as pd
 
df1=pd.read_csv('C:\\Users\\anil\\Documents\\ml notes\\project\\tmdb-5000-movie-dataset\\tmdb_5000_credits.csv')
df2=pd.read_csv('C:\\Users\\anil\\Documents\\ml notes\\project\\tmdb-5000-movie-dataset\\tmdb_5000_movies.csv')
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

df2['soup'] = df2.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

print(get_recommendations('The Avengers', cosine_sim2,indices,df2))


