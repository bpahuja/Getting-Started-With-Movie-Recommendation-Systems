import numpy as np
import pandas as pd



 
df1=pd.read_csv('C:\\Users\\anil\\Documents\\ml notes\\project\\tmdb-5000-movie-dataset\\tmdb_5000_credits.csv')
df2=pd.read_csv('C:\\Users\\anil\\Documents\\ml notes\\project\\tmdb-5000-movie-dataset\\tmdb_5000_movies.csv')
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')


C= df2['vote_average'].mean()
m= df2['vote_count'].quantile(0.9)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))