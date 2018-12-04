import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate
sns.set_style("darkgrid")


"""Simple Reccomender"""

df1 = pd.read_csv('ratings_small.csv', header = None, names = ['userId','movieId', 'rating'], usecols = [0,1,2])

df1.head()

df1 = df1.drop([0])
df1['rating'] = df1['rating'].astype(float)

print('Dataset 1 shape: {}'.format(df1.shape))

print(df1.iloc[::5000, :])

df = df1

p = df.groupby('rating')['rating'].agg(['count'])
p

movie_count = df['movieId'].nunique()
print(movie_count)

cust_count = df['userId'].nunique()

rating_count = df['userId'].count()

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plt.axis('off')
plt.show()
for i in range(1,11):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i/2, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')

df = df[pd.notnull(df['rating'])]
df['movieId'] = df['movieId'].astype(int)
df['userId'] = df['userId'].astype(int)
print('-Dataset examples-')
print(df.iloc[::5000, :])

f = ['count','mean']

df_movie_summary = df.groupby('movieId')['rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.9),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('userId')['rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

print('Customer minimum times of review: {}'.format(cust_benchmark))

print('Original Shape: {}'.format(df.shape))
df = df[~df['movieId'].isin(drop_movie_list)]
df = df[~df['userId'].isin(drop_cust_list)]
print('After Trim Shape: {}'.format(df.shape))
print('-Data Examples-')
print(df.iloc[::5000000, :])

Cust_Id_u = list(sorted(df['userId'].unique()))
Movie_Id_u = list(sorted(df['movieId'].unique()))
data = df['rating'].tolist()
row = df['userId'].astype('category', categories=Cust_Id_u).cat.codes
col = df['movieId'].astype('category', categories=Movie_Id_u).cat.codes
sparse_matrix = csr_matrix((data, (row, col)), shape=(len(Cust_Id_u), len(Movie_Id_u)))
df_p = pd.DataFrame(sparse_matrix.todense(), index=Cust_Id_u, columns=Movie_Id_u)
df_p = df_p.replace(0, np.NaN)

print(df_p.shape)

df_p.head()

df_title = pd.read_csv('movies.csv')
df_title.columns

df_title = df_title[['title','movieId']]
df_title.set_index('movieId', inplace = True)
df_title.head()

reader = Reader()

data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)

svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
df_1 = df[(df['userId'] == 430) & (df['rating'] == 5 )]
# d = df[df['movieId'] == 1287]
# df_1.set_index('movieId',inplace=True)
# print(df_1.shape)
# print(df_title.head())
# print(d.head())
# print(df_title[df_title['id'] == 1287])
df_1 = df_1.join(df_title,on='movieId',how='inner')
print(df_1.head())
print(df_1.shape)

# j = df[['movieId','rating']]
# total_map = j.join(df_title,on='movieId',how='inner')

user_1 = df_title.copy()
user_1 = user_1.reset_index()
# user_1 = user_1[~user_1['movieId'].isin(drop_movie_list)]
# getting full dataset
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd.fit(trainset)

# print(svd.predict(43, 110).est)
user_1['Estimate_Score'] = user_1['movieId'].apply(lambda x: svd.predict(430, x).est)
user_1 = user_1.drop('movieId', axis = 1)

user_1 = user_1.sort_values('Estimate_Score', ascending=False)
print(user_1[['title','Estimate_Score']][:10])


