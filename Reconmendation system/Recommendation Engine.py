"""
Created on Sat Apr  2 20:11:06 2022
"""

import pandas as pd
import numpy as np

movies_df = pd.read_csv('Movie.csv')
movies_df.shape
movies_df.head()

movies_df.sort_values('userId')

#number of unique users in the dataset
len(movies_df)
len(movies_df.userId.unique())

movies_df['rating'].value_counts()
movies_df['rating'].hist()


len(movies_df.movie.unique())

movies_df.movie.value_counts()

user_movies_df = movies_df.pivot(index='userId',
                                 columns='movie',
                                 values='rating')

user_movies_df
user_movies_df.iloc[0]
user_movies_df.iloc[200]
list(user_movies_df)

#Impute those NaNs with 0 values
user_movies_df.fillna(0, inplace=True)

user_movies_df.shape

# from scipy.spatial.distance import cosine correlation
# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances( user_movies_df.values,metric='cosine')

#user_sim = 1 - pairwise_distances( user_movies_df.values,metric='correlation')

user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index   = movies_df.userId.unique()
user_sim_df.columns = movies_df.userId.unique()

user_sim_df.iloc[0:5, 0:5]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#Most Similar Users
user_sim_df.max()

user_sim_df.idxmax(axis=1)[0:5]

movies_df[(movies_df['userId']==6) | (movies_df['userId']==168)]

user_6=movies_df[movies_df['userId']==6]

user_168=movies_df[movies_df['userId']==168]


user_3=movies_df[movies_df['userId']==3]
user_11=movies_df[movies_df['userId']==11]

user_3.movie
user_11.movie

pd.merge(user_3,user_11,on='movie',how='inner')
pd.merge(user_3,user_11,on='movie',how='outer')


#-------------------------------------------------------------------




