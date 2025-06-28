import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# from readme we check the structure of u.data, u.item nd u.genre and we will combine them together 
r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols) # now ratings is a dataframe with the names of columns as r_cols
i_cols = ['movie_id','title','release_date','video_release_date','IMDb_URL'] \
       + [f'genre_{i}' for i in range(19)]
movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1') #this gets the data of title and genre as asked and make a dataframe of the same
data = pd.merge(ratings, movies[['movie_id','title']], on='movie_id') #this will merge the two datasets on basis of movie_id

# we now make a matrix as asked so that it can be used for recommendations ahead
R     = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0) 
means = R.mean(axis=1)
R_dm  = R.sub(means, axis=0) #normalization

# using the mentioned method of train_test_split from sklearn 
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
#process repeat for training dataset
R_tr      = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
means_tr  = R_tr.mean(axis=1)
R_tr_dm   = R_tr.sub(means_tr, axis=0)

# applying the concept of knn on our previously made train matrix
K = 5
model_tr = NearestNeighbors(metric='cosine', algorithm='brute')
model_tr.fit(R_tr_dm.values)

# precompute neighbors nd similarities
dist_tr, ind_tr = model_tr.kneighbors(R_tr_dm.values, n_neighbors=K+1)
nbrs_mat  = ind_tr[:,1:]
sims_mat  = 1 - dist_tr[:,1:] #convert cos dist into similarity

def predict_rating(u_idx, m_idx):
    #finds 5 neighbors and their interest of movive 
    sims    = sims_mat[u_idx] 
    nbrs    = nbrs_mat[u_idx]
    user_m  = means_tr.values[u_idx] 
    ratings = R_tr_dm.values[nbrs, m_idx]
    num     = sims.dot(ratings) # sum of similarity*rating
    den     = np.abs(sims).sum() #sum of abosulte of similarities
    if den == 0:
        return float(user_m)
    return float(np.clip(user_m + num/den, 1, 5))

# calc of  RMSE as required by question:
y_true, y_pred = [], [] # ground truth and prediction
for _, row in test_df.iterrows():
    u,m,actual = int(row.user_id), int(row.movie_id), row.rating
    if u not in R_tr.index or m not in R_tr.columns:
        continue
    ui = R_tr.index .get_loc(u)
    mi = R_tr.columns.get_loc(m)
    y_true.append(actual)
    y_pred.append(predict_rating(ui, mi))

rmse = sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE: {rmse:.2f}")

model_full = NearestNeighbors(metric='cosine', algorithm='brute')
model_full.fit(R_dm.values)
dist_f, ind_f = model_full.kneighbors(R_dm.values, n_neighbors=K+1)
nbrs_full = ind_f[:,1:]
sims_full = 1 - dist_f[:,1:]

def recommend(user_id, n=5):
    if user_id not in R.index:
        raise ValueError("Unknown user")
    ui     = R.index.get_loc(user_id)
    sims   = sims_full[ui]
    nbrs   = nbrs_full[ui]
    mean_u = means.values[ui]
    mat    = R_dm.values[nbrs]  # shape (K, n_items), is a full mean centered rating matrix

    # weighted sum across neighbors → predictions vector, weightedsum + user mean, and  it is clipped to 1-5
    num   = (sims[:, None] * mat).sum(axis=0)
    den   = np.abs(sims).sum()
    preds = mean_u + num/den if den > 0 else mean_u
    preds = np.clip(preds, 1, 5)

    # mask the already rated movies
    already = R.values[ui] > 0
    preds[already] = -np.inf

    # top n index
    top_ids = np.argpartition(-preds, n)[:n] 
    top_ids = top_ids[np.argsort(-preds[top_ids])]

    return [
        (movies.loc[movies.movie_id == R.columns[idx], 'title'].iloc[0], preds[idx])
        for idx in top_ids
    ]

# example:
print("Top 5 recommendations for user 10:")
for title, score in recommend(10, 5):
    print(f"{title} (predicted rating: {score:.2f})")

# import pandas as pd
# import numpy as np
# from math import sqrt
# from sklearn.neighbors import NearestNeighbors
# 
# # from readme we check the structure of u.data, u.item nd u.genre and we will combine them together 
# r_cols = ['user_id','movie_id','rating','timestamp']
# ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols) # now ratings is a dataframe with the names of columns as r_cols
# i_cols = ['movie_id','title','release_date','video_release_date','IMDb_URL'] \
#        + [f'genre_{i}' for i in range(19)]
# movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1') #this gets the data of title and genre as asked and make a dataframe of the same
# data = pd.merge(ratings, movies[['movie_id','title']], on='movie_id') #this will merge the two datasets on basis of movie_id
# 
# # we now make a matrix as asked so that it can be used for recommendations ahead
# R     = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0) 
# means = R.mean(axis=1)
# R_dm  = R.sub(means, axis=0) #normalization
# 
# # full-model knn on the entire dataset
# K = 5
# model_full = NearestNeighbors(metric='cosine', algorithm='brute')
# model_full.fit(R_dm.values)
# 
# # precompute neighbors and similarities for full data
# dist_f, ind_f = model_full.kneighbors(R_dm.values, n_neighbors=K+1)
# nbrs_full = ind_f[:,1:]
# sims_full = 1 - dist_f[:,1:]  # convert cosine distances into similarities
# 
# def recommend(user_id, n=5):
#     if user_id not in R.index:
#         raise ValueError("Unknown user")
#     ui     = R.index.get_loc(user_id)
#     sims   = sims_full[ui]
#     nbrs   = nbrs_full[ui]
#     mean_u = means.values[ui]
#     mat    = R_dm.values[nbrs]  # shape (K, n_items), is a full mean centered rating matrix
# 
#     # weighted sum across neighbors → predictions vector, weightedsum + user mean, and  it is clipped to 1-5
#     num   = (sims[:, None] * mat).sum(axis=0)
#     den   = np.abs(sims).sum()
#     preds = mean_u + num/den if den > 0 else mean_u
#     preds = np.clip(preds, 1, 5)
# 
#     # mask the already rated movies
#     already = R.values[ui] > 0
#     preds[already] = -np.inf
# 
#     # top n index
#     top_ids = np.argpartition(-preds, n)[:n] 
#     top_ids = top_ids[np.argsort(-preds[top_ids])]
# 
#     return [
#         (movies.loc[movies.movie_id == R.columns[idx], 'title'].iloc[0], preds[idx])
#         for idx in top_ids
#     ]
# 
# # example:
# print("Top 5 recommendations for user 10:")
# for title, score in recommend(10, 5):
#     print(f"{title} (predicted rating: {score:.2f})")
# 
