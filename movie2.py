import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# from readme we check the structure of u.data, u.item nd u.genre and we will combine them together 
r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)  # now ratings is a dataframe with the names of columns as r_cols
i_cols = ['movie_id','title','release_date','video_release_date','IMDb_URL'] \
       + [f'genre_{i}' for i in range(19)]
movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')  # this gets the data of title and genre as asked and make a dataframe of the same
data = pd.merge(ratings, movies[['movie_id','title']], on='movie_id')  # this will merge the two datasets on basis of movie_id

# we now make a matrix as asked so that it can be used for recommendations ahead
R     = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0) 
means = R.mean(axis=1)
R_dm  = R.sub(means, axis=0)  # normalization

# using the mentioned method of train_test_split from sklearn 
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# process repeat for training dataset
R_tr      = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
means_tr  = R_tr.mean(axis=1)
R_tr_dm   = R_tr.sub(means_tr, axis=0)

# ---- HYPERPARAMETERS TO TUNE ----
K_VALUES = [5, 10, 20, 30, 50]
GAMMA    = 1.0   # shrinkage: adds stability when few neighbors have rated the item

def build_knn(K):
    """Fit a KNN model on the mean-centered train matrix."""
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(R_tr_dm.values)
    dist, ind = model.kneighbors(R_tr_dm.values, n_neighbors=K+1)
    # drop self (index 0) → neighbours only
    nbrs = ind[:, 1:]
    sims = 1 - dist[:, 1:]
    return nbrs, sims

def predict_rating(u_idx, m_idx, nbrs_mat, sims_mat, gamma=GAMMA):
    """Predict a single rating with shrinkage and filtering."""
    sims = sims_mat[u_idx]            # shape (K,)
    nbrs = nbrs_mat[u_idx]            # shape (K,)
    user_mean = means_tr.values[u_idx]
    # pull neighbor ratings for this movie (mean-centered)
    neigh_ratings = R_tr_dm.values[nbrs, m_idx]
    # filter out neighbours who never rated it
    mask = neigh_ratings != 0
    if not mask.any():
        return user_mean
    sims_f = sims[mask]
    ratings_f = neigh_ratings[mask]
    # weighted sum with shrinkage
    num = sims_f.dot(ratings_f)
    den = np.abs(sims_f).sum() + gamma
    return float(np.clip(user_mean + num/den, 1, 5))

def evaluate_rmse(K, gamma=GAMMA):
    nbrs_mat, sims_mat = build_knn(K)
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, m, actual = int(row.user_id), int(row.movie_id), row.rating
        if u not in R_tr.index or m not in R_tr.columns:
            continue
        ui = R_tr.index.get_loc(u)
        mi = R_tr.columns.get_loc(m)
        y_true.append(actual)
        y_pred.append(predict_rating(ui, mi, nbrs_mat, sims_mat, gamma))
    return sqrt(mean_squared_error(y_true, y_pred))

# grid-search best K
best = (None, 1e9)
for K in K_VALUES:
    rmse = evaluate_rmse(K)
    print(f"K={K:2d} → RMSE={rmse:.4f}")
    if rmse < best[1]:
        best = (K, rmse)
print(f"\n→ Best K = {best[0]}, RMSE = {best[1]:.4f}")

# ---- Final model on full data ----
# rebuild with best K
K_opt = best[0]
nbrs_full, sims_full = build_knn(K_opt)  # but use R_dm/means for full data if desired

def recommend(user_id, n=5):
    if user_id not in R.index:
        raise ValueError("Unknown user")
    ui = R.index.get_loc(user_id)
    sims = sims_full[ui]
    nbrs = nbrs_full[ui]
    mean_u = means.values[ui]
    mat    = R_dm.values[nbrs]       # (K, n_movies)
    # weighted predictions
    num = (sims[:, None] * mat).sum(axis=0)
    den = np.abs(sims).sum() + GAMMA
    preds = mean_u + num/den if den > 0 else mean_u
    preds = np.clip(preds, 1, 5)
    # mask already-seen
    preds[R.values[ui] > 0] = -np.inf
    top = np.argpartition(-preds, n)[:n]
    top = top[np.argsort(-preds[top])]
    return [
        (movies.loc[movies.movie_id == R.columns[idx], 'title'].iloc[0], preds[idx])
        for idx in top
    ]

# example:
print(f"\nTop {5} recommendations for user 10 (K={K_opt}):")
for title, score in recommend(10, 5):
    print(f"{title} (predicted rating: {score:.2f})")

