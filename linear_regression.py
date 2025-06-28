import pandas as pd
import numpy as np
from math import sqrt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

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

#compute a low‑rank SVD of the mean‑centered TRAIN matrix (k latent factors)
R_sparse = csr_matrix(R_tr_dm.values)
K_FACTORS = 40  # dimensionality of latent space (tune 20‑100)
U, S, Vt = svds(R_sparse, k=K_FACTORS)

# convert to latent representations P (users) and Q (items)
P = U @ np.diag(np.sqrt(S))          # shape (n_users, k)
Q = (np.diag(np.sqrt(S)) @ Vt).T     # shape (n_items, k)

# maps from ids to row/col indices
user_to_idx = {u: i for i, u in enumerate(R_tr.index)}
item_to_idx = {m: i for i, m in enumerate(R_tr.columns)}

# build feature matrix [P_u | Q_i] for each training instance
X_train = np.zeros((len(train_df), 2*K_FACTORS))
for row_idx, (u, m) in enumerate(zip(train_df.user_id, train_df.movie_id)):
    ui = user_to_idx[u]
    mi = item_to_idx[m]
    X_train[row_idx, :K_FACTORS]      = P[ui]
    X_train[row_idx, K_FACTORS:] = Q[mi]
y_train = train_df['rating'].values

# fit a ridge regressor on latent features
ridge = Ridge(alpha=5.0, fit_intercept=True)  # alpha tuned lightly; grid‑search for best
ridge.fit(X_train, y_train)

intercept = float(ridge.intercept_)
coef_user = ridge.coef_[:K_FACTORS]
coef_item = ridge.coef_[K_FACTORS:]

def predict_lr(u, m):
    if u not in user_to_idx or m not in item_to_idx:
        return 3.0  # default fallback
    ui, mi = user_to_idx[u], item_to_idx[m]
    pu, qi = P[ui], Q[mi]
    return float(np.clip(intercept + pu.dot(coef_user) + qi.dot(coef_item), 1, 5))

# RMSE on held‑out test
y_true, y_pred = [], []
for _, row in test_df.iterrows():
    y_true.append(row.rating)
    y_pred.append(predict_lr(row.user_id, row.movie_id))
rmse = sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE (Latent‑Factor Linear Regression): {rmse:.2f}")


def recommend_from_train(u_id, K=10):
    if u_id not in user_to_idx:
        raise ValueError("Unknown user in training data")
    ui = user_to_idx[u_id]
    pu = P[ui]
    raw_scores = intercept + pu.dot(coef_user) + Q.dot(coef_item)
    raw_scores = np.clip(raw_scores, 1, 5)
    already = R_tr.loc[u_id].values > 0
    raw_scores[already] = -np.inf
    top_idx = np.argpartition(-raw_scores, K)[:K]
    top_idx = top_idx[np.argsort(-raw_scores[top_idx])]
    return R_tr.columns[top_idx]

def ndcg_at_k(rels, k):
    dcg = sum(r / np.log2(i+2) for i, r in enumerate(rels))
    idcg = sum(1 / np.log2(i+2) for i in range(min(sum(rels), k)))
    return dcg / idcg if idcg else 0

def apk(actual_set, pred_list, k):
    hits, s = 0, 0.0
    for i, p in enumerate(pred_list[:k], 1):
        if p in actual_set:
            hits += 1
            s += hits / i
    return s / min(len(actual_set), k) if actual_set else 0

def evaluate_ranking(k=10, thr=4):
    precs, recs, f1s, ndcgs, maps = [], [], [], [], []
    for u in test_df.user_id.unique():
        rel_items = set(test_df.loc[(test_df.user_id==u)&(test_df.rating>=thr),'movie_id'])
        if not rel_items or u not in user_to_idx:
            continue
        ranked = recommend_from_train(u, k)
        hits   = [1 if m in rel_items else 0 for m in ranked]
        h      = sum(hits)
        prec, rec = h/k, h/len(rel_items)
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        ndcg = ndcg_at_k(hits, k)
        maps.append(apk(rel_items, ranked, k))
        precs.append(prec); recs.append(rec); f1s.append(f1); ndcgs.append(ndcg)
    return map(float, map(np.mean, [precs, recs, f1s, ndcgs, maps]))

p10, r10, f10, ndcg10, map10 = evaluate_ranking(10)
print("\nRanking metrics @10 on test users (Latent‑Factor Linear Regression):")
print(f"Precision@10: {p10:.4f}")
print(f"Recall@10:    {r10:.4f}")
print(f"F1@10:        {f10:.4f}")
print(f"NDCG@10:      {ndcg10:.4f}")
print(f"MAP@10:       {map10:.4f}\n")

# For deployment we retrain SVD on all data and reuse the same ridge weights.
U_full, S_full, Vt_full = svds(csr_matrix(R_dm.values), k=K_FACTORS)
P_full = U_full @ np.diag(np.sqrt(S_full))
Q_full = (np.diag(np.sqrt(S_full)) @ Vt_full).T

user_to_idx_full = {u: i for i, u in enumerate(R.index)}


def recommend(u_id, n=5):
    if u_id not in user_to_idx_full:
        raise ValueError("Unknown user")
    ui = user_to_idx_full[u_id]
    pu = P_full[ui]
    scores = intercept + pu.dot(coef_user) + Q_full.dot(coef_item)
    scores = np.clip(scores, 1, 5)
    already = R.loc[u_id].values > 0
    scores[already] = -np.inf
    top = np.argpartition(-scores, n)[:n]
    top = top[np.argsort(-scores[top])]
    return [(movies.loc[movies.movie_id==R.columns[i],'title'].iloc[0], scores[i]) for i in top]

print("Top 5 recommendations for user 10 (latent‑factor model):")
for t,s in recommend(10,5):
    print(f"{t} (pred: {s:.2f})")

