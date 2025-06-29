import numpy as np
import pandas as pd
from math import sqrt
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.model_selection import train_test_split

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
i_cols = ['movie_id', 'title', 'release_date', 'video_release_date',
          'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
movies = pd.read_csv('ml-100k/u.item', sep='|',
                     names=i_cols, encoding='latin-1')[['movie_id', 'title']]
data = ratings.merge(movies, on='movie_id')

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
R_tr = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
user_idx = {u: i for i, u in enumerate(R_tr.index)}
item_idx = {m: i for i, m in enumerate(R_tr.columns)}
R_tr_mask = R_tr.values > 0

item_vectors = R_tr.values.T                  # n_items × n_users
cos_dist = pairwise_distances(item_vectors, metric='cosine')
sim_full = 1.0 - cos_dist
np.fill_diagonal(sim_full, 0.0)

# keep only top-K neighbors per item
K_NEIGHBORS = 40
MIN_SIM = 0.05

knn_idx = np.argsort(-sim_full, axis=1)[:, :K_NEIGHBORS]
knn_sim = np.take_along_axis(sim_full, knn_idx, axis=1)

# drop weak links and build sparse matrix
rows = np.repeat(np.arange(sim_full.shape[0]), K_NEIGHBORS)
cols = knn_idx.flatten()
data_ = knn_sim.flatten()

mask = (cols >= 0) & (data_ >= MIN_SIM)
rows, cols, data_ = rows[mask], cols[mask], data_[mask]

sim_sparse = csr_matrix((data_, (rows, cols)),
                        shape=sim_full.shape,
                        dtype=np.float32)

# precompute normalization (sum of abs sims per item)
denom = np.array(np.abs(sim_sparse).sum(axis=1)).ravel() + 1e-8

def score_knn_vec(u_id):
    """Returns array of scores for all items for user u_id."""
    ui = user_idx.get(u_id, None)
    if ui is None:
        return None
    # user rating vector (n_items,)
    r_u = R_tr.values[ui]
    # weighted sums: each item_i score = sum_j sim(i,j)*r_u[j]
    raw = sim_sparse.dot(r_u)                # shape: (n_items,)
    return raw / denom                       # vectorized normalization

def recommend_knn_fast(u_id, K=10):
    scores = score_knn_vec(u_id)
    if scores is None:
        raise ValueError(f"Unknown user {u_id}")
    # mask already-rated
    rated = R_tr_mask[user_idx[u_id]]
    scores[rated] = -np.inf
    # top-K
    idx = np.argpartition(-scores, K)[:K]
    idx = idx[np.argsort(-scores[idx])]
    return [R_tr.columns[i] for i in idx], scores

# build rescaler from raw train-set scores
train_raw = []
for uid, mid in zip(train_df.user_id, train_df.movie_id):
    raw = score_knn_vec(uid)[item_idx[mid]]
    train_raw.append(raw)
train_raw = np.array(train_raw)
lo, hi = train_raw.min(), train_raw.max()

def rescale(x): return 1 + 4*(x - lo)/(hi - lo)

y_true, y_pred = [], []
for _, row in test_df.iterrows():
    uid, mid = row.user_id, row.movie_id
    if uid not in user_idx or mid not in item_idx:
        continue
    raw = score_knn_vec(uid)[item_idx[mid]]
    y_true.append(row.rating)
    y_pred.append(rescale(raw))

rmse = sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE (fast KNN-IBCF): {rmse:.3f}")

def ndcg_at_k(hits, k):
    dcg = np.sum(hits / np.log2(np.arange(2, k+2)))
    idcg = np.sum(1/np.log2(np.arange(2, min(hits.sum(), k)+2)))
    return dcg/idcg if idcg else 0.0

def apk(actual, pred, k):
    hits = 0; score = 0.0
    for i, p in enumerate(pred[:k], 1):
        if p in actual:
            hits += 1
            score += hits/i
    return score/min(len(actual), k) if actual else 0.0

ps, rs, f1s, ndcgs, maps = [], [], [], [], []
for u in test_df.user_id.unique():
    if u not in user_idx: continue
    rel = set(test_df.loc[
        (test_df.user_id==u)&(test_df.rating>=4), 'movie_id'])
    if not rel: continue
    recs, _ = recommend_knn_fast(u, 10)
    hits = np.array([1 if m in rel else 0 for m in recs])
    h = hits.sum()
    p, r = h/10, h/len(rel)
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    ps.append(p); rs.append(r); f1s.append(f1)
    ndcgs.append(ndcg_at_k(hits, 10))
    maps.append(apk(rel, recs, 10))

print("\nRanking @10 (fast KNN-IBCF):")
print(f"P@10:  {np.mean(ps):.4f}")
print(f"R@10:  {np.mean(rs):.4f}")
print(f"F1@10: {np.mean(f1s):.4f}")
print(f"NDCG:  {np.mean(ndcgs):.4f}")
print(f"MAP:   {np.mean(maps):.4f}")

print("\nTop-5 fast KNN-IBCF recs for user 10:")
recs, scores = recommend_knn_fast(10, 5)
for mid in recs:
    title = movies.loc[movies.movie_id==mid, 'title'].iloc[0]
    print(" •", title)

