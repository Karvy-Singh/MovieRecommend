import pandas as pd
import numpy as np
from math import sqrt
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor  # faster tree ensemble

# from readme we check the structure of u.data, u.item nd u.genre and we will combine them together 
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

i_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] \
         + [f'genre_{i}' for i in range(19)]
movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

data = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')

# full rating matrix (dense) for later demo use
R = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
means = R.mean(axis=1)
R_dm = R.sub(means, axis=0)

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
R_tr = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
means_tr = R_tr.mean(axis=1)
R_tr_dm = R_tr.sub(means_tr, axis=0)

dim_latent = 40  # speed/quality sweet‑spot; tweak 20‑40
U, S, Vt = svds(csr_matrix(R_tr_dm.values), k=dim_latent)
S_sqrt = np.sqrt(S)
P = U * S_sqrt            # user embeddings  (n_users × k)
Q = (Vt.T * S_sqrt)       # item embeddings  (n_items × k)

user_to_idx = {u: i for i, u in enumerate(R_tr.index)}
item_to_idx = {m: i for i, m in enumerate(R_tr.columns)}

ui_idx = train_df['user_id'].map(user_to_idx).to_numpy()
mi_idx = train_df['movie_id'].map(item_to_idx).to_numpy()
X_train = np.hstack((P[ui_idx], Q[mi_idx]))
y_train = train_df['rating'].to_numpy()

hgb = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=160,
    max_depth=4,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42
)

hgb.fit(X_train, y_train)  

def predict_fast(u, m):
    if u not in user_to_idx or m not in item_to_idx:
        return 3.0  # global mean fallback
    feat = np.hstack((P[user_to_idx[u]], Q[item_to_idx[m]]))[None, :]
    return float(np.clip(hgb.predict(feat)[0], 1, 5))

y_true = test_df['rating'].to_numpy()
ui_tst = test_df['user_id'].map(user_to_idx).fillna(-1).astype(int).to_numpy()
mi_tst = test_df['movie_id'].map(item_to_idx).fillna(-1).astype(int).to_numpy()

mask_known = (ui_tst != -1) & (mi_tst != -1)
X_test = np.hstack((P[ui_tst[mask_known]], Q[mi_tst[mask_known]]))
rmse = sqrt(mean_squared_error(y_true[mask_known], hgb.predict(X_test)))
print(f"Test RMSE (HistGB, 40‑dim): {rmse:.2f}")

def recommend_from_train(u_id, K=10):
    if u_id not in user_to_idx:
        raise ValueError('Unknown user in training data')
    ui = user_to_idx[u_id]
    user_vec = P[ui]
    # single matrix‑vector multiply for all items
    scores = hgb.predict(np.hstack((np.tile(user_vec, (Q.shape[0], 1)), Q)))
    scores = np.clip(scores, 1, 5)
    scores[R_tr.loc[u_id].values > 0] = -np.inf  # mask train ratings
    top = np.argpartition(-scores, K)[:K]
    top = top[np.argsort(-scores[top])]
    return R_tr.columns[top]

def ndcg_at_k(rels, k):
    dcg = np.sum(rels / np.log2(np.arange(2, k+2)))
    idcg = np.sum(1 / np.log2(np.arange(2, min(np.sum(rels), k)+2)))
    return dcg / idcg if idcg else 0.0

def apk(actual, pred, k):
    hits = 0; s = 0.0
    for i, p in enumerate(pred[:k], 1):
        if p in actual:
            hits += 1; s += hits / i
    return s / min(len(actual), k) if actual else 0.0

def evaluate_ranking(k=10, thr=4):
    """Compute Precision, Recall, F1, NDCG and MAP @k (fixed list‑initialisation bug)."""
    precs, recs, f1s, ndcgs, maps = [], [], [], [], []  
    for u in test_df['user_id'].unique():
        if u not in user_to_idx:
            continue
        rel = set(test_df.loc[(test_df.user_id == u) & (test_df.rating >= thr), 'movie_id'])
        if not rel:
            continue

        ranked = recommend_from_train(u, k)
        hits   = np.array([1 if m in rel else 0 for m in ranked])
        hit_cnt = hits.sum()

        prec = hit_cnt / k
        rec  = hit_cnt / len(rel)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ndcg_val = ndcg_at_k(hits, k)
        map_val  = apk(rel, ranked, k)

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        ndcgs.append(ndcg_val)
        maps.append(map_val)

    return (
        float(np.mean(precs)),
        float(np.mean(recs)),
        float(np.mean(f1s)),
        float(np.mean(ndcgs)),
        float(np.mean(maps))
    )

p10, r10, f10, ndcg10, map10 = evaluate_ranking()
print("\nRanking metrics @10 (HistGB, 40‑dim):")
print(f"Precision@10: {p10:.4f}")
print(f"Recall@10:    {r10:.4f}")
print(f"F1@10:        {f10:.4f}")
print(f"NDCG@10:      {ndcg10:.4f}")
print(f"MAP@10:       {map10:.4f}\n")

print("Top‑5 recommendations for user 10:")
for m in recommend_from_train(10, 5):
    print(movies.loc[movies.movie_id==m, 'title'].iloc[0])

