import pandas as pd
import numpy as np
from math import sqrt
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# from readme we check the structure of u.data, u.item nd u.genre and we will combine them together 
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
i_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] \
         + [f'genre_{i}' for i in range(19)]
movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

data = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')

R = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
R_tr = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

K_LAT = 40
U, S, Vt = svds(csr_matrix(R_tr.values - R_tr.values.mean()), k=K_LAT)
P = U * np.sqrt(S)          # users  (n_u × k)
Q = (Vt.T) * np.sqrt(S)     # items  (n_i × k)
user_idx = {u: i for i, u in enumerate(R_tr.index)}
item_idx = {m: i for i, m in enumerate(R_tr.columns)}


def f_vec(ui, mi):
    """Return k‑dim interaction feature (element‑wise product)."""
    return P[ui] * Q[mi]

# For each user, sample up to PAIRS_PER_USER preference pairs (pos > neg)
PAIRS_PER_USER = 30
X_pairs = []
y_pairs = []  # +1 means first movie preferred over second
np_random = np.random.default_rng(42)

for u in train_df['user_id'].unique():
    ui = user_idx[u]
    user_ratings = train_df[train_df.user_id == u]
    if user_ratings.rating.nunique() < 2:
        continue
    # separate positive (>=4) and negative (<=2) for clear signal
    pos = user_ratings[user_ratings.rating >= 4]['movie_id'].tolist()
    neg = user_ratings[user_ratings.rating <= 2]['movie_id'].tolist()
    if not pos or not neg:
        continue
    n_samples = min(PAIRS_PER_USER, len(pos) * len(neg))
    sampled = np_random.choice(len(pos) * len(neg), size=n_samples, replace=False)
    for s in sampled:
        i = s // len(neg)
        j = s % len(neg)
        mi_pos = item_idx[pos[i]]
        mi_neg = item_idx[neg[j]]
        # feature difference (pos − neg) labeled +1
        X_pairs.append(f_vec(ui, mi_pos) - f_vec(ui, mi_neg))
        y_pairs.append(1)
        # add the inverse pair to balance classes
        X_pairs.append(f_vec(ui, mi_neg) - f_vec(ui, mi_pos))
        y_pairs.append(-1)

X_pairs = np.vstack(X_pairs)
y_pairs = np.array(y_pairs)

# Standardise features for faster convergence
scaler = StandardScaler(with_mean=False)
X_pairs_std = scaler.fit_transform(X_pairs)

rank_svm = SGDClassifier(loss='hinge', alpha=1e-4, max_iter=3000, tol=1e-3, random_state=42)
rank_svm.fit(X_pairs_std, y_pairs)

w = rank_svm.coef_.ravel()  # weight vector for scoring
b = rank_svm.intercept_[0]

def score(u, m):
    if u not in user_idx or m not in item_idx:
        return 0.0
    vec = scaler.transform(f_vec(user_idx[u], item_idx[m]).reshape(1, -1))
    return float((vec.dot(w) + b).item())

train_scores = np.array([score(r.user_id, r.movie_id) for _, r in train_df.iterrows()])
min_s, max_s = train_scores.min(), train_scores.max()

def rescale(s):
    return 1 + 4 * (s - min_s) / (max_s - min_s + 1e-8)

y_true = []
y_pred = []
for _, row in test_df.iterrows():
    if row.user_id not in user_idx or row.movie_id not in item_idx:
        continue
    y_true.append(row.rating)
    y_pred.append(rescale(score(row.user_id, row.movie_id)))

rmse = sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE (RankSVM): {rmse:.2f}")

R_tr_mask = R_tr.values > 0  # mask array for speed

def recommend_from_train(u_id, K=10):
    if u_id not in user_idx:
        raise ValueError('Unknown user in training data')
    ui = user_idx[u_id]
    user_vec = scaler.transform((P[ui] * Q).reshape(Q.shape[0], -1))  # vectorised
    scores = user_vec.dot(w) + b
    scores = scores.ravel()
    scores[R_tr_mask[ui]] = -np.inf  # mask train‑rated movies
    top_idx = np.argpartition(-scores, K)[:K]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return R_tr.columns[top_idx]

def ndcg_at_k(rels, k):
    dcg = np.sum(rels / np.log2(np.arange(2, k + 2)))
    idcg = np.sum(1 / np.log2(np.arange(2, min(rels.sum(), k) + 2)))
    return dcg / idcg if idcg else 0.0

def apk(actual, pred, k):
    hits, score = 0, 0.0
    for i, p in enumerate(pred[:k], 1):
        if p in actual:
            hits += 1
            score += hits / i
    return score / min(len(actual), k) if actual else 0.0

def evaluate_ranking(k=10, thr=4):
    ps, rs, f1s, ndcgs, maps = [], [], [], [], []
    for u in test_df.user_id.unique():
        if u not in user_idx:
            continue
        rel = set(test_df.loc[(test_df.user_id == u) & (test_df.rating >= thr), 'movie_id'])
        if not rel:
            continue
        ranked = recommend_from_train(u, k)
        hits = np.array([1 if m in rel else 0 for m in ranked])
        h = hits.sum()
        prec, rec = h / k, h / len(rel)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ps.append(prec); rs.append(rec); f1s.append(f1)
        ndcgs.append(ndcg_at_k(hits, k)); maps.append(apk(rel, ranked, k))
    return map(float, map(np.mean, [ps, rs, f1s, ndcgs, maps]))

p10, r10, f10, ndcg10, map10 = evaluate_ranking()
print("\nRanking metrics @10 (RankSVM):")
print(f"Precision@10: {p10:.4f}")
print(f"Recall@10:    {r10:.4f}")
print(f"F1@10:        {f10:.4f}")
print(f"NDCG@10:      {ndcg10:.4f}")
print(f"MAP@10:       {map10:.4f}\n")

print("Top‑5 recommendations for user 10 (RankSVM):")
for m in recommend_from_train(10, 5):
    print(movies.loc[movies.movie_id == m, 'title'].iloc[0])

