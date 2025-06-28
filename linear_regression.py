import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

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

# --------------------  LINEAR REGRESSION WITH USER–GENRE INTERACTIONS --------------------
# To give the linear model *personalised* variation across movies, we add user–genre interaction
# features: a feature "ug_<user>_<genre>" is 1 when the movie belongs to <genre>.
# The model is still linear, but far more expressive than just user & movie biases.

# helper: list of genre column names in the movies DataFrame
GENRE_COLS = [col for col in movies.columns if col.startswith('genre_')]

# pre‑compute the genre indices present for every movie_id (speeds up prediction)
movie_genres = {
    row.movie_id: [g_idx for g_idx, gcol in enumerate(GENRE_COLS) if row[gcol] == 1]
    for _, row in movies.iterrows()
}

# build dict‑style feature representations for each training sample
def make_feature_dict(u, m):
    feats = {
        f'u_{u}': 1,
        f'm_{m}': 1,
    }
    for g in movie_genres[m]:
        feats[f'ug_{u}_{g}'] = 1  # user‑genre interaction feature
    return feats

train_dicts = [make_feature_dict(r.user_id, r.movie_id) for _, r in train_df.iterrows()]
vec = DictVectorizer()
X_train = vec.fit_transform(train_dicts)

y_train = train_df['rating']

# ridge (L2‑regularised) regression to prevent over‑fitting on sparse data
lin_reg = Ridge(alpha=2.0, fit_intercept=True)
lin_reg.fit(X_train, y_train)

# ----------  cache coefficients for fast scoring  ----------
coef = lin_reg.coef_
intercept = float(lin_reg.intercept_)

user_coef, movie_coef, user_genre_coef = {}, {}, {}
for idx, fname in enumerate(vec.get_feature_names_out()):
    if fname.startswith('u_'):
        user_coef[int(fname[2:])] = coef[idx]
    elif fname.startswith('m_'):
        movie_coef[int(fname[2:])] = coef[idx]
    elif fname.startswith('ug_'):
        _, uid, gid = fname.split('_')  # "ug_<user>_<genre>"
        user_genre_coef[(int(uid), int(gid))] = coef[idx]

# --------------------  PREDICT & RECOMMEND  --------------------

def predict_lr(u, m):
    """Predict rating using user, movie and user‑genre coefficients."""
    pred = intercept
    pred += user_coef.get(u, 0.0)
    pred += movie_coef.get(m, 0.0)
    for g in movie_genres[m]:
        pred += user_genre_coef.get((u, g), 0.0)
    return float(np.clip(pred, 1, 5))

# RMSE on the held‑out test split
y_true, y_pred = [], []
for _, row in test_df.iterrows():
    y_true.append(row.rating)
    y_pred.append(predict_lr(row.user_id, row.movie_id))
rmse = sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE (User–Genre Linear Regression): {rmse:.2f}")

# --------------------  RECOMMEND FROM TRAIN (for evaluation) --------------------

def recommend_from_train(u_id, K=10):
    if u_id not in R_tr.index:
        raise ValueError("Unknown user in training data")

    preds = np.full(R_tr.shape[1], intercept)
    # add user bias once per movie
    preds += user_coef.get(u_id, 0.0)
    # add movie biases vectorised
    preds += np.array([movie_coef.get(m, 0.0) for m in R_tr.columns])
    # add user‑genre interactions (vectorised)
    ug_contrib = np.zeros_like(preds)
    for idx, m in enumerate(R_tr.columns):
        for g in movie_genres[m]:
            ug_contrib[idx] += user_genre_coef.get((u_id, g), 0.0)
    preds += ug_contrib
    preds = np.clip(preds, 1, 5)

    already = R_tr.loc[u_id].values > 0  # mask TRAIN ratings only
    preds[already] = -np.inf

    top_idx = np.argpartition(-preds, K)[:K]
    top_idx = top_idx[np.argsort(-preds[top_idx])]
    return R_tr.columns[top_idx]

# helper metrics (unchanged)

def ndcg_at_k(relevances, k):
    dcg  = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    idcg = sum(1 / np.log2(idx + 2) for idx in range(min(sum(relevances), k)))
    return dcg / idcg if idcg else 0.0

def apk(actual_set, pred_list, k):
    hits = 0
    sum_precisions = 0.0
    for i, p in enumerate(pred_list[:k], start=1):
        if p in actual_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(actual_set), k) if actual_set else 0.0

# --------------------  RANKING METRICS EVALUATION  --------------------

def evaluate_ranking_lr(k=10, threshold=4):
    precisions, recalls, f1s, ndcgs, maps = [], [], [], [], []

    for u in test_df['user_id'].unique():
        true_items = set(test_df.loc[(test_df.user_id == u) & (test_df.rating >= threshold), 'movie_id'])
        if not true_items:
            continue
        if u not in R_tr.index:
            continue

        ranked_items = recommend_from_train(u, k)
        hits         = [1 if m in true_items else 0 for m in ranked_items]
        num_hits     = sum(hits)

        prec = num_hits / k
        rec  = num_hits / len(true_items)
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        ndcg_val = ndcg_at_k(hits, k)
        map_val  = apk(true_items, ranked_items, k)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        ndcgs.append(ndcg_val)
        maps.append(map_val)

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        float(np.mean(ndcgs)),
        float(np.mean(maps))
    )

mean_prec, mean_rec, mean_f1, mean_ndcg, mean_map = evaluate_ranking_lr(k=10)
print(f"\nRanking metrics @10 on test users (User–Genre Linear Regression):")
print(f"Precision@10: {mean_prec:.4f}")
print(f"Recall@10:    {mean_rec:.4f}")
print(f"F1@10:        {mean_f1:.4f}")
print(f"NDCG@10:      {mean_ndcg:.4f}")
print(f"MAP@10:       {mean_map:.4f}\n")

# --------------------  FULL‑DATA RECOMMENDER FOR DEMO  --------------------

def recommend(u_id, n=5):
    if u_id not in R.index:
        raise ValueError("Unknown user")

    preds = np.full(R.shape[1], intercept)
    preds += user_coef.get(u_id, 0.0)
    preds += np.array([movie_coef.get(m, 0.0) for m in R.columns])
    ug_contrib = np.zeros_like(preds)
    for idx, m in enumerate(R.columns):
        for g in movie_genres[m]:
            ug_contrib[idx] += user_genre_coef.get((u_id, g), 0.0)
    preds += ug_contrib
    preds = np.clip(preds, 1, 5)

    already = R.loc[u_id].values > 0
    preds[already] = -np.inf

    top_ids = np.argpartition(-preds, n)[:n]
    top_ids = top_ids[np.argsort(-preds[top_ids])]

    return [
        (movies.loc[movies.movie_id == R.columns[idx], 'title'].iloc[0], preds[idx])
        for idx in top_ids
    ]

# --------------------  EXAMPLE  --------------------
print("Top 5 recommendations for user 10 (demo – user–genre linear model):")
for title, score in recommend(10, 5):
    print(f"{title} (predicted rating: {score:.2f})")

