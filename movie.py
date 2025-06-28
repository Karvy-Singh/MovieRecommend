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

def recommend_from_train(u_id, K=10):
    if u_id not in R_tr.index:
        raise ValueError("Unknown user in training data")

    ui   = R_tr.index.get_loc(u_id)
    sims = sims_mat[ui]
    nbrs = nbrs_mat[ui]
    mu   = means_tr.values[ui]

    # score for every item (weighted sum of neighbour ratings)
    score_num = (sims[:, None] * R_tr_dm.values[nbrs]).sum(axis=0)
    score_den = np.abs(sims).sum()
    preds     = mu + score_num / score_den if score_den else mu
    preds     = np.clip(preds, 1, 5)

    # mask items the user has rated in the training split only
    already   = R_tr.values[ui] > 0
    preds[already] = -np.inf

    # top-K movie indices
    top_idx = np.argpartition(-preds, K)[:K]
    top_idx = top_idx[np.argsort(-preds[top_idx])]
    return R_tr.columns[top_idx]


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


def evaluate_ranking_knn(k=10, threshold=4):
    """Return mean Precision, Recall, F1, NDCG, MAP @k over users in *test_df*."""
    precisions, recalls, f1s, ndcgs, maps = [], [], [], [], []

    for u in test_df['user_id'].unique():
        true_items = set(test_df.loc[(test_df.user_id == u) & (test_df.rating >= threshold), 'movie_id'])
        if not true_items:
            continue  # skip users without relevant items
        if u not in R_tr.index:
            continue  # user not in training split

        ranked_items = recommend_from_train(u, k)
        hits         = [1 if m in true_items else 0 for m in ranked_items]
        num_hits     = sum(hits)

        # precision & recall
        prec = num_hits / k
        rec  = num_hits / len(true_items)
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

        # ndcg and map
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

mean_prec, mean_rec, mean_f1, mean_ndcg, mean_map = evaluate_ranking_knn(k=10)
print(f"\nRanking metrics @10 on test users:")
print(f"Precision@10: {mean_prec:.4f}")
print(f"Recall@10:    {mean_rec:.4f}")
print(f"F1@10:        {mean_f1:.4f}")
print(f"NDCG@10:      {mean_ndcg:.4f}")
print(f"MAP@10:       {mean_map:.4f}\n")

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
    mat    = R_dm.values[nbrs]  # shape (K, n_items)

    num   = (sims[:, None] * mat).sum(axis=0)
    den   = np.abs(sims).sum()
    preds = mean_u + num/den if den else np.full(R.shape[1], mean_u)
    preds = np.clip(preds, 1, 5)

    # mask items already rated in full data (training+test)
    already = R.values[ui] > 0
    preds[already] = -np.inf

    top_ids = np.argpartition(-preds, n)[:n] 
    top_ids = top_ids[np.argsort(-preds[top_ids])]

    return [
        (movies.loc[movies.movie_id == R.columns[idx], 'title'].iloc[0], preds[idx])
        for idx in top_ids
    ]

print("Top 5 recommendations for user 10 (demo using full data):")
for title, score in recommend(10, 5):
    print(f"{title} (predicted rating: {score:.2f})")
