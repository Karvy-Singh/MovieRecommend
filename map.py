import pandas as pd
import numpy as np
from math import log2
from sklearn.model_selection import train_test_split

r_cols = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

i_cols = ['movie_id','title','release_date','video_release_date','IMDb_URL'] \
       + [f'genre_{i}' for i in range(19)]
movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

data = pd.merge(ratings, movies[['movie_id','title']], on='movie_id')
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
popularity = train_df.groupby('movie_id')['rating'].mean()

def dcg_at_k(rels, k):
    return sum(rel / log2(idx + 2) for idx, rel in enumerate(rels[:k]))

def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def apk(actual_set, pred_list, k):
    hits = 0
    score = 0.0
    for i, p in enumerate(pred_list[:k]):
        if p in actual_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual_set), k) if actual_set else 0.0

def evaluate_ranking(train_df, test_df, popularity, K=10):
    ndcgs, maps = [], []
    for user in test_df['user_id'].unique():
        true_items = set(test_df.loc[test_df.user_id == user, 'movie_id'])
        seen_items = set(train_df.loc[train_df.user_id == user, 'movie_id'])
        candidates = [m for m in popularity.index if m not in seen_items]
        ranked = sorted(candidates, key=lambda m: popularity[m], reverse=True)[:K]
        rels = [1 if m in true_items else 0 for m in ranked]
        ndcgs.append(ndcg_at_k(rels, K))
        maps.append(apk(true_items, ranked, K))
    return np.mean(ndcgs), np.mean(maps)

K = 10
mean_ndcg, mean_map = evaluate_ranking(train_df, test_df, popularity, K)
print(f"NDCG@{K}: {mean_ndcg:.4f}")
print(f"MAP@{K}:  {mean_map:.4f}")

