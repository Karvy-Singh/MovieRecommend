from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, ndcg_score

RATING_THRESHOLD  = 4          # ≥ threshold ⇒ *relevant* (for Precision/Recall/MAP/F1)
TOP_K_NEIGHBOURS  = 50         # number of neighbours fed to the regression
DEFAULT_FILL      = 0          # value for unrated cells in the user‑item matrix
TEST_SIZE         = 0.20       # fraction of each user’s ratings withheld for testing

def load_movielens_100k(path: str | Path = "ml-100k") -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    ratings = pd.read_csv(path / "u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"], engine="python")
    movies  = pd.read_csv(path / "u.item", sep="|", encoding="latin-1", engine="python",
                          names=["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{i}" for i in range(19)])
    return ratings, movies[["movie_id", "title"]]


def build_user_item_matrix(ratings: pd.DataFrame, fill: int | float = DEFAULT_FILL) -> pd.DataFrame:
    """Pivot ratings into a user × item table filled with fill."""
    return ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(fill)


def mean_centre(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return a matrix where every user’s non‑zero ratings have her mean subtracted."""
    user_means = matrix.replace(0, np.nan).mean(axis=1)
    return matrix.sub(user_means, axis=0).fillna(0)

def split_user_ratings(user_row: pd.Series, test_size: float = TEST_SIZE, seed: int = 42) -> Tuple[List[int], List[int]]:
    rated_items = user_row[user_row > 0].index.tolist()
    return train_test_split(rated_items, test_size=test_size, random_state=seed)

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    mask = (u != 0) | (v != 0)
    if not mask.any():
        return 0.0
    num   = np.dot(u[mask], v[mask])
    denom = np.linalg.norm(u[mask]) * np.linalg.norm(v[mask])
    return num / denom if denom else 0.0


def top_neighbours(target_vector: pd.Series, matrix_norm: pd.DataFrame, k: int = TOP_K_NEIGHBOURS) -> List[int]:
    sims = [(uid, cosine(target_vector.values, row.values))
            for uid, row in matrix_norm.iterrows() if uid != target_vector.name]
    sims.sort(key=lambda t: t[1], reverse=True)
    return [uid for uid, sim in sims[:k] if sim > 0]


def fit_lr(neighbours: List[int], matrix_norm: pd.DataFrame, uid: int, train_items: List[int]) -> LinearRegression:
    X = matrix_norm.loc[neighbours, train_items].T.values  # (n_items, n_neigh)
    y = matrix_norm.loc[uid, train_items].values           # (n_items,)
    return LinearRegression().fit(X, y)

def predict_centered(model: LinearRegression, neighbours: List[int], matrix_norm: pd.DataFrame, items: List[int]) -> pd.Series:
    X = matrix_norm.loc[neighbours, items].T.values
    return pd.Series(model.predict(X), index=items)


def recommend(user_id: int, user_item: pd.DataFrame, user_item_norm: pd.DataFrame, movies: pd.DataFrame,
              n_recs: int = 5, k_neigh: int = TOP_K_NEIGHBOURS, verbose: bool = False) -> Tuple[List[str], Dict[str, float]]:
    #Hold‑out split for this user
    train_items, test_items = split_user_ratings(user_item.loc[user_id])

    # Neighbour selection + regression 
    neigh_ids = top_neighbours(user_item_norm.loc[user_id], user_item_norm, k=k_neigh)
    if not neigh_ids:
        raise RuntimeError("No suitable neighbours – try increasing TOP_K_NEIGHBOURS.")
    model = fit_lr(neigh_ids, user_item_norm, user_id, train_items)

    # Score all unseen items
    unseen = user_item.columns.difference(train_items + test_items)
    preds_centered = predict_centered(model, neigh_ids, user_item_norm, unseen)
    topN_ids = preds_centered.sort_values(ascending=False).head(n_recs).index.tolist()
    titles = movies.set_index("movie_id").loc[topN_ids, "title"].tolist()

    # Evaluation
    user_mean = user_item.replace(0, np.nan).loc[user_id].mean()

    # RMSE on centred ratings
    y_true_c = user_item_norm.loc[user_id, test_items]
    y_pred_c = predict_centered(model, neigh_ids, user_item_norm, test_items)
    rmse = mean_squared_error(y_true_c, y_pred_c)

    # Ranking metrics use raw (positive) ratings
    y_true_raw = user_item.loc[user_id, test_items]
    y_pred_raw = y_pred_c + user_mean
    ndcg = ndcg_score([y_true_raw.values], [y_pred_raw.values])

    # Binary relevance for P@K, R@K, MAP, F1
    relevant = [i for i in test_items if user_item.loc[user_id, i] >= RATING_THRESHOLD]
    rank_order = y_pred_raw.sort_values(ascending=False).index.tolist()
    k = n_recs
    hits = [1 if itm in relevant else 0 for itm in rank_order[:k]]
    precision = sum(hits) / k if k else 0
    recall    = sum(hits) / len(relevant) if relevant else 0
    f1 = (2*precision*recall) / (precision + recall) if (precision + recall) else 0
    # MAP@k
    ap, hit_cnt = 0.0, 0
    for idx, itm in enumerate(rank_order[:k], start=1):
        if itm in relevant:
            hit_cnt += 1
            ap += hit_cnt / idx
    map_k = ap / len(relevant) if relevant else 0

    metrics = dict(RMSE=rmse, Precision_at_k=precision, Recall_at_k=recall, F1=f1, nDCG=ndcg, MAP=map_k)

    if verbose:
        print(f"\nRecommendations for user U{user_id} (top {n_recs}):")
        for t in titles:
            print(f"  • {t}")
        print("\nMetrics:")
        for k_, v_ in metrics.items():
            print(f"  {k_:14s}: {v_: .4f}")

    return titles, metrics

def demo(user_id: int = 10, n_recs: int = 5):
    ratings, movies = load_movielens_100k()
    ui  = build_user_item_matrix(ratings)
    uin = mean_centre(ui)
    return recommend(user_id, ui, uin, movies, n_recs=n_recs, verbose=True)

if __name__ == "__main__":
    import sys
    uid  = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    nrec = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    demo(uid, nrec)

