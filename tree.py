from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, ndcg_score

RATING_THRESHOLD  = 4       # ≥ threshold ⇒ relevant for Precision/Recall/MAP/F1
TOP_K_NEIGHBOURS  = 50      # neighbors to consider
DEFAULT_FILL      = 0       # fill value for missing ratings
TEST_SIZE         = 0.20    # fraction of a user’s ratings to hold out

#load preprocess data
def load_movielens_100k(path: str | Path = "ml-100k") -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    ratings = pd.read_csv(path / "u.data", sep="\t", names=["user_id","movie_id","rating","timestamp"], engine="python")
    movies  = pd.read_csv(path / "u.item", sep="|", encoding="latin-1", engine="python",
                          names=["movie_id","title","release_date","video_release_date","imdb_url"] + [f"genre_{i}" for i in range(19)])
    return ratings, movies[["movie_id","title"]]


def build_user_item_matrix(ratings: pd.DataFrame, fill: float = DEFAULT_FILL) -> pd.DataFrame:
    return ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(fill)


def mean_centre(matrix: pd.DataFrame) -> pd.DataFrame:
    user_means = matrix.replace(0, np.nan).mean(axis=1)
    return matrix.sub(user_means, axis=0).fillna(0)

def split_user_ratings(user_row: pd.Series, test_size: float = TEST_SIZE, seed: int = 42) -> Tuple[List[int], List[int]]:
    items = user_row[user_row > 0].index.tolist()
    return train_test_split(items, test_size=test_size, random_state=seed)

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    mask = (u != 0) | (v != 0)
    if not mask.any(): return 0.0
    num = np.dot(u[mask], v[mask])
    den = np.linalg.norm(u[mask]) * np.linalg.norm(v[mask])
    return num/den if den else 0.0


def top_neighbours(target: pd.Series, norm_matrix: pd.DataFrame, k: int = TOP_K_NEIGHBOURS) -> List[int]:
    sims = [(uid, cosine(target.values, row.values))
            for uid,row in norm_matrix.iterrows() if uid!=target.name]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid,sim in sims[:k] if sim>0]

def fit_dt(neighbours: List[int], norm_matrix: pd.DataFrame, uid: int, train_items: List[int]) -> DecisionTreeRegressor:
    # X: n_items × n_neigh, y: centred ratings
    X = norm_matrix.loc[neighbours, train_items].T.values
    y = norm_matrix.loc[uid, train_items].values
    return DecisionTreeRegressor(random_state=42).fit(X, y)

def predict_centered(model: DecisionTreeRegressor, neighbours: List[int], norm_matrix: pd.DataFrame, items: List[int]) -> pd.Series:
    X = norm_matrix.loc[neighbours, items].T.values
    return pd.Series(model.predict(X), index=items)

def recommend(user_id: int, ui: pd.DataFrame, uin: pd.DataFrame, movies: pd.DataFrame,
              n_recs: int = 5, k_neigh: int = TOP_K_NEIGHBOURS, verbose: bool = False) -> Tuple[List[str], Dict[str,float]]:
    train_items, test_items = split_user_ratings(ui.loc[user_id])
    neigh = top_neighbours(uin.loc[user_id], uin, k_neigh)
    if not neigh:
        raise RuntimeError("No neighbours found—try increasing TOP_K_NEIGHBOURS")
    model = fit_dt(neigh, uin, user_id, train_items)
    unseen = ui.columns.difference(train_items + test_items)
    preds_c = predict_centered(model, neigh, uin, unseen)
    top_ids = preds_c.nlargest(n_recs).index.tolist()
    titles = movies.set_index("movie_id").loc[top_ids, "title"].tolist()

    # Metrics
    user_mean = ui.replace(0, np.nan).loc[user_id].mean()
    y_true_c = uin.loc[user_id, test_items]
    y_pred_c = predict_centered(model, neigh, uin, test_items)
    rmse = mean_squared_error(y_true_c, y_pred_c)
    # raw predictions for ranking
    y_pred_raw = y_pred_c + user_mean
    y_true_raw = ui.loc[user_id, test_items]
    ndcg = ndcg_score([y_true_raw.values], [y_pred_raw.values])
    # P@K, R@K, MAP, F1
    relevant = [i for i in test_items if ui.loc[user_id,i]>=RATING_THRESHOLD]
    order = pd.Series(y_pred_raw, index=test_items).sort_values(ascending=False).index.tolist()
    hits = [1 if i in relevant else 0 for i in order[:n_recs]]
    precision = sum(hits)/n_recs if n_recs else 0
    recall    = sum(hits)/len(relevant) if relevant else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0
    # MAP@K
    ap, cnt = 0.0, 0
    for idx,item in enumerate(order[:n_recs],1):
        if item in relevant:
            cnt += 1
            ap += cnt/idx
    map_k = ap/len(relevant) if relevant else 0

    metrics = {"RMSE":rmse, "Precision_at_k":precision, "Recall_at_k":recall,
               "F1":f1, "nDCG":ndcg, "MAP":map_k}

    if verbose:
        print(f"\nRecommendations for U{user_id} (top {n_recs}):")
        for t in titles: print(f" • {t}")
        print("\nMetrics:")
        for m,v in metrics.items(): print(f" {m:12s}: {v:.4f}")
    return titles, metrics

def demo(user_id: int = 10, n_recs: int = 5):
    ratings, movies = load_movielens_100k()
    ui  = build_user_item_matrix(ratings)
    uin = mean_centre(ui)
    return recommend(user_id, ui, uin, movies, n_recs, verbose=True)

if __name__ == "__main__":
    import sys
    uid  = int(sys.argv[1]) if len(sys.argv)>1 else 10
    nrec = int(sys.argv[2]) if len(sys.argv)>2 else 5
    demo(uid, nrec)

