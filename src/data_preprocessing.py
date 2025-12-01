## data preprocessing

## 1. train / validation / test set segmentation (60/20/20)
# src/data_preprocessing.py

import pandas as pd
import numpy as np

def load_data(path="../data/fetal_health.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["fetal_health"])
    y = df["fetal_health"]
    return X, y

def split_indices(
    y: pd.Series,
    train_ratio: float = 0.8,
    ## test_ratio = 1 - train_ratio - val_ratio
    random_seed: int = 42
):

    rng = np.random.default_rng(random_seed)

    classes = sorted(y.unique())
    train_idx = []
    test_idx = []

    for c in classes:
        idx_c = np.where(y.values == c)[0]

        rng.shuffle(idx_c)

        n = len(idx_c)
        n_train = int(n * train_ratio)

        idx_train_c = idx_c[:n_train]
        idx_test_c  = idx_c[n_train:]

        train_idx+=idx_train_c.tolist()
        test_idx+=idx_test_c.tolist()

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return train_idx, test_idx

def make_splits(X, y, train_idx, test_idx):
    X_train = X.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test  = y.iloc[test_idx].copy()

    return X_train, X_test, y_train, y_test


## 2. train set 내 클래스 불균형 해소를 위한 SMOTE

def smote_minority_class(X, y, minority_class, target_num, k=5, random_state=42):
    rng = np.random.default_rng(random_state)
    
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    
    mask_min = (y == minority_class)
    X_min = X[mask_min]
    n_min= X_min.shape[0]
    
    if target_num <= n_min:
        print(f"[SMOTE] class {minority_class}: 이미 {n_min}개 ≥ target {target_num}개라서 그대로 사용.")
        return X, y
    
    n_new = target_num - n_min
    print(f"[SMOTE] class {minority_class}: {n_min}개 → {target_num}개로 늘리기 (synthetic {n_new}개).")
    
    diff = X_min[:, None, :] - X_min[None, :, :]   ## 브로드캐스팅
    dist = np.linalg.norm(diff, axis=2)   # shape: (n_min, n_min)
    
    np.fill_diagonal(dist, np.inf) ## 자기 자신과의 거리 무한대로
    
    knn_indices = np.argsort(dist, axis=1)[:, :k]  # shape: (n_min, k)
    
    synthetic_samples = []
    for _ in range(n_new):
        i = rng.integers(0, n_min)
        x_i = X_min[i]
        
        neighbor_idx = rng.choice(knn_indices[i])
        x_nn = X_min[neighbor_idx]
        
        lam = rng.random()
        
        x_new = x_i + lam * (x_nn - x_i)
        synthetic_samples.append(x_new)
    
    synthetic_samples = np.vstack(synthetic_samples)
    synthetic_labels = np.full(n_new, minority_class)
    
    X_new = np.vstack([X, synthetic_samples])
    y_new = np.concatenate([y, synthetic_labels])
    
    return X_new, y_new


## 3. logistic regression에 넣게 될 train set "standard scaler"


def scale_train_test_np(X_train_np: np.array, X_test_np: np.array):

    mean = X_train_np.mean(axis=0)
    std = X_train_np.std(axis=0)

    eps = 1e-8
    std_safe = np.where(std < eps, 1.0, std)

    X_train_scaled = (X_train_np - mean) / std_safe
    X_test_scaled  = (X_test_np  - mean) / std_safe


    return X_train_scaled, X_test_scaled