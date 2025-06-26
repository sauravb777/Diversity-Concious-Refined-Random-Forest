# # traditionalrf_titanic.py
# # RUN THIS RANDOM FORREST INSIDE OF INFO_GAIN_RATIO_RF FOLDER USING THIS CODE:-python -m traditional_rf.traditionalrf_titanic
# import random
# import pandas as pd
# import numpy as np
# import math
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# from r_q_str_corr.find_r_q_s_c import (
#     compute_r, compute_q, compute_strength, compute_correlation
# )
# from feature_weight_update.feature_weight_update import compute
# from treenum.treenum import (
#     compute_accuracy, compute_qu_qv, compute_nu, compute_l, compute_deltaB
# )
# from feature_ranking.feature_ranking import LocalGlobalWt
# from dataCleaning.data_cleaning import load_and_clean_titanic

# # --- Utilities: entropy, IG, IG_ratio ---
# def entropy(p):
#     if p == 0 or p == 1:
#         return 0.0
#     return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

# def information_gain(left_y, right_y):
#     parent = left_y + right_y
#     n = len(parent)
#     if n == 0:
#         return 0.0
#     p = parent.count(1) / n
#     H_p = entropy(p)
#     nl = len(left_y)
#     pl = left_y.count(1) / nl if nl>0 else 0.0
#     H_l = entropy(pl)
#     nr = len(right_y)
#     pr = right_y.count(1) / nr if nr>0 else 0.0
#     H_r = entropy(pr)
#     return H_p - (nl/n)*H_l - (nr/n)*H_r

# def information_gain_ratio(left_y, right_y):
#     ig = information_gain(left_y, right_y)
#     parent = left_y + right_y
#     n = len(parent)
#     if n == 0:
#         return 0.0
#     p = parent.count(1) / n
#     H_p = entropy(p)
#     return ig / H_p if H_p>0 else 0.0

# # --- Bootstrapping & OOB score ---
# def draw_bootstrap(X, y):
#     idxs = np.random.choice(len(X), len(X), replace=True)
#     oob = [i for i in range(len(X)) if i not in idxs]
#     return (
#         X.iloc[idxs].values, y.iloc[idxs].values,
#         X.iloc[oob].values,    y.iloc[oob].values
#     )

# def oob_score(tree, X_oob, y_oob):
#     mis = 0
#     for i, x in enumerate(X_oob):
#         if predict_tree(tree, x) != y_oob[i]:
#             mis += 1
#     return mis / len(y_oob)

# # --- Find best split by raw IG, store IG_ratio as quality ---
# def find_split_point(Xb, yb, max_features):
#     best_ig = -1.0
#     best_node = None
#     n_feat = Xb.shape[1]
#     feat_idxs = random.sample(range(n_feat), max_features)

#     for fi in feat_idxs:
#         for sp in np.unique(Xb[:,fi]):
#             mask = Xb[:,fi] <= sp
#             left_y  = yb[mask].tolist()
#             right_y = yb[~mask].tolist()
#             ig = information_gain(left_y, right_y)
#             if ig > best_ig:
#                 best_ig = ig
#                 best_node = {
#                   'feature_idx'     : fi,
#                   'split_point'     : sp,
#                   'quality_of_split': information_gain_ratio(left_y, right_y),
#                   'left_child'      : {
#                       'X_bootstrap': Xb[mask],
#                       'y_bootstrap': yb[mask]
#                   },
#                   'right_child'     : {
#                       'X_bootstrap': Xb[~mask],
#                       'y_bootstrap': yb[~mask]
#                   }
#                 }
#     return best_node

# # --- Build a single tree ---
# def terminal_node(node):
#     yb = node['y_bootstrap']
#     vals, counts = np.unique(yb, return_counts=True)
#     return vals[np.argmax(counts)]

# def split_node(node, f, min_samp, max_depth, depth):
#     left, right = node['left_child'], node['right_child']
#     del node['left_child'], node['right_child']

#     # base case: no data or depth limit
#     if (len(left['y_bootstrap']) == 0 or
#         len(right['y_bootstrap']) == 0 or
#         depth >= max_depth):
#         merged = np.concatenate((left['y_bootstrap'], right['y_bootstrap']))
#         vals, counts = np.unique(merged, return_counts=True)
#         node['left_split']  = vals[np.argmax(counts)]
#         node['right_split'] = node['left_split']
#         return

#     # left branch
#     if len(left['X_bootstrap']) <= min_samp:
#         node['left_split'] = terminal_node(left)
#     else:
#         node['left_split'] = find_split_point(
#             left['X_bootstrap'], left['y_bootstrap'], f
#         )
#         split_node(node['left_split'], f, min_samp, max_depth, depth+1)

#     # right branch
#     if len(right['X_bootstrap']) <= min_samp:
#         node['right_split'] = terminal_node(right)
#     else:
#         node['right_split'] = find_split_point(
#             right['X_bootstrap'], right['y_bootstrap'], f
#         )
#         split_node(node['right_split'], f, min_samp, max_depth, depth+1)

# def build_tree(Xb, yb, f, max_depth, min_samp):
#     root = find_split_point(Xb, yb, f)
#     split_node(root, f, min_samp, max_depth, 1)
#     return root

# # --- Random Forest with nav return ---
# def random_forest(X_train, y_train, n_estimators, f, max_depth, min_samp):
#     trees, oob_ls, local_wts = [], [], []
#     ranker = LocalGlobalWt(X_train.shape[1])

#     for _ in range(n_estimators):
#         Xb, yb, Xo, yo = draw_bootstrap(X_train, y_train)
#         tree = build_tree(Xb, yb, f, max_depth, min_samp)
#         trees.append(tree)
#         oob_ls.append(oob_score(tree, Xo, yo))
#         local_wts.append(ranker.find_local_weight_feature(tree))

#     norm_tree_wt = ranker.normalized_weight_of_tree(oob_ls)
#     total_splits = ranker.counter
#     nav = total_splits / len(trees)
#     return trees, local_wts, norm_tree_wt, nav

# # --- Prediction ---
# def predict_tree(tree, x):
#     fi = tree['feature_idx']; sp = tree['split_point']
#     branch = tree['left_split'] if x[fi] <= sp else tree['right_split']
#     return predict_tree(branch, x) if isinstance(branch, dict) else branch

# def predict_rf(trees, X):
#     preds = []
#     for xi in X.values:
#         votes = [predict_tree(t, xi) for t in trees]
#         preds.append(max(set(votes), key=votes.count))
#     return np.array(preds)

# # --- Main IRF loop on Titanic ---
# if __name__ == "__main__":
#     df = load_and_clean_titanic("titanic.csv")  
#     # Make sure load_and_clean_titanic() reads "titanic.csv" and returns a DataFrame.

#     # Encode the target
#     df['Survived'] = df['Survived'].astype(int)
#     label = 'Survived'

#     # Encode categorical features
#     df['Sex']      = df['Sex'].map({'male':0,'female':1}).astype(int)
#     df['Embarked'] = df['Embarked'].map({'C':0,'Q':1,'S':2}).astype(int)

#     f = int(math.sqrt(len(df.columns)-1))   # #features to try at each split
#     v = len(df.columns)-1                   # total features
#     n_estimators = 20

#     while v >= f:
#         X = df.drop(columns=[label])
#         y = df[label]
#         Xt, Xs, yt, ys = train_test_split(
#             X, y, test_size=0.3, random_state=42, stratify=y
#         )

#         trees, local_wts, norm_tree_wt, nav = random_forest(
#             Xt, yt, n_estimators, f, max_depth=6, min_samp=4
#         )

#         global_wt = LocalGlobalWt(Xt.shape[1]).global_wt(local_wts, norm_tree_wt)
#         updated, u, v, du, dv = compute(global_wt)

#         r    = compute_r(u, v, f)
#         q    = compute_q(u, v, f)
#         strength = compute_strength(q, nav, n_estimators)
#         corr, rho = compute_correlation(u, v, f, nav, n_estimators)
#         accuracy = compute_accuracy(strength, corr)

#         qu, qv = compute_qu_qv(u, v, f)
#         nu      = compute_nu(q, rho, nav, n_estimators)
#         l       = compute_l(q, nav, n_estimators)
#         deltaB  = compute_deltaB(qu, qv, du, dv, l, nu)

#         preds    = predict_rf(trees, Xs)
#         test_acc = (preds == ys.values).mean()
#         print(f"Test acc={test_acc:.4f}, ΔB={deltaB}")

#         # prune & prepare next iteration
#         n_estimators += deltaB
#         keep = set(updated.keys())
#         df = df[[c for i,c in enumerate(df.columns) if i in keep or c==label]].reset_index(drop=True)
#         print(f"Remaining features: {df.columns.tolist()}")
#         print();
#         print("f", f)
#         print("v", v)
#         print("del_b", deltaB)
#         print("n_estimators---------", n_estimators)
#         print("-------------///////////////////////////////////////////iteration end////////////////////////////////////------------------------------------")
#         print()



























# traditional_rf/traditionalrf_titanic.py
# traditional_rf/traditionalrf_titanic.py

import random
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, _tree

from r_q_str_corr.find_r_q_s_c import compute_r, compute_q, compute_strength, compute_correlation
from feature_weight_update.feature_weight_update import compute
from treenum.treenum import compute_accuracy, compute_qu_qv, compute_nu, compute_l, compute_deltaB
from feature_ranking.feature_ranking import LocalGlobalWt
from dataCleaning.data_cleaning import load_and_clean_titanic

# --- Utilities: entropy, info-gain & ratio ---
def entropy_from_counts(counts):
    probs = counts / counts.sum()
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def info_gain_and_ratio(tree, nid):
    T = tree.tree_
    parent = T.value[nid][0]
    H_p = entropy_from_counts(parent)
    left, right = T.children_left[nid], T.children_right[nid]
    if left == _tree.TREE_LEAF or right == _tree.TREE_LEAF or H_p == 0:
        return None, None
    lc, rc = T.value[left][0], T.value[right][0]
    Hl, Hr = entropy_from_counts(lc), entropy_from_counts(rc)
    n, nl, nr = sum(parent), sum(lc), sum(rc)
    ig = H_p - (nl/n)*Hl - (nr/n)*Hr
    igr = ig / H_p if H_p > 0 else 0.0
    return ig, igr

# --- Bootstrapping & OOB ---
def draw_bootstrap(X, y):
    idxs = np.random.choice(len(X), len(X), replace=True)
    oob  = [i for i in range(len(X)) if i not in idxs]
    return X.iloc[idxs], y.iloc[idxs], X.iloc[oob], y.iloc[oob]

def oob_score(tree, Xo, yo):
    if len(yo) == 0:
        return 1.0
    return np.mean(tree.predict(Xo) != yo)

# --- Random Forest using sklearn DT ---
def random_forest(X_train, y_train, B, f, max_depth, min_samples_leaf):
    trees, oob_ls, local_wts = [], [], []
    split_counts = []
    ranker = LocalGlobalWt(X_train.shape[1])

    for _ in range(B):
        Xb, yb, Xo, yo = draw_bootstrap(X_train, y_train)
        dt = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_features=f,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        dt.fit(Xb, yb)
        trees.append(dt)
        oob_ls.append(oob_score(dt, Xo, yo))

        # collect IGR per split
        splits = []
        for nid in range(dt.tree_.node_count):
            feat = dt.tree_.feature[nid]
            if feat < 0: 
                continue
            ig, igr = info_gain_and_ratio(dt, nid)
            if ig is not None:
                splits.append((feat, igr))
        split_counts.append(len(splits))
        local_wts.append(ranker.compute_local_from_list(splits))

    norm_tree_wt = ranker.normalized_weight_of_tree(oob_ls)
    nav = sum(split_counts) / len(trees)
    return trees, local_wts, norm_tree_wt, nav

def predict_rf(trees, X):
    all_preds = np.array([t.predict(X) for t in trees]).T
    return np.array([max(set(row), key=list(row).count) for row in all_preds])

# --- Main IRF Loop for Titanic dataset ---
if __name__ == "__main__":
    # 1) Load
    df = load_and_clean_titanic("titanic.csv")

    # 2) Drop unused text columns
    df = df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

    # 3) Fill missing numeric fields
    df["Age"]  = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # 4) Encode target and categoricals
    df["Survived"] = df["Survived"].astype(int)
    df["Sex"]      = df["Sex"].map({"male":0, "female":1}).astype(int)
    df["Embarked"] = df["Embarked"].map({"C":0, "Q":1, "S":2}).astype(int)

    label = "Survived"
    features = [c for c in df.columns if c != label]

    # IRF initial settings
    f = int(math.sqrt(len(features)))
    B = 20
    v= len(features)-f
    iteration = 0

    while v >= f:
        iteration += 1
        print(f"\n========== Iteration {iteration} ==========")
        print(f"Start features: {len(features)}, f={f}, B={B}")

        X = df[features]
        y = df[label]
        Xt, Xs, yt, ys = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train + weight computations
        trees, local_wts, norm_tree_wt, nav = random_forest(
            Xt, yt, B, f, max_depth=6, min_samples_leaf=4
        )
        global_wt_list = LocalGlobalWt(Xt.shape[1]).global_wt(local_wts, norm_tree_wt)
        global_wt = dict(zip(features, global_wt_list))

        # Prune & promote
        updated, u, v, du, dv, pruned = compute(global_wt)
        print(f"Promoted: {du}, Pruned: {dv} -> New v: {v}")
        print(f"Remaining Important: {u}, Unimportant: {v}, Total next: {u+v}")
        print(f"Updated features: {updated}\n")

        # IRF metrics
        r = compute_r(u, v, f)
        q = compute_q(u, v, f)
        strength = compute_strength(q, nav, B)
        corr, rho = compute_correlation(u, v, f, nav, B)
        print(f"Strength={strength:.4f}, Correlation={corr:.4f}")

        qu, qv = compute_qu_qv(u, v, f)
        nu = compute_nu(q, rho, nav, B)
        l_val = compute_l(q, nav, B)
        deltaB = compute_deltaB(qu, qv, du, dv, l_val, nu)
        print(f"DeltaB (trees to add): {deltaB}, Next B = {B+deltaB}")

        # Test accuracy
        preds = predict_rf(trees, Xs)
        acc = (preds == ys.values).mean()
        print(f"Test-set accuracy: {acc:.4f}")

        # Apply updates
        B += deltaB
        features = updated
        df = df[features + [label]].reset_index(drop=True)

        print(f"----- End of Iteration {iteration} -----")

    print("\nIRF complete.")
