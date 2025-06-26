# # RUN THIS RANDOM FORREST INSIDE OF INFO_GAIN_RATIO_RF FOLDER USING THIS CODE:
# #    python -m traditional_rf.traditionalrf

# import random
# import pandas as pd
# import numpy as np
# import math
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, _tree

# from r_q_str_corr.find_r_q_s_c import compute_r, compute_q, compute_strength, compute_correlation
# from feature_weight_update.feature_weight_update import compute
# from treenum.treenum import compute_accuracy, compute_qu_qv, compute_nu, compute_l, compute_deltaB
# from feature_ranking.feature_ranking import LocalGlobalWt, local_feature_weights
# from dataCleaning.data_cleaning import load_and_clean_titanic

# # --- Utilities: compute entropy from class counts ---
# def entropy_from_counts(counts):
#     probs = counts / counts.sum()
#     return -np.sum([p * np.log2(p) for p in probs if p > 0])

# # --- Compute IG and IGR = IG / H_parent for a sklearn tree node ---
# def info_gain_and_ratio(tree, nid):
#     T = tree.tree_
#     parent_counts = T.value[nid][0]
#     H_p = entropy_from_counts(parent_counts)

#     left, right = T.children_left[nid], T.children_right[nid]
#     if left == _tree.TREE_LEAF or right == _tree.TREE_LEAF or H_p == 0:
#         return None, None

#     lc = T.value[left][0]
#     rc = T.value[right][0]
#     Hl = entropy_from_counts(lc)
#     Hr = entropy_from_counts(rc)

#     n, nl, nr = sum(parent_counts), sum(lc), sum(rc)
#     ig = H_p - (nl/n)*Hl - (nr/n)*Hr
#     igr = ig / H_p if H_p > 0 else 0.0
#     return ig, igr

# # --- Bootstrapping & OOB error ---
# def draw_bootstrap(X, y):
#     idxs = np.random.choice(len(X), len(X), replace=True)
#     oob  = [i for i in range(len(X)) if i not in idxs]
#     return X.iloc[idxs], y.iloc[idxs], X.iloc[oob], y.iloc[oob]

# def oob_score(tree, X_oob, y_oob):
#     if len(y_oob) == 0:
#         return 1.0
#     preds = tree.predict(X_oob)
#     return np.mean(preds != y_oob)

# # --- Random Forest using sklearn DecisionTreeClassifier ---
# def random_forest(X_train, y_train, n_estimators, f, max_depth, min_samp):
#     trees, oob_ls, local_wts = [], [], []
#     splits_counts = []
#     ranker = LocalGlobalWt(X_train.shape[1])

#     for _ in range(n_estimators):
#         Xb, yb, Xo, yo = draw_bootstrap(X_train, y_train)

#         # Fit tree
#         dt = DecisionTreeClassifier(
#             criterion='entropy',
#             splitter='best',
#             max_features=f,
#             max_depth=max_depth,
#             min_samples_leaf=min_samp,
#             random_state=None
#         )
#         dt.fit(Xb, yb)
#         trees.append(dt)

#         # OOB error
#         oob_ls.append(oob_score(dt, Xo, yo))

#         # Extract IG and IGR per internal node
#         splits = []
#         T = dt.tree_
#         for nid in range(T.node_count):
#             feat = T.feature[nid]
#             if feat < 0:
#                 continue
#             ig, igr = info_gain_and_ratio(dt, nid)
#             if ig is not None:
#                 splits.append((feat, igr))

#         # Record split counts and compute local weights (avg IGR per feature)
#         splits_counts.append(len(splits))
#         local_wts.append(local_feature_weights(splits))

#     # Normalize tree weights and compute avg splits per tree
#     norm_tree_wt = ranker.normalized_weight_of_tree(oob_ls)
#     nav = sum(splits_counts) / len(trees)

#     return trees, local_wts, norm_tree_wt, nav

# # --- Prediction by majority vote ---
# def predict_rf(trees, X):
#     preds = []
#     for _, row in X.iterrows():
#         votes = [t.predict(row.values.reshape(1, -1))[0] for t in trees]
#         preds.append(max(set(votes), key=votes.count))
#     return np.array(preds)

# # --- Main IRF loop ---
# if __name__ == "__main__":
#     df    = load_and_clean_titanic("breast-cancer.csv")
#     label = "diagnosis"

#     f            = int(math.sqrt(len(df.columns) - 1))
#     v            = len(df.columns) - 1
#     n_estimators = 20

#     while v >= f:
#         X = df.drop(columns=[label])
#         y = pd.Series(
#             LabelEncoder().fit_transform(df[label]),
#             name=label
#         )
#         Xt, Xs, yt, ys = train_test_split(
#             X, y, test_size=0.3,
#             random_state=42, stratify=y
#         )

#         # Train forest and get weights
#         trees, local_wts, norm_tree_wt, nav = random_forest(
#             Xt, yt, n_estimators, f, max_depth=6, min_samp=4
#         )
#         global_wt = LocalGlobalWt(Xt.shape[1]).global_wt(local_wts, norm_tree_wt)

#         # IRF metrics unchanged
#         updated, u, v, du, dv = compute(global_wt)
#         r  = compute_r(u, v, f)
#         q  = compute_q(u, v, f)
#         strength = compute_strength(q, nav, n_estimators)
#         corr, rho = compute_correlation(u, v, f, nav, n_estimators)
#         accuracy  = compute_accuracy(strength, corr)
#         qu, qv    = compute_qu_qv(u, v, f)
#         nu        = compute_nu(q, rho, nav, n_estimators)
#         l_val     = compute_l(q, nav, n_estimators)
#         deltaB    = compute_deltaB(qu, qv, du, dv, l_val, nu)

#         # Evaluate on hold‐out
#         preds    = predict_rf(trees, Xs)
#         test_acc = (preds == ys.values).mean()
#         print(f"Test acc={test_acc:.4f}, ΔB={deltaB}")

#         # Prepare next iteration
#         n_estimators += deltaB
#         df = df[[c for i, c in enumerate(df.columns) if i in updated or c == label]]
#         df.reset_index(drop=True, inplace=True)

#         print(f"Remaining features: {df.columns.tolist()}")
#         print(f"f={f}, v={v}, ΔB={deltaB}, new B={n_estimators}")
#         print("— iteration end —\n")

# RUN THIS RANDOM FOREST INSIDE OF INFO_GAIN_RATIO_RF:
#    python -m traditional_rf.traditionalrf

# RUN THIS RANDOM FOREST INSIDE OF INFO_GAIN_RATIO_RF:
#    python -m traditional_rf.traditionalrf

# RUN THIS RANDOM FOREST INSIDE OF INFO_GAIN_RATIO_RF FOLDER
#   python -m traditional_rf.traditionalrf








# traditional_rf/traditionalrf.py
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

# --- Utilities ---
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

def draw_bootstrap(X, y):
    idxs = np.random.choice(len(X), len(X), replace=True)
    oob = [i for i in range(len(X)) if i not in idxs]
    return X.iloc[idxs], y.iloc[idxs], X.iloc[oob], y.iloc[oob]

def oob_score(tree, X_oob, y_oob):
    if len(y_oob) == 0:
        return 1.0
    return np.mean(tree.predict(X_oob) != y_oob)

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
    preds = np.array([t.predict(X) for t in trees]).T
    return np.array([max(set(row), key=list(row).count) for row in preds])

# --- Main IRF Loop ---
if __name__ == "__main__":
    df = load_and_clean_titanic("breast-cancer.csv")
    label = "diagnosis"

    f = int(math.sqrt(len(df.columns) - 1))
    features = [c for c in df.columns if c != label]
    B = 20
    iteration = 0
    v = len(df.columns) - f

    while v >= f:
        iteration += 1
        print(f"\n========== Iteration {iteration} ==========")
        print(f"Start features: {len(features)}, f={f}, B={B}")

        X = df[features]
        y = pd.Series(LabelEncoder().fit_transform(df[label]), name=label)
        Xt, Xs, yt, ys = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        trees, local_wts, norm_tree_wt, nav = random_forest(
            Xt, yt, B, f, max_depth=6, min_samples_leaf=4
        )
        # compute global weights
        global_wt_list = LocalGlobalWt(Xt.shape[1]).global_wt(local_wts, norm_tree_wt)
        global_wt = dict(zip(features, global_wt_list))

        # feature update
        updated, u, v, du, dv, pruned = compute(global_wt)
        print(f"Promoted: {du}, Pruned: {dv} -> New v: {v}")
        print(f"Remaining Important: {u}, Unimportant: {v}, Total next: {u+v}")
        print()
        print(f"updated : {updated}")
        print()

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

        # test accuracy
        preds = predict_rf(trees, Xs)
        acc = (preds == ys.values).mean()
        print(f"Test-set accuracy: {acc:.4f}")

        # apply updates
        B += deltaB
        features = updated
        df = df[features + [label]].reset_index(drop=True)

        print(f"----- End of Iteration {iteration} -----")

    print("\nIRF complete.")
