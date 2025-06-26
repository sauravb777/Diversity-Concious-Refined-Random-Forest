import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, _tree

from r_q_str_corr.find_r_q_s_c import compute_r, compute_q, compute_strength, compute_correlation
from feature_weight_update.feature_weight_update import compute
from treenum.treenum import compute_accuracy, compute_qu_qv, compute_nu, compute_l, compute_deltaB
from feature_ranking.feature_ranking import LocalGlobalWt

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


def fit(df:pd.DataFrame,features:list,label:str,f:int,B:int,v:int,iteration:int=0,max_depth=6,min_samples_leaf=4):
    from traditional_rf.erf import save_to_json
    accuracy_list = []
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
            Xt, yt, B, f, max_depth=max_depth, min_samples_leaf=min_samples_leaf
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
        accuracy_list.append({"accuracy": acc, "B": B})

        print(f"Test-set accuracy: {acc:.4f}")

        # apply updates
        B += deltaB
        features = updated
        df = df[features + [label]].reset_index(drop=True)

        print(f"----- End of Iteration {iteration} -----")
    save_to_json("irf_breast_cancer.json",accuracy_list)
    return trees



