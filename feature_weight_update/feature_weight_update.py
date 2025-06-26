# import math, numpy as np

# important_features = {}
# unimportant_features = {}
# first_time = True
# del_u = del_v = 0

# def compute(global_list):
#     global important_features, unimportant_features, first_time, del_u, del_v
#     if first_time:
#         sorted_feats = dict(sorted(global_list.items(), key=lambda x: x[1], reverse=True))
#         u0 = math.isqrt(len(sorted_feats))
#         items = list(sorted_feats.items())
#         important_features = dict(items[:u0])
#         unimportant_features = dict(items[u0:])
#         first_time = False
#     else:
#         for k,v in global_list.items():
#             if k in important_features:
#                 important_features[k] = v
#             elif k in unimportant_features:
#                 unimportant_features[k] = v

#     before_u = len(important_features)
#     before_v = len(unimportant_features)

#     # remove low
#     vals = np.array(list(unimportant_features.values()))
#     μ, σ = vals.mean(), vals.std()
#     thr = μ - 2*σ
#     print("length of Unimp feature Before Pruning--------------", len(unimportant_features))
#     print("unnnnnnnnnnnnnnnnnnnn", unimportant_features)
#     print()
#     print("This was the threshold value", thr)
#     print()
#     unimportant_features = {k:v for k,v in unimportant_features.items() if v>thr}
#     if len(unimportant_features)==before_v:
#         worst = min(unimportant_features, key=unimportant_features.get)
#         del unimportant_features[worst]
#         print("This RAN Instead")
        
#     print("length of Unimp feature After Pruning---------------", len(unimportant_features))
#     print("unnnnnnnnnnnnnnnnnnnn", unimportant_features)
#     print()

#     # promote
#     if important_features:
#         min_imp = min(important_features.values())
#         to_prom = [k for k,v in unimportant_features.items() if v>=min_imp]
#         for k in to_prom:
#             important_features[k] = unimportant_features.pop(k)

#     del_u = len(important_features) - before_u
#     del_v = len(unimportant_features) - before_v
#     merged = {**important_features, **unimportant_features}
    
#     return merged, len(important_features), len(unimportant_features), del_u, del_v












# feature_weight_update/feature_weight_update.py
import math
import numpy as np

# These persist across iterations
important_features = {}
unimportant_features = {}
first_time = True

def compute(global_weights):
    global important_features, unimportant_features, first_time

    # On first pass, split by top sqrt rule
    if first_time:
        sorted_feats = sorted(global_weights.items(), key=lambda x: x[1], reverse=True)
        u0 = int(math.sqrt(len(sorted_feats)))
        important_features = {k: v for k, v in sorted_feats[:u0]}
        unimportant_features = {k: v for k, v in sorted_feats[u0:]}
        first_time = False
    else:
        # Update existing buckets
        for f, w in global_weights.items():
            if f in important_features:
                important_features[f] = w
            elif f in unimportant_features:
                unimportant_features[f] = w

    before_u = len(important_features)
    before_v = len(unimportant_features)

    # Prune: anything ≤ (mean − 2σ)
    vals = np.array(list(unimportant_features.values()))
    if len(vals) > 0:
        μ, σ = vals.mean(), vals.std()
        threshold = μ - 2*σ
        pruned = [f for f, w in unimportant_features.items() if w <= threshold]
        # If none pruned, remove worst
        if not pruned:
            worst = min(unimportant_features, key=unimportant_features.get)
            pruned = [worst]
            print("this ran insted")
            print()
    else:
        pruned = []

    # Apply pruning
    for f in pruned:
        unimportant_features.pop(f, None)

    # Promote those whose weight ≥ min important
    if important_features:
        min_imp = min(important_features.values())
        to_promote = [f for f, w in unimportant_features.items() if w >= min_imp]
        print(f"how many getting prmoted in this loop? {len(to_promote)}")
        print()
        print(to_promote)
        print()
        for f in to_promote:
            important_features[f] = unimportant_features.pop(f)

    # Deltas
    du = len(important_features) - before_u
    dv = len(pruned)

    # Build updated feature list
    updated_features = list(important_features.keys()) + list(unimportant_features.keys())

    return updated_features, len(important_features), len(unimportant_features), du, dv, pruned
