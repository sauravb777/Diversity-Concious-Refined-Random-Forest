# from collections import defaultdict

# class LocalGlobalWt:
#     def __init__(self, number_of_features):
#         self.feature_quality_list = []
#         self.N = 0
#         self.number_of_features = number_of_features
#         self.counter = 0

#     def collect_features(self, tree):
#         if isinstance(tree, dict) and "feature_idx" in tree and "quality_of_split" in tree:
#             self.counter += 1
#             self.feature_quality_list.append((tree["feature_idx"], tree["quality_of_split"]))
#             # dive into children if they’re sub-trees
#             left = tree.get("left_split")
#             right = tree.get("right_split")
#             if isinstance(left, dict):
#                 self.collect_features(left)
#             if isinstance(right, dict):
#                 self.collect_features(right)

#     def find_local_weight_feature(self, tree):
#         # reset for this tree
#         self.feature_quality_list.clear()
#         self.counter = 0

#         # collect (feat, quality) pairs
#         self.collect_features(tree)

#         # group by feature index
#         feature_to_qualities = defaultdict(list)
#         for feat_idx, quality in self.feature_quality_list:
#             feature_to_qualities[feat_idx].append(quality)

#         # ensure unused features get a zero entry
#         for feat in range(self.number_of_features):
#             feature_to_qualities.setdefault(feat, [0.0])

#         # total splits (avoid div by zero)
#         self.N = max(len(self.feature_quality_list), 1)

#         # compute local weights = sum(qualities)/N
#         return {
#             feat: sum(vals) / self.N
#             for feat, vals in feature_to_qualities.items()
#         }

#     def normalized_weight_of_tree(self, oob_list):
#         inv = [1.0 / err if err > 0 else 0.0 for err in oob_list]
#         top = max(inv) or 1.0
#         return [v / top for v in inv]

#     def global_wt(self, local_wts, tree_wts):
#         agg = {}
#         for t, lw in enumerate(local_wts):
#             gamma = tree_wts[t]
#             for feat, w in lw.items():
#                 agg[feat] = agg.get(feat, 0.0) + w * gamma

#         # normalize across features
#         max_val = max(agg.values()) or 1.0
#         return {feat: val / max_val for feat, val in agg.items()}


from collections import defaultdict

class LocalGlobalWt:
    def __init__(self, number_of_features):
        self.feature_quality_list = []
        self.N = 0
        self.number_of_features = number_of_features
        self.counter = 0

    def collect_features(self, tree):
        """
        Recursively traverse a built tree (dict) collecting
        (feature_idx, quality_of_split) at every split-node.
        """
        if isinstance(tree, dict) and "feature_idx" in tree and "quality_of_split" in tree:
            self.counter += 1
            self.feature_quality_list.append((tree["feature_idx"], tree["quality_of_split"]))
            left = tree.get("left_split")
            right = tree.get("right_split")
            if isinstance(left, dict):
                self.collect_features(left)
            if isinstance(right, dict):
                self.collect_features(right)

    def find_local_weight_feature(self, tree):
        """
        For one tree: gather all split qualities, then
        return a dict {feature_idx: avg_split_quality}.
        """
        self.feature_quality_list.clear()
        self.counter = 0
        self.collect_features(tree)

        feature_to_qualities = defaultdict(list)
        for feat_idx, quality in self.feature_quality_list:
            feature_to_qualities[feat_idx].append(quality)
        for feat in range(self.number_of_features):
            feature_to_qualities.setdefault(feat, [0.0])

        self.N = max(len(self.feature_quality_list), 1)
        return {
            feat: sum(vals) / self.N
            for feat, vals in feature_to_qualities.items()
        }

    def compute_local_from_list(self, split_list):
        """
        Given a list of (feature_idx, igr) pairs, compute local weights
        as avg(igr) per feature, filling unused features with 0.
        """
        feats = defaultdict(list)
        for feat_idx, igr in split_list:
            feats[feat_idx].append(igr)
        for feat in range(self.number_of_features):
            feats.setdefault(feat, [0.0])
        total = sum(len(v) for v in feats.values()) or 1
        return {feat: sum(vals) / total for feat, vals in feats.items()}

    def normalized_weight_of_tree(self, oob_list):
        inv = [1.0 / err if err > 0 else 0.0 for err in oob_list]
        top = max(inv) or 1.0
        return [v / top for v in inv]

    def global_wt(self, local_wts, tree_wts):
        agg = {}
        for t, lw in enumerate(local_wts):
            gamma = tree_wts[t]
            for feat, w in lw.items():
                agg[feat] = agg.get(feat, 0.0) + w * gamma

        max_val = max(agg.values()) or 1.0
        return {feat: val / max_val for feat, val in agg.items()}

    def compute_pruned_and_updated(self, global_weights, current_features, threshold):

        pruned_feats = [feat for feat, score in global_weights.items() if score <= threshold]
        updated_features = [feat for feat in current_features if feat not in pruned_feats]

        u = len(updated_features)
        v = len(pruned_feats)
        du = u - len(current_features)  # promoted features count
        dv = len(pruned_feats)          # pruned features count

        return updated_features, u, v, du, dv, pruned_feats

