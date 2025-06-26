import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from traditional_rf import irf
from dataCleaning.data_cleaning import TitanicDataCleaner,BreastCancerLoader,AdultDataLoader,DiabetesLoader
import pandas as pd
import math
import os
import json
from tqdm import tqdm
def save_to_json(file_name, data):
    # Check if the file exists
    if os.path.exists(file_name):
        # If file exists, read the current content and append new data
        with open(file_name, 'r') as file:
            existing_data = json.load(file)
        existing_data.append(data)  # Append the new data to the existing data
    else:
        # If file doesn't exist, start with an empty list and add the data
        existing_data = [data]
    
    # Save the updated data back to the file
    with open(file_name, 'w') as file:
        json.dump(existing_data, file, indent=4)
import sys
class EnhancedRandomForest:
    def __init__(self, n_estimators=100, correlation_threshold=0.9):
        self.n_estimators = n_estimators
        self.threshold = correlation_threshold
        # self.random_state = random_state
        # self.rf = RandomForestClassifier(n_estimators=n_estimators)
        self.selected_trees = []
        self.selected_indices = []
        self.auc_final_score = []

   

    def fit(self,df:pd.DataFrame,features:list,label:str,f:int,B:int,v:int,X_val:pd.DataFrame=None,y_val:pd.Series=None,ds_name=None):
     
        if X_val is None or y_val is None:
            # Get 30% of the data randomly
            val_df = df.sample(frac=0.3, random_state=42)
            y_val = val_df[label]
            df = df.drop(val_df.index)
            
        trees = irf.fit(df, features, label, f, B,v)
        # sys.exit()
        final_features = trees[0].feature_names_in_
        if X_val is None or y_val is None:
            print("---------here-----------")
            X_val = val_df[final_features]
        else:
            print("---------here-----------")

            X_val = X_val[final_features]

        total_irf_trees = len(trees)
        print(f"Trained {total_irf_trees} trees using improved random forest.")
        tree_probs = []
        auc_scores = []
        for tree in trees:
            prob = tree.predict_proba(X_val)[:, 1]
            tree_probs.append(prob)
            auc_scores.append(roc_auc_score(y_val, prob))

        tree_probs = np.column_stack(tree_probs)
        correlation_matrix = np.corrcoef(tree_probs, rowvar=False)

        if save_headmap:
            plt.figure(figsize=(30, 28))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            # plt.title(f'Correlated Matrix Heatmap {ds_name}')
            plt.title(f'RF With Correlated Trees')

            # Save the heatmap
            # plt.savefig(f'{ds_name}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'heatmap_before.eps', dpi=30, bbox_inches='tight')
            print(f"Saved heatmap before removing correlated trees.")


        # Build adjacency list based on correlation threshold
        n_trees = correlation_matrix.shape[0]
        adjacency = [[] for _ in range(n_trees)]
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                if correlation_matrix[i, j] >= self.threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)


        # Find connected components (clusters)
        def dfs(node, visited, component):
            visited[node] = True
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    dfs(neighbor, visited, component)

        visited = [False] * n_trees
        components = []
        for i in range(n_trees):
            if not visited[i]:
                comp = []
                dfs(i, visited, comp)
                components.append(comp)

        # Select best AUC tree from each cluster
        self.selected_indices = []
        for component in components:
            print(component)
            best_idx = max(component, key=lambda idx: auc_scores[idx])
            self.selected_indices.append(best_idx)

        # Save selected trees
        self.selected_trees = [trees[i] for i in self.selected_indices]


        erf_auc_scores = []
        tree_probs = []
        for tree in self.selected_trees:
            prob = tree.predict_proba(X_val)[:, 1]
            tree_probs.append(prob)
            erf_auc_scores.append(roc_auc_score(y_val, prob))

        tree_probs = np.column_stack(tree_probs)
        correlation_matrix = np.corrcoef(tree_probs, rowvar=False)

        if save_headmap:
            plt.figure(figsize=(30, 28))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            # plt.title(f'Correlated Matrix Heatmap {ds_name}')
            plt.title(f'RRF Without Correlated Trees')

            # Save the heatmap
            # plt.savefig(f'{ds_name}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'heatmap_after.eps', dpi=30, bbox_inches='tight')
            print(f"Saved heatmap after removing correlated trees.")

        ###############Random Forest #################
        # rf_auc = []
        # for i in [60,80,100,120,140]:
        #     sk_rf_model = RandomForestClassifier(n_estimators=len(self.selected_trees))
        #     sk_rf_model.fit(df[final_features], df[label])
        #     y_probs = sk_rf_model.predict_proba(val_df[final_features])[:, 1]
        #     rf_auc_score = roc_auc_score(y_val, y_probs)
        #     rf_auc.append((i,rf_auc_score))
        #     rf_auc_scores = []
        #     for tree in sk_rf_model.estimators_:
        #         prob = tree.predict_proba(X_val)[:, 1]
        #         roc_auc = roc_auc_score(y_val, prob)
        #         rf_auc_scores.append(roc_auc)
        #         # rf_auc.append((i,roc_auc))

        #     rf_auc_score = sum(rf_auc_scores)/len(sk_rf_model.estimators_)


        print(f"-----------IRF------------- {sum(auc_scores)/len(trees)}")
        print(f"Selected {len(self.selected_trees)} trees for Enhanced RF.")
        print(f"------------ERF------------ {sum(erf_auc_scores)/len(self.selected_trees)}")
        irf_acc = sum(auc_scores)/len(trees)
        erf_acc = sum(erf_auc_scores)/len(self.selected_trees)
        total_erf_trees = len(self.selected_trees)
        data_to_save = {
            "total_erf_trees": total_erf_trees,
            "total_irf_trees":total_irf_trees,
            "irf_auc": irf_acc,
            "erf_auc": erf_acc,
            # "rf_auc_score": rf_auc_score,
            # "rf_ntrees":rf_auc
        }
        # save_to_json('tester2.json', data_to_save)


    def predict_proba(self, X):
        probs = np.array([tree.predict_proba(X)[:, 1] for tree in self.selected_trees])
        avg_probs = np.mean(probs, axis=0)
        return np.column_stack([1 - avg_probs, avg_probs])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    

if __name__ == "__main__":
    # Usage of the class
    
    import pandas as pd
    import random
    save_headmap = True


    # correlation_thresholds = [0.91,0.92,0.93,0.94]



    loader = BreastCancerLoader()
    ddf = loader.get_dataframe()
    label="target"
    correlation_threshold = 0.91
    ds_name = "breast_cancer"
    print("Working on: ",ds_name)

    # cleaner = TitanicDataCleaner()
    # ddf = cleaner.clean_data()
    # label = "survived"
    # correlation_threshold = 0.92
    # ds_name = "Titanic"


    # loader = AdultDataLoader()
    # ddf = loader.get_dataframe()
    # label="label"
    # correlation_threshold = 0.90
    # ds_name = "Adult Income"


    # loader = DiabetesLoader()
    # ddf = loader.get_dataframe()
    # label = "class"
    # correlation_threshold = 0.77
    # ds_name = "Diabetes"


    # preprocessor = PimaPreprocessor(scale_features=True)  # or False
    # label = "class"





    # print(df.columns)
    # print(df['label'].value_counts())

    for _ in tqdm(range(1)):

        f = int(math.sqrt(len(ddf.columns) - 1))
        features = [c for c in ddf.columns if c != label]
        B = 28
        iteration = 0
        v = len(ddf.columns) - f
        erf = EnhancedRandomForest(n_estimators=30, correlation_threshold=correlation_threshold)
        erf.fit(ddf, features, label, f, B,v,ds_name=ds_name)

    