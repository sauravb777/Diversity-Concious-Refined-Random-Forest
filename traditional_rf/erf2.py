import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from traditional_rf import irf
from dataCleaning.data_cleaning import IrisPreprocessor,MNISTLoader,WineDataLoader,LetterDataLoader,YeastDataLoader,OptDigitsDataLoader,MiceProteinLoader,CovertypeLoader,StatlogImageSegmentation,DataLoader
import pandas as pd
import math
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
def create_head_map(probs_erf,probs_rf, acc_file):
    probs_erf = np.array(probs_erf)
    probs_erf = probs_erf.reshape(probs_erf.shape[0], -1)  # shape: (3, 48)
    cor_mat_erf = np.corrcoef(probs_erf)

    probs_erf = np.array(probs_erf)
    probs_erf = probs_erf.reshape(probs_erf.shape[0], -1)  # shape: (3, 48)
    cor_mat_rf = np.corrcoef(probs_erf)

    plt.figure(figsize=(30, 28))
    sns.heatmap(cor_mat_erf, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix Heatmap {acc_file} RRF')
    plt.savefig(f'{acc_file}_correlation_heatmap_after.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(30, 28))
    sns.heatmap(cor_mat_rf, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix Heatmap {acc_file} RF')
    plt.savefig(f'{acc_file}_correlation_heatmap_before.png', dpi=300, bbox_inches='tight')

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

class EnhancedRandomForest:
    def __init__(self, n_estimators=100, correlation_threshold=0.9):
        self.n_estimators = n_estimators
        self.threshold = correlation_threshold
        self.selected_trees = []
        self.selected_indices = []
        self.auc_final_score = []

   

    def fit(self,df:pd.DataFrame,features:list,label:str,f:int,B:int,v:int,X_val:pd.DataFrame=None,y_val:pd.Series=None,max_depth=6,min_samples_leaf=4,acc_file:str= None):
        if not acc_file:
            raise "Enter file name to save acc."
     
        if X_val is None or y_val is None:
            val_df = df.sample(frac=0.3) #random_state=42
            y_val = val_df[label]
            df = df.drop(val_df.index)
            
      

        trees = irf.fit(df, features, label, f, B,v,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
        # sys.exit()
        final_features = trees[0].feature_names_in_
        final_feature_count = len(final_features)
        feature_count = len(features)

        if X_val is None or y_val is None:
            X_val = val_df[final_features]
        else:
            X_val = X_val[final_features]

        total_irf_trees = len(trees)
        print(f"Trained {total_irf_trees} trees using improved random forest.")
        tree_probs = []
        auc_scores = []
        for tree in trees:
            prob = tree.predict_proba(X_val)#[:, 1]
            tree_probs.append(prob) 
            auc_scores.append(roc_auc_score(y_val, prob,average='macro', multi_class='ovr'))
        tree_probs = np.array(tree_probs)
        flat_data = tree_probs.reshape(tree_probs.shape[0], -1)  # shape: (3, 48)
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(flat_data)
        # if save_headmap:
        #     plt.figure(figsize=(30, 28))
        #     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        #     plt.title(f'Correlation Matrix Heatmap {acc_file} Before')
        #     # Save the heatmap
        #     plt.savefig(f'{acc_file}_correlation_heatmap.png', dpi=300, bbox_inches='tight')

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
        # print(components)
        # print(auc_scores)
        for component in components:
            best_idx = max(component, key=lambda idx: auc_scores[idx])
            self.selected_indices.append(best_idx)

        # Save selected trees
        self.selected_trees = [trees[i] for i in self.selected_indices]



        erf_auc_scores = []
        erf_probs = []
        for tree in self.selected_trees:
            prob = tree.predict_proba(X_val)#[:, 1]
            erf_probs.append(prob)
            erf_auc_scores.append(roc_auc_score(y_val, prob,average='macro', multi_class='ovr'))

        # ----------------------------------------------------------------------------------------------------

        sk_rf_model = RandomForestClassifier(n_estimators=len(self.selected_trees))
        sk_rf_model.fit(df[features], df[label])
        # y_probs = sk_rf_model.predict_proba(val_df[final_features])[:, 1]
        # rf_auc_score = roc_auc_score(y_val, y_probs)
        rf_auc_scores = []
        rf_probs = []
        for tree in sk_rf_model.estimators_:
            prob = tree.predict_proba(val_df[features])#[:, 1]
            rf_probs.append(prob)
            rf_auc_scores.append(roc_auc_score(y_val, prob,average='macro', multi_class='ovr'))
        rf_auc_score = sum(rf_auc_scores)/len(sk_rf_model.estimators_)
        if save_headmap:
            erf_probs = np.array(erf_probs)
            rf_probs = np.array(rf_probs)
            create_head_map(erf_probs,rf_probs,acc_file)

        print(f"-----------IRF------------- {sum(auc_scores)/len(trees)}")
        print(f"Selected {len(self.selected_trees)} trees for Enhanced RF.")
        print(f"------------ERF------------ {sum(erf_auc_scores)/len(self.selected_trees)}")
        irf_acc = sum(auc_scores)/len(trees)
        erf_acc = sum(erf_auc_scores)/len(self.selected_trees)
        total_erf_trees = len(self.selected_trees)
        # if erf_acc>rf_auc_score:
        data_to_save = {
            "total_erf_trees": total_erf_trees,
            "total_irf_trees":total_irf_trees,
            "irf_auc": irf_acc,
            "erf_auc": erf_acc,
            "rf_auc_score": rf_auc_score,
            "final_feature_count":final_feature_count,
            "feature_count":feature_count
        }
        # save_to_json(f'{acc_file}.json', data_to_save)
        save_to_json(f'tester2.json', data_to_save)



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
    save_headmap = True

    # preprocessor = IrisPreprocessor(scale_features=True)  # or False
    # ddf = preprocessor.preprocess()
    # label = "species_encoded"
    # acc_file = 'iris'
    # correlation_threshold = 0.999


    loader = MNISTLoader()
    ddf = loader.get_dataframe()
    label = "class"
    acc_file = 'mnist'
    correlation_threshold=0.86

    # loader = WineDataLoader()
    # ddf = loader.get_dataframe()
    # label = "label"
    # acc_file = 'wine'
    # correlation_threshold=0.95


    # loader = LetterDataLoader()
    # ddf = loader.get_dataframe()
    # label = "label"
    # acc_file = 'Letter'
    # correlation_threshold=0.82
    
    
    # loader = OptDigitsDataLoader()
    # ddf = loader.get_dataframe()
    # label = "label"
    # acc_file = 'OptDigits'
    # correlation_threshold=0.73


    # loader = CovertypeLoader()
    # ddf = loader.get_dataframe()
    # label = "class"
    # acc_file = 'forest_cover_type'
    # correlation_threshold=0.93
    

    # loader = StatlogImageSegmentation()
    # ddf = loader.get_dataframe()
    # label = "class" Not used
    # acc_file = 'statlog_image_segmentation'
    # correlation_threshold=0.93

    # loader = AutoUnivLoader()
    # loader = DataLoader(182)
    # ddf = loader.get_dataframe()
    # label = "label"
    # acc_file = 'satimage'
    # correlation_threshold=0.89
    # {"satimage":"182"}

    print(ddf.columns)
    print(ddf[label].value_counts())



    for _ in tqdm(range(1)):
        f = int(math.sqrt(len(ddf.columns) - 1))
        features = [c for c in ddf.columns if c != label]
        B = 28
        iteration = 0
        v = len(ddf.columns) - f
        erf = EnhancedRandomForest(n_estimators=30, correlation_threshold=correlation_threshold)
        erf.fit(ddf, features, label, f, B,v,acc_file=acc_file)

    