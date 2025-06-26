from collections import defaultdict
import os
from traditional_rf.erf2 import save_to_json
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset_files = {
    "Titanic": "titanic.json",
    "Breast Cancer": "brest_cancer.json",
    "Diabetes": "diabetes2.json",
    "Adult Income":"adult_income.json",

    # "Statlog \n Image Segmentation":"statlog_image_segmentation.json",
    "MNIST": "mnist.json",
    "Letter": "letter.json",
    # "Forest Cover Type":"forest_cover_type.json",
    "OptDigits":"OptDigits.json",
    "sat Image": "satimage.json",

    # "Auto Univ":"auto_univ.json",
    # "amazon \n commerce_reviews":"amazon_commerce_reviews.json"
    
}

# for dataset_name, filename in dataset_files.items():
#     with open(filename, 'r') as f:
#         data = json.load(f)

#     grouped = defaultdict(lambda: {'erf_auc': [], 'rf_auc_score': []})

#     for entry in data:
#         n_trees = entry["total_erf_trees"]
#         if n_trees > 100:
#             continue
#         grouped[n_trees]['erf_auc'].append(entry["erf_auc"])
#         grouped[n_trees]['rf_auc_score'].append(entry["rf_auc_score"])

#     # Prepare plot data
#     num_trees = []
#     accuracy_custom = []
#     accuracy_sklearn = []

#     for n in sorted(grouped):
#         num_trees.append(n)
#         accuracy_custom.append(sum(grouped[n]['erf_auc']) / len(grouped[n]['erf_auc']))
#         accuracy_sklearn.append(sum(grouped[n]['rf_auc_score']) / len(grouped[n]['rf_auc_score']))

#     # Plot line chart
#     plt.figure(figsize=(10, 6))
#     plt.plot(num_trees, accuracy_custom, marker='o', label='Custom Random Forest (AUC)')
#     plt.plot(num_trees, accuracy_sklearn, marker='s', label='Sklearn Random Forest (AUC)')
#     plt.title(f'AUC vs. Number of Trees - {dataset_name}')
#     plt.xlabel('Number of Trees')
#     plt.ylabel('AUC Score')

#     #  # Set axis limits
#     # plt.xlim(0, 100)
#     plt.ylim(0.6, 0.99)

#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_auc_comparison.eps", dpi=300)
#     plt.close()

    # # Create DataFrame for pairplot
    # df = pd.DataFrame({
    #     "Number_of_Trees": num_trees,
    #     "Custom_RF_AUC": accuracy_custom,
    #     "Sklearn_RF_AUC": accuracy_sklearn
    # })

    # sns.pairplot(df)
    # plt.suptitle(f'Pairplot of AUC Scores - {dataset_name}', y=1.02)
    # plt.tight_layout()
    # plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_pairplot.eps", dpi=300)
    # plt.show()


from collections import defaultdict

for dataset_name, filename in dataset_files.items():
    with open(filename, 'r') as f:
        data = json.load(f)

    grouped = defaultdict(lambda: {'erf_auc': [], 'irf_auc': [], 'rf_auc_score': []})

    for entry in data:
        n_trees = entry["total_erf_trees"]
        if n_trees > 100:
            continue
        grouped[n_trees]['erf_auc'].append(entry["erf_auc"])
        # Assuming irf_auc exists in every entry
        grouped[n_trees]['irf_auc'].append(entry.get("irf_auc", None))
        grouped[n_trees]['rf_auc_score'].append(entry["rf_auc_score"])

    num_trees = []
    erf_avg = []
    # irf_avg = []
    rf_avg = []

    for n in sorted(grouped):
        num_trees.append(n)
        erf_auc_values = [v for v in grouped[n]['erf_auc'] if v is not None]
        # irf_auc_values = [v for v in grouped[n]['irf_auc'] if v is not None]
        rf_auc_values = [v for v in grouped[n]['rf_auc_score'] if v is not None]

        # Average ignoring None values
        erf_avg.append(round(sum(erf_auc_values) / len(erf_auc_values),2) if erf_auc_values else None)
        # irf_avg.append(sum(irf_auc_values) / len(irf_auc_values) if irf_auc_values else None)
        rf_avg.append(round(sum(rf_auc_values) / len(rf_auc_values),2) if rf_auc_values else None)

    plt.figure(figsize=(10, 6))
    plt.plot(num_trees, erf_avg, marker='o', label='RRF AUC')
    # plt.plot(num_trees, irf_avg, marker='^', label='IRF AUC')
    plt.plot(num_trees, rf_avg, marker='s', label='RF AUC')

    plt.title(f'AUC VS Number Of Trees On {dataset_name} Dataset')
    plt.xlabel('Number of Trees')
    plt.ylabel('AUC Score')
    plt.ylim(0.6, 0.99)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_auc_comparison.eps", dpi=300)
    plt.close()




'''

dataset_files = {

    "Titanic": "titanic.json",
    "Breast Cancer": "brest_cancer.json",
    "Diabetes": "diabetes.json",
    "MNIST": "mnist.json",
    "Letter":"/Users/girijabhusal/Desktop/INFO_GAIN_RATIO_IRF/letter.json",
  
}


for dataset_name, filename in dataset_files.items():
    with open(filename, 'r') as f:
        data = json.load(f)

    grouped = defaultdict(lambda: {'erf_auc': [], 'rf_auc_score': []})

    for entry in data:
        n_trees = entry["total_erf_trees"]
        if n_trees>250:
            continue

        grouped[n_trees]['erf_auc'].append(entry["erf_auc"])
        grouped[n_trees]['rf_auc_score'].append(entry["rf_auc_score"])

    num_trees = []
    accuracy_custom = []
    accuracy_sklearn = []

    for n in sorted(grouped):
        num_trees.append(n)
        accuracy_custom.append(sum(grouped[n]['erf_auc']) / len(grouped[n]['erf_auc']))
        accuracy_sklearn.append(sum(grouped[n]['rf_auc_score']) / len(grouped[n]['rf_auc_score']))


    plt.figure(figsize=(10, 6))
    plt.plot(num_trees, accuracy_custom, marker='o', label='Custom Random Forest (AUC)')
    plt.plot(num_trees, accuracy_sklearn, marker='s', label='Sklearn Random Forest (AUC)')

    plt.title(f'AUC vs. Number of Trees-{dataset_name}')
    plt.xlabel('Number of Trees')
    plt.ylabel('AUC Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save or show the plot
    plt.savefig("auc_comparison.eps", dpi=300)
    # plt.show()

'''







#   "OptDigits":"/Users/girijabhusal/Desktop/INFO_GAIN_RATIO_IRF/OptDigits.json",
# Store average AUCs
rf_aucs = []
erf_aucs = []
dataset_names = []
mean_auc_results = {}
# Load AUCs from each file
for dataset_name, filename in dataset_files.items():
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue
    
    with open(filename, 'r') as f:
        data = json.load(f)

    # Average AUCs (assumes list of results; can be just one item)
    rf_auc = sum(item['rf_auc_score'] for item in data) / len(data)
    erf_auc = sum(item['erf_auc'] for item in data) / len(data)
    mean_auc_results[f"{dataset_name}"] = {"rf_auc":rf_auc,"erf_aucs":erf_auc}

    dataset_names.append(dataset_name)

    rf_aucs.append(rf_auc)
    erf_aucs.append(erf_auc)

# Plotting
x = range(len(dataset_names))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x, rf_aucs, width=bar_width, label='Random Forest (RF)', color='skyblue')
plt.bar([p + bar_width for p in x], erf_aucs, width=bar_width, label='Refined RF (RRF)', color='steelblue')

plt.xlabel('Dataset')
plt.ylabel('Mean AUC-ROC')
plt.title('Mean AUC-ROC Comparison Across Datasets')
plt.xticks([p + bar_width / 2 for p in x], dataset_names)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0.5, 0.99)

# Save or show the plot
save_to_json(f'mean_auc.json', mean_auc_results)
plt.savefig("auc_comparison.eps", dpi=300)
plt.show()

