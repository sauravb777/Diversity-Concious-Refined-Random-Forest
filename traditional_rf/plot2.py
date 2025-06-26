import json
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_b(json_file_path):
    """
    Reads a JSON file containing accuracy and B values,
    and plots Accuracy vs B as a line graph.

    Parameters:
    json_file_path (str): Path to the JSON file.
    """
    # Load data from JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Flatten the nested list
    flattened_data = [item for sublist in data for item in sublist]

    # Extract B and accuracy
    b_values = [item['B'] for item in flattened_data]
    accuracy_values = [item['accuracy'] for item in flattened_data]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(b_values, accuracy_values, marker='o', linestyle='-')
    plt.xlabel('B')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs B')
    plt.grid(True)
    plt.show()


def plot_auc_comparison(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    dataset_names = [entry["Dataset"] for entry in data]
    rf_auc = [entry["RF (Mean AUC)"] for entry in data]
    erf_auc = [entry["RRF (Mean AUC)"] for entry in data]

    x = np.arange(len(dataset_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(x, rf_auc, width=bar_width, label='RF (Mean AUC)', color='skyblue')
    plt.bar(x + bar_width, erf_auc, width=bar_width, label='RRF (Mean AUC)', color='steelblue')

    # This is what you're asking about:
    plt.xticks([p + bar_width / 2 for p in x], dataset_names)
    plt.xlabel('Datasets')
    plt.ylabel('Mean AUC')
    plt.title('RF vs RRF Mean AUC Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.ylim(0.5, 0.99)
    plt.savefig("auc_comparison.eps", dpi=70)
    plt.show()

plot_auc_comparison('/Users/austin/Desktop/INFO_GAIN_RATIO_IRF/acc.json')
# plot_accuracy_vs_b('/Users/austin/Desktop/INFO_GAIN_RATIO_IRF/irf_data.json')




import matplotlib.pyplot as plt

datasets = {
    "Breast cancer": [
        {"accuracy": 0.94,"B": 28},
        {"accuracy": 0.92,"B": 39},
        {"accuracy": 0.94,"B": 45},
        {"accuracy": 0.95,"B": 54}
    ],
    "Adult Incode": [
        {"accuracy": 0.84, "B": 28},
        {"accuracy": 0.80, "B": 72},
        {"accuracy": 0.83, "B": 123},
        {"accuracy": 0.81, "B": 150},
        {"accuracy": 0.84, "B": 187}
    ],
    "Diabetes": [
    
        {"accuracy": 0.68,"B": 28},
        {"accuracy": 0.67,"B": 83},
        {"accuracy": 0.69,"B": 105},
        {"accuracy": 0.72,"B": 166},
        {"accuracy": 0.71,"B": 181}
    
    ],
    "Titanic": [
        {"accuracy": 0.80,"B": 28},
        {"accuracy": 0.78,"B": 43},
        {"accuracy": 0.82,"B": 60},
        {"accuracy": 0.82,"B": 94}
    ],
    "Mnist": [
        {"accuracy": 0.88,"B": 28},
        {"accuracy": 0.86,"B": 38},
        {"accuracy": 0.87,"B": 51},
        {"accuracy": 0.86,"B": 65},
        {"accuracy": 0.91,"B": 76},
        {"accuracy": 0.90,"B": 89}

    ],
    "Letter": [
        {"accuracy": 0.89,"B": 28},
        {"accuracy": 0.87,"B": 64},
        {"accuracy": 0.91,"B": 87},
        {"accuracy": 0.90,"B": 104},
        {"accuracy": 0.91,"B": 118}
    ],


    "SatImage": [
        {"accuracy": 0.88,"B": 28},
        {"accuracy": 0.87,"B": 47},
        {"accuracy": 0.91,"B": 54},
        {"accuracy": 0.90,"B": 62},
        {"accuracy": 0.92,"B": 77},
        {"accuracy": 0.92,"B": 98},

    ],
    "OptDigits": [
        {"accuracy": 0.93,"B": 28},
        {"accuracy": 0.92,"B": 37},
        {"accuracy": 0.91,"B": 42},
        {"accuracy": 0.94,"B": 56},
        {"accuracy": 0.95,"B": 63}]
}

# # Create one plot per dataset
# for name, data in datasets.items():
#     x = [entry["B"] for entry in data]
#     y = [entry["accuracy"] for entry in data]
#     y_min = max(0, min(y) - 0.05)
#     y_max = min(1, max(y) + 0.03)
#     plt.figure(figsize=(6, 4))
#     plt.plot(x, y, marker='o', color='blue')
#     plt.title(f'{name}: Accuracy vs No of Trees')
#     plt.xlabel('No of Trees')
#     plt.ylabel('Accuracy')
#     plt.grid(True)
#     plt.tight_layout()
#     # plt.ylim(0.66, 0.97)
#     plt.ylim(y_min, y_max)

#     plt.savefig(f"IRF_{name.lower().replace(' ', '_')}_accuracy_vs_trees.eps", dpi=90)
#     plt.show()


# # Create subplots (4 rows x 2 columns)
# fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
# axes = axes.flatten()  # Flatten to 1D array for easy indexing

# # Plot each dataset
# for idx, (name, data) in enumerate(datasets.items()):
#     x = [entry["B"] for entry in data]
#     y = [entry["accuracy"] for entry in data]

#     ax = axes[idx]
#     ax.plot(x, y, marker='o', color='blue')
#     ax.set_title(name)
#     ax.set_xlabel("No of Trees")
#     ax.set_ylabel("Accuracy")
#     ax.grid(True)

# # Hide any unused subplots (in case there are less than 8)
# for i in range(len(datasets), len(axes)):
#     fig.delaxes(axes[i])

# # Adjust layout to avoid overlap
# plt.tight_layout()
# plt.suptitle("Accuracy vs No of Trees (B) for All Datasets", fontsize=16, y=1.02)
# plt.subplots_adjust(top=0.95)
# plt.show()


