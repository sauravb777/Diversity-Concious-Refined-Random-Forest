import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_and_clean_breast_cancer(path="dataCleaning/breast-cancer.csv"):
    path = os.path.join(os.path.dirname(__file__), path)
    df = pd.read_csv(path)
    # strip whitespace
    for col in df.select_dtypes('object'):
        df[col] = df[col].str.strip()
    # impute numeric
    for col in df.select_dtypes('number'):
        df[col] = df[col].fillna(df[col].median())
    # impute categorical
    for col in df.select_dtypes('object'):
        df[col] = df[col].fillna(df[col].mode()[0])
    return df



from sklearn.datasets import fetch_openml

class DiabetesLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        print("Loading Diabetes dataset from OpenML...")
        diabetes = fetch_openml(name='diabetes', version=1, as_frame=True)
        self.df = diabetes.frame
        self.X = diabetes.data
        self.y = diabetes.target
        print("Diabetes dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y
    

diabetes = fetch_openml(name='diabetes', version=1, as_frame=True)


from sklearn.datasets import load_breast_cancer

class BreastCancerLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        print("Loading Breast Cancer dataset...")
        cancer = load_breast_cancer(as_frame=True)
        self.df = pd.concat([cancer.data, cancer.target.rename("target")], axis=1)
        self.X = cancer.data
        self.y = cancer.target
        print("Breast Cancer dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y




class TitanicDataCleaner:
    def __init__(self):
        # Load the Titanic dataset when the class is instantiated
        self.titanic = sns.load_dataset('titanic')
    
    def clean_data(self):
        """
        Clean and preprocess the Titanic dataset:
        - Handle missing values
        - Convert categorical variables to numerical
        - Drop irrelevant columns
        """
        # Handle missing values
        self.titanic['age'].fillna(self.titanic['age'].median(), inplace=True)
        self.titanic['embarked'].fillna(self.titanic['embarked'].mode()[0], inplace=True)
        self.titanic.drop(columns=['deck'], inplace=True)  # Drop 'deck' column due to excessive missing values
        self.titanic['embark_town'].fillna(self.titanic['embark_town'].mode()[0], inplace=True)

        # Convert 'sex' and 'who' to numerical values using Label Encoding
        self.titanic['sex'] = self.titanic['sex'].map({'male': 0, 'female': 1})
        self.titanic['who'] = self.titanic['who'].map({'man': 0, 'woman': 1})

        # One-Hot Encoding for 'embarked' (since it has multiple categories)
        self.titanic = pd.get_dummies(self.titanic, columns=['embarked'], drop_first=True)

        # One-Hot Encoding for 'embark_town' (since it has multiple categories)
        self.titanic = pd.get_dummies(self.titanic, columns=['embark_town'], drop_first=True)

        # Drop irrelevant columns
        self.titanic.drop(columns=['class', 'alive', 'alone'], inplace=True)

        # Ensure all columns are numerical
        self.titanic = self.titanic.apply(pd.to_numeric, errors='ignore')  # Ignore columns like 'sex', 'who'

        return self.titanic




    
class PimaPreprocessor:
    def __init__(self, scale_features=True):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if self.scale_features else None
        self.df = None

    def load_data(self):
        """Load Pima Indians Diabetes dataset from OpenML."""
        data = fetch_openml(name='diabetes', version=1, as_frame=True)
        self.df = data.frame
        print("✅ Dataset loaded with shape:", self.df.shape)

    def handle_missing_values(self):
        """Replace biologically invalid 0s with NaN, then impute with median."""
        zero_invalid_cols = ['plas', 'pres', 'skin', 'insu', 'mass']
        self.df[zero_invalid_cols] = self.df[zero_invalid_cols].replace(0, np.nan)
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        print("✅ Handled invalid/missing values.")

    def encode_target(self):
        """Convert class column from strings to binary integers."""
        self.df['class'] = self.df['class'].map({'tested_negative': 0, 'tested_positive': 1})
        print("✅ Encoded target labels.")

    def scale_features_func(self):
        """Scale numeric features if requested."""
        X = self.df.drop(columns=['class'])
        y = self.df['class']

        if self.scaler:
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            self.df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
            print("✅ Features scaled.")
        else:
            self.df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            print("✅ Scaling skipped.")

    def preprocess(self):
        self.load_data()
        self.handle_missing_values()
        self.encode_target()
        self.scale_features_func()
        return self.df
    


class MNISTLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=True)
        self.df = mnist.frame
        self.X = mnist.data
        self.y = mnist.target.astype(int)
        print("MNIST dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y



class LetterDataLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        print("Loading Letter dataset from OpenML...")
        letter = fetch_openml(name='letter', version=1, as_frame=True)
        self.X = letter.data

        # Convert target labels from letters to integers (A-Z → 0–25)
        encoder = LabelEncoder()
        self.y = pd.Series(encoder.fit_transform(letter.target), name="label")

        self.df = pd.concat([self.X, self.y], axis=1)
        print("Letter dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y
    


class OptDigitsDataLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.load_data()

    def load_data(self):
        print("Loading OptDigits dataset from OpenML...")
        optdigits = fetch_openml(data_id=28, as_frame=True)
        self.X = optdigits.data

        # Encode target labels if needed (digits already numeric but stored as strings)
        self.y = pd.Series(self.label_encoder.fit_transform(optdigits.target.astype(str)), name="label")

        self.df = pd.concat([self.X, self.y], axis=1)
        print("OptDigits dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y

    def get_label_encoder(self):
        return self.label_encoder



    
class CovertypeLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        print("Loading Covertype dataset...")
        dataset = fetch_openml(data_id=180, as_frame=True)
        self.df = dataset.frame
        self.X = dataset.data

        # Convert categorical labels to integer values
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(dataset.target)

        print("Covertype dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y
    
# class AdultIncome:
#     def __init__(self):
#         self.df = None

#     def load_df(self):
#         adult = fetch_openml("adult", version=2, as_frame=True)
#         self.df = adult.frame
#     def get_dataframe(self):
#         return self.df
    

# adult = fetch_openml("segment", version=1, as_frame=True)


class StatlogImageSegmentation:
    def __init__(self):
        self.df = None

    def load_df(self):
        segment = fetch_openml("segment", version=1, as_frame=True)
        self.df = segment.frame
    def get_dataframe(self):
        segment = fetch_openml("segment", version=1, as_frame=True)
        df = segment.frame
        return df


class AdultDataLoader:
    def __init__(self):
        self.df = None
        self.label_encoders = {}  # To store encoders for categorical features
        self.label_encoder = LabelEncoder()  # For the target column
        self.load_and_prepare_data()

    def load_and_prepare_data1(self):
        print("Loading Adult dataset from OpenML...")
        adult = fetch_openml(name='adult', version=2, as_frame=True)
        X = adult.data
        y = adult.target

        # Replace missing values represented by '?' with NaN and drop rows with missing values
        X = X.replace('?', pd.NA).dropna()
        y = y[X.index]  # Align y with cleaned X

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)

        # Label encode all categorical features
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include='category').columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            self.label_encoders[col] = le  # Save encoder for future use

        # Combine into a single DataFrame
        X_encoded["label"] = y_encoded
        self.df = X_encoded

        print("Adult dataset loaded and all categorical features label-encoded.")

    def load_and_prepare_data(self):
        print("Loading Adult dataset from OpenML...")
        
        # Load data
        adult = fetch_openml(name='adult', version=2, as_frame=True)
        X = adult.data
        y = adult.target

        # Replace '?' with pd.NA safely and drop missing rows
        X = X.applymap(lambda x: pd.NA if x == '?' else x)
        X = X.dropna()
        y = y.loc[X.index]  # Align target with cleaned features

        # Encode target column
        y_encoded = self.label_encoder.fit_transform(y)

        # Label encode all object and category type features
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            self.label_encoders[col] = le  # Save encoder for future use

        # Combine features and label into one DataFrame
        X_encoded["label"] = y_encoded
        self.df = X_encoded

        print("Adult dataset loaded and all categorical features label-encoded.")


    def get_dataframe(self):
        return self.df.copy()

    def get_label_encoder(self):
        return self.label_encoder

    def get_feature_encoders(self):
        return self.label_encoders




# class DataLoader:
#     def __init__(self,id):
#         self.id  = id
#         self.df = None
#         self.label_encoders = {}  # To store encoders for categorical features
#         self.label_encoder = LabelEncoder()  # For the target column
#         self.load_and_prepare_data()

#     def load_and_prepare_data(self):
#         print("Loading AutoUniv dataset from OpenML...")
#         dataset = fetch_openml(data_id=self.id, as_frame=True)
#         X = dataset.data
#         y = dataset.target

#         # Replace missing values represented by '?' with NaN and drop rows with missing values
#         X = X.replace('?', pd.NA).dropna()
#         y = y[X.index]  # Align y with cleaned X

#         # Encode target
#         y_encoded = self.label_encoder.fit_transform(y)

#         # Label encode all categorical features
#         X_encoded = X.copy()
#         for col in X_encoded.select_dtypes(include='category').columns:
#             le = LabelEncoder()
#             X_encoded[col] = le.fit_transform(X_encoded[col])
#             self.label_encoders[col] = le  # Save encoder for future use

#         # Combine into a single DataFrame
#         X_encoded["label"] = y_encoded
#         self.df = X_encoded

#         print("AutoUnivLoader dataset loaded and all categorical features label-encoded.")

#     def get_dataframe(self):
#         return self.df.copy()

#     def get_label_encoder(self):
#         return self.label_encoder

#     def get_feature_encoders(self):
#         return self.label_encoders
    

class DataLoader:
    def __init__(self, id):
        self.id = id
        self.df = None
        self.label_encoders = {}  # For categorical features
        self.label_encoder = LabelEncoder()  # For the target
        self.scaler = StandardScaler()  # You can switch to MinMaxScaler()
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        print("Loading dataset from OpenML...")
        dataset = fetch_openml(data_id=self.id, as_frame=True)
        X = dataset.data
        y = dataset.target

        # Handle missing values
        X = X.replace('?', pd.NA).dropna()
        y = y[X.index]  # Align y with cleaned X

        # Encode categorical features
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include='category').columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            self.label_encoders[col] = le

        # Scale numerical features
        numeric_cols = X_encoded.select_dtypes(include=['number']).columns
        X_encoded[numeric_cols] = self.scaler.fit_transform(X_encoded[numeric_cols])

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)

        # Combine X and y
        X_encoded["label"] = y_encoded
        self.df = X_encoded

        print("Dataset loaded, categorical features label-encoded, and numerical features scaled.")

    def get_dataframe(self):
        return self.df.copy()

    def get_label_encoder(self):
        return self.label_encoder

    def get_feature_encoders(self):
        return self.label_encoders

    def get_scaler(self):
        return self.scaler



# _______________________________________________________________

import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class IrisPreprocessor:
    def __init__(self, scale_features=False):
        self.scale_features = scale_features
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler() if self.scale_features else None
        self.data = None

    def load_data(self):
        self.data = sns.load_dataset('iris')
        print("Dataset loaded.")

    def check_missing_values(self):
        missing = self.data.isnull().sum()
        print("Missing values:\n", missing)

    def encode_target(self):
        self.data['species_encoded'] = self.label_encoder.fit_transform(self.data['species'])

    def scale_features_func(self):
        feature_cols = self.data.drop(['species', 'species_encoded'], axis=1)
        if self.scaler:
            scaled = self.scaler.fit_transform(feature_cols)
            scaled_df = pd.DataFrame(scaled, columns=feature_cols.columns)
            self.data = pd.concat([scaled_df, self.data[['species_encoded']]], axis=1)
            print("Features scaled and combined with target.")
        else:
            self.data = pd.concat([feature_cols, self.data[['species_encoded']]], axis=1)
            print("Features kept unscaled and combined with target.")

    def preprocess(self):
        self.load_data()
        self.check_missing_values()
        self.encode_target()
        self.scale_features_func()
        return self.data
    


from sklearn.datasets import load_wine
class WineDataLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        print("Loading Wine dataset from sklearn...")
        wine = load_wine()
        self.X = pd.DataFrame(wine.data, columns=wine.feature_names)
        self.y = pd.Series(wine.target, name="label")
        self.df = pd.concat([self.X, self.y], axis=1)
        print("Wine dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y
    




class YeastDataLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.load_data()

    def load_data(self):
        print("Loading Yeast dataset from OpenML...")
        yeast = fetch_openml(data_id=181, as_frame=True)
        self.X = yeast.data

        # Encode target labels (e.g., 'CYT', 'NUC', etc.) to integers
        self.y = pd.Series(self.label_encoder.fit_transform(yeast.target), name="label")

        # Combine features and label into one DataFrame
        self.df = pd.concat([self.X, self.y], axis=1)
        print("Yeast dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y

    def get_label_encoder(self):
        return self.label_encoder
    


class MiceProteinLoader:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        print("Loading Mice Protein Expression dataset...")
        dataset = fetch_openml(data_id=1464, as_frame=True)
        self.df = dataset.frame
        self.X = dataset.data
        self.y = dataset.target
        print("Mice Protein dataset loaded successfully.")

    def get_dataframe(self):
        return self.df

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.y