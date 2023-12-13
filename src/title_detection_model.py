from feature_extractors.binary_feature_extractor import Binary_Feature_Extractor
from feature_extractors.pdf_loc_feature_extractor import PDF_Loc_Feature_Extractor
from feature_extractors.text_embedding_extractor import Embedding_Extractor
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from joblib import dump, load



class Title_Detection_Model:

    def __init__(self):
        self.binary_extractor = Binary_Feature_Extractor()
        self.pdf_loc_feature_extractor = PDF_Loc_Feature_Extractor()
        self.text_embedding_extractor = Embedding_Extractor()
        self.all_extractors = [self.binary_extractor, self.pdf_loc_feature_extractor, self.text_embedding_extractor]

    def extract_features_from_record(self, record):
        """use feature extrctors to extract all relevant features from a record and combine in numpy array"""

        concatenated_features = np.array([])
        for extractor in self.all_extractors:
            concatenated_features = np.concatenate([concatenated_features, extractor.extract_feature(record)])

        return concatenated_features

    def load_datasets(self):
        dataset_dict = {"train": pd.read_csv("data/train_sections_data.csv", header=0), "test": pd.read_csv("data/test_sections_data.csv", header=0)}
        return dataset_dict

    def create_feature_dataset(self, df):
        output_list = []

        for id, record in df.iterrows():
            output_list.append(self.extract_features_from_record(record))

        return {"x":output_list, "y": df["Label"].values}


    def train_MLP(self, feature_dataset):

        X, y = make_classification(n_samples=100, random_state=1)
        clf = MLPClassifier(random_state=1, max_iter=300).fit(feature_dataset["x"], feature_dataset["y"])
        return clf

    def predict_model(self, feature_set, clf):
        return clf.predict(feature_set)

    def evaluate_model(self, test_feature_dataset, clf):
        score = clf.score(test_feature_dataset["x"], test_feature_dataset["y"])
        return score


    def end_to_end_training(self):
        dataset_dict = load_datasets()

        # create feature sets
        train_feature_dataset = self.create_feature_dataset(dataset_dict["train"])
        test_feature_dataset = self.create_feature_dataset(dataset_dict["test"])

        # train mlp
        trained_model = self.train_MLP(train_feature_dataset)

        #save model
        dump(trained_model, 'model/mlp_ensemble.joblib')

        #evaluate model
        eval_result = self.evaluate_model(test_feature_dataset, trained_model)
        print(eval_result)

        return trained_model