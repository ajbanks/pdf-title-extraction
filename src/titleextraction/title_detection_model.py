from feature_extractors.binary_feature_extractor import Binary_Feature_Extractor
from feature_extractors.pdf_loc_feature_extractor import PDF_Loc_Feature_Extractor
from feature_extractors.text_embedding_extractor import Embedding_Extractor
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.metrics import classification_report
from typing import List, Tuple, Dict

import logging
_logger = logging.getLogger("title-extraction-logger")
logging.basicConfig(level=logging.INFO)

class Title_Detection_Model:

    def __init__(self):
        self.binary_extractor = Binary_Feature_Extractor()
        self.pdf_loc_feature_extractor = PDF_Loc_Feature_Extractor()
        self.text_embedding_extractor = Embedding_Extractor()
        self.all_extractors = [self.binary_extractor, self.pdf_loc_feature_extractor, self.text_embedding_extractor]

    def extract_features_from_record(self, record: Dict) -> np.array:
        """use feature extractors to extract all relevant features from a record and combine in numpy array

            Args:
                Dict: this is a record from a pandas dataframe containing the following keys:

                [Text	IsBold	IsItalic	IsUnderlined	Left	Right	Top	  Bottom	 FontType	Label]

            Returns: A numpy array containing the feature representation of the data sample

        """

        concatenated_features = np.array([])
        for extractor in self.all_extractors:
            concatenated_features = np.concatenate([concatenated_features, extractor.extract_feature(record)])

        return concatenated_features

    def load_datasets(self):

        """Load the train and test datasets located in the data folder

            Args:
            Returns: A dictionary in the format {"train": train DataFrame, "test": test DataFrame}

        """
        dataset_dict = {"train": pd.read_csv("data/train_sections_data.csv", header=0, encoding='unicode_escape').sample(frac=1).reset_index(drop=True),
                        "test": pd.read_csv("data/test_sections_data.csv", header=0, encoding='unicode_escape').sample(frac=1).reset_index(drop=True)}
        return dataset_dict

    def create_feature_dataset(self, df) -> Dict:
        """Convert a train or test dataset into a feature set of numerical representations

            Args:
                df: this is a pandas dataframe containing data with in the following format as

                [Text	IsBold	IsItalic	IsUnderlined	Left	Right	Top	  Bottom	 FontType	Label]

            Returns: A dictionary in the format {"x":list contatining records converted to features, "y": list of labels}

        """
        output_list = []

        for id, record in df.iterrows():
            output_list.append(self.extract_features_from_record(record))

        return {"x":output_list, "y": df["Label"].values}


    def train_MLP(self, feature_dataset):
        """Train a multi layer perceptron on a feature set of numerical representations

            Args:
                feature_dataset: A numpy array of dataset records that have be converted in to feature representations

            Returns: clf: The Trained MLP

        """
        clf = MLPClassifier(random_state=42, max_iter=300).fit(feature_dataset["x"], feature_dataset["y"])
        return clf

    def predict_model(self, feature_set, clf):
        """Predict model on feature set

            Args:
                feature_set: A numpy array of dataset records that have be converted in to feature representations
                clf: A trained Multi layer Perceptron

            Returns: A numpy array of Model predictions

        """
        return clf.predict(feature_set)

    def evaluate_model(self, test_feature_dataset, clf) -> str:
        """Evaluate the model on a feature set. Returns a classification report detailing F1 and accuracy
            Returns:
                F1 classification report: str
        """
        preds = self.predict_model(test_feature_dataset["x"], clf)
        classification_rep = classification_report(test_feature_dataset["y"], preds)

        return str(classification_rep)


    def end_to_end_training(self):
        """Performs end to end training of an mlp model. loades dataset, processes data and evaluates the model

            Returns: trained_model: The Trained MLP

        """
        _logger.info("loading dataset")
        dataset_dict = self.load_datasets()
        _logger.info(" dataset loaded")

        _logger.info("creatng feature datasets")

        # create feature sets
        train_feature_dataset = self.create_feature_dataset(dataset_dict["train"])
        test_feature_dataset = self.create_feature_dataset(dataset_dict["test"])

        _logger.info("training mlp")

        # train mlp
        trained_model = self.train_MLP(train_feature_dataset)

        #save model
        dump(trained_model, 'models/mlp_ensemble.joblib')

        _logger.info("evaluating")

        #evaluate model
        eval_result = self.evaluate_model(test_feature_dataset, trained_model)
        _logger.info(str(eval_result))

        return trained_model



if __name__ == "__main__":
    title_detection_model = Title_Detection_Model()
    title_detection_model.end_to_end_training()