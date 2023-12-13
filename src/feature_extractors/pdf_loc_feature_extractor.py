import numpy as np

class PDF_Loc_Feature_Extractor(Extractor):

    def __init__(self):
        self.extractor_label = "PDF_Loc_Feature_Extractor"

    def extract_feature(self, record: Dict) -> np.array:
        "convertjust returns the location features atm"

        feature_dict = {"Left": 0, "Right": 0, "Top": 0, "Bottom": 0}

        return np.array([record["Left"], record["Right"], record["Top"], record["Bottom"]])