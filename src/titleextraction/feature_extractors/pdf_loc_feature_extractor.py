import numpy as np
from titleextraction.feature_extractors.feature_extractor import Extractor
from typing import List, Tuple, Dict

class PDF_Loc_Feature_Extractor(Extractor):

    def __init__(self):
        self.extractor_label = "PDF_Loc_Feature_Extractor"

    def extract_feature(self, record: Dict) -> np.array:
        """just returns the location features as a numpy array """

        return np.array([record["Left"], record["Right"], record["Top"], record["Bottom"]])