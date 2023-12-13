import numpy as np



class Binary_Feature_Extractor(Extractor):

    def __init__(self):
        self.extractor_label = "Binary_Feature_Extractor"
        self.feature_labels = ["IsBold", "IsItalic", "IsUnderlined"]

    def extract_feature(self, record: Dict) -> np.array:
        "returns the length of the text"
        feature_dict = {"IsBold": 0, "IsItalic":0, "IsUnderlined":0}
        if record["IsBold"] == True:
            feature_dict["IsBold"] = 1
        if record["IsItalic"] == True:
            feature_dict["IsItalic"] = 1
        if record["IsUnderlined"] == True:
            feature_dict["IsUnderlined"] = 1

        return np.array([feature_dict["IsBold"], feature_dict["IsItalic"], feature_dict["IsUnderlined"]])