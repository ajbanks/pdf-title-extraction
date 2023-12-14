
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import logging
from sklearn.utils import shuffle
from tqdm import trange
from itertools import groupby

_logger = logging.getLogger(__name__)


class Extractor(object):
    """Abstract class to extract features from records in our database"""

    def extract_feature(self, record: Dict) -> np.array:
        pass


class dummy_extract(Extractor):
    """Really stupid feature extractor to help development/testing.
    It just counts the length of the first few words of a bit of text."""


    def extract_feature(self, record: Dict) -> np.array:
        "returns the length of the text"
        text = record["Text"]
        text_length = len(text)
        return np.array([text_length])

