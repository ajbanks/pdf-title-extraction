from sentence_transformers import SentenceTransformer, util
import numpy as np
from titleextraction.feature_extractors.feature_extractor import Extractor
from typing import List, Tuple, Dict


class Embedding_Extractor(Extractor):

    def __init__(self):
        self.extractor_label = "Embedding_Extractor"
        self.embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


    def embedder(self, text: str) -> np.array:
        """converts text in to a sentence embedding representation"""
        text_embedding = self.embedding_model.encode('How big is London')
        return text_embedding

    def extract_feature(self, record: Dict) -> np.array:
        """converts the text feature to a sentence embedding"""
        text = record["Text"]
        return self.embedder(text)