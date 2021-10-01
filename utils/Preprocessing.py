import spacy_udpipe
import re
import time
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.cs.stop_words import STOP_WORDS as STOP_WORDS_CS
from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_EN


class Preprocessing:
    def __init__(self, lang='EN'):
        self.lang = lang
        self.tokenize_functions = {
            "EN": self._tokenizeDataEN,
            "CS": self._tokenizeDataCS
        }
        self.is_trained = False  # todo
        self.nlp = spacy_udpipe.load("en") if lang == 'EN' else spacy_udpipe.load("cs")
        self.default_stop_words = False
        self.stop_words = []
        self._vectorizer = None

    def fit(self, train, n_tokens=None, default_stop_words=False, stop_words=None):
        if stop_words is None:
            self.stop_words = []
        else:
            self.stop_words = stop_words
        start_time = time.time()
        self.default_stop_words = default_stop_words
        text = train
        self._vectorizer = CountVectorizer(
            decode_error='replace',
            lowercase=False,
            tokenizer=self.tokenize_functions[self.lang],
            stop_words=None,
            ngram_range=(1, 1),
            max_df=0.99,
            min_df=1,
            max_features=n_tokens,
            binary=False)

        X = self._vectorizer.fit_transform(text)
        print("--- Finished: %s seconds ---" % (time.time() - start_time))
        return X

    def transform(self, input):
        input = self._vectorizer.transform(input)
        return input

    def getFeatureNames(self):
        return self._vectorizer.get_feature_names()

    def _tokenizeDataCS(self, text):
        doc = self.nlp(text)
        stems = [token.lemma_ for token in doc
                 if ((token.lemma_ not in STOP_WORDS_CS) if self.default_stop_words else True)
                 and (token.lemma_ not in self.stop_words)
                 and re.match(r'[a-žA-Ž]+$', token.lemma_)]
        return stems

    def _tokenizeDataEN(self, text):
        doc = self.nlp(text)
        stems = [token.lemma_ for token in doc
                 if ((token.lemma_ not in STOP_WORDS_EN) if self.default_stop_words else True)
                 and (token.lemma_ not in self.stop_words)
                 and re.match(r'[a-zA-Z]+$', token.lemma_)]
        return stems
