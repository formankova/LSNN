import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Preprocessing():
    def createBafOfWords(self, train, test, n_tokens):
        text = np.concatenate((train, test))
        self._vectorizer = CountVectorizer(
            decode_error='replace',
            lowercase=False,
            tokenizer=self._tokenizeData,
            stop_words=None,
            ngram_range=(1, 1),
            max_df=0.99,  # document frequency
            min_df=2,
            max_features=n_tokens,
            binary=False)

        X = self._vectorizer.fit_transform(text)
        self._tf_transformer = TfidfTransformer().fit(X)

        X_train = self.transform(train)
        X_test = self.transform(test)

        return X_train, X_test

    def transform(self, input):
        input = self._vectorizer.transform(input)
        input = self._tf_transformer.transform(input)
        return input.toarray()

    def _tokenizeData(self, text):
        lower_text = text.lower()
        tokenizer = nltk.tokenize.WordPunctTokenizer()
        stemmer = nltk.stem.PorterStemmer()
        tokens = tokenizer.tokenize(lower_text)
        stems = [stemmer.stem(token) for token in tokens]
        return stems