import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class TextUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def cleanReview( review, remove_stopwords=False ):
        # Here we will be removing the unwanted noise from our text. Since we are only interested in the words that
        # add meaning to the review, we will also be filtering out the stopwords.

        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Replace numbers with NUM
        review_text = re.sub("[0-9]]","NUM" , review_text)

        # 3 - Remove all non-alphabets.
        review_text = re.sub("[^a-zA-Z]"," ", review_text)


        # 4. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 5. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(" ".join(words))

    @staticmethod
    def cleanReviewAndSequenize( traindata , testdata , max_review_length = 200 ):

        train_clean_reviews = [TextUtility.cleanReview(t) for t in traindata ]
        test_clean_reviews = [TextUtility.cleanReview(t) for t in testdata ]

        all_reviews = train_clean_reviews + test_clean_reviews
        print("cleaned the reviews")
        # Tokenize the reviews
        tokenizer = Tokenizer(num_words=50000)
        tokenizer.fit_on_texts(all_reviews)
        train_seq = tokenizer.texts_to_sequences(train_clean_reviews)
        test_seq = tokenizer.texts_to_sequences(test_clean_reviews)

        # Total number of words found and sequenced
        numberOfWords = len(tokenizer.word_index)
        #print(numberOfWords)
        # restrict all the reviews to a strict length... by default it is 200 for below
        train_pad = pad_sequences(train_seq, maxlen=max_review_length)
        test_pad = pad_sequences(test_seq, maxlen=max_review_length)

        return train_pad,test_pad,numberOfWords,max_review_length