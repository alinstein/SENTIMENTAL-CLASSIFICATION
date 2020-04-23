from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from torch.utils.data.dataset import random_split
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import re
import tqdm
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class Sentiment_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='./train.tsv',choice=1, split='train',transform=None):

        self.landmarks_frame = pd.read_csv(csv_file,sep="\t")
        if choice==1:
            self.temp='Phrase';
        if choice==2:
            self.temp='review'

        self.reviews = []
        lemmatizer = WordNetLemmatizer()
        print("Preprocessing :")
        for sent in (self.landmarks_frame[self.temp]):
            review_text = BeautifulSoup(sent).get_text()
            review_text = re.sub("[^a-zA-Z]"," ", review_text)
            words = word_tokenize(review_text.lower())
            lemma_words = [lemmatizer.lemmatize(i) for i in words]
            self.reviews.append(lemma_words)

        self.landmarks_frame[self.temp] = self.reviews
        self.landmarks_frame[self.temp]=self.landmarks_frame[self.temp].apply(lambda x: ' '.join(x))
        self.unique_words = set()
        self.len_max = 0
        for sent in (self.reviews):
            self.unique_words.update(sent)
            if(self.len_max<len(sent)):
                self.len_max = len(sent)

        self.target = np.array(self.landmarks_frame['Sentiment'])

        self.word_to_ix = {word: i for i, word in enumerate(self.unique_words)}
        # Use 90% for training and 10% for validation.
        print(len(self.reviews), len(self.target))
        if split == 'train':
            self.reviews, self.target  = self.reviews[:140000], self.target[:140000]

        elif split == 'test' :
           self.reviews, self.target  = self.reviews[140000:156000], self.target[140000:156000]

    def unique(self):
        return self.unique_words

    def max_length(self):
        return self.len_max
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):

            N=self.len_max-len(self.reviews[idx])
            lemma_words= np.array([self.word_to_ix[x] for x in self.reviews[idx]])

            lemma_words = np.pad(lemma_words, (0, N), 'constant')

     
      
            rating = np.array(self.target[idx])

            sample={'Sentiment':torch.from_numpy(rating).type(torch.int64)
                    ,'review':torch.from_numpy(lemma_words).type((torch.int64))}

            return sample

