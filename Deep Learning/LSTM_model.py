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

class LSTM_M(torch.nn.Module):
    def __init__(self,  vocab_size, embed_dim,BATCH_SIZE, hidden_size=75,output_size=5):
        super(LSTM_M, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.hidden_size = hidden_size
        self.batch_size= BATCH_SIZE
        self.lstm = torch.nn.LSTM(embed_dim, self.hidden_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.lin =torch.nn.Linear( self.hidden_size,output_size)

    def forward(self, etexts):

        batch_size = etexts.shape[0]
        x=self.embedding(etexts).permute(1, 0, 2)
        first_hidden_h = torch.zeros(1, self.batch_size, self.hidden_size)
        first_hidden_c = torch.zeros(1, self.batch_size, self.hidden_size)
        lstm_output, hidden_output = self.lstm(x,(first_hidden_h,first_hidden_c))

        return self.softmax(self.lin(hidden_output[0].squeeze()))
