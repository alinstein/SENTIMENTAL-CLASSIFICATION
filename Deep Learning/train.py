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
from dataset import Sentiment_Dataset
from GRU_model import GRU_M
from LSTM_model import LSTM_M


def main():
        model_type="gru"
        #for LSTM :
        #model_type ="lstm"
        N_EPOCHS = 25
        Emb_dim=50
        min_valid_loss = float('inf')
        #dataset_ = Sentiment_Dataset()
        
        BATCH_SIZE=100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.CrossEntropyLoss().to(device)
        sub_train_ = Sentiment_Dataset()
        sub_test_ = Sentiment_Dataset(split="test")
        vocab_size=len(sub_train_.unique())
        dataloader = DataLoader(sub_train_, batch_size=BATCH_SIZE,
                        shuffle=True)

        dataloader_test = DataLoader(sub_test_, batch_size=BATCH_SIZE)
        if model_type =='lstm':
            print("Training with LSTM")
            model=LSTM_M(vocab_size,Emb_dim,BATCH_SIZE)
        else :
            print("Training with GRU")
            model=GRU_M(vocab_size,Emb_dim,BATCH_SIZE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        def train_func(data):
                optimizer.zero_grad()
                text, label = data['review'], data['Sentiment']
                output = model(text)
                loss = criterion(output, label)
                train_loss = loss.item()
                loss.backward()
                optimizer.step()
                train_acc = (output.argmax(1) == label).sum().item()
                return train_loss , train_acc

        def test_func(data):
                text, label = data['review'], data['Sentiment']
                output = model(text)
                loss = criterion(output, label)
                test_loss = loss.item()
                test_acc = (output.argmax(1) == label).sum().item()
                return test_loss , test_acc

        for epoch in range(N_EPOCHS):
            # Train the model
            train_loss = 0
            train_acc = 0
            train_acc_bat=0


            if epoch % 3==0 :
                  test_loss = 0
                  test_acc = 0
                  print("Validation")
                  model.eval()
                  for i, xdata in enumerate(dataloader_test):

                        loss, acc = test_func(xdata)
                        test_loss+= loss
                        test_acc+= acc

                  test_loss, test_acc  = test_loss/ len(sub_test_), test_acc/ len(sub_test_)
                  print("Validation Loss :",test_loss)
                  print("Validation Acc:",test_acc)


            start_time = time.time()
            model.train()
            for i, xdata in enumerate(dataloader):

                loss, acc = train_func(xdata)
                train_loss+= loss
                train_acc+= acc
                train_acc_bat += acc
                if i % 100==0 :
                    print("Batch",i)
                    print(train_acc_bat/100)
                    train_acc_bat=0

            train_loss, train_acc  = train_loss/ len(sub_train_), train_acc/ len(sub_train_)
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            print("Loss :",train_loss)
            print("Acc:",train_acc)



if __name__ == '__main__':
  main()
