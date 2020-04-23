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

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir,choice==1. transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, , sep="\t")
        if choice==1:
            self.temp='Phrase';
        if choice==2:
            self.temp='review'

        self.reviews = []

        for sent in tqdm(df[choice]):
            review_text = BeautifulSoup(sent).get_text()
            review_text = re.sub("[^a-zA-Z]"," ", review_text)
            words = word_tokenize(review_text.lower())
            lemma_words = [lemmatizer.lemmatize(i) for i in words]
            self.reviews.append(lemma_words)

        self.reviews=self.reviews.apply(lambda x: ' '.join(x))
        self.unique_words = set()
        self.len_max = 0

        for sent in (self.reviews):
            self.unique_words.update(sent)
            if(self.len_max<len(sent)):
                self.len_max = len(sent)

        self.target=torch.nn.functional.one_hot(self.landmarks_frame['Sentiment'],2)


    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
            self.reviews[self.temp][idx]

            rating = self.target[idx]
            sample={'Sentiment':rating,'review':lemma_words}

        return sample


class RNN(nn.Module):
    def __init__(self,  vocab_size, embed_dim, hidden_size=128):
        super(RNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.hidden_size = hidden_size

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, etexts, hidden):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(1, batch_size, self.hidden_size),
                        torch.zeros(1, batch_size, self.hidden_size))
        first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        self.softmax(lstm_output)
        return lstm_output


def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(FaceLandmarksDataset, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

main():

        N_EPOCHS = 5
        min_valid_loss = float('inf')

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

        # train_len = int(len(train_dataset) * 0.95)
        # sub_train_, sub_valid_ = \
        #     random_split(train_dataset, [train_len, len(train_dataset) - train_len])

        for epoch in range(N_EPOCHS):

            start_time = time.time()
            train_loss, train_acc = train_func(sub_train_)
            # valid_loss, valid_acc = test(sub_valid_)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            # print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


if __name__ == '__main__':
  main()
