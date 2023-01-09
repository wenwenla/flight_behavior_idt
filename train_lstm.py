import os
import pickle
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report


DEVICE = 'cuda:3' #'cuda:0'


class RNN(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn_layear = torch.nn.LSTM(input_size=30, hidden_size=128, batch_first=True)
        self.fc0 = torch.nn.Linear(128, 128)
        self.fc1 = torch.nn.Linear(128, 11)

    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.zeros((1, batch_size, 128)).to(DEVICE)
        c_0 = torch.zeros((1, batch_size, 128)).to(DEVICE)
        output, (h, c) = self.rnn_layear(x, (h_0, c_0))
        y0 = torch.relu(self.fc0(h[-1]))
        y1 = self.fc1(y0)
        return y1


class FBDataSet(Dataset):

    def __init__(self, split_index, lb, rb) -> None:
        super().__init__()
        self.split_index = split_index
        # dataset [lb, rb)
        self.lb = lb
        self.rb = rb
        
    def __len__(self):
        return 11 * (self.rb - self.lb)

    def __getitem__(self, idx):
        L = (self.rb - self.lb)
        label = idx // L
        bn = ['cppds', 'defenseds2', 'flockingds', 'formation', 'hideds', 'leaderfollowers_3d', 'lineds',
         'patrol', 'pass3d', 'poi', 'treesds2'][label]
        pkl_index = idx %  L + self.lb
        fn = f'./new_3d_normalized/{bn}/{pkl_index}.pkl'
        # print(idx, fn, label)
        with open(fn, 'rb') as fin:
            data = pickle.load(fin)
        L_S = np.random.randint(0, data.shape[0] - self.split_index - 1)
        data = data[L_S:L_S + self.split_index, :10, :].reshape(-1, 30)
        return data.astype('float32'), label


class MixedSeqDataset(Dataset):

    def __init__(self, root_path, label, split_index, mean_std) -> None:
        super().__init__()
        self.root_path = root_path
        self.label = label
        self.split_index = split_index
        self.mean_std = mean_std

    def __len__(self):
        return 100

    def __getitem__(self, index):
        fn = os.path.join(self.root_path, f'{index}.pkl')
        with open(fn, 'rb') as fin:
            data = pickle.load(fin)
        mean = self.mean_std[0]
        std = self.mean_std[1]
        data = (data - mean) / std
        L_S = np.random.randint(0, data.shape[0] - self.split_index - 1)
        data = data[L_S:L_S + self.split_index, :10, :].reshape(-1, 30)
        label = self.label
        return data.astype('float32'), label



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def main():
    N = 15
    # --- fix randomness ---
    setup_seed(19971023)
    # --- fix randomness ---
    
    # 80% for training, 20% for evaluation
    ds_train = FBDataSet(N, 0, 8000)
    ds_eval = FBDataSet(N, 8000, 10000)

    train_dl = DataLoader(ds_train, batch_size=512, shuffle=True, num_workers=10)
    eval_dl = DataLoader(ds_eval, batch_size=512, shuffle=True, num_workers=10)
    net = RNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for episode in range(100):
        total_loss = 0
        for batch in tqdm(train_dl):
            images, labels = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # print(images.shape)

            output = net(images)
            loss = loss_fn(output, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # if episode < 40:
        #     continue
        # eval
        if episode > 0:
            correct = 0
            total = 0
            for test_batch in eval_dl:
                images, labels = test_batch
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                with torch.no_grad():
                    output = net(images)
                    _, predicted = torch.max(output, 1)
                    total += images.shape[0]
                    correct += (predicted == labels).sum().item()

            torch.save(net.state_dict(), f'state_dict_{N}/model_EP{episode}_{N}.sd')
            print(f'Episode: {episode} Loss: {total_loss} Acc: {correct / total, correct, total}')


def evaluation():
    setup_seed(19971023)
    N = 15
    ds_eval = FBDataSet(N, 8000, 10000)
    ds_loader = DataLoader(ds_eval, 2000 * 10)
    net = RNN().to(DEVICE)
    net.load_state_dict(torch.load(f'./state_dict_{N}/model_EP99_{N}.sd'))
    pred = []
    real_label = []
    for d in ds_loader:
        images, labels = d
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            output = net(images)
            _, predicted = torch.max(output, 1)

        pred.extend(predicted.cpu().numpy())
        real_label.extend(labels.cpu().numpy())
    
    print(f'==========result===========')
    m = classification_report(real_label, pred)
    print(m)
    print(f'===========================')

    m = confusion_matrix(real_label, pred)
    print(m)


def eval_real_data():
    setup_seed(19971023)
    N = 15
    mean = [8.07585196, 4.89015749, 1.07418603] 
    std = [3.85663151, 1.94566411, 0.34342566]

    mean = [27.2549865, 16.41504084, 0.        ] 
    std = [15.03331502, 10.08243792, 1.        ]
    ds_eval = MixedSeqDataset('real_trees/mixed', 10, N, [mean, std])
    ds_loader = DataLoader(ds_eval, 100)
    net = RNN().to(DEVICE)
    net.load_state_dict(torch.load(f'./state_dict_{N}/model_EP99_{N}.sd'))
    pred = []
    real_label = []
    for d in ds_loader:
        images, labels = d
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            output = net(images)
            _, predicted = torch.max(output, 1)

        pred.extend(predicted.cpu().numpy())
        real_label.extend(labels.cpu().numpy())
    
    print(f'==========result===========')
    m = classification_report(real_label, pred)
    print(m)
    print(f'===========================')

    m = confusion_matrix(real_label, pred)
    print(m)


if __name__ == '__main__':
    # main()
    # evaluation()
    eval_real_data()
