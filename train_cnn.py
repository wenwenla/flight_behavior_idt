import os
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image, write_jpeg
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, Normalize, ConvertImageDtype, RandomRotation, RandomApply
from tqdm import trange, tqdm


DEVICE = 'cuda:3'


class CNNModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 11)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FBImageDataset(Dataset):

    def __init__(self, lb, rb, transform=None) -> None:
        super().__init__()
        self.transform = transform
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
        fn = f'./new_3d/{bn}/{pkl_index}.jpg'
        image = read_image(fn)
        if self.transform:
            image = self.transform(image)
        return image, label


class MixedImageDataset(Dataset):

    def __init__(self, root_path, label, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.root_path = root_path
        self.label = label

    def __len__(self):
        return 100

    def __getitem__(self, index):
        fn = os.path.join(self.root_path, f'{index}.jpg')
        image = read_image(fn)
        if self.transform:
            image = self.transform(image)
        label = self.label
        return image, label


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def main():

    setup_seed(19971023)
    transformer = Compose([
        ConvertImageDtype(torch.float),
        Normalize(0, 1),
    ])

    # 80% for training, 20% for evaluation
    ds_train = FBImageDataset(0, 8000, transformer)
    ds_eval = FBImageDataset(8000, 10000, transformer)

    train_dl = DataLoader(ds_train, batch_size=512, shuffle=True, num_workers=10)
    eval_dl = DataLoader(ds_eval, batch_size=512, shuffle=True, num_workers=10)
    net = CNNModule().to(DEVICE)
    opt = optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for episode in range(5):
        total_loss = 0
        for batch in tqdm(train_dl):
            images, labels = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            output = net(images)
            loss = loss_fn(output, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if True:
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

            torch.save(net.state_dict(), f'state_dict_img/model_EP{episode}.sd')
            print(f'Episode: {episode} Loss: {total_loss} Acc: {correct / total, correct, total}')


def evaluation():
    transformer = Compose([
        ConvertImageDtype(torch.float),
        Normalize(0, 1),
    ])
    ds_eval = FBImageDataset(8000, 10000, transformer)
    ds_loader = DataLoader(ds_eval, 2000 * 10)
    net = CNNModule().to(DEVICE)
    net.load_state_dict(torch.load(f'./state_dict_img/model_EP4.sd'))
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
    transformer = Compose([
        ConvertImageDtype(torch.float),
        Normalize(0, 1),
    ])
    ds = MixedImageDataset('real_pass/mixed', 7, transformer)
    ds_loader = DataLoader(ds, 100)
    net = CNNModule().to(DEVICE)
    net.load_state_dict(torch.load(f'./state_dict_img/model_EP4.sd'))

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
    main()
    evaluation()
    # eval_real_data()
