
import os
import pickle
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


DEVICE = 'cuda:3' #'cuda:0'


"""
数据集3D:  n * 8 * 3
进行了3D识别 - 二分类 - 识别准确率可达95%
"""


class RNN(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn_layear = torch.nn.LSTM(input_size=24, hidden_size=128, batch_first=True)
        self.fc0 = torch.nn.Linear(128, 128)
        self.fc1 = torch.nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.zeros((1, batch_size, 128)).to(DEVICE)
        c_0 = torch.zeros((1, batch_size, 128)).to(DEVICE)
        output, (h, c) = self.rnn_layear(x, (h_0, c_0))
        y0 = torch.relu(self.fc0(h[-1]))
        y1 = self.fc1(y0)
        return y1


def rename_data():
    import os
    rt, _, path = next(os.walk('3d_data/pass_the_hole'))
    print(path)
    for p in path:
        new_p = int(p[17:23])
        print(new_p)
        os.rename(os.path.join(rt, p), os.path.join(rt, f'{new_p}.pkl'))



class FlightBehaviorDataset(Dataset):

    def __init__(self, split_index) -> None:
        super().__init__()
        self.split_index = split_index
        
    def __len__(self):
        return 2 * 10000

    def __getitem__(self, idx):
        label = idx // 10000
        bn = ['pass_the_hole', 'leaderfollowers_3d'][idx // 10000]
        fn = f'./3d_data/{bn}/{idx % 10000}.pkl'
        with open(fn, 'rb') as fin:
            data = pickle.load(fin)
        L_S = np.random.randint(0, data.shape[0] - self.split_index - 1)
        data = data[L_S:L_S + self.split_index, :8, :].reshape(-1, 24)
        return data.astype('float32'), label


def main():
    ds = FlightBehaviorDataset(2)
    train_ds, eval_ds = random_split(ds, [0.1, 0.9])
    
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=10)
    eval_dl = DataLoader(eval_ds, batch_size=512, shuffle=True, num_workers=10)
    net = RNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for episode in range(50):
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

        if episode < 40:
            continue
        # eval
        correct = 0
        total = 0
        errors = []
        errors_index = 0
        for test_batch in eval_dl:
            images, labels = test_batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.no_grad():
                output = net(images)
                _, predicted = torch.max(output, 1)
                # if episode > 20:
                #     print(predicted)
                total += images.shape[0]
                correct += (predicted == labels).sum().item()


            with torch.no_grad():
                indices = (predicted != labels)
                error_images = images[indices].cpu()
                error_indices = indices.nonzero().cpu()

                for index, e in enumerate(error_images):
                    errors_index += 1
                    p = predicted[error_indices[index]]
                    t = labels[error_indices[index]]
                    errors.append((p.item(), t.item()))
        # if episode > 40:
        #     print(errors)

        torch.save(net.state_dict(), f'state_dict/model_EP{episode}_10.sd')
        
        print(f'Episode: {episode} Loss: {total_loss} Acc: {correct / total, correct, total}')


if __name__ == '__main__':
    main()
    # rename_data()
