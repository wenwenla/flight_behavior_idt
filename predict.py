import torch
import numpy as np
from scipy.special import softmax
import pickle
import matplotlib.pyplot as plt


"""
MUST NORMALIZE THE DATA:

D = (D - MEAN) / STD

SUGGEST MEAN AND STD

hideds [20.79729158 20.30498147] [12.74107724  8.49194497]
flockingds [ 0.06114862 -0.12371165] [0.11280865 0.21755442]
lineds [18.73829483 16.40642671] [3.42504071 2.17322052]
patrol [311.19842472 331.43508942] [181.25028419 182.97797351]
cppds [21.80146768 18.74824417] [11.11151168 10.73092679]
treesds [29.06641607 13.12324931] [13.30089166 10.26865916]
poi [0.50895584 0.5965058 ] [0.21000028 0.24624227]
leaderfollowers [308.23919025 303.58594929] [172.42308291 179.5639752 ]
lambdads [18.01425921 21.73481789] [2.92321643 2.88203447]
defenseds [34.92932669 32.81388117] [ 8.20954526 11.39624929]
"""

DEVICE = 'cpu'



class RNN(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn_layear = torch.nn.LSTM(input_size=20, hidden_size=128, batch_first=True)
        self.fc0 = torch.nn.Linear(128, 128)
        self.fc1 = torch.nn.Linear(128, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.zeros((1, batch_size, 128)).to(DEVICE)
        c_0 = torch.zeros((1, batch_size, 128)).to(DEVICE)
        output, (h, c) = self.rnn_layear(x, (h_0, c_0))
        y0 = torch.relu(self.fc0(h[-1]))
        y1 = self.fc1(y0)
        return y1


class Predictor:

    def __init__(self, model_fn) -> None:
        self.model = RNN()
        self.model.load_state_dict(torch.load(model_fn, map_location='cpu'))

    def _get_score(self, data):
        assert data.shape == (1, 10, 20)
        x = torch.from_numpy(data).float().to(DEVICE)
        with torch.no_grad():
            y = self.model(x)
        return y.numpy()

    def get_prob(self, data):
        print(self._get_score(data))
        return softmax(self._get_score(data))    
    
    def get_cat_name(self, index):
        return [
            'cpp', 'defense', 'flocking', 'hide', 'lambda', 'leaderfollower', 'line', 'patrol', 'poi', 'tree'
        ][index]

    def save_image(self, data, fn):
        score = self._get_score(data)
        plt.figure(figsize=(10, 3))
        ax = plt.gca()
        ax.imshow(score, cmap='binary')
        labels = ['cpp', 'defense', 'flocking', 'hide', 'lambda', 'leaderfollower', 'line', 'patrol', 'poi', 'tree']
        ax.set_xticks([i for i in range(10)], labels=labels)
        ax.set_yticks([0], ['Pr'])
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
        plt.savefig(f'{fn}.jpg')



def main():

    NAME = 'cppds'
    DI = 3123
    START = 23
    
    for START in range(0, 100):
        if START == 50:
            NAME = 'flockingds'
        with open(f'./mean_std_ds/{NAME}/{DI}.pkl', 'rb') as fin:
            data_seq = pickle.load(fin)
        print(data_seq.shape)

        
        rnd_seq = data_seq[START:START + 10, 0:10, 0:2].reshape(1, 10, 20)
        print(rnd_seq.shape)

        p = Predictor('./state_dict/model_EP40_10.sd')
        
        prob = p.get_prob(rnd_seq)
        for i in range(10):
            print(f'{prob[0, i]:.4f}')

        p.save_image(rnd_seq, f'seq/heatmap_{START}')


def to_gif():
    import imageio
    images = []
    for f in range(100):
        fn = f'seq/heatmap_{f}.jpg'
        images.append(imageio.imread(fn))

    imageio.mimsave('seq.gif', images, duration=0.1)


if __name__ == '__main__':
    main()
    to_gif()
