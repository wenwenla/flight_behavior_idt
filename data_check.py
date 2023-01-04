import os
import numpy as np
import pickle

def check(fn):
    for i in range(10):
        for j in range(i + 1, 10000):
            with open(os.path.join(fn, f'{i}.pkl'), 'rb') as fin:
                di = pickle.load(fin)
            with open(os.path.join(fn, f'{j}.pkl'), 'rb') as fin:
                dj = pickle.load(fin)

            if di.shape == dj.shape and np.all(di == dj):
                print(f'ERROR {fn}: {i} == {j}')
                return False
    return True


if __name__ == '__main__':

    dataset = ['cppds', 'defenseds', 'flockingds', 'formation', 'hideds', 'leaderfollowers', 'lineds', 'patrol', 'poi', 'treesds2']

    for d in dataset:
        if check(f'./mean_std_ds/{d}'):
            print(f'Dataset {d} passed similarity check.')