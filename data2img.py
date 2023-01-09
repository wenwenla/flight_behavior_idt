import os
import sys
import numpy as np
import pickle
from tqdm import trange
from PIL import Image, ImageDraw


def load_ds(fn):
    with open(fn, 'rb') as fin:
        data = pickle.load(fin)
    # print(data)
    # input('...')
    return data


def normalize_all(root, path, target):
    # _, path, fn = next(os.walk(root))
    # print(path)
    tmp = []
    # print('calc-mean-std...')
    for i in range(10000):
        data = load_ds(os.path.join(root, path, f'{i}.pkl'))
        tmp.extend(data)
    tmp = np.array(tmp)
    m = np.mean(tmp, axis=(0, 1))
    s = np.std(tmp, axis=(0, 1))
    if s[2] == 0.0:
        s[2] = 1.0
    print(path, m, s)
    
    # print('save files...')
    # for i in trange(10000):
    #     new_path = os.path.join(target, path)
    #     if not os.path.exists(new_path):
    #         os.makedirs(new_path)
    #     data = load_ds(os.path.join(root, path, f'{i}.pkl'))
    #     data = (data - m) / s
    #     with open(os.path.join(new_path, f'{i}.pkl'), 'wb') as fout:
    #         pickle.dump(data, fout)


def data_processing(folder):
    W = 128
    for i in trange(10000):
        data = load_ds(f'{folder}/{i}.pkl')[:,:,:2]
        min_vals = np.min(data, axis=(0, 1))
        max_vals = np.max(data, axis=(0, 1))

        print(i, min_vals, max_vals)

        data = (data - min_vals) / (max_vals - min_vals)
        img = Image.new('L', (W, W))
        drawer = ImageDraw.Draw(img)

        for agent in range(data.shape[1]):
            seq = (data[:, agent, :] * W)
            seq = seq.astype('int')
            seq = [(s[0], s[1]) for s in seq]
            drawer.line(seq, fill='white', width=0)
        img.save(f'{folder}/{i}.jpg')


def main():
    rt, path, files = next(os.walk('./new_3d'))
    
    # for p in path:
    #     folder = os.path.join(rt, p)
    #     print(f'Processing {folder}...')
    #     data_processing(folder)

    
    # for p in path:
    #     normalize_all('./new_3d', p, './new_3d_normalized')
    data_processing('./real_pass/mixed')


if __name__ == '__main__':
    main()
