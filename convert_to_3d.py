import os
import pickle
import numpy as np


def convert(src, dest):
    rt, ph, files = next(os.walk(src))
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    for f in files:
        if f[-3:] != 'pkl' :
            continue
        if f[:3] == 'obs':
            continue

        src_fn = os.path.join(rt, f)
        dest_fn = os.path.join(dest, f)
        print(src_fn)
        with open(src_fn, 'rb') as fin:
            data = pickle.load(fin)
            assert len(data.shape) == 3
            assert data.shape[2] == 2
            new_shape = (data.shape[0], data.shape[1], 1)
            n = np.zeros(new_shape)

            new_data = np.concatenate((data, n), axis=2)
            print(new_data.shape)

        with open(dest_fn, 'wb') as fout:
            pickle.dump(new_data, fout)


def main():
    # convert('./dataset/treesds2', './new_3d/treesds2')
    # convert('./dataset/formation', './new_3d/formation')
    # convert('./dataset/flockingds', './new_3d/flockingds')
    # convert('./dataset/cppds', './new_3d/cppds')
    # convert('./dataset/hideds', './new_3d/hideds')
    # convert('./dataset/poi', './new_3d/poi')
    # convert('./dataset/lineds', './new_3d/lineds')
    # convert('./dataset/patrol', './new_3d/patrol')
    convert('./dataset/defenseds2', './new_3d/defenseds2')


if __name__ == '__main__':
    main()
