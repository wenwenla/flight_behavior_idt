import pickle
import numpy as np


def main():
    with open('./real_pass/2.pkl', 'rb') as fin:
        real_data = pickle.load(fin)

    print(real_data.shape)
    for i in range(100):
        with open(f'./real_pass/data_for_mix/{i}.pkl', 'rb') as fin:
            simu_data = pickle.load(fin)
        # print(simu_data.shape)
        max_len = max(real_data.shape[0], simu_data.shape[0])
        real_pack = np.copy(real_data)
        simu_pack = np.copy(simu_data)

        while real_pack.shape[0] < max_len:
            real_pack = np.concatenate([real_pack, [real_pack[-1]]], axis=0)

        real_pack[:,:,0] += 1.0
        real_pack[:,:,1] += 4.0

        # print(np.min(real_pack, axis=(0, 1)))
        # print(np.max(real_pack, axis=(0, 1)))
        # sys.exit(0)

        while simu_pack.shape[0] < max_len:
            simu_pack = np.concatenate([simu_pack, [simu_pack[-1]]], axis=0)
        # print(real_pack.shape)
        # print(simu_pack.shape)
        simu_pack[:,:4,:] = real_pack
        with open(f'./real_pass/mixed/{i}.pkl', 'wb') as fout:
            pickle.dump(simu_pack, fout)


if __name__ == '__main__':
    main()
