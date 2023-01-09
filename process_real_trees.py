import pickle
import numpy as np
import pandas as pd

def to_tuple(s):
    return eval(s)


def main():
    for i in range(10):
        df = pd.read_excel(f'./real_trees/{i}.xlsx', sheet_name='action_test')
        pos_seq = []
        for p in df['position']:
            p = to_tuple(p)
            pos_seq.append([p[0], p[1], 0])

        result = np.array(pos_seq)
        result = result[:, np.newaxis, :]
        # print(result)
        with open(f'./real_trees/{i}.pkl', 'wb') as fout:
            pickle.dump(result, fout)


if __name__ == '__main__':
    main()

