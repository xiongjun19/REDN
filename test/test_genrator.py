# coding=utf8

import numpy as np
import torch


def init_matrix(rows, cols):
    res = np.zeros([rows, cols], np.int)
    for i in range(rows):
        for j in range(cols):
            res[i][j] = np.abs(i - j)
    return res


def init_pe(eb_lookup, pe_shape):
    pos_mat = init_matrix(pe_shape[-2], pe_shape[-1])
    pos_mat = pos_mat.reshape([1, -1])
    ts = torch.LongTensor(pos_mat)
    pos_tensor = eb_lookup(ts)
    pos_tensor = pos_tensor.view([pe_shape[-2], pe_shape[-1], -1])
    new_size = pe_shape + [pos_tensor.shape[-1]]
    new_tensor = pos_tensor.expand(new_size)
    return new_tensor


if __name__ == '__main__':
    emb_dict = torch.nn.Embedding(10, 20)
    pe = init_pe(emb_dict, [3, 2, 3, 3])
    print(pe)
    print(pe.shape)
