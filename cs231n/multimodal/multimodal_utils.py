import numpy as np


def y2pair_id(y, N):
    """

    """

    region2pair_id = np.zeros(yy.shape[0])

    word2pair_id = np.zeros(yy.shape[1])

    i = 0
    k_region = 0
    k_word = 0
    col_counter = 0
    row_counter = 0
    while i < N:

        num_regions = len(np.where(y[:, col_counter] == 1)[0])
        num_words = len(np.where(y[row_counter, :] == 1)[0])

        region2pair_id[k_region: k_region + num_regions] = i
        word2pair_id[k_word: k_word + num_words] = i

        i += 1
        k_region += num_regions
        k_word += num_words

        row_counter += num_regions
        col_counter += num_words

    return region2pair_id, word2pair_id


def pair_id2y(region2pair_id, word2pair_id):

    N = np.max(region2pair_id)
    assert N == np.max(word2pair_id)

    n_regions = region2pair_id.shape[0]
    n_words = word2pair_id.shape[0]
    y = -np.ones((n_regions, n_words))

    for i in range(N + 1):
        MEQ = np.outer(region2pair_id == i, word2pair_id == i)
        y[MEQ] = 1

    return y


if __name__ == '__main__':
    yy = np.array([[1, 1, -1, -1, -1, -1, -1],
                  [1, 1, -1, -1, -1, -1, -1],
                  [-1, -1, 1, 1, -1, -1, -1],
                  [-1, -1, 1, 1, -1, -1, -1],
                  [-1, -1, 1, 1, -1, -1, -1],
                  [-1, -1, -1, -1, 1, 1, 1],
                  [-1, -1, -1, -1, 1, 1, 1],
                  [-1, -1, -1, -1, 1, 1, 1]], dtype=np.float)

    r2p = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    w2p = np.array([0, 0, 1, 1, 2, 2, 2])

    assert np.allclose(pair_id2y(r2p, w2p), yy)

    print y2pair_id(yy, N=3)