import numpy as np
from cs231n.multimodal.data_provider.json_data import JsonFile
import itertools

def mk_toy_img_id2region_indices(json_fname, num_regions_per_img, subset_num_items=-1):

    json_file = JsonFile(json_fname, num_items=subset_num_items)

    img_ids = json_file.get_img_ids()

    img_id2region_indices = {}

    region_index = 0
    for img_id in img_ids:
        img_id2region_indices[img_id] = []
        for i in range(num_regions_per_img):
            img_id2region_indices[img_id].append(region_index)
            region_index += 1

    return img_id2region_indices


def get_num_lines_from_file(fname):
    counts = itertools.count()
    with open(fname) as f:
        for _ in f: counts.next()
    return counts.next()


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
    #
    # print y2pair_id(yy, N=3)

    from cs231n.multimodal.data_provider.data_tests import data_config

    fname = data_config.dc['json_path_test']
    imgid2regionind = mk_toy_img_id2region_indices(json_fname=fname, num_regions_per_img=5, subset_num_items=3)
    correct = {}
    correct[6] = [0,1,2,3,4]
    correct[80] = [5,6,7,8,9]
    correct[147] = [10, 11, 12, 13, 14]
    print imgid2regionind  # {80: [5, 6, 7, 8, 9], 147: [10, 11, 12, 13, 14], 6: [0, 1, 2, 3, 4]}
    # assert imgid2regionind == correct

    fname = data_config.dc['cnn_regions_path_test']
    print get_num_lines_from_file(fname)