import numpy as np
import csv
import time
import json


def gen_word_to_pair_id(word_seq):
    """(list of np arrays) -> np array

    """

    N = len(word_seq)

    word2pair_id = np.array([])  # empty array

    for i in range(N):
        word_ids = word_seq[i]
        pair_ids = i * np.ones(word_ids.shape, dtype=int)

        word2pair_id = np.hstack((word2pair_id, pair_ids))

    return word2pair_id


class AlignmentData(object):

    def __init__(self, data_config, split):
        self.d = data_config
        self.split = split
        if split == 'train':
            fname = self.d['json_path_train']
        elif split == 'val':
            fname = self.d['json_path_val']
        elif split == 'test':
            fname = self.d['json_path_test']
        else:
            raise ValueError("only treain, val and test splits supported")
        self.json_file = JsonFile(fname)

        return

    @staticmethod
    def make_region2pair_id(img_ids, num_regions_per_img):
        region2pair_id = np.zeros((len(img_ids) * num_regions_per_img, ))

        i = 0
        for img_id in img_ids:
            region2pair_id[i: i + num_regions_per_img] = i
            i += num_regions_per_img

        return region2pair_id

    def make_word2pair_id(self, img_ids):
        word2pair_id = np.array([])  # empty array

        counter = 0
        for img_id in img_ids:
            unique_word_list = self.json_file.get_word_list_of_img_id(img_id)

            n_words = len(unique_word_list)
            pair_ids = counter * np.ones(n_words, dtype=int)

            word2pair_id = np.hstack((word2pair_id, pair_ids))
            counter += 1

        return word2pair_id

    @staticmethod
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

    def make_y_true_txt2img(self):
        """
        y is y_true_zappos_img for txt2img
        """

        return

    def make_y_true_img2txt(self):
        """
        y is y_true_all_vocab for img2txt
        """


        return


class JsonFile(object):

    def __init__(self, json_fname):
        with open(json_fname, 'r') as f:
            self.dataset = json.load(f)

        return

    def get_ids_split(self, target_split='train'):
        ids = []
        for item in self.dataset['items']:
            imgid = item['imgid']
            split = item['split']
            if split == target_split:
                ids.append(imgid)
        return ids

    def get_item_from_img_id(self, target_img_id):
        item = None
        for item in self.dataset['items']:
            imgid = item['imgid']
            if imgid == target_img_id:
                break  # found teh right item
        return item

    def get_word_list_of_img_id(self, img_id):
        """
        return a list of unique words that correspond to the img_id
        """
        word_list = []
        item = self.get_item_from_img_id(img_id)
        if item is None:
            return word_list
        txt = item['text']
        word_list = list(set([w.replace('\n') for w in txt.split(" ")]))
        return word_list


    def get_text_of_img_ids(self, img_ids):
        id2text = {}
        for img_id in img_ids:
            # get item from json
            item = self.get_item_from_img_id(img_id)
            if item is None:
                print img_id, " not found"
                continue
            txt = item['text']
            id2text[img_id] = txt
        return id2text


class JsonData(object):

    def __init__(self, data_config):
        self.d = data_config

        self.json_train = {}
        self.json_val = {}
        self.json_test = {}

        return

    def set_json_split(self, split='test'):
        if split == 'train':
            fname = self.d['cnn_regions_path_train']
            self.json_train = np.loadtxt(fname, delimiter=',')
        elif split == 'val':
            fname = self.d['cnn_regions_path_val']
            self.json_val = np.loadtxt(fname, delimiter=',')
        elif split == 'test':
            fname = self.d['cnn_regions_path_test']
            self.json_test = np.loadtxt(fname, delimiter=',')
        else:
            raise ValueError("only train, val and test splits supported")
        return


class CnnData(object):

    def __init__(self, data_config):
        self.d = data_config

        # full image cnn
        self.cnn_full_img_train = np.array([])
        self.cnn_full_img_val = np.array([])
        self.cnn_full_img_test = np.array([])

        # full image + regions cnn
        self.cnn_region_train = np.array([])
        self.cnn_region_val = np.array([])
        self.cnn_region_test = np.array([])

        return

    def set_cnn_full_img_split(self, split='test'):
        if split == 'train':
            fname = self.d['cnn_full_img_path_train']
            self.cnn_full_img_train = np.loadtxt(fname, delimiter=',')
        elif split == 'val':
            fname = self.d['cnn_full_img_path_val']
            self.cnn_full_img_val = np.loadtxt(fname, delimiter=',')
        elif split == 'test':
            fname = self.d['cnn_full_img_path_test']
            self.cnn_full_img_test = np.loadtxt(fname, delimiter=',')
        else:
            raise ValueError("only train, val and test splits supported")
        return

    def set_cnn_regions_split(self, split):
        if split == 'train':
            fname = self.d['cnn_regions_path_train']
            self.cnn_region_train = np.loadtxt(fname, delimiter=',')
        elif split == 'val':
            fname = self.d['cnn_regions_path_val']
            self.cnn_region_val = np.loadtxt(fname, delimiter=',')
        elif split == 'test':
            fname = self.d['cnn_regions_path_test']
            self.cnn_region_test = np.loadtxt(fname, delimiter=',')
        else:
            raise ValueError("only train, val and test splits supported")
        return

    def get_cnn_regions_split(self, split):
        if split == 'train':
            if self.cnn_region_train.size == 0:
                self.set_cnn_regions_split(split)
            return self.cnn_region_train
        elif split == 'val':
            if self.cnn_region_val.size == 0:
                self.set_cnn_regions_split(split)
            return self.cnn_region_val
        elif split == 'test':
            if self.cnn_region_test.size == 0:
                self.set_cnn_regions_split(split)
            return self.cnn_region_test
        else:
            raise ValueError("only train, val and test splits supported")

    def get_cnn_full_img_split(self, split):
        if split == 'train':
            if self.cnn_full_img_train.size == 0:
                self.set_cnn_full_img_split(split)
            return self.cnn_full_img_train
        elif split == 'val':
            if self.cnn_full_img_val.size == 0:
                self.set_cnn_full_img_split(split)
            return self.cnn_full_img_val
        elif split == 'test':
            if self.cnn_full_img_test.size == 0:
                self.set_cnn_full_img_split(split)
            return self.cnn_full_img_test
        else:
            raise ValueError("only train, val and test splits supported")


class Word2VecData(object):

    def __init__(self, data_config):
        self.d = data_config

        self.word2vec_dim = 0
        self.word2vec_vocab = []
        self.word2vec_vectors = np.array([])  # load from word2vec_vectors

        self.external_vocab = []
        self.external_word_vectors = np.array([])
        return

    def _load_cnn_features(self, cnn_fname):
        self.cnn = np.loadtxt(cnn_fname, delimiter=',')

    def get_split(self, split='test', num_samples=-1):
        X_img = np.array([])
        X_txt = np.array([])
        region2pair_id = []
        word2pair_id = []

        return X_img, X_txt, region2pair_id, word2pair_id

    def set_word_vectors(self, verbose=False):
        # TODO: consider using linecache to load only the lines that you need and not the entire word2vec matrix
        # load word2vec_vocab if not already loaded
        if len(self.word2vec_vocab) == 0:
            if verbose:
                print "loading word2vec vocab... \n"
            self.set_word2vec_vocab()

        # load word2vec vectors
        if verbose:
            print "loading word2vec vectors... \n"

        # start = time.time()
        self.word2vec_vectors = np.loadtxt(self.d['word2vec_vectors'])
        # end = time.time()

        # set word2vec_dim
        self.word2vec_dim = self.word2vec_vectors.shape[1]

        if verbose:
            print "word2vec shape", self.word2vec_vectors.shape
            # print str((end - start)) + "s"

        return

    def set_word2vec_vocab(self):
        with open(self.d['word2vec_vocab'], 'rb') as f:
            self.word2vec_vocab = [w.replace('\n', '') for w in f.readlines()]

    def set_external_vocab(self):
        with open(self.d['external_vocab'], 'rb') as f:
            self.external_vocab = [w.replace('\n', '') for w in f.readlines()]

    def set_external_word_vectors(self):

        if self.word2vec_vectors.size == 0:
            self.set_word_vectors()

        word2id, id2word = self.get_vocab_dicts(vocab_name='word2vec')

        self.external_word_vectors = np.zeros((len(self.external_vocab), self.word2vec_dim))
        i = 0
        for word in self.external_vocab:
            if word not in word2id:
                # TODO: change ngrams from _ to -.
                # TODO: initialize weights randomly if not found in word2vec
                print word + " not in word2vec"
                i += 1
                continue
            id = word2id[word]
            self.external_word_vectors[i, :] = self.word2vec_vectors[id, :]
            i += 1
        return

    def get_external_word_vectors(self):
        """
        Convinience method to return X_txt_zappos
        """
        if self.external_word_vectors.size == 0:
            self.set_external_word_vectors()
        return self.external_word_vectors

    def get_word_vectors(self, external_vocab=False):
        """
        Convinience method to return either
        X_txt_word2vec (external_vocab=False) or
        X_txt_zappos (external_vocab=True)
        """

        # return the subset of word vectors for the external vocabulary
        if external_vocab:
            if self.external_word_vectors.size == 0:
                self.set_external_word_vectors()
            return self.external_word_vectors

        # return the ALL word vectors
        else:
            if self.word2vec_vectors.size == 0:
                self.set_word_vectors()
            return self.word2vec_vectors

    def get_word_vectors_of_word_list(self, word_list):
        if self.word2vec_dim == 0:
            print "loading word vectors ..."
            self.set_word_vectors()

        word2id, id2word = self.get_vocab_dicts(vocab_name='word2vec')

        X_txt = np.zeros((len(word_list), self.word2vec_dim))

        i = 0
        for word in word_list:
            if word not in word2id:
                # TODO: initiliaze weights randomly if word not found in word2vec (or set it to a common word?)
                print word, " not in word2vec"
                i += 1
                continue

            id = word2id[word]
            X_txt[i, :] = self.word2vec_vectors[id, :]
            i += 1

        return X_txt

    def get_vocab(self, vocab_name='word2vec'):

        if vocab_name == 'word2vec':
            if len(self.word2vec_vocab) == 0:
                self.set_word2vec_vocab()
            return self.word2vec_vocab

        if vocab_name == 'external':
            if len(self.external_vocab) == 0:
                self.set_external_vocab()
            return self.external_vocab

    def get_vocab_dicts(self, vocab_name='word2vec'):
        """
        Return word2id, id2word of the vocab_name
        """
        if vocab_name == 'word2vec':
            if len(self.word2vec_vocab) == 0:
                self.set_word2vec_vocab()
            vocab = self.word2vec_vocab

        elif vocab_name == 'external':
            if len(self.external_vocab) == 0:
                self.set_external_vocab()
            vocab = self.external_vocab

        else:
            raise ValueError("vocabs supported are word2vec and external only")

        word2id = {}
        id2word = {}
        i = 0
        for word in vocab:
            word2id[word] = i
            id2word[i] = word
            i += 1
        return word2id, id2word



if __name__ == '__main__':

    # dd = Data(d)
    #
    # dd.set_word_vectors()

    pass

