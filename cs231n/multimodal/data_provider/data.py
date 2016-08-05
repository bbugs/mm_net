import numpy as np
import csv
import time


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
                # TODO: chnge ngrams from _ to -.
                # TODO: initialize weights randomly if not found in word2vec
                print word + " not in word2vec"
                i += 1
                continue
            id = word2id[word]
            self.external_word_vectors[i, :] = self.word2vec_vectors[id, :]
            i += 1
        return

    def get_external_word_vectors(self):
        if self.external_word_vectors.size == 0:
            self.set_external_word_vectors()
        return self.external_word_vectors

    def get_word_vectors(self, external_vocab=False):

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
            self.set_word_vectors()

        word2id, id2word = self.get_vocab_dicts(vocab_name='word2vec')

        X_txt = np.zeros((len(word_list), self.word2vec_dim))

        i = 0
        for word in word_list:
            id = word2id[word]
            X_txt[i, :] = self.word2vec_vectors[id, :]

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

