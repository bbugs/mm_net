import numpy as np
import os
import json
import collections
import linecache

from cs231n.multimodal.data_provider.vocab_data import Vocabulary


class Word2VecData(object):

    def __init__(self, w2v_vocab_fname, w2v_vectors_fname):

        self.w2v_vocab_fname = w2v_vocab_fname
        self.w2v_vectors_fname = w2v_vectors_fname

        self.word2vec_dim = 0
        self.word2vec_vocab = None

        # self.external_vocab = None
        # self.external_word_vectors = np.array([])
        return

    def set_word2vec_vocab(self):
        self.word2vec_vocab = Vocabulary(self.w2v_vocab_fname)

        # with open(self.d['word2vec_vocab'], 'rb') as f:
        #     self.word2vec_vocab = [w.replace('\n', '') for w in f.readlines()]

    def set_word2vec_dim(self):
        # read the first line of word2vec vector file and check the dimension
        vec = np.fromstring(linecache.getline(self.w2v_vectors_fname, 1), sep=" ")
        self.word2vec_dim = vec.shape[0]

    def get_word2vec_dim(self):
        if self.word2vec_dim == 0:
            self.set_word2vec_dim()
        return self.word2vec_dim

    def get_word_vectors_of_word_list(self, word_list):

        if self.word2vec_vocab is None:
            self.set_word2vec_vocab()

        if self.word2vec_dim == 0:
            self.set_word2vec_dim()

        word2id, id2word = self.word2vec_vocab.get_vocab_dicts()

        X_txt = np.zeros((len(word_list), self.word2vec_dim))

        i = 0
        for word in word_list:
            if word not in word2id:
                # seed is set so that we don't get different results every time this is called
                np.random.seed(42)
                X_txt[i, :] = np.random.randn(self.word2vec_dim)
                print word, " not in word2vec"
                i += 1
                continue

            w_id = word2id[word]
            # X_txt[i, :] = self.word2vec_vectors[w_id, :]
            # Note that linecache line numbers start at 1.
            X_txt[i, :] = np.fromstring(linecache.getline(self.w2v_vectors_fname, w_id + 1), sep=" ")
            i += 1

        return X_txt

    # def get_vocab(self, vocab_name='word2vec'):
    #
    #     if vocab_name == 'word2vec':
    #         if len(self.word2vec_vocab) == 0:
    #             self.set_word2vec_vocab()
    #         return self.word2vec_vocab
    #
    #     if vocab_name == 'external':
    #         if len(self.external_vocab) == 0:
    #             self.set_external_vocab()
    #         return self.external_vocab
    #
    # def get_external_vocab(self):
    #     if len(self.external_vocab) == 0:
    #         self.set_external_vocab()
    #     return self.external_vocab
    #
    # def get_vocab_dicts(self, vocab_name='word2vec'):
    #     """
    #     Return word2id, id2word of the vocab_name
    #     """
    #     if vocab_name == 'word2vec':
    #         if len(self.word2vec_vocab) == 0:
    #             self.set_word2vec_vocab()
    #         vocab = self.word2vec_vocab
    #
    #     elif vocab_name == 'external':
    #         if len(self.external_vocab) == 0:
    #             self.set_external_vocab()
    #         vocab = self.external_vocab
    #
    #     else:
    #         raise ValueError("vocabs supported are word2vec and external only")
    #
    #     word2id = {}
    #     id2word = {}
    #     i = 0
    #     for word in vocab:
    #         word2id[word] = i
    #         id2word[i] = word
    #         i += 1
    #     return word2id, id2word
    #
    #
    #
    # def _load_cnn_features(self, cnn_fname):
    #     self.cnn = np.loadtxt(cnn_fname, delimiter=',')
    #
    # def get_split(self, split='test', num_samples=-1):
    #     X_img = np.array([])
    #     X_txt = np.array([])
    #     region2pair_id = []
    #     word2pair_id = []
    #
    #     return X_img, X_txt, region2pair_id, word2pair_id
