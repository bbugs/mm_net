import numpy as np
import os
import json
import collections
import linecache

from cs231n.multimodal.data_provider.vocab_data import Vocabulary


class Word2VecData(object):

    # TODO: eliminate dependency to data_config. Inputs: word2vec_vocab_fname,
    # word2vec_vectors_fname. Most useful method is  get_word_vectors_of_word_list

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
        # read the first line of word2vec vector file and check the dimenision
        vec = np.fromstring(linecache.getline(self.w2v_vectors_fname, 1), sep=" ")
        self.word2vec_dim = vec.shape[0]

    # def set_external_vocab(self):
    #     self.external_vocab = Vocabulary(self.d['external_vocab'])
    #     # with open(self.d['external_vocab'], 'rb') as f:
    #     #     self.external_vocab = [w.replace('\n', '') for w in f.readlines()]
    #
    # def set_word_vectors(self, verbose=False):
    #     # TODO: consider using linecache to load only the lines that you need and not the entire word2vec matrix
    #     # set word2vec_vocab if not already loaded
    #     if self.word2vec_vocab is None:
    #         if verbose:
    #             print "loading word2vec vocab... \n"
    #         self.set_word2vec_vocab()
    #
    #     # load word2vec vectors
    #     if verbose:
    #         print "loading word2vec vectors... \n"
    #
    #     # start = time.time()
    #     self.word2vec_vectors = np.loadtxt(self.d['word2vec_vectors'])
    #     # end = time.time()
    #
    #     # set word2vec_dim
    #     self.word2vec_dim = self.word2vec_vectors.shape[1]
    #
    #     if verbose:
    #         print "word2vec shape", self.word2vec_vectors.shape
    #         # print str((end - start)) + "s"
    #
    #     return

        # def set_external_word_vectors(self):
    #
    #     if self.word2vec_vectors.size == 0:
    #         self.set_word_vectors()
    #
    #     if self.external_vocab is None:
    #         self.set_external_vocab()
    #
    #     word2id, id2word = self.word2vec_vocab.get_vocab_dicts()
    #
    #     ext_vocab = self.external_vocab.get_vocab()
    #     self.external_word_vectors = np.zeros((len(ext_vocab), self.word2vec_dim))
    #     i = 0
    #     for word in ext_vocab:
    #         if word not in word2id:
    #             # TODO: change ngrams from _ to -.
    #             # TODO: initialize weights randomly if not found in word2vec
    #             print word + " not in word2vec"
    #             i += 1
    #             continue
    #         w_id = word2id[word]
    #         self.external_word_vectors[i, :] = self.word2vec_vectors[w_id, :]
    #         i += 1
    #     return

    # def get_word_vectors(self, external_vocab=False):
    #     """
    #     Convinience method to return either
    #     X_txt_word2vec (external_vocab=False) or
    #     X_txt_zappos (external_vocab=True)
    #     """
    #
    #     # return the subset of word vectors for the external vocabulary
    #     if external_vocab:
    #         if self.external_word_vectors.size == 0:
    #             self.set_external_word_vectors()
    #         return self.external_word_vectors
    #
    #     # return the ALL word vectors
    #     else:
    #         if self.word2vec_vectors.size == 0:
    #             self.set_word_vectors()
    #         return self.word2vec_vectors

    # def get_external_word_vectors(self):
    #     """
    #     Convinience method to return X_txt_zappos
    #     """
    #     if self.external_word_vectors.size == 0:
    #         self.set_external_word_vectors()
    #     return self.external_word_vectors

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
                # TODO: initiliaze weights randomly if word not found in word2vec (or set it to a common word?)
                print word, " not in word2vec"
                i += 1
                continue

            w_id = word2id[word]
            # X_txt[i, :] = self.word2vec_vectors[w_id, :]
            # Note that linecache line numbers start at 1.
            X_txt[i, :] = np.fromstring(linecache.getline(self.w2v_vectors_fname, w_id + 1), sep=" ")
            #TODO: test that linecache reads in the right word vecto
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
