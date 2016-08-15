from cs231n.multimodal.data_provider import word2vec_data
from cs231n.multimodal.data_provider.data_tests import data_config

d = data_config.dc

dd = word2vec_data.Word2VecData(w2v_vocab_fname=d['word2vec_vocab'], w2v_vectors_fname=d['word2vec_vectors'])

# X_txt_zappos = dd.get_external_word_vectors()
# print X_txt_zappos

# raw_input("enter key to continue")

# X_txt_zappos = dd.get_word_vectors(external_vocab=True)
# print X_txt_zappos

# raw_input("enter key to continue")

X_txt = dd.get_word_vectors_of_word_list(['random_stuff', 'cat', 'is', 'nice', 'cat'])

print X_txt
# print X_txt.shape
