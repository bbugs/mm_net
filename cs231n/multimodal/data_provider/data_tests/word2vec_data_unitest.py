from cs231n.multimodal.data_provider import word2vec_data
from cs231n.multimodal.data_provider.data_tests import data_config

d = data_config.dc

dd = word2vec_data.Word2VecData(d)

dd.set_word_vectors(verbose=True)

# dd.set_external_vocab()
X_txt_zappos = dd.get_external_word_vectors()
print X_txt_zappos

# raw_input("enter key to continue")

X_txt_zappos = dd.get_word_vectors(external_vocab=True)
print X_txt_zappos

# raw_input("enter key to continue")

X_txt = dd.get_word_vectors_of_word_list(['random_stuff', 'cat', 'is', 'nice', 'cat'])

print X_txt
