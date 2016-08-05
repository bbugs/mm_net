
from cs231n.multimodal.data_provider import data

d = {}

d['root_path'] = root_path = '../../../../DeepFashion/'  # assume module is run from assignment2

d['num_regions_per_img'] = 4 + 1

# Image CNN features
d['cnn_regions_path'] = root_path + 'data/fashion53k/img_regions/4_regions_cnn/per_split/'
d['cnn_regions_path_train'] = d['cnn_regions_path'] + '/cnn_fc7_train.txt'
d['cnn_regions_path_val'] = d['cnn_regions_path'] + '/cnn_fc7_val.txt'
d['cnn_regions_path_test'] = d['cnn_regions_path'] + '/cnn_fc7_test.txt'

# Text features
d['json_path'] = root_path + 'data/fashion53k/json/with_ngrams/'
d['json_path_train'] = d['json_path'] + 'dataset_dress_all_train.clean.json'
d['json_path_val'] = d['json_path'] + 'dataset_dress_all_val.clean.json'
d['json_path_test'] = d['json_path'] + 'dataset_dress_all_test.clean.json'

# Word2vec vectors and vocab
d['word2vec_vocab'] = root_path + 'data/word_vects/glove/vocab.txt'
d['word2vec_vectors'] = root_path + 'data/word_vects/glove/vocab_vecs.txt'
d['word2vec_dim'] = 200

# External vocabulary
d['external_vocab'] = root_path + 'data/fashion53k/external_vocab/zappos.vocab.txt'


dd = data.Word2VecData(d)
dd.set_word2vec_vocab()
# print dd.get_vocab(vocab_name='word2vec')

# dd.set_word_vectors(verbose=True)

dd.set_external_vocab()
# X_txt_zappos = dd.set_external_word_vectors()

X_txt_zappos = dd.get_word_vectors(external_vocab=True)

print ""
# print dd.get_vocab_dicts(vocab_name='external')