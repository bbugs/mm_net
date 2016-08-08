
from cs231n.multimodal.data_provider import word2vec_data

d = {}

d['root_path'] = root_path = '../../../../DeepFashion/'  # assume module is run from assignment2

# Image CNN (Full Image + Regions) features
d['num_regions_per_img'] = 4 + 1
d['cnn_regions_path'] = root_path + 'data/fashion53k/img_regions/4_regions_cnn/per_split/'
d['cnn_regions_path_train'] = d['cnn_regions_path'] + '/cnn_fc7_train.txt'
d['cnn_regions_path_val'] = d['cnn_regions_path'] + '/cnn_fc7_val.txt'
d['cnn_regions_path_test'] = d['cnn_regions_path'] + '/cnn_fc7_test.txt'

# Image CNN (Full Image only) features
# TODO: Make cnn features for full images only
d['cnn_full_img_path'] = root_path + '/data/fashion53k/full_img/per_split/'
d['cnn_full_img_path_train'] = d['cnn_full_img_path'] + '/cnn_fc7_train.txt'
d['cnn_full_img_path_val'] = d['cnn_full_img_path'] + '/cnn_fc7_val.txt'
d['cnn_full_img_path_test'] = d['cnn_full_img_path'] + '/cnn_fc7_test.txt'

# Text features
d['json_path'] = root_path + 'data/fashion53k/json/with_ngrams/'
d['json_path_train'] = d['json_path'] + 'dataset_dress_all_train.clean.json'
d['json_path_val'] = d['json_path'] + 'dataset_dress_all_val.clean.json'
d['json_path_test'] = d['json_path'] + 'dataset_dress_all_test.clean.json'

# Word2vec vectors and vocab
d['word2vec_vocab'] = root_path + 'data/word_vects/glove/vocab.txt'
d['word2vec_vectors'] = root_path + 'data/word_vects/glove/vocab_vecs.txt'

# External vocabulary
d['external_vocab'] = root_path + 'data/fashion53k/external_vocab/zappos.vocab.txt'

# target vocab
d['target_vocab'] = 'fname'  # TODO: create this


dd = word2vec_data.Word2VecData(d)
# dd.set_word2vec_vocab()
# print dd.get_vocab(vocab_name='word2vec')

# dd.set_word_vectors(verbose=True)

# dd.set_external_vocab()
# X_txt_zappos = dd.set_external_word_vectors()

# X_txt_zappos = dd.get_word_vectors(external_vocab=True)


print dd.get_word_vectors_of_word_list(['random_stuff', 'cat', 'is', 'nice', 'cat'])

print ""
# print dd.get_vocab_dicts(vocab_name='external')