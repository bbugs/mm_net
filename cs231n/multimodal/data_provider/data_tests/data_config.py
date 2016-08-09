# Set data config dc

dc = {}

# assume all data has been precomputed in dc['root_path']
dc['root_path'] = root_path = '../../../../DeepFashion/'  # assume module is run from assignment2

# Image CNN (Full Image + Regions) features
dc['num_regions_per_img'] = 4 + 1
dc['cnn_regions_path'] = root_path + 'data/fashion53k/img_regions/4_regions_cnn/per_split/'
dc['cnn_regions_path_train'] = dc['cnn_regions_path'] + '/cnn_fc7_train.txt'
dc['cnn_regions_path_val'] = dc['cnn_regions_path'] + '/cnn_fc7_val.txt'
dc['cnn_regions_path_test'] = dc['cnn_regions_path'] + '/cnn_fc7_test.txt'

# Image CNN (Full Image only) features
# TODO: Make cnn features for full images only
# TODO: see where num_regions apply
dc['cnn_full_img_path'] = root_path + '/data/fashion53k/full_img/per_split/'
dc['cnn_full_img_path_train'] = dc['cnn_full_img_path'] + '/cnn_fc7_train.txt'
dc['cnn_full_img_path_val'] = dc['cnn_full_img_path'] + '/cnn_fc7_val.txt'
dc['cnn_full_img_path_test'] = dc['cnn_full_img_path'] + '/cnn_fc7_test.txt'

# Text features
dc['json_path'] = root_path + 'data/fashion53k/json/with_ngrams/'
dc['json_path_train'] = dc['json_path'] + 'dataset_dress_all_train.clean.json'
dc['json_path_val'] = dc['json_path'] + 'dataset_dress_all_val.clean.json'
dc['json_path_test'] = dc['json_path'] + 'dataset_dress_all_test.clean.json'

# Word2vec vectors and vocab
dc['word2vec_vocab'] = root_path + 'data/word_vects/glove/vocab.txt'
dc['word2vec_vectors'] = root_path + 'data/word_vects/glove/vocab_vecs.txt'

# External vocabulary
dc['external_vocab'] = root_path + 'data/fashion53k/external_vocab/zappos.vocab.txt'

# target vocab (used in alignment_data.py on make_y_true_img2txt)
#TODO: see where does min_freq applies
dc['train_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_train_min_freq_5.txt'  #
dc['val_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_val_min_freq_5.txt'  # do we need this ??
dc['test_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_test_min_freq_5.txt'  # do we need this ??

dc['target_vocab_fname'] = dc['test_vocab']
