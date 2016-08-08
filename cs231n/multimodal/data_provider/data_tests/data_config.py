# Set data config dc

dc = {}

dc['root_path'] = root_path = '../../../../DeepFashion/'  # assume module is run from assignment2

# Image CNN (Full Image + Regions) features
dc['num_regions_per_img'] = 4 + 1
dc['cnn_regions_path'] = root_path + 'data/fashion53k/img_regions/4_regions_cnn/per_split/'
dc['cnn_regions_path_train'] = dc['cnn_regions_path'] + '/cnn_fc7_train.txt'
dc['cnn_regions_path_val'] = dc['cnn_regions_path'] + '/cnn_fc7_val.txt'
dc['cnn_regions_path_test'] = dc['cnn_regions_path'] + '/cnn_fc7_test.txt'

# Image CNN (Full Image only) features
# TODO: Make cnn features for full images only
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
dc['train_vocab'] = 'fname'  # TODO: create this (with JsonFile.get_vocab_words+from_json(min_word_freq=5).
dc['val_vocab'] = 'fname'  # do we need this ??
dc['test_vocab'] = 'fname'  # do we need this ??

