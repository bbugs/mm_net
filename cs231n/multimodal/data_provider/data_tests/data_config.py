# Set data config dc

config = {}

# assume all data has been precomputed in dc['root_path']
config['root_path'] = root_path = '../../../../DeepFashion/'  # assume module is run from assignment2

# Image CNN (Full Image + Regions) features
config['num_regions_per_img'] = 4 + 1
config['cnn_regions_path'] = root_path + 'data/fashion53k/img_regions/4_regions_cnn/per_split/'
config['cnn_regions_path_train'] = config['cnn_regions_path'] + '/cnn_fc7_train.txt'
config['cnn_regions_path_val'] = config['cnn_regions_path'] + '/cnn_fc7_val.txt'
config['cnn_regions_path_test'] = config['cnn_regions_path'] + '/cnn_fc7_test.txt'

# Image CNN (Full Image only) features
# TODO: Make cnn features for full images only
# TODO: see where num_regions apply
config['cnn_full_img_path'] = root_path + '/data/fashion53k/full_img/per_split/'
config['cnn_full_img_path_train'] = config['cnn_full_img_path'] + '/cnn_fc7_train.txt'
config['cnn_full_img_path_val'] = config['cnn_full_img_path'] + '/cnn_fc7_val.txt'
config['cnn_full_img_path_test'] = config['cnn_full_img_path'] + '/cnn_fc7_test.txt'

# Text features
config['json_path'] = root_path + 'data/fashion53k/json/with_ngrams/'
config['json_path_train'] = config['json_path'] + 'dataset_dress_all_train.clean.json'
config['json_path_val'] = config['json_path'] + 'dataset_dress_all_val.clean.json'
config['json_path_test'] = config['json_path'] + 'dataset_dress_all_test.clean.json'

# Word2vec vectors and vocab
config['word2vec_vocab'] = root_path + 'data/word_vects/glove/vocab.txt'
config['word2vec_vectors'] = root_path + 'data/word_vects/glove/vocab_vecs.txt'

# External vocabulary
config['external_vocab'] = root_path + 'data/fashion53k/external_vocab/zappos.vocab.txt'

# target vocab (used in alignment_data.py on make_y_true_img2txt)
#TODO: see where does min_freq applies (perhaps make the clean json alread remove words with less than min_freq)
config['train_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_train_min_freq_5.txt'  #
config['val_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_val_min_freq_5.txt'  # do we need this ??
config['test_vocab'] = root_path + '/fashion53k/vocab_per_split/vocab_test_min_freq_5.txt'  # do we need this ??

config['target_vocab_fname'] = config['test_vocab']

####################################################################
# Set loss parameters
####################################################################
config['reg'] = 1  # regularization
config['hidden_dim'] = 700  # size of multimodal space
config['finetune_cnn'] = False
config['finetune_w2v'] = False

# local loss params
config['uselocal'] = True
config['local_margin'] = 1.
config['local_scale'] = 1.
config['do_mil'] = False

# global loss params
config['useglobal'] = False
config['global_margin'] = 40.
config['global_scale'] = 1.
config['smooth_num'] = 5.
config['global_method'] = 'sum'  # maxaccum
config['thrglobalscore'] = False

####################################################################
# Set optimization parameters
####################################################################

config['lr'] = 1e-6  # learning rate
config['lr_decay'] = 0.95
config['num_epochs'] = 10
config['batch_size'] = 100



