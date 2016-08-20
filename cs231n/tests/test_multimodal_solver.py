
from cs231n.multimodal.data_provider.experiment_data import BatchData, EvaluationData
from cs231n.multimodal.data_provider.cnn_data import CnnData
from cs231n.multimodal.data_provider.word2vec_data import Word2VecData
from cs231n.multimodal import multimodal_net
from cs231n.multimodal.multimodal_solver import MultiModalSolver

from cs231n.multimodal.data_provider.data_tests import data_config
from cs231n.multimodal import multimodal_utils
import math

##############################################
# Set batch data object
##############################################
print "setting batch data"
json_fname_train = data_config.config['json_path_train']
cnn_fname_train = data_config.config['cnn_regions_path_train']
num_regions_per_img = data_config.config['num_regions_per_img']
imgid2region_indices_train = multimodal_utils.mk_toy_img_id2region_indices(json_fname_train,
                                                                           num_regions_per_img=num_regions_per_img,
                                                                           subset_num_items=-1)
num_items_train = len(imgid2region_indices_train)
w2v_vocab_fname = data_config.config['word2vec_vocab']
w2v_vectors_fname = data_config.config['word2vec_vectors']

batch_data = BatchData(json_fname_train, cnn_fname_train,
                       imgid2region_indices_train,
                       w2v_vocab_fname, w2v_vectors_fname,
                       subset_num_items=1000)


##############################################
# Set evaluation data objects for train and val splits
##############################################
# ______________________________________________
# Train Evaluation Data
# ----------------------------------------------
print "setting evaluation data for train split"
# TODO: the evaluation data for both train and val should be with cnn for the full region
external_vocab_fname = data_config.config['external_vocab']

eval_data_train = EvaluationData(json_fname_train, cnn_fname_train, imgid2region_indices_train,
                                 w2v_vocab_fname, w2v_vectors_fname,
                                 external_vocab_fname, subset_num_items=100)
# ______________________________________________
# Val Evaluation Data
# ----------------------------------------------
print "setting evaluation data for val split"
json_fname_val = data_config.config['json_path_val']
cnn_fname_val = data_config.config['cnn_regions_path_val']
imgid2region_indices_val = multimodal_utils.mk_toy_img_id2region_indices(json_fname_val,
                                                                         num_regions_per_img=num_regions_per_img,
                                                                         subset_num_items=-1)

eval_data_val = EvaluationData(json_fname_val, cnn_fname_val, imgid2region_indices_val,
                               w2v_vocab_fname, w2v_vectors_fname,
                               external_vocab_fname, subset_num_items=25)

##############################################
# Set the model
##############################################
print "setting the model"
img_input_dim = CnnData(cnn_fname=data_config.config['cnn_regions_path_test']).get_cnn_dim()
txt_input_dim = Word2VecData(w2v_vocab_fname, w2v_vectors_fname).get_word2vec_dim()

# hyperparameters
reg = data_config.config['reg']
hidden_dim = data_config.config['hidden_dim']

# fine tuning
finetune_cnn = data_config.config['finetune_cnn']
finetune_w2v = data_config.config['finetune_w2v']

# local loss settings
uselocal = data_config.config['uselocal']
local_margin = data_config.config['local_margin']
local_scale = data_config.config['local_scale']
do_mil = data_config.config['do_mil']

# global loss settings
useglobal = data_config.config['useglobal']
global_margin = data_config.config['global_margin']
global_scale = data_config.config['global_scale']
smooth_num = data_config.config['smooth_num']
global_method = data_config.config['global_method']
thrglobalscore = data_config.config['thrglobalscore']

# weight scale for weight initialzation
std_img = math.sqrt(2. / img_input_dim)
std_txt = math.sqrt(2. / txt_input_dim)
weight_scale = {'img': std_img, 'txt': std_txt}

mm_net = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim, hidden_dim, weight_scale, reg=reg, seed=None,
                                      finetune_w2v=finetune_w2v, finetune_cnn=finetune_cnn)

mm_net.set_global_score_hyperparams(global_margin=global_margin, global_scale=global_scale,
                                    smooth_num=smooth_num, global_method=global_method,
                                    thrglobalscore=thrglobalscore)

mm_net.set_local_hyperparams(local_margin=local_margin, local_scale=local_scale, do_mil=do_mil)

##############################################
# Set optimization parameters
##############################################
print "setting the optimization parameters"
lr = data_config.config['lr'] # learning rate
lr_decay = data_config.config['lr_decay']
num_epochs = data_config.config['num_epochs']
batch_size = data_config.config['batch_size']


##############################################
# Train model with solver
##############################################
print "starting training"
solver = MultiModalSolver(mm_net, batch_data, eval_data_train, eval_data_val, num_items_train,
                          uselocal=uselocal,
                          useglobal=useglobal,
                          update_rule='sgd',
                          optim_config={'learning_rate': lr},
                          lr_decay=lr_decay,
                          num_epochs=num_epochs,
                          batch_size=batch_size,
                          print_every=2)

solver.train()


## Later
# for i in xrange(len(lr)):
#     for j in xrange(len(reg)):
#         model = TwoLayerNet(hidden_dim=300, reg=reg[j], weight_scale=1e-2)
#         solver = Solver(model, data,update_rule='rmsprop', optim_config={'learning_rate': lr[i],},
#                       lr_decay=0.95,
#                       num_epochs=10, batch_size=1024,
#                       print_every=1024)
#         solver.train()
#         if solver.best_val_acc > best_val:
#             best_model = model
#             best_val = solver.best_val_acc
#         results[lr[i],reg[j]] = solver.best_val_acc