"""
Test the MultiModalNet against matlab implementation
susy_code_bare_bones/check_python_cost.m
"""

import os
from cs231n.multimodal import multimodal_net
import numpy as np
import math
from cs231n.layers import *
import scipy.io as sio
from cs231n.tests import test_utils

rpath = '../../nips2014_karpathy/susy_code_bare_bones/'

####################################################################
# Set loss parameters
####################################################################
loss_params = {}

loss_params['reg'] = 0.
loss_params['finetuneCNN'] = False

# local loss params
loss_params['uselocal'] = uselocal = True
loss_params['local_margin'] = local_margin = 1.
loss_params['local_scale'] = local_scale = 1.
loss_params['do_mil'] = do_mil = True

# global loss params
loss_params['useglobal'] = useglobal = False
loss_params['global_margin'] = global_margin = 40.
loss_params['global_scale'] = global_scale = 1.
loss_params['smooth_num'] = smotth_num = 5.
loss_params['global_method'] = global_method = 'maxaccum'
loss_params['thrglobalscore'] = thrglobalscore = False

sio.savemat(rpath + 'loss_params.mat', {'loss_params': loss_params})

####################################################################
# Create random data
####################################################################
np.random.seed(42)

N = 23  # number of image-sentence pairs in batch

n_sent_per_img = 5
n_sent = n_sent_per_img * N

n_region_per_img = 3
n_regions = n_region_per_img * N

n_words_per_img = 4
n_words = n_words_per_img * N

region2pair_id = test_utils.mk_random_pair_id(n_region_per_img, N)
word2pair_id = test_utils.mk_random_pair_id(n_words_per_img, N)

img_input_dim = 13 # size of cnn
txt_input_dim = 11  # size of word2vec pretrained vectors
hidden_dim = 7  # size of multimodal space

std_img = math.sqrt(2. / img_input_dim)
std_txt = math.sqrt(2. / txt_input_dim)
weight_scale = {'img': std_img, 'txt': std_txt}

X_img = np.random.randn(n_regions, img_input_dim)
X_txt = np.random.randn(n_words, txt_input_dim)

####################################################################
# Initialize multimodal net
####################################################################

mmnet = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim, hidden_dim, weight_scale, seed=42)

Wi2s = mmnet.params['Wi2s']
Wsem = mmnet.params['Wsem']
bi2s = mmnet.params['bi2s']
bsem = mmnet.params['bsem']

sio.savemat(rpath + 'X_img.mat', {'X_img': X_img})
sio.savemat(rpath + 'X_txt.mat', {'X_txt': X_txt})
sio.savemat(rpath + 'Wi2s.mat', {'Wi2s': Wi2s})
sio.savemat(rpath + 'Wsem.mat', {'Wsem': Wsem})
sio.savemat(rpath + 'bi2s.mat', {'bi2s': bi2s})
sio.savemat(rpath + 'bsem.mat', {'bsem': bsem})
sio.savemat(rpath + 'region2pair_id.mat', {'region2pair_id': region2pair_id})
sio.savemat(rpath + 'word2pair_id.mat', {'word2pair_id': word2pair_id})


mmnet.set_global_score_hyperparams(global_margin=global_margin, global_scale=global_scale,
                                   smooth_num=smotth_num, global_method=global_method,
                                   thrglobalscore=thrglobalscore)

mmnet.set_local_hyperparams(local_margin=local_margin, local_scale=local_scale, do_mil=do_mil)

loss, grads = mmnet.loss(X_img, X_txt, region2pair_id, word2pair_id,
                         useglobal=useglobal, uselocal=uselocal)

print "loss", loss
# print grads

###################
# Call Matlab check_python_cost.m to create the

# TODO: in the matlab code, save the cost and gradients, then load them here and compare resutls.
# For now, I just used a visual inspection and everything checks, except for the gradient wrt Wsem
# because I cannot compute it in matlab because I don't have depTrees and have to spend more time
# re-writing BackwardSents to not use depTrees.

os.system("matlab -nojvm -nodesktop < {0}/check_python_cost.m".format(rpath))

matlab_output = sio.loadmat(rpath + 'matlab_output.mat')

print matlab_output['cost']
print matlab_output['df_Wi2s']

assert np.allclose(loss, matlab_output['cost'][0][0])
assert np.allclose(grads['Wi2s'].T, matlab_output['df_Wi2s'])

# y = np.random.randint(N_words, size=N_img)  # indicate which word is correct (happens together) with each image






