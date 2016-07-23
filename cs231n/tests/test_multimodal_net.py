"""
Test MultiModalNet against numerical gradient
"""

from cs231n.multimodal import multimodal_net
import numpy as np
import math
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.tests.test_utils import rel_error, mk_random_pair_id
from cs231n.layers import *

img_input_dim = 16  # size of cnn
txt_input_dim = 8  # size of word2vec pretrained vectors
hidden_dim = 15  # size of multimodal space

N = 25  # number of images in batch
n_regions_per_image = 4  # number of regions per image
n_regions = n_regions_per_image * N

n_words_per_img = 7
n_words = n_words_per_img * N # number of words in the batch

X_img = np.random.randn(n_regions, img_input_dim)
X_txt = np.random.randn(n_words, txt_input_dim)

std_img = math.sqrt(2. / img_input_dim)
std_txt = math.sqrt(2. / txt_input_dim)
weight_scale = {'img': std_img, 'txt': std_txt}

region2pair_id = mk_random_pair_id(n_regions_per_image, N)
word2pair_id = mk_random_pair_id(n_words_per_img, N)

# print region2pair_id
# print word2pair_id

useglobal = True
uselocal = False
mm_net = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim,
                                      hidden_dim, weight_scale=weight_scale, reg=0.0, seed=None)

mm_net.set_global_score_hyperparams(global_margin=40., global_scale=1., smooth_num=5.,
                                    global_method='sum', thrglobalscore=False)

mm_net.set_local_hyperparams(local_margin=1., local_scale=1., do_mil=True)


print 'Testing initialization ... '
Wi2s_std = abs(mm_net.params['Wi2s'].std() - std_img)
bi2s = mm_net.params['bi2s']
Wsem_std = abs(mm_net.params['Wsem'].std() - std_txt)
bsem = mm_net.params['bsem']
assert Wi2s_std < std_img, 'First layer weights do not seem right'
assert np.all(bi2s == 0), 'First layer biases do not seem right'
assert Wsem_std < std_txt, 'Second layer weights do not seem right'
assert np.all(bsem == 0), 'Second layer biases do not seem right'


print 'Testing training loss (no regularization)'
loss, grads = mm_net.loss(X_img, X_txt, region2pair_id, word2pair_id,
                          uselocal=uselocal, useglobal=useglobal)

print loss
print grads.keys()


print "Testing the gradients"

for reg in [0.0, 0.7, 10, 100, 1000]:
    print 'Running numeric gradient check with reg = ', reg
    mm_net.reg = reg
    loss, grads = mm_net.loss(X_img, X_txt, region2pair_id, word2pair_id,
                              uselocal=uselocal, useglobal=useglobal)

    for name in sorted(grads):
        f = lambda _: mm_net.loss(X_img, X_txt, region2pair_id, word2pair_id,
                                  uselocal=uselocal, useglobal=useglobal)[0]
        grad_num = eval_numerical_gradient(f, mm_net.params[name], verbose=False)
        print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))



