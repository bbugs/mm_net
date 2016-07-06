

from cs231n.multimodal import multimodal_net
import numpy as np
import math

img_input_dim = 16 # size of cnn
txt_input_dim = 8  # size of word2vec pretrained vectors
hidden_dim = 4  # size of multimodal space

N_img = 3  # number of regions in the batch
N_words = 5  # number of words in the batch

X_img = np.random.randn(N_img, img_input_dim)
X_txt = np.random.randn(N_words, txt_input_dim)
y = np.random.randint(N_words, size=N_img)  # indicate which word is correct (happens together) with each image

std_img = math.sqrt(2. / img_input_dim)
std_txt = math.sqrt(2. / txt_input_dim)
weight_scale = {'img': std_img, 'txt': std_txt}

print y
print weight_scale

model = multimodal_net.MultiModalNet(img_input_dim, txt_input_dim,
                                     hidden_dim, weight_scale=weight_scale, reg=0.0)

print 'Testing initialization ... '
Wi2s_std = abs(model.params['Wi2s'].std() - std_img)
bi2s = model.params['bi2s']
Wsem_std = abs(model.params['Wsem'].std() - std_txt)
bsem = model.params['bsem']
assert Wi2s_std < std_img, 'First layer weights do not seem right'
assert np.all(bi2s == 0), 'First layer biases do not seem right'
assert Wsem_std < std_txt, 'Second layer weights do not seem right'
assert np.all(bsem == 0), 'Second layer biases do not seem right'


print 'Testing training loss (no regularization)'
y = np.asarray([0, 4, 1])
loss, grads = model.loss(X_img, X_txt, y)


