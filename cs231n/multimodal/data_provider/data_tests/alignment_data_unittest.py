
from cs231n.multimodal.data_provider.alignment_data import AlignmentData
from cs231n.multimodal.data_provider.data_tests import data_config
import numpy as np

d = data_config.dc

ad = AlignmentData(d, split='test', num_items=10)

###################################################
#  Test make_region2pair_id
###################################################
img_ids = [2, 3, 4, 65, 45]
region2pair_id = ad.make_region2pair_id(img_ids, num_regions_per_img=5)

correct = np.array([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4])

assert np.allclose(region2pair_id, correct)

###################################################
#  Test make_word2pair_id
###################################################
img_ids = [2, 3, 4, 65, 45]
region2pair_id = ad.make_region2pair_id(img_ids, num_regions_per_img=5)

correct = np.array([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4])

assert np.allclose(region2pair_id, correct)

