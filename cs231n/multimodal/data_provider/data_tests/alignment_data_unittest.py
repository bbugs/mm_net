
from cs231n.multimodal.data_provider.alignment_data import AlignmentData
from cs231n.multimodal.data_provider.data_tests import data_config
import numpy as np

d = data_config.dc

ad = AlignmentData(d, split='test', num_items=10)


###################################################
#  Test make_region2pair_id
###################################################
def test_make_region2pair_id():
    img_ids = [2, 3, 4, 65, 45]
    region2pair_id = ad.make_region2pair_id(img_ids, num_regions_per_img=5)

    correct = np.array([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4])

    assert np.allclose(region2pair_id, correct)

    return


###################################################
#  Test make_word2pair_id
###################################################
def test_make_word2pair_id():
    img_ids = [6, 80, 385]

    word2pair_id = ad.make_word2pair_id(img_ids, verbose=True)

    print word2pair_id

    correct = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 2., 2., 2., 2., 2.,
                        2., 2.])

    assert np.allclose(word2pair_id, correct)

    return


###################################################
#  Test pair_id2y
###################################################
def test_pair_id2y():
    print "\n\n\n"
    print "testing pair_id2y"
    img_ids = [212, 261, 373, 385]
    region2pair_id = ad.make_region2pair_id(img_ids, num_regions_per_img=5)
    word2pair_id = ad.make_word2pair_id(img_ids, verbose=True)

    y = ad.pair_id2y(region2pair_id, word2pair_id)

    print y


###################################################
#  Test make_y_true_img2txt
###################################################
def test_make_y_true_img2txt()
    ad.make_y_true_img2txt(num_regions_per_img=5, d)



def main():
    # test_make_region2pair_id()
    # test_make_word2pair_id()
    test_pair_id2y()

if __name__ == "__main__":
    main()