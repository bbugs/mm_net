"""
Write some tests for the functions in layers

"""
import numpy as np
from cs231n import layers, gradient_check
from cs231n.tests import test_utils

__author__ = "Susana Zoghbi"


def test_svm_struct_loss():
    """
    Test svm_struct_loss against pen & paper computations
    """
    # scores
    x = np.array([[4, 2, 1, 4],
                  [5, 7, 5, 4],
                  [7, 6, 8, 6],
                  [5, 9, 7, 5]], dtype=np.float)
    # correct classes
    y = np.array(np.arange(4))
    # true loss
    true_loss = 5.75
    # true gradient wrt the socres (computed on paper)
    dx_true = np.array([[-4, 0, 0, 1],
                        [1, -2, 0, 0],
                        [1, 0, -3, 1],
                        [2, 2, 1, -5]]) / 4.

    # compute loss and gradient wrt scores
    loss, dx = layers.svm_struct_loss(x, y)

    print "computed loss \t", loss
    print "true loss \t", true_loss
    print "computed gradient \n", dx
    print "true gradient \n", dx_true

    if loss == true_loss:
        print "svm_struct_loss passed loss test"

    if np.allclose(dx, dx_true, rtol=1e-05, atol=1e-08):
        print "svm_struct_loss passed gradient test"

    return


def test_svm_struct_loss_with_num_gradient():
    """
    Test svm_struct_loss agains numerical gradient.
    Note the numerical gradient is right only within a certain number range.
    I found the numerical gradient is wrong when x is large, e.g., between 1 and 10.
    When x is between 0 and 1, the numerical gradient is right.
    """

    x = 1e-2 * np.array([[4, 2, 1, 4],
                         [5, 7, 5, 4],
                         [7, 6, 8, 6],
                         [5, 9, 7, 5]], dtype=np.float)
    # correct classes
    y = np.array(np.arange(4))

    # compute loss and gradient wrt scores
    loss, dx = layers.svm_struct_loss(x, y)

    # compute numerical gradient
    dx_num = gradient_check.eval_numerical_gradient(
        lambda x: layers.svm_struct_loss(x, y)[0], x, verbose=False)

    print 'Testing svm_struct_loss:'
    print 'loss: ', loss
    print 'dx error: ', test_utils.rel_error(dx_num, dx)

    return


def run_suite():
    suite = test_utils.TestSuite()

    x = np.array([[4, 2, 1, 4],
                  [5, 7, 5, 4],
                  [7, 6, 8, 6],
                  [5, 9, 7, 5]], dtype=np.float)

    # correct classes
    y = np.array(np.arange(4))

    dx_true = np.array([[-4, 0, 0, 1],
                        [1, -2, 0, 0],
                        [1, 0, -3, 1],
                        [2, 2, 1, -5]]) / 4.

    suite.run_test(layers.svm_struct_loss(x, y)[0], 5.75, message="Test loss")
    suite.run_test(layers.svm_struct_loss(x, y)[1], dx_true, message="Test gradient against pen and paper")

    suite.report_results()


def test_svm_two_classes():
    print "\n\ntesting svm_two_classes against pen and paper values"
    x = np.array([[3, -8, -4, 5],
                  [-6, -2, -4, -8],
                  [-8, 2, -3, -4],
                  [-2, -5, -6, 9],
                  [-3, 6, -5, -1]], dtype=np.float)
    y = np.array([[1, 1, -1, -1],
                  [1, 1, -1, -1],
                  [-1, -1, 1, 1],
                  [-1, -1, 1, 1],
                  [-1, -1, 1, 1]], dtype=np.float)
    loss, dx = layers.svm_two_classes(x, y, delta=1, do_mil=False, normalize=False)

    ##############################################
    # from pen and paper
    ##############################################
    true_loss = 59.0
    dx_true = np.array([[0, -1, 0, 1],
                        [-1, -1, 0, 0],
                        [0, 1, -1, -1],
                        [0, 0, -1, 0],
                        [0, 1, -1, -1]], dtype=np.float)

    print "computed loss \t", loss
    print "true loss \t", true_loss
    print "computed gradient \n", dx
    print "true gradient \n", dx_true

    assert loss == true_loss, "svm_two_classes did NOT pass loss test"

    assert np.allclose(dx, dx_true, rtol=1e-05, atol=1e-08),\
        "svm_two_classes did NOT pass gradient test"

    ##############################################
    # Test when normalize is True. (True values from matlab code toy_example_local_cost.m)
    ##############################################
    print "\n\nTest when normalize is True"
    loss, dx = layers.svm_two_classes(x, y, delta=1, do_mil=False, normalize=True)
    true_loss = 23.83333333  # from matlab code (toy_example_local_cost.m)
    dx_true = np.array([[0, -0.5, 0, 0.5],  # from matlab code (toy_example_local_cost.m)
                        [-0.5, -0.5, 0, 0],
                        [0, 0.33333333, -0.33333333, -0.33333333],
                        [0, 0, -0.33333333, 0],
                        [0, 0.33333333, -0.33333333, -0.33333333]])

    print "computed loss", loss
    print "true loss", true_loss

    print "computed gradient", dx
    print "true gradient", dx_true

    assert abs(loss - true_loss) < 1e-8, "svm_two_classes did NOT pass loss test with normalization"
    assert np.allclose(dx, dx_true, rtol=1e-05, atol=1e-08), \
            "svm_two_classes did NOT pass gradient test with normalization"

    ##############################################
    # Test when both normalize and do_mil are True. (True values from matlab code toy_example_local_cost.m)
    ##############################################
    print "\n\nTest when normalize is True and do_mil is True"
    loss, dx = layers.svm_two_classes(x, y, delta=1, do_mil=True, normalize=True)

    true_loss = 11.0
    dx_true = np.array([[0, 0, 0, 0.2500],
                        [0, -1.0000, 0, 0],
                        [0, 0.2500, -1.0000, 0],
                        [0, 0, 0, 0],
                        [0, 0.2500, 0, 0]])
    print loss
    print "true loss", true_loss

    print "computed gradient", dx
    print "true gradient", dx_true

    assert abs(loss - true_loss) < 1e-8, "svm_two_classes did NOT pass loss test with normalization and do_mil"
    assert np.allclose(dx, dx_true, rtol=1e-05, atol=1e-08), \
        "svm_two_classes did NOT pass gradient test with normalization and do_mil"

    ##############################################
    # Test with numerical gradient
    ##############################################
    print "\n\nTest with numerical gradient svm_two_classes"
    loss, dx = layers.svm_two_classes(x, y, delta=1, do_mil=True, normalize=True)

    true_loss = 11.0

    print loss
    print "true loss", true_loss

    print "computed gradient", dx
    print "true gradient", dx_true

    # compute numerical gradient
    dx_num = gradient_check.eval_numerical_gradient(
        lambda x: layers.svm_two_classes(x, y, delta=1, do_mil=True, normalize=True)[0], x, verbose=False)
    print "num gradient", dx_num
    print 'dx error: ', test_utils.rel_error(dx_num, dx)


def test_get_normalization_weights():
    y = np.array(
        [[1, 1, -1, -1],
         [1, 1, -1, -1],
         [-1, -1, 1, 1],
         [-1, -1, 1, 1],
         [-1, -1, 1, 1]])

    W_true = np.array(
        [[0.5, 0.5, 0.5, 0.5],
         [0.5, 0.5, 0.5, 0.5],
         [0.33333333, 0.33333333, 0.33333333, 0.33333333],
         [0.33333333, 0.33333333, 0.33333333, 0.33333333],
         [0.33333333, 0.33333333, 0.33333333, 0.33333333]])

    W_computed = layers.get_normalization_weights(y)

    print W_true
    print W_computed

    assert np.allclose(W_computed, W_true, rtol=1e-05, atol=1e-08), \
        "get_normalization_weights did NOT pass test"


def test_perform_mil():
    x = np.array([[3, -8, -4, 5],
                  [-6, -2, -4, -8],
                  [-8, 2, -3, -4],
                  [-2, -5, -6, 9],
                  [-3, 6, -5, -1]], dtype=np.float)

    y = np.array([[1, 1, -1, -1],
                  [1, 1, -1, -1],
                  [-1, -1, 1, 1],
                  [-1, -1, 1, 1],
                  [-1, -1, 1, 1]], dtype=np.float)

    y_new = layers.perform_mil(x, y)

    y_new_true = np.array([[1, -1, -1, -1],
                           [-1, 1, -1, -1],
                           [-1, -1, 1, -1],
                           [-1, -1, -1, 1],
                           [-1, -1, -1, -1]])

    print "y_new", y_new
    print "y_new true", y_new_true

    assert np.allclose(y_new, y_new_true, rtol=1e-05, atol=1e-08), \
        "do_mil did NOT pass test"


def test_sigmoid_cross_entropy_loss():
    """
    Test svm_struct_loss agains numerical gradient.
    Note the numerical gradient is right only within a certain number range.
    I found the numerical gradient is wrong when x is large, e.g., between 1 and 10.
    When x is between 0 and 1, the numerical gradient is right.
    """

    # x = np.array([-2, -.5, 0, .4, 1], dtype=np.float64)
    # y = np.array([1, 0, 1, 0, 1], dtype=np.float64)

    # x = 1e-2*np.random.randn(4, 5)

    x = np.random.uniform(-2, -1, (4,5))
    y = np.random.randint(0, 2, x.shape)
    print x

    # compute loss and gradient wrt scores
    loss, dx = layers.sigmoid_cross_entropy_loss(x, y)
    #
    # correct_loss = 0.9064835704  # pen and paper

    correct_loss = 1.3043201335824406 # pen and paper

    # compute numerical gradient
    dx_num = gradient_check.eval_numerical_gradient(
        lambda x: layers.sigmoid_cross_entropy_loss(x, y)[0], x, verbose=True, h=1e-4)

    print 'Testing sigmoid_cross_entropy_loss:'
    print 'loss: ', loss
    print "correct_loss", correct_loss
    print "computed gradient", dx
    print "num gradient", dx_num
    print 'dx error: ', test_utils.rel_error(dx_num, dx)
    print "dx_num - dx computed", dx_num - dx

    return


if __name__ == "__main__":

    # TODO: Use the test_suite from test_utils.py
    # test_svm_struct_loss()
    # test_svm_struct_loss_with_num_gradient()
    #
    # run_suite()
    #
    # test_svm_two_classes()
    #
    # test_get_normalization_weights()
    #
    # test_perform_mil()
    test_sigmoid_cross_entropy_loss()