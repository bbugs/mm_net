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
    suite.run_test(layers.svm_struct_loss(x, y)[1], dx_true, message="Test gradient agains pen and paper")

    suite.report_results()


if __name__ == "__main__":
    test_svm_struct_loss()
    test_svm_struct_loss_with_num_gradient()

    run_suite()
