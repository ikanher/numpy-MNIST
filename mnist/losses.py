__author__ = 'Aki Rehn'
__project__ = 'mnist'

import numpy as np

class CrossEntropy(object):
    """
    Cross-Entropy Loss

    https://en.wikipedia.org/wiki/Cross_entropy
    """
    def loss(self, y_pred, y):
        """
        Calculates Cross-Entropy Loss
        """
        predictions = np.diag(y_pred[:,y])
        return -np.mean(np.log(predictions))

