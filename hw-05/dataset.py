import numpy as np
import idx2numpy
import os


def load_mnist(folder):
    train_x = np.expand_dims(
        idx2numpy.convert_from_file(os.path.join(folder, "train_images")), axis=3
    )
    train_y = idx2numpy.convert_from_file(os.path.join(folder, "train_labels"))
    test_x = np.expand_dims(
        idx2numpy.convert_from_file(os.path.join(folder, "test_images")), axis=3
    )
    test_y = idx2numpy.convert_from_file(os.path.join(folder, "test_labels"))
    return train_x, train_y, test_x, test_y


def random_split_train_val(X, y, random_seed):
    np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:]
    train_X = X[train_indices]
    train_y = y[train_indices]

    return train_X, train_y
