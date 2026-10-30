import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    l = len(y)
    range_list = np.arange(l)
    np.random.seed(seed)
    np.random.shuffle(range_list)
    train_end = int(train_frac * l)
    #print(train_end)
    val_end = train_end + int(val_frac * l)
    #print(val_end)
    train_idx = range_list[:train_end]
    val_idx = range_list[train_end:val_end]
    test_idx = range_list[val_end:]
    test_predictions = y[list(test_idx)]
    #print(train_idx)
    #print(val_idx)
    #print(test_idx)
    return test_predictions, train_idx, val_idx, test_idx
    pass