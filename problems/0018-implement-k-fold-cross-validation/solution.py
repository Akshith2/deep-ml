import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    # create a range(n_samples), shuffle it, calcuate number of test_indices
    # from backwards add indices 
    index_list = list(range(n_samples))
    if shuffle:
        np.random.shuffle(index_list)
    
    n_elements = n_samples//k
    extra_sample = n_samples % k

    fold_list = []

    start, end = 0,0

    #print(index_list)

    for i in range(k):
        if i == 0:
            train_indices = index_list[(n_elements+extra_sample):]
            test_indices = index_list[:(n_elements+extra_sample)]
            start = 0
            end = n_elements + extra_sample
        else:
            train_indices = index_list[:start] + index_list[end:]
            test_indices = index_list[start:end]

        start = end
        end = end + n_elements

        fold_list.append((train_indices,test_indices))

    return fold_list