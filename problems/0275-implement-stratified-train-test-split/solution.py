import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)

    # {class name : [list of index of class]} for each class Name:
    class_index_list = {}
    for index,target in enumerate(y):
        index_list = class_index_list.get(target,np.array([],dtype=int))
        index_list = np.append(index_list,index)
        class_index_list[target] = index_list

    #print(f" class_index_list:{class_index_list}")

    #sample n_test values from each class 
    test_index_list = {}
    for target, indices in class_index_list.items():
        n_test = int(test_size * len(indices))
        test_index_list[target] = np.random.choice(class_index_list[target],size = n_test, replace = False)
    
    #print(f" test_index_list:{test_index_list}")
    
    #concatinate all test indexes values in one list
    test_index = np.array([],dtype=int)
    for target,index_list in test_index_list.items():
        test_index = np.concatenate((test_index,index_list))
    
    #print(f"test_index:{test_index}")

    #create x_train,x_test,y_train,y_test
    x_train = np.array([],dtype=int)
    x_test = np.array([],dtype=int)
    y_train = np.array([],dtype=int)
    y_test = np.array([],dtype =int)
    for i in range(len(X)):
        if i in test_index:
            x_test = np.append(x_test,X[i])
            y_test = np.append(y_test,y[i])
        else:
            x_train = np.append(x_train,X[i])
            y_train = np.append(y_train,y[i])
    
    #print((x_train,x_test,y_train,y_test))
    
    return (x_train,x_test,y_train,y_test)

