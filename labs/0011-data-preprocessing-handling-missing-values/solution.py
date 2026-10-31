import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()

    #to fill missing values we have to use hard impute
    n_rows,n_columns = X_clean.shape
    #print(n_rows,n_columns)

    X_t = X_clean

    for feature in X_t:
        mean = np.nanmean(feature)
        feature[np.isnan(feature)] = mean
        #print(mean,feature)

    X_clean = X_t

    #print(X_clean)

    #at this point we can start svd first determine rank and new matrix
    X_reconstructed = X_clean
    rank = int(np.max([1,0.15 * np.min([n_rows,n_columns])]))
    mse_prev = np.nan 
    missing_mask = np.isnan(X)

    for i in range(100):
        #get U,s, Vt

        U,s,Vt = np.linalg.svd(X_reconstructed)
        X_reconstructed = U[:,:rank] @ np.diag(s[:rank]) @ Vt[:rank,:]

        # replace with actual value
        X_reconstructed[~missing_mask] = X[~missing_mask]
        

    return X_reconstructed