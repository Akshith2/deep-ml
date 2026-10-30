import numpy as np

def bias_variance_decomp(predictions, y_true):

    predictions = np.array(predictions)
    y_true = np.array(y_true)
    #print(np.mean(predictions,axis=0))
    mean_predictions = np.mean(predictions,axis=0)
    squared_bias = np.mean(np.square(mean_predictions-y_true))
    variance = np.mean(np.var(predictions,axis=0))
    avg_bias = np.mean(squared_bias)
    mse = squared_bias + variance

    return {'bias_squared': float(squared_bias), 'variance': float(variance), 'mse': float(mse)}
    pass
