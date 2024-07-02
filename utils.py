import tensorflow as tf
import numpy as np
import scipy

from sklearn.metrics import roc_auc_score

def BCE_loss(true, pred, avg='sum_over_batch_size'):
    # binary cross-entropy loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=avg)
    return bce(true, pred)

def CCE_loss(true, pred, avg='sum_over_batch_size'):
    # categorical cross-entropy loss
    scce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=avg)
    return scce(true, pred)

def entropy(x):
    value,counts = np.unique(x, return_counts=True)
    return scipy.stats.entropy(counts)

def accuracy(true, pred):
    return np.mean((true == pred).astype(np.float32))

def auc_score(true, pred):
    return roc_auc_score(true, pred)

# demographic parity difference
def ddp(y_pred, u):
    val = 0.
    for i in range(int(np.max(u)+1)):
        val += abs(np.mean(y_pred[u==i])-np.mean(y_pred))
    return val

# equalized odds difference
def deo(y, y_pred, u):
    val = 0.
    for i in range(int(np.max(u)+1)):
        for j in range(int(np.max(y)+1)):
            if len(u.shape) == 1:
                u = np.expand_dims(u, axis=1)
            val += abs(np.mean(y_pred[np.logical_and(u==i, y==j)])-np.mean(y_pred[y==j]))
    return val

# equalized opportunity difference
def deopp(y, y_pred, u):
    val = 0.
    for i in range(int(np.max(u)+1)):
        if len(u.shape) == 1:
            u = np.expand_dims(u, axis=1)
        val += abs(np.mean(y_pred[np.logical_and(u==i, y==1)])-np.mean(y_pred[y==1]))
    return val