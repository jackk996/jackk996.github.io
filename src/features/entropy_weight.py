import numpy as np

def entropy_weights(X, eps=1e-12):
    X=np.asarray(X,dtype=float); X=np.clip(X,0,None)
    colsum=X.sum(axis=0,keepdims=True); P=X/np.maximum(colsum,eps)
    n=max(X.shape[0],2); E=-(P*np.log(P+eps)).sum(axis=0)/np.log(n)
    d=1-E
    return np.ones(X.shape[1])/X.shape[1] if d.sum() <= eps else d/d.sum()
