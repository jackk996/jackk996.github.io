from __future__ import annotations
import pickle
import numpy as np, pandas as pd
from .entropy_weight import entropy_weights

class TemporalWeightedPCA:
    def __init__(self, alpha=0.03, variance_threshold=0.85, n_components=None):
        self.alpha=alpha; self.variance_threshold=variance_threshold; self.n_components=n_components
    def fit(self, X, timestamps):
        X=np.asarray(X,dtype=float); ts=pd.to_datetime(timestamps)
        latest=ts.max(); delta=((latest-ts)/np.timedelta64(1,'D'))/30.4375
        theta=np.exp(-self.alpha*np.asarray(delta,dtype=float)); theta=theta/np.maximum(theta.sum(),1e-12)
        self.scaler_min_=X.min(axis=0); self.scaler_max_=X.max(axis=0); Xs=(X-self.scaler_min_)/np.where(self.scaler_max_-self.scaler_min_==0,1,self.scaler_max_-self.scaler_min_)
        self.weighted_mean_=(theta[:,None]*Xs).sum(axis=0); C=(Xs-self.weighted_mean_).T @ ((Xs-self.weighted_mean_)*theta[:,None])
        vals, vecs=np.linalg.eigh(C); order=np.argsort(vals)[::-1]; self.eigenvalues_=vals[order]; self.loadings_=vecs[:,order]
        total=max(self.eigenvalues_.sum(),1e-12); self.explained_variance_ratio_=self.eigenvalues_/total
        if self.n_components is None: self.n_components_=int(np.searchsorted(np.cumsum(self.explained_variance_ratio_), self.variance_threshold)+1)
        else: self.n_components_=int(self.n_components)
        self.entropy_weights_=entropy_weights(Xs); V=self.loadings_[:,:self.n_components_]
        denom=np.maximum((V**2).sum(axis=0),1e-12); self.gamma_=((self.entropy_weights_[:,None]*(V**2)).sum(axis=0))/denom
        return self
    def transform(self, X):
        X=np.asarray(X,dtype=float); Xs=(X-self.scaler_min_)/np.where(self.scaler_max_-self.scaler_min_==0,1,self.scaler_max_-self.scaler_min_)
        return (Xs-self.weighted_mean_) @ self.loadings_[:,:self.n_components_] @ np.diag(self.gamma_)
    def fit_transform(self,X,timestamps): return self.fit(X,timestamps).transform(X)
    def inverse_transform(self,Z):
        Z=np.asarray(Z,dtype=float); approx=(Z @ np.linalg.pinv(np.diag(self.gamma_)) @ self.loadings_[:,:self.n_components_].T)+self.weighted_mean_
        return approx*np.where(self.scaler_max_-self.scaler_min_==0,1,self.scaler_max_-self.scaler_min_)+self.scaler_min_
    def get_loadings(self): return self.loadings_[:,:self.n_components_]
    def get_explained_variance_ratio(self): return self.explained_variance_ratio_[:self.n_components_]
    def save(self,path): pickle.dump(self, open(path,'wb'))
    @classmethod
    def load(cls,path): return pickle.load(open(path,'rb'))
