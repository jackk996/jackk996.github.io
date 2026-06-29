import numpy as np

def rmse(y_true,y_pred): return float(np.sqrt(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2)))
def mae(y_true,y_pred): return float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))))
def safe_mape(y_true,y_pred,epsilon=1e-6):
    y=np.asarray(y_true); return float(np.mean(np.abs((y-np.asarray(y_pred))/np.maximum(np.abs(y),epsilon)))*100)
def nonzero_mape(y_true,y_pred,epsilon=1e-6):
    y=np.asarray(y_true); mask=np.abs(y)>epsilon; return float(np.mean(np.abs((y[mask]-np.asarray(y_pred)[mask])/y[mask]))*100) if mask.any() else float('nan')
def evaluate_predictions(y_true,y_pred,epsilon=1e-6):
    y=np.asarray(y_true).ravel(); p=np.asarray(y_pred).ravel(); corr=float(np.corrcoef(y,p)[0,1]) if len(y)>1 and np.std(y)>0 and np.std(p)>0 else float('nan')
    ss_res=((y-p)**2).sum(); ss_tot=((y-y.mean())**2).sum(); r2=float(1-ss_res/ss_tot) if ss_tot else float('nan')
    return {'RMSE':rmse(y,p),'MAE':mae(y,p),'MAPE':safe_mape(y,p,epsilon),'R2':r2,'Pearson':corr,'n':int(len(y)),'pred_mean':float(p.mean()),'pred_std':float(p.std())}
