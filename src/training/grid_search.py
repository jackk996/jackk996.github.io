import itertools, pandas as pd

def parameter_grid(window_size=(3,6,9,12), lstm_units=(32,64,128), lstm_layers=(1,2)):
    for w,u,l in itertools.product(window_size,lstm_units,lstm_layers): yield {'window_size':w,'lstm_units':u,'lstm_layers':l}
def save_grid_results(rows,path): pd.DataFrame(rows).to_csv(path,index=False)
