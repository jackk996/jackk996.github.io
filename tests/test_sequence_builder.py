import pytest
np = pytest.importorskip('numpy'); pd = pytest.importorskip('pandas')
from src.data.sequence_builder import build_subsystem_windows

def test_no_cross_subsystem_windows():
    rows=[]
    for s in ['A','B']:
        for i,m in enumerate(pd.period_range('2020-01', periods=8, freq='M').astype(str)):
            rows.append({'project_id':'P','subsystem_id':s,'month':m,'risk_score':0.1*i,'pc1':i,'pc2':i,'pc3':i,'pc4':i})
    ds=build_subsystem_windows(pd.DataFrame(rows), ['pc1','pc2','pc3','pc4'], window_size=3)
    assert ds.X_seq.shape[0] == 10
    assert set(ds.metadata['subsystem_id']) == {'A','B'}
