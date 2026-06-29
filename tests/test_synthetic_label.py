import pytest
pd = pytest.importorskip('pandas')
from src.data.synthetic_generator import generate_synthetic_risk_data

def test_synthetic_demo_label_present():
    df=generate_synthetic_risk_data(projects=1,subsystems=1,months=8)
    assert set(df['dataset_label']) == {'synthetic_demo'}
