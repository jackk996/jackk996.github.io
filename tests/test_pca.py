import pytest
np = pytest.importorskip('numpy'); pd = pytest.importorskip('pandas')
from src.features.temporal_weighted_pca import TemporalWeightedPCA

def test_pca_dimension_and_variance():
    rng=np.random.default_rng(1); X=rng.random((30,20)); ts=pd.date_range('2020-01-01', periods=30, freq='MS')
    p=TemporalWeightedPCA(alpha=0.01, variance_threshold=0.85).fit(X,ts)
    Z=p.transform(X)
    assert Z.shape[1] == p.n_components_
    assert p.get_explained_variance_ratio().sum() <= 1.0 + 1e-9
