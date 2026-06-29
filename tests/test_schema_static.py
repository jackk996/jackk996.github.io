from src.data.schema import INDICATOR_COLUMNS, REQUIRED_COLUMNS

def test_20_indicator_columns():
    assert len(INDICATOR_COLUMNS) == 20
    assert len(set(INDICATOR_COLUMNS)) == 20
    for c in ['project_id','subsystem_id','month','risk_score']:
        assert c in REQUIRED_COLUMNS
