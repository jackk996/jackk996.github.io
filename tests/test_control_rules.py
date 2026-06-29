import pytest
from src.explainability.control_rules import risk_level, recommended_actions

def test_four_level_boundaries():
    assert risk_level(0.00) == '低风险'
    assert risk_level(0.2499) == '低风险'
    assert risk_level(0.25) == '中风险'
    assert risk_level(0.50) == '高风险'
    assert risk_level(0.75) == '极高风险'
    assert risk_level(1.00) == '极高风险'
    with pytest.raises(ValueError): risk_level(1.01)
    assert '专家诊断' in recommended_actions(0.9)
