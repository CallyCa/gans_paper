import pytest
import pandas as pd
from src.ahp_saaty import AHPSaaty

@pytest.fixture
def judgment_matrix():
    return pd.DataFrame({
        'Criterion 1': [1, 1/3, 1/5],
        'Criterion 2': [3, 1, 1/3],
        'Criterion 3': [5, 3, 1]
    }, index=['Criterion 1', 'Criterion 2', 'Criterion 3'])

@pytest.fixture
def decision_matrix():
    return pd.DataFrame({
        'Criterion 1': [0.2, 0.4, 0.4],
        'Criterion 2': [0.5, 0.3, 0.2],
        'Criterion 3': [0.3, 0.3, 0.4]
    }, index=['Alternative 1', 'Alternative 2', 'Alternative 3'])

def test_ahp_saaty_consistency_ratio(judgment_matrix):
    ahp = AHPSaaty(judgment_matrix)
    assert ahp.cr <= 0.1

def test_ahp_saaty_local_preference(judgment_matrix, decision_matrix):
    ahp = AHPSaaty(judgment_matrix)
    local_pref = ahp.local_preference(decision_matrix)
    assert not local_pref.empty
    assert local_pref.shape == decision_matrix.shape

def test_ahp_saaty_global_preference(judgment_matrix, decision_matrix):
    ahp = AHPSaaty(judgment_matrix)
    global_pref = ahp.global_preference(decision_matrix)
    assert not global_pref.empty
    assert 'Ranking' in global_pref.columns
    assert 'Score' in global_pref.columns

def test_invalid_judgment_matrix():
    invalid_matrix = pd.DataFrame({
        'Criterion 1': [1, 2, 3],
        'Criterion 2': [0.5, 1, 2],
        'Criterion 3': [0.33, 0.5, 1]
    }, index=['Criterion 1', 'Criterion 2', 'Criterion 3'])
    with pytest.raises(ValueError):
        AHPSaaty(invalid_matrix)

pytest.main()