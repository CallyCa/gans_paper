import pytest
import pandas as pd
from src.ahp_gaussian import AHPGaussian
import numpy as np

@pytest.fixture
def decision_matrix():
    return pd.DataFrame({
        'Criterion 1': [0.2, 0.4, 0.4],
        'Criterion 2': [0.5, 0.3, 0.2],
        'Criterion 3': [0.3, 0.3, 0.4]
    }, index=['Alternative 1', 'Alternative 2', 'Alternative 3'])

def validate_decision_matrix(matrix):
    """
    Simplified validation logic (might not be the exact function in your code)
    """
    lengths = [len(matrix[col]) for col in matrix.columns]
    return len(set(lengths)) == 1

def test_ahp_gaussian_global_preference(decision_matrix):
    ahp = AHPGaussian(decision_matrix)
    global_pref = ahp.global_preference()
    assert not global_pref.empty
    assert 'Ranking' in global_pref.columns
    assert 'Score' in global_pref.columns

def test_ahp_gaussian_normalized_weights(decision_matrix):
    ahp = AHPGaussian(decision_matrix)
    normalized_weights = ahp._normalize_gaussian_factor()
    assert abs(normalized_weights.sum() - 1.0) < 1e-6

def test_invalid_decision_matrix():
    valid_matrix = pd.DataFrame({
        'Criterion 1': [1, 2, 3],
        'Criterion 2': [0.5, 1, 2],
        'Criterion 3': [0.3, 0.6, 0.9]
    }, index=['Alternative 1', 'Alternative 2', 'Alternative 3'])

    invalid_matrix = valid_matrix.copy()
    invalid_matrix.loc['Alternative 3', 'Criterion 2'] = np.nan

    if not validate_decision_matrix(invalid_matrix):
        print("Validation failed as expected")

pytest.main()
