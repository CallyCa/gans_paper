import pytest
import pandas as pd
from src.decision_helper import DecisionHelper

def test_check_objective():
    criteria = pd.DataFrame({
        'Criterion 1': [1, 2, 3],
        'Criterion 2': [4, 5, 6]
    })
    objective = ['Criterion 1']
    result = DecisionHelper.check_objective(criteria, objective)
    assert result['Criterion 1'] == 'min'
    assert result['Criterion 2'] == 'max'

def test_normalize_decision_matrix():
    decision_matrix = pd.DataFrame({
        'Criterion 1': [1, 2, 3],
        'Criterion 2': [4, 5, 6]
    })
    normalized_matrix = DecisionHelper.normalize_decision_matrix(decision_matrix)
    assert normalized_matrix is not None
    assert (normalized_matrix.sum(axis=0) - 1).abs().max() < 1e-6

def test_aggregate_matrix():
    decision_matrix = pd.DataFrame({
        'Criterion 1': [1, 2, 3],
        'Criterion 2': [4, 5, 6]
    })
    weights = pd.Series([0.5, 0.5], index=['Criterion 1', 'Criterion 2'])
    aggregated_matrix = DecisionHelper.aggregate_matrix(decision_matrix, weights)
    assert aggregated_matrix is not None
    assert not aggregated_matrix.isna().any().any()  # Ensure no NaN values
    # Ensure sum of weighted normalized values is close to sum of weights
    assert (aggregated_matrix.sum(axis=0) - weights).abs().max() < 1e-6

def test_get_results():
    decision_matrix = pd.DataFrame({
        'Criterion 1': [1, 2, 3],
        'Criterion 2': [4, 5, 6]
    })
    weights = pd.Series([0.5, 0.5])
    results = DecisionHelper.get_results(decision_matrix, weights)
    assert not results.empty
    assert 'Ranking' in results.columns
    assert 'Score' in results.columns

def test_validate_parameters():
    df = pd.DataFrame({
        'Criterion 1': [1, 2, 3],
        'Criterion 2': [4, 5, 6]
    })
    valid_columns = ['Criterion 1', 'Criterion 2']
    DecisionHelper.validate_parameters(df, valid_columns)
    invalid_columns = ['Criterion 3']
    with pytest.raises(ValueError):
        DecisionHelper.validate_parameters(df, invalid_columns)

pytest.main()