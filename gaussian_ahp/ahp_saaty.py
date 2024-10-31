import pandas as pd
import matplotlib.pyplot as plt
from decision_helper import DecisionHelper
from config_manager import ConfigManager


class AHPSaaty:
    """
    Analytic Hierarchy Process (AHP) is a method created by Thomas Saaty to assist in multi-criteria decision-making.
    Characterized by a decision matrix composed of alternatives and criteria weighted by the decision-maker.
    Based on mathematics and psychology, AHP not only determines the best decision but also justifies the choice.
    
    Note: Assists decision-making for problems with up to 15 criteria.
          For problems with more than 15 criteria, refer to AHPGaussian.

    Parameters
    ----------
    :param judgment_matrix: DataFrame (required), square matrix of decision criteria filled with Saaty's fundamental scale.
    :param objective: list, vector containing only the decision criteria that should be minimized.
                      default = 'max', assumes that all criteria should be maximized.

    Attributes
    ----------
    :ivar optimization: dict, contains the criteria and their respective optimizations ('max', 'min').
    :ivar weights: Series, contains the weighting of each criterion.
    :ivar cr: float, consistency ratio that checks if pairwise evaluations respect the principle of transitivity.
              cr should be ≤ 0.1 for the judgment matrix to be consistent and the result satisfactory.

    Methods
    ----------
    local_preference(decision_matrix): calculates the local preference for each criterion.
    global_preference(decision_matrix): calculates the global preference of the alternatives.
    visualize_judgment_matrix(): visualizes the judgment matrix.
    visualize_local_preference(): visualizes the local preference.
    visualize_global_preference(): visualizes the global preference.
    """

    def __init__(self, judgment_matrix, objective='max'):
        self._judgment_matrix = judgment_matrix
        self._objective = objective
        self._config_manager = ConfigManager('settings/ahp_settings.yaml')
        self._consistency_index = self._config_manager.get_config('consistency_index')
        self._validate_judgment_matrix()

        self.optimization = DecisionHelper.check_objective(judgment_matrix, objective)
        self._normalized_judgments = self._normalize_judgments()
        self._n = self._judgment_matrix.shape[1]
        self.weights = self._calculate_priority_vector()
        self._consistency_matrix = self._consistency_analysis()
        self._max_lambda = self._calculate_max_lambda()
        self._ri = self._consistency_index[str(self._n)]
        self._ci = (self._max_lambda - self._n) / (self._n - 1)
        self.cr = self._ci / self._ri

    def _validate_judgment_matrix(self):
        """
        Validate if the judgment matrix adheres to Saaty's fundamental scale.
        The judgment matrix should contain values from 1 to 9 and their respective reciprocals.
        The judgment matrix should be square.
        """
        if (self._judgment_matrix.max().max() > 9) or (self._judgment_matrix.min().min() < 1 / 9):
            raise ValueError('Judgment matrix violates Saaty’s fundamental scale')
        if self._judgment_matrix.shape[0] != self._judgment_matrix.shape[1]:
            raise ValueError('Judgment matrix must be square')
        if list(self._judgment_matrix.columns) != list(self._judgment_matrix.index):
            raise ValueError('Columns and indices of the judgment matrix must be ordered equally')
        for row in self._judgment_matrix.index:
            for col in self._judgment_matrix.columns:
                if (self._judgment_matrix.loc[row, col]) != (1 / self._judgment_matrix.loc[col, row]):
                    raise ValueError(f'Reciprocal error in judgment matrix: item ({row},{col})/({col},{row})')

    def _normalize_judgments(self):
        """Normalize the columns of the judgment matrix and return a DataFrame."""
        col_sum = [self._judgment_matrix[col].sum() for col in self._judgment_matrix.columns]
        normalized_matrix = pd.DataFrame()
        for index, col in enumerate(self._judgment_matrix.columns):
            normalized_matrix[col] = self._judgment_matrix[col].apply(lambda x: x / col_sum[index])
        return normalized_matrix

    def _calculate_priority_vector(self):
        """Calculate the weighted average of each criterion and return a Series."""
        return self._normalized_judgments.sum(axis=1) / self._n

    def _consistency_analysis(self):
        """Multiply the judgment matrix by the priority vector and return a DataFrame."""
        consistency_matrix = pd.DataFrame()
        for index, col in enumerate(self._judgment_matrix.columns):
            consistency_matrix[col] = self._judgment_matrix[col] * self.weights[index]
        return consistency_matrix

    def _calculate_max_lambda(self):
        """Scalar necessary to calculate the consistency ratio (cr). Return a float."""
        row_sum = self._consistency_matrix.sum(axis=1)
        return ((row_sum / self.weights).sum()) / self._n

    def _validate_decision_matrix(self, decision_matrix):
        """Validate the structure of the judgment and decision matrices."""
        if decision_matrix.shape[1] > len(self._consistency_index):
            raise ValueError('Decision matrix must contain a maximum of 15 evaluation criteria')
        if self._judgment_matrix.shape[1] != decision_matrix.shape[1]:
            raise ValueError('Number of columns in judgment and decision matrices must be equal')
        if list(self._judgment_matrix.columns) != list(decision_matrix.columns):
            raise ValueError('Columns of decision and judgment matrices must be equal')

    def local_preference(self, decision_matrix):
        """
        Calculate the local preference for each criterion.

        :param decision_matrix: DataFrame, matrix with rows containing the alternatives and columns containing the criteria.
        :return: DataFrame, matrix containing the local preferences for each criterion.
        """
        return DecisionHelper.normalize_decision_matrix(decision_matrix, self._objective)

    def global_preference(self, decision_matrix):
        """
        Calculate the global preference of the alternatives.

        :param decision_matrix: DataFrame, matrix with rows containing the alternatives and columns containing the criteria.
        :return: DataFrame, global preference containing the ranking of the alternatives and their scores.
        """
        self._validate_decision_matrix(decision_matrix)
        return DecisionHelper.get_results(decision_matrix, self.weights, self._objective)

    def visualize_judgment_matrix(self):
        """Visualize the judgment matrix."""
        DecisionHelper.plot_decision_matrix(self._judgment_matrix, "Judgment Matrix")

    def visualize_local_preference(self, decision_matrix):
        """Visualize the local preferences."""
        local_pref = self.local_preference(decision_matrix)
        DecisionHelper.plot_preferences(local_pref, "Local Preferences")

    def visualize_global_preference(self, decision_matrix):
        """Visualize the global preferences."""
        global_pref = self.global_preference(decision_matrix)
        DecisionHelper.plot_results(global_pref, "Global Preferences")
