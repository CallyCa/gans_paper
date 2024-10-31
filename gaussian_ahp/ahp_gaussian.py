import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from decision_helper import DecisionHelper

class AHPGaussian:
    """
    Multi-criteria decision-making method derived from the Analytic Hierarchy Process developed by Marcos dos Santos.
    Presents a new approach to the original method by Thomas Saaty based on sensitivity analysis from the Gaussian factor.
    In this method, pairwise evaluation between criteria is not necessary to obtain their respective weights.
    Note: The model's feasibility is only satisfied in scenarios where the alternatives have cardinal entries in the criteria under analysis.

    Parameters
    ----------
    :param decision_matrix: DataFrame (required), matrix with rows containing the alternatives and columns containing the criteria.
    :param objective: list, vector containing only the decision criteria that should be minimized.
                      default = 'max', assumes that all criteria should be maximized.

    Attributes
    ----------
    :ivar optimization: dict, contains the criteria and their respective optimizations ('max', 'min').
    :ivar local_preference: DataFrame, contains the local preference for each criterion.
    :ivar gaussian_factor: Series, contains the coefficient of variation relative to the mean.
    :ivar weights: Series, contains the weighting of each criterion.

    Methods
    ----------
    global_preference(): calculates the global preference of the alternatives.
    visualize_decision_matrix(): visualizes the decision matrix.
    visualize_local_preference(): visualizes the local preference.
    visualize_global_preference(): visualizes the global preference.
    """

    def __init__(self, decision_matrix, objective='max'):
        self._validate_decision_matrix(decision_matrix)
        self._decision_matrix = decision_matrix
        self._objective = objective

        self.optimization = DecisionHelper.check_objective(decision_matrix, objective)
        self.local_preference = DecisionHelper.normalize_decision_matrix(decision_matrix, objective)
        self._mean = self._calculate_mean()
        self._std_dev = self._calculate_std_dev()
        self.gaussian_factor = self._calculate_gaussian_factor()
        self.weights = self._normalize_gaussian_factor()

    def _validate_decision_matrix(self, decision_matrix):
        """
        Validate if the decision matrix has consistent column lengths.

        :param decision_matrix: DataFrame, the decision matrix to validate.
        :raises ValueError: if the decision matrix columns have inconsistent lengths.
        """
        lengths = [len(decision_matrix[col]) for col in decision_matrix.columns]
        if len(set(lengths)) > 1:
            raise ValueError("All columns in the decision matrix must have the same length.")

    def _calculate_mean(self):
        return self.local_preference.sum() / len(self._decision_matrix)

    def _calculate_std_dev(self):
        return self.local_preference.std()

    def _calculate_gaussian_factor(self):
        return self._std_dev / self._mean

    def _normalize_gaussian_factor(self):
        normalized_weights = self.gaussian_factor.copy()
        total = self.gaussian_factor.sum()
        return normalized_weights.apply(lambda x: x / total)

    def global_preference(self):
        """
        Calculate the global preference of the alternatives.

        :return: DataFrame, containing the global preference, ranking of the alternatives, and their scores.
        """
        return DecisionHelper.get_results(self._decision_matrix, self.weights, self._objective)

    def visualize_decision_matrix(self):
        """Visualize the decision matrix."""
        DecisionHelper.plot_decision_matrix(self._decision_matrix, "Decision Matrix")

    def visualize_local_preference(self):
        """Visualize the local preferences."""
        DecisionHelper.plot_preferences(self.local_preference, "Local Preferences")

    def visualize_global_preference(self):
        """Visualize the global preferences."""
        global_pref = self.global_preference()
        DecisionHelper.plot_results(global_pref, "Global Preferences")