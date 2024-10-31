import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List

class DecisionHelper:
    """Utility class to assist with decision-making processes."""

    @staticmethod
    def check_objective(criteria: pd.DataFrame, objective: Union[str, List[str]] = 'max') -> dict:
        """
        Check if the specified objective is valid. If valid, minimize all criteria in the decision matrix
        that are listed. If no objective is specified, all criteria in the decision matrix will be maximized.
        
        :param criteria: DataFrame, decision matrix or judgment matrix.
        :param objective: Union[str, List[str]], vector containing only the decision criteria that should be minimized.
                          default = 'max', assumes that all criteria should be maximized.
        :return: dict, dictionary where the keys are the evaluation criteria and the values are their optimizations (max/min).
        """
        optimization = {col: 'max' for col in criteria.columns}
        if objective != 'max':
            if not isinstance(objective, list):
                raise TypeError('Objective must be a list')
            for item in objective:
                if item in optimization:
                    optimization[item] = 'min'
                else:
                    raise KeyError(f'{item} is not a valid criterion')
        return optimization

    @staticmethod
    def normalize_decision_matrix(decision_matrix: pd.DataFrame, objective: Union[str, List[str]] = 'max') -> pd.DataFrame:
        """
        Normalize the decision matrix. The normalized decision matrix indicates the local preferences
        of the alternatives in relation to the evaluation criterion.

        :param decision_matrix: DataFrame, decision matrix.
        :param objective: Union[str, List[str]], vector containing only the decision criteria that should be minimized.
                          default = 'max', assumes that all criteria should be maximized.
        :return: DataFrame, contains the normalized decision matrix.
        """
        optimization = DecisionHelper.check_objective(decision_matrix, objective)
        normalized_matrix = decision_matrix.copy()
        for item in optimization:
            if optimization[item] == 'max':
                total = normalized_matrix[item].sum()
                normalized_matrix[item] = normalized_matrix[item] / total
            else:
                normalized_matrix[item] = normalized_matrix[item].apply(lambda x: 1 / x)
                total = normalized_matrix[item].sum()
                normalized_matrix[item] = normalized_matrix[item] / total
        return normalized_matrix

    @staticmethod
    def aggregate_matrix(decision_matrix: pd.DataFrame, weights: pd.Series, objective: Union[str, List[str]] = 'max') -> pd.DataFrame:
        """
        Create a comparison indicator between the alternatives in the decision matrix.
        Multiply local preference by the weight of the evaluation criterion.
        The sum of the aggregation matrix by alternative informs the global preference.

        :param decision_matrix: DataFrame, decision matrix.
        :param weights: Series, contains the weighting of each criterion.
        :param objective: Union[str, List[str]], vector containing only the decision criteria that should be minimized.
                          default = 'max', assumes that all criteria should be maximized.
        :return: DataFrame, contains the aggregation matrix.
        """
        aggregated_matrix = DecisionHelper.normalize_decision_matrix(decision_matrix, objective)
        for index, item in enumerate(aggregated_matrix.columns):
            aggregated_matrix[item] = aggregated_matrix[item] * weights[index]
        return aggregated_matrix

    @staticmethod
    def get_results(decision_matrix: pd.DataFrame, weights: pd.Series, objective: Union[str, List[str]] = 'max') -> pd.DataFrame:
        """
        Create a DataFrame with global preferences and rank the alternatives.

        :param decision_matrix: DataFrame, decision matrix.
        :param weights: Series, contains the weighting of each criterion.
        :param objective: Union[str, List[str]], vector containing only the decision criteria that should be minimized.
                          default = 'max', assumes that all criteria should be maximized.
        :return: DataFrame, contains the alternative, its ranking, and its score.
        """
        results_df = DecisionHelper.aggregate_matrix(decision_matrix, weights, objective)
        results_df['Score'] = results_df.sum(axis=1)
        results_df = results_df.sort_values('Score', ascending=False)
        results_df['Ranking'] = list(range(1, len(results_df) + 1))
        return results_df[['Ranking', 'Score']]

    @staticmethod
    def validate_parameters(df: pd.DataFrame, valid_columns: List[str]):
        """
        Validate that the DataFrame columns match the expected criteria.

        :param df: DataFrame, the input DataFrame to validate.
        :param valid_columns: list, expected column names.
        :raises ValueError: if the DataFrame does not contain the expected columns.
        """
        if not all(col in df.columns for col in valid_columns):
            raise ValueError("DataFrame does not contain the expected columns")

    @staticmethod
    def plot_decision_matrix(matrix: pd.DataFrame, title: str = "Decision Matrix"):
        """
        Plot the decision matrix as a heatmap.

        :param matrix: DataFrame, the decision matrix to plot.
        :param title: str, the title of the plot.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, annot=True, cmap='YlGnBu', cbar=True, linewidths=.5, fmt='.2f', annot_kws={"size": 10})
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.xlabel('Criteria', fontsize=14)
        plt.ylabel('Alternatives', fontsize=14)
        plt.show()

    @staticmethod
    def plot_preferences(matrix: pd.DataFrame, title: str = "Preferences"):
        """
        Plot the preferences matrix as a bar chart.

        :param matrix: DataFrame, the preferences matrix to plot.
        :param title: str, the title of the plot.
        """
        ax = matrix.plot(kind='bar', figsize=(14, 7), colormap='Paired', edgecolor='black')
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xlabel('Alternatives', fontsize=14)
        plt.ylabel('Preference Score', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Criteria', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Add annotations
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=10, color='black')
        
        plt.show()

    @staticmethod
    def plot_results(results: pd.DataFrame, title: str = "Results"):
        """
        Plot the results as a bar chart.

        :param results: DataFrame, the results to plot.
        :param title: str, the title of the plot.
        """
        ax = results.plot(kind='bar', x='Ranking', y='Score', figsize=(14, 7), colormap='Set2', edgecolor='black')
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xlabel('Ranking', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Add annotations
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=10, color='black')
        
        plt.show()
