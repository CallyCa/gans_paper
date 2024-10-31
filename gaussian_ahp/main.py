# from fractions import Fraction
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from ahp_saaty import AHPSaaty
from ahp_gaussian import AHPGaussian
from decision_helper import DecisionHelper


def ensure_directory_exists(directory):
    """Ensure that a directory exists. If it doesn't, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# def convert_fractions(df):
#     """Convert fraction strings in a DataFrame to numerical values."""
#     for col in df.columns:
#         df[col] = df[col].apply(lambda x: float(Fraction(x)) if isinstance(x, str) and '/' in x else float(x))
#     return df

def main():
    # Load configuration
    config_manager = ConfigManager('settings/ahp_settings.yaml')
    consistency_index = config_manager.get_config('consistency_index')

    # Paths to data files
    decision_matrix_path = 'data/decision_matrix.csv'
    judgment_matrix_path = 'data/judgment_matrix.csv'

    # Check if data files exist
    if not os.path.exists(decision_matrix_path):
        print(f"Error: File '{decision_matrix_path}' not found.")
        return

    if not os.path.exists(judgment_matrix_path):
        print(f"Error: File '{judgment_matrix_path}' not found.")
        return

    # Load data
    decision_matrix = pd.read_csv(decision_matrix_path)
    judgment_matrix = pd.read_csv(judgment_matrix_path, index_col=0)

    # Convert fractions in judgment matrix
    # judgment_matrix = convert_fractions(judgment_matrix)

    # Validate parameters
    valid_columns = list(decision_matrix.columns)
    DecisionHelper.validate_parameters(judgment_matrix, valid_columns)

    # Define objective
    objective = 'max'

    # Ensure the results directory exists
    ensure_directory_exists('results')

    # AHP Saaty
    ahp_saaty = AHPSaaty(judgment_matrix, objective)
    ahp_saaty.visualize_judgment_matrix()
    local_pref_saaty = ahp_saaty.local_preference(decision_matrix)
    ahp_saaty.visualize_local_preference(decision_matrix)
    global_pref_saaty = ahp_saaty.global_preference(decision_matrix)
    ahp_saaty.visualize_global_preference(decision_matrix)
    
    # Save AHP Saaty results
    local_pref_saaty.to_csv('results/local_pref_saaty.csv')
    global_pref_saaty.to_csv('results/global_pref_saaty.csv')

    # Ensure the results directory exists
    ensure_directory_exists('results/figures')

    # AHP Gaussian
    ahp_gaussian = AHPGaussian(decision_matrix, objective)
    ahp_gaussian.visualize_decision_matrix()
    ahp_gaussian.visualize_local_preference()
    global_pref_gaussian = ahp_gaussian.global_preference()
    ahp_gaussian.visualize_global_preference()
    
    # Save AHP Gaussian results
    global_pref_gaussian.to_csv('results/global_pref_gaussian.csv')

if __name__ == "__main__":
    main()