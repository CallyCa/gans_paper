import pandas as pd

# Criar a matriz de decisão
decision_matrix = pd.DataFrame({
    'Criterion 1': [0.2, 0.4, 0.4],
    'Criterion 2': [0.5, 0.3, 0.2],
    'Criterion 3': [0.3, 0.3, 0.4]
}, index=['Alternative 1', 'Alternative 2', 'Alternative 3'])

# Salvar a matriz de decisão como CSV
decision_matrix.to_csv('data/decision_matrix.csv', index=False)

# Criar a matriz de julgamento
judgment_matrix = pd.DataFrame({
    'Criterion 1': [1, 3, 5],
    'Criterion 2': [1/3, 1, 3],
    'Criterion 3': [1/5, 1/3, 1]
}, index=['Criterion 1', 'Criterion 2', 'Criterion 3'])

# Salvar a matriz de julgamento como CSV
judgment_matrix.to_csv('data/judgment_matrix.csv', index=True)

print("CSV files created successfully.")
