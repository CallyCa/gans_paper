import yaml
import json
import random

class ConfigHandler:
    """Classe para carregar, salvar e gerar novos parâmetros de configuração."""

    @staticmethod
    def load_yaml(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def save_yaml(config, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuração salva em YAML em: {filepath}")
    
    @staticmethod
    def save_json(data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_metrics_to_csv(metrics_df, csv_file_path):
        """Salva as métricas calculadas em um arquivo CSV.

        Args:
            metrics_df (pd.DataFrame): DataFrame contendo as métricas a serem salvas.
            csv_file_path (str): Caminho para salvar o arquivo CSV.
        """
        if metrics_df.empty:
            print("Nenhuma métrica para salvar.")
        else:
            print(f"Métricas registradas para salvar: {metrics_df.columns.tolist()}")
        metrics_df.to_csv(csv_file_path, index=False)
        print(f"Métricas salvas em {csv_file_path}")

    @staticmethod
    def generate_new_params():
        """Gera novos parâmetros de hiperparâmetros para o GAN com maior variabilidade e complexidade no treinamento."""
        return {
            'settings': {
                'seed': random.randint(1, 1000),  
                'codings_size': random.randint(128, 512),  # Aumentado para melhorar a diversidade
                'batch_size': random.choice([64, 128, 256]),  # Definido em múltiplos comuns para estabilidade
                'learning_rate': round(random.uniform(0.00002, 0.0003), 6)  # Abaixo para aprendizado mais suave
            },
            'gan': {
                'generator_layers': [
                    {
                        'units': random.randint(256, 1024),  # Aumentado para melhorar a capacidade do modelo
                        'activation': random.choice(["relu", "leaky_relu"]),
                        'kernel_initializer': random.choice(["he_normal", "lecun_normal"]),
                        'dropout': round(random.uniform(0, 0.3), 2) if random.random() > 0.3 else None,  
                        'batch_normalization': random.choice([True, False])  # Adicionado para estabilidade
                    }
                    for _ in range(random.randint(4, 6))  # Mais camadas para maior complexidade
                ],
                'discriminator_layers': [
                    {
                        'units': random.randint(256, 1024),
                        'activation': random.choice(["relu", "leaky_relu"]),
                        'kernel_initializer': random.choice(["he_normal", "glorot_uniform"]),
                        'dropout': round(random.uniform(0, 0.3), 2) if random.random() > 0.3 else None,  
                        'batch_normalization': random.choice([True, False])  
                    }
                    for _ in range(random.randint(4, 6))
                ],
                'learning_rate': round(random.uniform(0.0001, 0.0003), 6),  # Similar para melhor equilíbrio com o gerador
                'epochs': random.randint(5, 15),  # Aumentado para permitir mais tempo de aprendizado
                'loss_function': "binary_crossentropy",  # Mantido para garantir compatibilidade com a arquitetura GAN
                'optimizer': random.choice(['adam', 'rmsprop']),  
                'beta_1': round(random.uniform(0.5, 0.9), 3),  # Ajuste para estabilidade no Adam
                'beta_2': round(random.uniform(0.9, 0.999), 3),
                'rho': round(random.uniform(0.5, 0.95), 2) if random.choice(['rmsprop']) else None
            }
        }
