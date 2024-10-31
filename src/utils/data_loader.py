import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """Classe para carregar e preparar dados para o treinamento de GANs."""

    @staticmethod
    def load_fashion_mnist():
        """
        Carrega o conjunto de dados Fashion MNIST, normaliza e expande as dimensões para incluir canais de cores.

        Returns:
            tuple: Conjuntos de treino, validação e teste (X_train, X_valid, X_test, y_train, y_valid, y_test).
        """
        (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        # Normalização das imagens para o intervalo [0, 1]
        X_train_full = np.expand_dims(X_train_full.astype(np.float32) / 255, axis=-1)
        X_test = np.expand_dims(X_test.astype(np.float32) / 255, axis=-1)

        # Divisão dos dados para obter conjunto de validação
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    @staticmethod
    def create_tf_datasets(X_train, X_valid, X_test, batch_size=32):
        """
        Converte conjuntos de dados de treino, validação e teste para objetos tf.data.Dataset com shuffling e prefetching.

        Args:
            X_train (np.array): Conjunto de dados de treino.
            X_valid (np.array): Conjunto de dados de validação.
            X_test (np.array): Conjunto de dados de teste.
            batch_size (int, optional): Tamanho do batch para o treinamento. Default é 32.

        Returns:
            tuple: Conjuntos de dados tf.data.Dataset (train_dataset, valid_dataset, test_dataset).
        """
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(X_valid).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)

        return train_dataset, valid_dataset, test_dataset
