import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class BaseModel:
    """Classe base para implementação de modelos de deep learning com TensorFlow."""

    def __init__(self, config):
        """
        Inicializa a classe base do modelo com as configurações fornecidas.

        Args:
            config (dict): Dicionário contendo configurações de hiperparâmetros.
        """
        self.config = config
        tf.random.set_seed(config['settings']['seed'])

    def compile_model(self, model, optimizer='adam', loss='mse'):
        """
        Compila o modelo com o otimizador e a função de perda especificados.

        Args:
            model (tf.keras.Model): Modelo a ser compilado.
            optimizer (str ou tf.keras.optimizers): Otimizador para o treinamento.
            loss (str ou tf.keras.losses): Função de perda para o treinamento.

        Returns:
            tf.keras.Model: Modelo compilado.
        """
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def generate_images(self, model, n_images_per_row, n_rows, codings_size, save_path=None):
        """
        Gera e exibe (ou salva) um grid de imagens usando o gerador.

        Args:
            model (tf.keras.Model): Modelo gerador para gerar as imagens.
            n_images_per_row (int): Número de imagens por linha.
            n_rows (int): Número de linhas.
            codings_size (int): Tamanho do vetor de entrada para o gerador.
            save_path (str, opcional): Caminho para salvar o grid de imagens. Se None, exibe as imagens.
        """
        noise = tf.random.normal(shape=[n_images_per_row * n_rows, codings_size])
        generated_images = model(noise).numpy()
        generated_images = (generated_images + 1) / 2.0

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_images_per_row, figsize=(n_images_per_row * 2, n_rows * 2))
        for i in range(n_rows):
            for j in range(n_images_per_row):
                ax = axes[i, j]
                ax.imshow(generated_images[i * n_images_per_row + j], cmap="binary")
                ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)