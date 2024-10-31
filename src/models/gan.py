# src/models/gan.py

import tensorflow as tf
from matplotlib import pyplot as plt
from .base_model import BaseModel
from src.utils.callbacks import create_checkpoint_callback
from src.constants.constants import IMAGE_SHAPE, IMAGE_CHANNELS, GENERATOR_OUTPUT_ACTIVATION, OPTIMIZERS

class GAN(BaseModel):
    """Classe GAN com configuração modular para fácil ajuste de hiperparâmetros."""

    def __init__(self, config):
        """
        Inicializa o modelo GAN com a configuração fornecida.

        Args:
            config (dict): Dicionário contendo configurações de hiperparâmetros.
        """
        super().__init__(config)
        self.codings_size = self.config['settings']['codings_size']
        self.batch_size = self.config['settings']['batch_size']
        self.generator_layers = self.config['gan']['generator_layers']
        self.discriminator_layers = self.config['gan']['discriminator_layers']
        self.loss_function = self.config['gan']['loss_function']
        self.learning_rate = self.config['gan']['learning_rate']
        self.optimizer_name = self.config['gan']['optimizer']
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

        # Configuração do diretório de checkpoints
        self.checkpoint_dir = config['settings'].get('checkpoint_dir', 'checkpoints')
        self.checkpoint_callback = create_checkpoint_callback(self.checkpoint_dir)

    def build_generator(self):
        # Constrói o gerador com base na configuração
        model = tf.keras.Sequential()
        for layer_cfg in self.generator_layers:
            model.add(tf.keras.layers.Dense(
                units=layer_cfg['units'],
                activation=layer_cfg['activation'],
                kernel_initializer=layer_cfg['kernel_initializer']
            ))
            if layer_cfg.get('dropout') is not None:
                model.add(tf.keras.layers.Dropout(layer_cfg['dropout']))
            if layer_cfg.get('batch_normalization'):
                model.add(tf.keras.layers.BatchNormalization())
        
        # Ajuste final para o formato de saída
        model.add(tf.keras.layers.Dense(IMAGE_SHAPE[0] * IMAGE_SHAPE[1], activation=GENERATOR_OUTPUT_ACTIVATION))
        model.add(tf.keras.layers.Reshape([*IMAGE_SHAPE, IMAGE_CHANNELS]))
        
        return model

    def build_discriminator(self):
        # Constrói o discriminador com base na configuração
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        for layer_cfg in self.discriminator_layers:
            model.add(tf.keras.layers.Dense(
                units=layer_cfg['units'],
                activation=layer_cfg['activation'],
                kernel_initializer=layer_cfg['kernel_initializer']
            ))
            if layer_cfg.get('dropout') is not None:
                model.add(tf.keras.layers.Dropout(layer_cfg['dropout']))
            if layer_cfg.get('batch_normalization'):
                model.add(tf.keras.layers.BatchNormalization())
        optimizer = self.get_optimizer()
        model.compile(loss=self.loss_function, optimizer=optimizer)
        return model

    def build_gan(self):
        # Constrói o modelo GAN combinando gerador e discriminador
        self.discriminator.trainable = False
        model = tf.keras.Sequential([self.generator, self.discriminator])
        optimizer = self.get_optimizer()
        model.compile(loss=self.loss_function, optimizer=optimizer)
        return model

    def get_optimizer(self):
        # Retorna o otimizador com base na configuração
        optimizer_class = OPTIMIZERS.get(self.optimizer_name)
        if not optimizer_class:
            raise ValueError(f"Otimizador desconhecido: {self.optimizer_name}")
        return optimizer_class(learning_rate=self.learning_rate)

    def train_epoch(self, train_dataset):
        """
        Treina o modelo GAN para uma única época e retorna as perdas médias.

        Args:
            train_dataset (tf.data.Dataset): Dataset de treino.

        Returns:
            tuple: (media_discriminator_loss, media_generator_loss)
        """
        epoch_discriminator_loss = 0
        epoch_generator_loss = 0
        batches = 0

        for X_batch in train_dataset:
            batch_size = X_batch.shape[0]  # Obter o tamanho do batch dinamicamente

            # Gerar imagens falsas e combinar com imagens reais
            noise = tf.random.normal(shape=[batch_size, self.codings_size])
            generated_images = self.generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            
            # Ajustar y1 para o número correto de amostras
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

            # Treinar discriminador
            discriminator_loss = self.discriminator.train_on_batch(X_fake_and_real, y1)
            epoch_discriminator_loss += discriminator_loss

            # Treinar GAN com rótulos invertidos
            noise = tf.random.normal(shape=[batch_size, self.codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            generator_loss = self.gan.train_on_batch(noise, y2)
            epoch_generator_loss += generator_loss

            batches += 1
        
        # Calcula a média das perdas por epoch
        avg_discriminator_loss = epoch_discriminator_loss / batches
        avg_generator_loss = epoch_generator_loss / batches

        print(f"Discriminator Loss: {avg_discriminator_loss}, Generator Loss: {avg_generator_loss}")

        return avg_discriminator_loss, avg_generator_loss

    def train_gan(self, train_dataset, n_epochs):
        """
        Treina o modelo GAN no conjunto de dados fornecido por várias épocas e retorna as perdas.

        Args:
            train_dataset (tf.data.Dataset): Dataset de treino.
            n_epochs (int): Número de épocas para o treinamento.

        Returns:
            tuple: (discriminator_losses, generator_losses) listas das perdas de cada época.
        """
        discriminator_losses = []
        generator_losses = []

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            avg_discriminator_loss, avg_generator_loss = self.train_epoch(train_dataset)

            discriminator_losses.append(avg_discriminator_loss)
            generator_losses.append(avg_generator_loss)
            
            print(f"Discriminator Loss: {avg_discriminator_loss}, Generator Loss: {avg_generator_loss}")
            
            self.gan.save_weights(self.checkpoint_callback.filepath.format(epoch=epoch + 1))

        return discriminator_losses, generator_losses